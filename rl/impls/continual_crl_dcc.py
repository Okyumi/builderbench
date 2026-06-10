"""Decomposed Contrastive Critic (DCC) driver for BuilderBench.

Mirror of ``continual_crl.py`` for the DCC algorithm. The shared
infrastructure (env, buffer, evaluator, wandb wiring, padding wrapper,
SLURM auto-resume) is duplicated rather than imported so this file stays
self-contained and we don't risk reaching into the existing CKA-grid
driver and changing its behaviour by accident.

Key differences vs. ``continual_crl.py``:

  * Critic is the decomposed family from ``decomposed_networks.py``:
    z_sa = combine(h_phi(b_shared([s;a])), phi_task([s;a]))
    z_g  = psi(g)                              (variants below)
    score = -|| z_sa - z_g ||_2

  * Additional dynamics loss
    L_dyn = || h_dyn(b_shared([s;a])) - next_cube_positions ||^2

  * Continual rule at task k > 0:
      b_shared, h_phi, h_dyn, psi_shared, (psi_proj)  -> carry forward
      phi_task, (psi_task)                            -> reinitialised

Config flags (see ``Args`` below) cover the four asks:

  --use_dcc                     # turn the whole algorithm on/off
  --dcc_use_dyn                 # dynamic loss on/off
  --dcc_combine_mode add|concat # z_sa = z_shared + z_task  vs  concat
  --dcc_goal_encoder_mode {shared,task_specific,partial_shared,
                           decomposed,projected}

The script is designed for HPC use: one process per SLURM task, auto-
resume from checkpoints, no interactive prompts, idempotent runs.
"""
from __future__ import annotations

import functools
import os
import pickle
import pprint
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState
from wandb_osh.hooks import TriggerWandbSyncHook

from utils.buffer import TrajectoryUniformSamplingQueue
from utils.wrapper import wrap_env
from utils.pad_wrapper import (
    PaddedEnvWrapper, UNIFIED_OBS_DIM, UNIFIED_GOAL_DIM,
    MAX_CUBES, FIXED_OBS_PREFIX, PER_CUBE_OBS,
)
from utils.evaluation import Evaluator
from utils.networks import save_params, load_params

# Prefer RL-local builderbench package over top-level namespace.
RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))
from builderbench.env_utils import make_env

from decomposed_networks import (
    make_decomposed_networks,
    cube_position_indices,
    DecomposedCriticNetworks,
)
import rl_metrics


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclass
class Args:
    # experiment
    agent: str = 'continual_crl_dcc'
    seed: int = 1
    exp_name: str = os.path.basename(__file__)[: -len('.py')]

    # logging and checkpointing
    track: bool = True
    wandb_project_name: str = 'buildstuff'
    wandb_entity: str = 'nyuad_mmvc'
    wandb_mode: str = 'online'
    wandb_dir: str = './'
    wandb_group: str = 'dccablation'
    wandb_name_tag: str = ''

    num_eval_steps: int = 50
    num_reset_steps: int = 1

    save_checkpoint: bool = True

    # environment (overridden per-task by the continual loop)
    env_id: str = 'cube-2-task1'
    num_envs: int = 2048
    num_eval_envs: int = 128
    num_threads: int = 12
    env_early_termination: bool = True
    permutation_invariant_reward: bool = True

    # algorithm (single-task hyperparameters)
    num_timesteps: int = 50_000_000
    rollout_length: int = 64
    batch_size: int = 4096
    sequence_length: int = 512
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    discount: float = 0.99
    entropy_cost: float = 0.1
    logsumexp_cost: float = 0.1
    rep_size: int = 64
    max_replay_size: int = 10000
    min_replay_size: int = 1000

    # continual learning
    task_sequence: str = ('cube-1-task1,cube-1-task2,cube-2-task1,cube-2-task2,'
                          'cube-2-task3,cube-3-task1,cube-3-task3,cube-2-task4,'
                          'cube-2-task5,cube-3-task2,cube-3-task4,cube-3-task5')
    steps_per_task: int = 50_000_000
    base_steps: int = 50_000_000
    checkpoint_dir: str = './continual_checkpoints_dcc'

    # ---- Decomposed Contrastive Critic ----
    # If ``use_dcc=False`` and the user runs this script, it will refuse;
    # baseline runs should use ``continual_crl.py`` with --actor_mode reset
    # --critic_mode reset (or persistent) instead. The flag is here for
    # symmetry with the ablation grid.
    use_dcc: bool = True

    # Dynamic loss: when True, b_shared receives gradients from
    # L_NCE + dcc_dyn_weight * L_dyn; h_dyn receives gradients from L_dyn.
    # When False, h_dyn is built but not optimised (so checkpoint shape
    # stays the same), and L_dyn contributes nothing to b_shared.
    dcc_use_dyn: bool = True
    dcc_dyn_weight: float = 1.0

    # State-action combination: how to fuse z_shared and z_task.
    dcc_combine_mode: str = 'add'   # 'add' | 'concat'

    # Goal-encoder mode: see decomposed_networks.make_decomposed_networks.
    dcc_goal_encoder_mode: str = 'shared'
    #   'shared'         : single psi (full encoder), reuse across tasks.
    #   'task_specific'  : full psi is reset every task.
    #   'partial_shared' : full psi_shared + small psi_task, combined like z_sa.
    #   'decomposed'     : full psi_shared + full psi_task, combined like z_sa.
    #   'projected'      : single psi + learnable projection to z_sa_dim.

    # phi_task / psi_task geometry (only used by the modes that have a
    # task-specific encoder).
    dcc_phi_task_width: int = 256
    dcc_phi_task_depth: int = 4

    # Whether to actually carry forward shared params between tasks
    # (sanity flag for ablation: if False, every group is reset each task,
    # which is effectively the 'reset' baseline with DCC architecture).
    dcc_carry_shared: bool = True

    # representation metrics
    log_rl_metrics: bool = True


# ---------------------------------------------------------------------------
# Actor (identical to continual_crl.py for fair comparison)
# ---------------------------------------------------------------------------


class Actor(nn.Module):
    action_size: int
    norm_type = 'layer_norm'

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, s, g_repr):
        normalize = (lambda x: nn.LayerNorm()(x)) if self.norm_type == 'layer_norm' \
            else (lambda x: x)
        lecun = variance_scaling(1 / 3, 'fan_in', 'uniform')
        bias_init = nn.initializers.zeros

        x = jnp.concatenate([s, g_repr], axis=-1)
        for _ in range(4):
            x = nn.Dense(1024, kernel_init=lecun, bias_init=bias_init)(x)
            x = normalize(x)
            x = nn.swish(x)

        mean = nn.Dense(self.action_size,
                        kernel_init=lecun, bias_init=bias_init,
                        name='mean_head')(x)
        log_std = nn.Dense(self.action_size,
                           kernel_init=lecun, bias_init=bias_init,
                           name='log_std_head')(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std


# ---------------------------------------------------------------------------
# Training state (carry the five critic groups separately + actor + optims)
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class DCCTrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState

    # Five (or six, depending on goal mode) param groups, each with its own
    # optimiser state. Names line up with the decomposed_networks bundle.
    b_shared_params: Any
    b_shared_opt_state: Any
    h_phi_params: Any
    h_phi_opt_state: Any
    h_dyn_params: Any
    h_dyn_opt_state: Any
    phi_task_params: Any
    phi_task_opt_state: Any

    psi_shared_params: Any
    psi_shared_opt_state: Any
    psi_task_params: Any                # may be a zero pytree if unused
    psi_task_opt_state: Any
    psi_proj_params: Any                # may be a zero pytree if unused
    psi_proj_opt_state: Any


class Transition(NamedTuple):
    observation: jnp.ndarray
    achieved_goal: jnp.ndarray
    action: jnp.ndarray
    extras: Any = ()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def make_inference_fn(actor: Actor, decomp: DecomposedCriticNetworks):
    """Like continual_crl.make_inference_fn but uses decomp.apply_g_repr."""
    def make_policy(params, deterministic: bool = False):
        def policy(observations, goals, key_sample):
            g_repr = decomp.apply_g_repr(params['critic'], goals)
            means, log_stds = actor.apply(params['actor'], observations, g_repr)
            if deterministic:
                return nn.tanh(means), {}
            stds = jnp.exp(log_stds)
            return (nn.tanh(means + stds * jax.random.normal(
                key_sample, shape=means.shape, dtype=means.dtype)), {})
        return policy
    return make_policy


# ---------------------------------------------------------------------------
# Checkpoint helpers (mirror continual_crl.py with a DCC suffix so the two
# drivers do not collide on the same checkpoint dir).
# ---------------------------------------------------------------------------


def _ckpt_dir(base_dir, args: Args):
    tag = (f'dcc__{args.dcc_combine_mode}__{args.dcc_goal_encoder_mode}'
           f'__dyn-{int(args.dcc_use_dyn)}__seed{args.seed}')
    return Path(base_dir) / tag


def _ckpt_path(base_dir, task_idx, args: Args):
    return _ckpt_dir(base_dir, args) / f'task_{task_idx:02d}.pkl'


def save_ckpt(base_dir, task_idx, args: Args, data: Dict[str, Any]):
    path = _ckpt_path(base_dir, task_idx, args)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'  Saved checkpoint to {path}', flush=True)


def load_ckpt(base_dir, task_idx, args: Args):
    path = _ckpt_path(base_dir, task_idx, args)
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def auto_resume(base_dir, num_tasks, args: Args) -> int:
    """Return the index of the last fully-completed task (or -1)."""
    last = -1
    for probe in range(num_tasks):
        if _ckpt_path(base_dir, probe, args).exists():
            last = probe
        else:
            break
    return last


# ---------------------------------------------------------------------------
# Task-id parsing
# ---------------------------------------------------------------------------


def _parse_num_cubes(task_id: str) -> int:
    """'cube-2-task3' -> 2"""
    parts = task_id.split('-')
    return int(parts[1])


# ---------------------------------------------------------------------------
# Single-task training loop
# ---------------------------------------------------------------------------


def train_single_task(
    args: Args,
    task_idx: int,
    task_id: str,
    actor_state: TrainState,
    ts_groups: Dict[str, Tuple[Any, Any]],     # name -> (params, opt_state)
    actor: Actor,
    decomp: DecomposedCriticNetworks,
    key: jax.Array,
):
    """Inner CRL loop for one task. Returns updated (actor_state, groups)."""
    args.env_id = task_id

    # Use base_steps for task 0, steps_per_task for the rest.
    task_steps = args.base_steps if task_idx == 0 else args.steps_per_task
    num_training_step = task_steps // (args.num_envs * args.rollout_length)
    num_training_steps_per_eval = max(num_training_step // args.num_eval_steps, 1)
    metrics_every = max(num_training_step // args.num_eval_steps, 1)

    # ---- environment -------------------------------------------------------
    key, env_key, eval_key, buffer_key = jax.random.split(key, 4)
    num_cubes = _parse_num_cubes(task_id)

    env_class, default_config = make_env(args)
    raw_env = env_class(num_envs=args.num_envs,
                        num_threads=args.num_threads,
                        config=default_config)
    env = PaddedEnvWrapper(wrap_env(raw_env, default_config.episode_length),
                           actual_cubes=num_cubes)
    raw_eval_env = env_class(num_envs=args.num_eval_envs,
                             num_threads=args.num_threads,
                             config=default_config)
    eval_env = PaddedEnvWrapper(wrap_env(raw_eval_env, default_config.episode_length),
                                actual_cubes=num_cubes)
    episode_length = default_config.episode_length

    reset_fn = jax.jit(env.reset)
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = reset_fn(env_keys)
    obs_size = UNIFIED_OBS_DIM
    action_size = env.action_size
    goal_size = UNIFIED_GOAL_DIM

    log_data_metric_keys = []
    for k in ('obj_reached_once', 'obj_lifted', 'obj_moved'):
        if k in env_state.metrics.keys():
            log_data_metric_keys.append(k)
    log_data_metric_keys = tuple(log_data_metric_keys)

    # ---- optimisers --------------------------------------------------------
    actor_opt = optax.adam(learning_rate=args.actor_learning_rate)
    critic_opt = optax.adam(learning_rate=args.critic_learning_rate)

    # ---- training state ----------------------------------------------------
    training_state = DCCTrainingState(
        env_steps=jnp.zeros((), dtype=jnp.float64),
        gradient_steps=jnp.zeros((), dtype=jnp.float64),
        actor_state=actor_state,
        b_shared_params=ts_groups['b_shared'][0],
        b_shared_opt_state=ts_groups['b_shared'][1],
        h_phi_params=ts_groups['h_phi'][0],
        h_phi_opt_state=ts_groups['h_phi'][1],
        h_dyn_params=ts_groups['h_dyn'][0],
        h_dyn_opt_state=ts_groups['h_dyn'][1],
        phi_task_params=ts_groups['phi_task'][0],
        phi_task_opt_state=ts_groups['phi_task'][1],
        psi_shared_params=ts_groups['psi_shared'][0],
        psi_shared_opt_state=ts_groups['psi_shared'][1],
        psi_task_params=ts_groups['psi_task'][0],
        psi_task_opt_state=ts_groups['psi_task'][1],
        psi_proj_params=ts_groups['psi_proj'][0],
        psi_proj_opt_state=ts_groups['psi_proj'][1],
    )

    # ---- replay buffer -----------------------------------------------------
    dummy_obs = jnp.zeros((obs_size,))
    dummy_goal = jnp.zeros((goal_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(
        observation=dummy_obs,
        achieved_goal=dummy_goal,
        action=dummy_action,
        extras={'state_extras': {'traj_id': 0.0}},
    )

    def jit_wrap(buffer):
        buffer.insert = jax.jit(buffer.insert)
        buffer.sample = jax.jit(buffer.sample)
        return buffer

    replay_buffer = jit_wrap(TrajectoryUniformSamplingQueue(
        max_replay_size=args.max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=args.batch_size,
        num_envs=args.num_envs,
        sequence_length=args.sequence_length + 1,
    ))
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    # ---- evaluator ---------------------------------------------------------
    make_policy = make_inference_fn(actor, decomp)
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=episode_length,
        key=eval_key,
    )

    # ---- helpers to pack a critic params dict ------------------------------

    def _pack_critic(ts: DCCTrainingState):
        d = dict(
            b_shared=ts.b_shared_params,
            h_phi=ts.h_phi_params,
            h_dyn=ts.h_dyn_params,
            phi_task=ts.phi_task_params,
            psi_shared=ts.psi_shared_params,
        )
        if decomp.psi_task is not None:
            d['psi_task'] = ts.psi_task_params
        if decomp.psi_proj is not None:
            d['psi_proj'] = ts.psi_proj_params
        return d

    # ---- actor / data-collect step ----------------------------------------

    def actor_step(training_state, env, env_state, key, extra_fields, metrics_fields):
        critic_p = _pack_critic(training_state)
        g_repr = decomp.apply_g_repr(critic_p, env_state.info['target_goal'])

        means, log_stds = actor.apply(training_state.actor_state.params,
                                       env_state.obs, g_repr)
        stds = jnp.exp(log_stds)
        actions = nn.tanh(means + stds * jax.random.normal(
            key, shape=means.shape, dtype=means.dtype))

        nstate = env.pre_step(env_state, actions)
        physics_state, sensor_data = env.step(nstate, actions)
        nstate = env.post_step(nstate, physics_state, sensor_data)

        state_extras = {x: nstate.info[x] for x in extra_fields}
        metrics = {x: nstate.metrics[x] for x in metrics_fields}

        return training_state, nstate, Transition(
            observation=env_state.obs,
            achieved_goal=env_state.info['achieved_goal'],
            action=actions,
            extras={'state_extras': state_extras},
        ), metrics

    @jax.jit
    def data_collect_step(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            ts, es, k = carry
            k, nk = jax.random.split(k)
            ts, es, tr, m = actor_step(ts, env, es, k,
                                       extra_fields=('traj_id',),
                                       metrics_fields=log_data_metric_keys)
            return (ts, es, nk), (tr, m)

        (training_state, env_state, _), (data, metrics) = jax.lax.scan(
            f, (training_state, env_state, key), (), length=args.rollout_length)
        training_state = training_state.replace(
            env_steps=training_state.env_steps + (args.num_envs * args.rollout_length),
        )
        buffer_state = replay_buffer.insert(buffer_state, data)
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            ts, es, bs, k = carry
            k, nk = jax.random.split(k)
            ts, es, bs, _ = data_collect_step(ts, es, bs, k)
            return (ts, es, bs, nk), ()
        return jax.lax.scan(
            f, (training_state, env_state, buffer_state, key), (),
            length=int(np.ceil(args.min_replay_size / args.rollout_length)))[0]

    # ---- losses ------------------------------------------------------------

    # Cube-position indices into the padded observation. h_dyn predicts these
    # for the NEXT state.
    cube_idx = jnp.array(cube_position_indices(
        FIXED_OBS_PREFIX, PER_CUBE_OBS, MAX_CUBES), dtype=jnp.int32)

    def contrastive_loss(critic_p, state, action, goal):
        z_sa = decomp.apply_sa_repr(critic_p, state, action)   # (B, z)
        z_g = decomp.apply_g_repr(critic_p, goal)              # (B, z)
        logits = -jnp.sqrt(
            jnp.sum((z_sa[:, None, :] - z_g[None, :, :]) ** 2, axis=-1) + 1e-6)
        c_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
        lse = jax.nn.logsumexp(logits + 1e-6, axis=1)
        c_loss = c_loss + args.logsumexp_cost * jnp.mean(lse ** 2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
        aux = dict(
            categorical_accuracy=jnp.mean(correct.astype(jnp.float32)),
            logits_pos=logits_pos,
            logits_neg=logits_neg,
            logsumexp=jnp.mean(lse),
        )
        return c_loss, aux

    def dyn_loss(critic_p, state, action, next_state):
        pred = decomp.apply_h_dyn(critic_p, state, action)
        target = next_state[..., cube_idx]
        mse = jnp.mean((pred - target) ** 2)
        return mse

    # ---- critic update -----------------------------------------------------

    @jax.jit
    def update_critic(transitions, ts: DCCTrainingState, key):
        # ``transitions`` is a flattened CRL batch: ``observation`` = s,
        # ``extras['future_goal']`` = g, ``extras['next_observation']`` = s'
        # (the latter from ``flatten_crl_dcc_fn`` in learn_step).
        state = transitions.observation
        action = transitions.action
        goal = transitions.extras['future_goal']
        next_state = transitions.extras['next_observation']

        def crit_loss_fn(packed):
            # ``packed`` is a flat dict that we pass to apply_sa/apply_g.
            cl, aux = contrastive_loss(packed, state, action, goal)
            total = cl
            if args.dcc_use_dyn:
                dl = dyn_loss(packed, state, action, next_state)
                total = total + args.dcc_dyn_weight * dl
                aux = {**aux, 'dyn_mse': dl, 'critic_total_loss': total,
                       'critic_infonce_loss': cl}
            else:
                aux = {**aux, 'dyn_mse': jnp.zeros(()),
                       'critic_total_loss': total,
                       'critic_infonce_loss': cl}
            return total, aux

        packed = _pack_critic(ts)
        (loss, aux), grads = jax.value_and_grad(crit_loss_fn, has_aux=True)(packed)

        # Apply per-group updates with each group's own optimiser state.
        def _apply(opt, p, g, opt_state):
            upd, new_opt_state = opt.update(g, opt_state, p)
            return optax.apply_updates(p, upd), new_opt_state

        new_b, new_b_opt = _apply(critic_opt,
                                  ts.b_shared_params, grads['b_shared'],
                                  ts.b_shared_opt_state)
        new_phi, new_phi_opt = _apply(critic_opt,
                                      ts.h_phi_params, grads['h_phi'],
                                      ts.h_phi_opt_state)
        new_task, new_task_opt = _apply(critic_opt,
                                        ts.phi_task_params, grads['phi_task'],
                                        ts.phi_task_opt_state)
        new_psi, new_psi_opt = _apply(critic_opt,
                                      ts.psi_shared_params, grads['psi_shared'],
                                      ts.psi_shared_opt_state)
        if args.dcc_use_dyn:
            new_dyn, new_dyn_opt = _apply(critic_opt,
                                          ts.h_dyn_params, grads['h_dyn'],
                                          ts.h_dyn_opt_state)
        else:
            new_dyn, new_dyn_opt = ts.h_dyn_params, ts.h_dyn_opt_state

        new_psi_t = ts.psi_task_params
        new_psi_t_opt = ts.psi_task_opt_state
        if decomp.psi_task is not None:
            new_psi_t, new_psi_t_opt = _apply(
                critic_opt, ts.psi_task_params, grads['psi_task'],
                ts.psi_task_opt_state)

        new_psi_p = ts.psi_proj_params
        new_psi_p_opt = ts.psi_proj_opt_state
        if decomp.psi_proj is not None:
            new_psi_p, new_psi_p_opt = _apply(
                critic_opt, ts.psi_proj_params, grads['psi_proj'],
                ts.psi_proj_opt_state)

        ts = ts.replace(
            b_shared_params=new_b, b_shared_opt_state=new_b_opt,
            h_phi_params=new_phi, h_phi_opt_state=new_phi_opt,
            h_dyn_params=new_dyn, h_dyn_opt_state=new_dyn_opt,
            phi_task_params=new_task, phi_task_opt_state=new_task_opt,
            psi_shared_params=new_psi, psi_shared_opt_state=new_psi_opt,
            psi_task_params=new_psi_t, psi_task_opt_state=new_psi_t_opt,
            psi_proj_params=new_psi_p, psi_proj_opt_state=new_psi_p_opt,
        )
        return ts, {**aux, 'critic_loss': loss}

    # ---- actor update ------------------------------------------------------

    @jax.jit
    def update_actor(transitions, ts: DCCTrainingState, key):
        def actor_loss(actor_params, critic_p, transitions, key):
            state = transitions.observation
            goal = transitions.extras['future_goal']
            critic_p_sg = jax.lax.stop_gradient(critic_p)

            g_repr = decomp.apply_g_repr(critic_p_sg, goal)
            means, log_stds = actor.apply(actor_params, state, g_repr)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(
                key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)

            z_sa = decomp.apply_sa_repr(critic_p_sg, state, action)
            qf_pi = -jnp.sqrt(jnp.sum((z_sa - g_repr) ** 2, axis=-1) + 1e-6)
            return jnp.mean(args.entropy_cost * log_prob - qf_pi), log_prob

        critic_p = _pack_critic(ts)
        (a_loss, log_prob), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            ts.actor_state.params, critic_p, transitions, key)
        new_actor_state = ts.actor_state.apply_gradients(grads=grads)
        ts = ts.replace(actor_state=new_actor_state)
        return ts, {'actor_loss': a_loss, 'sample_entropy': -log_prob}

    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, ka, kc = jax.random.split(key, 3)
        training_state, ma = update_actor(transitions, training_state, ka)
        training_state, mc = update_critic(transitions, training_state, kc)
        training_state = training_state.replace(
            gradient_steps=training_state.gradient_steps + 1)
        return (training_state, key), {**ma, **mc}

    @jax.jit
    def learn_step(training_state, buffer_state, key):
        ek, sk, tk = jax.random.split(key, 3)
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        batch_keys = jax.random.split(sk, transitions.observation.shape[0])
        transitions = jax.vmap(
            TrajectoryUniformSamplingQueue.flatten_crl_dcc_fn,
            in_axes=(None, 0, 0),
        )((args.discount,), transitions, batch_keys)

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order='F'),
            transitions,
        )
        permutation = jax.random.permutation(ek, len(transitions.action))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

        (training_state, _), metrics = jax.lax.scan(
            sgd_step, (training_state, tk), transitions)
        return training_state, buffer_state, metrics

    # ---- prefill -----------------------------------------------------------
    print(f'  Prefilling replay buffer...', flush=True)
    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key)

    # ---- training loop -----------------------------------------------------
    next_metrics_frequent = metrics_every if args.log_rl_metrics else float('inf')
    next_metrics_occasional = 5 * metrics_every if args.log_rl_metrics else float('inf')
    metrics = None
    xt = time.time()

    if args.save_checkpoint:
        task_save_path = (Path(args.wandb_dir) /
                          f'checkpoints/{args.exp_name}/task{task_idx}_{task_id}/')
        os.makedirs(task_save_path, exist_ok=True)

    print(f'  Training for {num_training_step} steps...', flush=True)
    for ts in range(1, num_training_step + 1):
        key, k_sgd, k_roll = jax.random.split(key, 3)
        training_state, env_state, buffer_state, data_metrics = data_collect_step(
            training_state, env_state, buffer_state, k_roll)
        training_state, buffer_state, training_metrics = learn_step(
            training_state, buffer_state, k_sgd)
        merged = data_metrics | training_metrics
        if metrics is None:
            metrics = merged
        else:
            metrics = jax.tree_util.tree_map(lambda x, y: x + y, metrics, merged)

        if args.log_rl_metrics and ts >= next_metrics_frequent:
            if ts >= next_metrics_occasional:
                rl_level = 'occasional'
                next_metrics_occasional = ts + 5 * metrics_every
                next_metrics_frequent = ts + metrics_every
            else:
                rl_level = 'frequent'
                next_metrics_frequent = ts + metrics_every

            try:
                critic_p = _pack_critic(training_state)
                _bs, sample_transitions = replay_buffer.sample(buffer_state)
                obs_sample = sample_transitions.observation[0, :args.batch_size]
                act_sample = sample_transitions.action[0, :args.batch_size]
                goal_sample = sample_transitions.achieved_goal[0, :args.batch_size]

                m = rl_metrics.compute_all_metrics_dcc(
                    decomp,
                    training_state.actor_state.params,
                    critic_p,
                    obs_sample, act_sample, goal_sample,
                    action_dim=action_size,
                    level=rl_level,
                )
                if args.track:
                    wandb_m = {f'rl_metrics/{k}': v for k, v in m.items()}
                    wandb_m['rl_metrics/env_steps'] = float(training_state.env_steps)
                    wandb.log(wandb_m)
            except Exception as e:
                print(f'  [rl_metrics] Warning: {e}', flush=True)

        if ts % num_training_steps_per_eval == 0:
            metrics = jax.tree_util.tree_map(
                lambda x: x / num_training_steps_per_eval, metrics)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            sps = (num_training_steps_per_eval * args.num_envs *
                   args.rollout_length) / (time.time() - xt)

            eval_actor_p = training_state.actor_state.params
            eval_critic_p = _pack_critic(training_state)
            metrics = {
                'training/sps': sps,
                'training/env_steps': training_state.env_steps,
                'training/task_idx': task_idx,
                **{f'training/{name}': value for name, value in metrics.items()},
                'buffer_current_size': replay_buffer.size(buffer_state),
            }
            metrics = evaluator.run_evaluation(
                policy_params={'actor': eval_actor_p, 'critic': eval_critic_p},
                training_metrics=metrics,
            )
            pprint.pprint(metrics)
            if args.track:
                wandb.log(metrics)
            metrics = None
            xt = time.time()

    print(f'  Task {task_idx} [{task_id}] training complete.', flush=True)

    new_groups = dict(
        b_shared=(training_state.b_shared_params, training_state.b_shared_opt_state),
        h_phi=(training_state.h_phi_params, training_state.h_phi_opt_state),
        h_dyn=(training_state.h_dyn_params, training_state.h_dyn_opt_state),
        phi_task=(training_state.phi_task_params, training_state.phi_task_opt_state),
        psi_shared=(training_state.psi_shared_params, training_state.psi_shared_opt_state),
        psi_task=(training_state.psi_task_params, training_state.psi_task_opt_state),
        psi_proj=(training_state.psi_proj_params, training_state.psi_proj_opt_state),
    )
    return training_state.actor_state, new_groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _maybe_init(init_fn, key):
    return None if init_fn is None else init_fn(key)


def main(args: Args):
    tasks = [t.strip() for t in args.task_sequence.split(',')]
    num_tasks = len(tasks)

    print('=' * 60)
    print(f'DCC Continual CRL — {num_tasks} tasks')
    print(f'Sequence: {tasks}')
    print(f'use_dyn={args.dcc_use_dyn} weight={args.dcc_dyn_weight}  '
          f'combine={args.dcc_combine_mode}  goal_mode={args.dcc_goal_encoder_mode}')
    print(f'Steps per task: {args.steps_per_task} | Base steps: {args.base_steps}')
    print('=' * 60)

    args.exp_name = (
        f"{args.wandb_name_tag + '__' if args.wandb_name_tag else ''}"
        f'dcc__{args.dcc_combine_mode}_{args.dcc_goal_encoder_mode}'
        f"__dyn-{int(args.dcc_use_dyn)}__{args.seed}__{int(time.time())}"
    )
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    last_completed = auto_resume(args.checkpoint_dir, num_tasks, args)
    start_task = last_completed + 1
    if start_task >= num_tasks:
        print(f'  All {num_tasks} tasks already completed. Nothing to do.')
        return

    # ---- networks ----------------------------------------------------------
    args.env_id = tasks[0]
    env_class, default_config = make_env(args)
    probe_env = wrap_env(
        env_class(num_envs=1, num_threads=1, config=default_config),
        default_config.episode_length)
    action_size = probe_env.action_size

    actor = Actor(action_size=action_size)
    decomp = make_decomposed_networks(
        obs_size=UNIFIED_OBS_DIM,
        action_size=action_size,
        goal_size=UNIFIED_GOAL_DIM,
        rep_size=args.rep_size,
        phi_task_width=args.dcc_phi_task_width,
        phi_task_depth=args.dcc_phi_task_depth,
        combine_mode=args.dcc_combine_mode,
        goal_encoder_mode=args.dcc_goal_encoder_mode,
        use_dyn=args.dcc_use_dyn,
        dyn_target_dim=UNIFIED_GOAL_DIM,    # 9 cube position coords
    )

    # ---- fresh params ------------------------------------------------------
    def fresh_actor(k):
        return actor.init(k, np.ones([1, UNIFIED_OBS_DIM]),
                          np.ones([1, decomp.z_g_dim]))

    def _fresh_groups(rng):
        keys = jax.random.split(rng, 7)
        psi_task = _maybe_init(decomp.init_psi_task, keys[5])
        psi_proj = _maybe_init(decomp.init_psi_proj, keys[6])
        return dict(
            b_shared=decomp.init_b_shared(keys[0]),
            h_phi=decomp.init_h_phi(keys[1]),
            h_dyn=decomp.init_h_dyn(keys[2]),
            phi_task=decomp.init_phi_task(keys[3]),
            psi_shared=decomp.init_psi_shared(keys[4]),
            psi_task=psi_task,
            psi_proj=psi_proj,
        )

    # ---- restore ----------------------------------------------------------
    actor_opt = optax.adam(learning_rate=args.actor_learning_rate)
    critic_opt = optax.adam(learning_rate=args.critic_learning_rate)

    key, k_actor, k_fresh = jax.random.split(key, 3)
    actor_params = fresh_actor(k_actor)
    actor_state = TrainState.create(apply_fn=actor.apply,
                                    params=actor_params, tx=actor_opt)

    fresh = _fresh_groups(k_fresh)
    # Group store: name -> (params, opt_state). For optional groups
    # (psi_task, psi_proj) where the module is None we keep a None param.
    def _opt_init(p):
        return critic_opt.init(p) if p is not None else None

    groups = {n: (fresh[n], _opt_init(fresh[n])) for n in fresh}

    if start_task > 0:
        ckpt = load_ckpt(args.checkpoint_dir, start_task - 1, args)
        if ckpt is not None:
            actor_state = ckpt['actor_state']
            groups = ckpt['critic_groups']
        else:
            print('  Could not load checkpoint; starting from scratch.')
            start_task = 0

    # ---- task loop ---------------------------------------------------------
    for task_idx in range(start_task, num_tasks):
        task_id = tasks[task_idx]
        print(f'\n{"=" * 60}\nTask {task_idx}/{num_tasks - 1}: {task_id}\n{"=" * 60}')

        key, task_key, reinit_key = jax.random.split(key, 3)

        if task_idx > 0:
            # Continual rule:
            #   shared groups (b_shared, h_phi, h_dyn, psi_shared, psi_proj)
            #     carry over when dcc_carry_shared=True (default).
            #   task-specific groups (phi_task, psi_task) are always reset.
            keys = jax.random.split(reinit_key, 7)
            new_phi_task = decomp.init_phi_task(keys[3])
            new_phi_task_opt = critic_opt.init(new_phi_task)
            groups['phi_task'] = (new_phi_task, new_phi_task_opt)

            if decomp.psi_task is not None:
                new_psi_task = decomp.init_psi_task(keys[5])
                new_psi_task_opt = critic_opt.init(new_psi_task)
                groups['psi_task'] = (new_psi_task, new_psi_task_opt)

            if not args.dcc_carry_shared:
                # Reset everything else too (sanity / ablation).
                new_b = decomp.init_b_shared(keys[0])
                new_h_phi = decomp.init_h_phi(keys[1])
                new_h_dyn = decomp.init_h_dyn(keys[2])
                new_psi = decomp.init_psi_shared(keys[4])
                groups['b_shared'] = (new_b, critic_opt.init(new_b))
                groups['h_phi'] = (new_h_phi, critic_opt.init(new_h_phi))
                groups['h_dyn'] = (new_h_dyn, critic_opt.init(new_h_dyn))
                groups['psi_shared'] = (new_psi, critic_opt.init(new_psi))
                if decomp.psi_proj is not None:
                    new_psi_proj = decomp.init_psi_proj(keys[6])
                    groups['psi_proj'] = (new_psi_proj,
                                           critic_opt.init(new_psi_proj))

            # Always reset the actor (Sawyer-side R/R schedule was the
            # best-performing actor mode in the workshop paper; we keep
            # the same default here).
            key, k_actor = jax.random.split(key)
            actor_params = fresh_actor(k_actor)
            actor_state = TrainState.create(apply_fn=actor.apply,
                                            params=actor_params, tx=actor_opt)

        actor_state, groups = train_single_task(
            args, task_idx, task_id, actor_state, groups, actor, decomp,
            task_key,
        )

        if args.save_checkpoint:
            save_ckpt(args.checkpoint_dir, task_idx, args, dict(
                actor_state=actor_state,
                critic_groups=groups,
            ))


if __name__ == '__main__':
    args = tyro.cli(Args)
    if not args.use_dcc:
        raise SystemExit(
            'continual_crl_dcc.py: use_dcc=False. Run continual_crl.py for '
            'baseline R/P/C grids instead.'
        )
    if args.track:
        wandb.init(project=args.wandb_project_name,
                   entity=args.wandb_entity,
                   mode=args.wandb_mode,
                   dir=args.wandb_dir,
                   group=args.wandb_group,
                   name=f'{args.wandb_name_tag}__{int(time.time())}',
                   config=vars(args))
    main(args)
