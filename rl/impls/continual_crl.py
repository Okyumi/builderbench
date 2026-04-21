"""Continual Contrastive RL training loop.

Wraps the single-task CRL agent (crl.py) with a sequential continual learning
driver.  For each task in the sequence the inner loop collects data and trains;
between tasks, actor and critic state are transferred according to actor_mode
and critic_mode.

Supports three transfer modes for each of actor and critic:
  - reset:      reinitialise from scratch every task.
  - persistent: carry forward the trained params.
  - cka:        CKA-RL decomposition: θ' = θ_base + Σ α_j v_j + v_k.

Flags that control CKA behaviour:
  --adapt_heads_only  (default True)
      When True, only the actor head (mean + log_std Dense layers) are
      decomposed into pool vectors.  After each task the body part of v_k
      is folded into θ_base so the encoder evolves, while the pool stores
      only head deltas.  This matches CKA-RL.

  --encoder_from_base  (default False)
      When True AND adapt_heads_only is True, body gradients are zeroed
      during training so the encoder stays exactly at the base-task values.
      CKA-RL default is False: the encoder receives gradients and evolves.

Usage (default – reset both networks each task):
  python continual_crl.py --task_sequence cube-2-task1,cube-2-task2,cube-2-task3

CKA actor, persistent critic:
  python continual_crl.py --actor_mode cka --critic_mode persistent

Quick test (2 tasks, 1M steps each):
  python continual_crl.py --task_sequence cube-2-task1,cube-2-task2 \\
      --steps_per_task 1000000
"""
import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

import pickle
import re
import time
import sys
import tyro
import numpy as np
import functools
import pprint
import wandb
import wandb_osh

import jax
import flax
import optax
import distrax
import flax.linen as nn
import jax.numpy as jnp

from mujoco import rollout
from pathlib import Path
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState
from dataclasses import dataclass
from typing import NamedTuple, Optional, Dict, Any
from wandb_osh.hooks import TriggerWandbSyncHook

from utils.buffer import TrajectoryUniformSamplingQueue
from utils.wrapper import wrap_env
from utils.pad_wrapper import PaddedEnvWrapper, UNIFIED_OBS_DIM, UNIFIED_GOAL_DIM
from utils.evaluation import Evaluator
from utils.networks import MLP, save_params, load_params

# Prefer RL-local builderbench package over top-level package namespace.
RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))
from builderbench.env_utils import make_env
from knowledge_pool import KnowledgePool, pytree_zeros_like
import rl_metrics


# ---------------------------------------------------------------------------
# Flax Actor head/body detection
# ---------------------------------------------------------------------------
# The Flax Actor uses @nn.compact with auto-naming:
#   Dense_0 .. Dense_3 (1024-width)  → body
#   LayerNorm_0 .. LayerNorm_3       → body
#   Dense_4 (mean projection)        → head
#   Dense_5 (log_std projection)     → head
#
# A param leaf's *path* (from tree_flatten_with_path) looks like:
#   ('params', 'Dense_4', 'kernel')  or  ('params', 'Dense_5', 'bias')
# We detect head leaves by checking for 'Dense_4' or 'Dense_5' in the path.

ACTOR_HEAD_LAYER_NAMES = ('Dense_4', 'Dense_5')


def _is_actor_head_leaf(path) -> bool:
    """Return True if a Flax param path belongs to the actor head."""
    path_str = '/'.join(str(p) for p in path)
    return any(name in path_str for name in ACTOR_HEAD_LAYER_NAMES)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    # experiment
    agent: str = "continual_crl"
    seed: int = 1
    exp_name: str = os.path.basename(__file__)[: -len(".py")]

    # logging and checkpointing
    track: bool = True
    wandb_project_name: str = "buildstuff"
    wandb_entity: str = 'nyuad_mmvc'
    wandb_mode: str = 'online'
    wandb_dir: str = './'
    wandb_group: str = 'test2'
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
    task_sequence: str = 'cube-1-task1,cube-1-task2,cube-2-task1,cube-2-task2,cube-2-task3,cube-3-task1,cube-3-task3,cube-2-task4,cube-2-task5,cube-3-task2,cube-3-task4,cube-3-task5'
    actor_mode: str = 'reset'       # 'reset', 'persistent', 'cka'
    critic_mode: str = 'reset'      # 'reset', 'persistent', 'cka'
    steps_per_task: int = 50_000_000
    base_steps: int = 50_000_000    # env steps for base task (task 0)
    checkpoint_dir: str = './continual_checkpoints'
    k_max: int = 10                 # knowledge pool capacity
    alpha_scale: float = 1.0        # softmax temperature for pool blending

    # CKA behaviour
    adapt_heads_only: bool = True   # only store head deltas in pool (CKA-RL)
    encoder_from_base: bool = False # freeze encoder body during CKA training

    # evaluation
    eval_episodes: int = 128        # envs used for cross-task evaluation

    # representation metrics
    log_rl_metrics: bool = True     # log weight norms, feature rank, NRC, etc.


# ---------------------------------------------------------------------------
# Network definitions (identical to crl.py)
# ---------------------------------------------------------------------------

class SA_encoder(nn.Module):
    rep_size: int
    norm_type = "layer_norm"

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = jnp.concatenate([s, a], axis=-1)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.rep_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class G_encoder(nn.Module):
    rep_size: int
    norm_type = "layer_norm"

    @nn.compact
    def __call__(self, g: jnp.ndarray):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(g)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.rep_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, s, g_repr):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = jnp.concatenate([s, g_repr], axis=-1)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)  # Dense_0
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)  # Dense_1
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)  # Dense_2
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)  # Dense_3
        x = normalize(x)
        x = nn.swish(x)

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)     # Dense_4 (HEAD)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)   # Dense_5 (HEAD)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

@flax.struct.dataclass
class CRLTrainingState:
    """Contains training state for the learner."""
    env_steps: np.ndarray
    gradient_steps: np.ndarray
    actor_state: TrainState
    critic_state: TrainState


class Transition(NamedTuple):
    """Container for a transition."""
    observation: jnp.ndarray
    achieved_goal: jnp.ndarray
    action: jnp.ndarray
    extras: jnp.ndarray = ()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def make_inference_fn(policy_network, g_encoder_network):
    """Creates params and inference function for the CRL agent."""
    def make_policy(params, deterministic: bool = False):
        def policy(observations, goals, key_sample):
            goals = g_encoder_network.apply(params['g_encoder'], goals)
            means, log_stds = policy_network.apply(params['actor'], observations, goals)

            if deterministic:
                return nn.tanh(means), {}

            stds = jnp.exp(log_stds)
            raw_actions = means + stds * jax.random.normal(key_sample, shape=means.shape, dtype=means.dtype)
            postprocessed_actions = nn.tanh(raw_actions)

            log_prob = jax.scipy.stats.norm.logpdf(raw_actions, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(postprocessed_actions)) + 1e-6)
            log_prob = log_prob.sum(-1)

            return postprocessed_actions, {
                'log_prob': log_prob,
                'raw_action': raw_actions,
            }
        return policy
    return make_policy


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def _ckpt_dir(base_dir, actor_mode, critic_mode, adapt_heads_only, seed):
    """Checkpoint directory keyed by ablation config."""
    config_key = f'actor_{actor_mode}_critic_{critic_mode}_heads_{adapt_heads_only}'
    return os.path.join(base_dir, config_key, f'seed_{seed}')


def _ckpt_path(base_dir, task_idx, actor_mode, critic_mode, adapt_heads_only, seed):
    return os.path.join(
        _ckpt_dir(base_dir, actor_mode, critic_mode, adapt_heads_only, seed),
        f'task_{task_idx}.pkl')


def save_ckpt(base_dir, task_idx, actor_mode, critic_mode, adapt_heads_only, seed, data):
    path = _ckpt_path(base_dir, task_idx, actor_mode, critic_mode, adapt_heads_only, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert JAX arrays to numpy for robust pickling
    data_np = jax.tree_util.tree_map(
        lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, data)
    with open(path, 'wb') as f:
        pickle.dump(data_np, f)
    print(f'  [ckpt] Saved -> {path}', flush=True)


def load_ckpt(base_dir, task_idx, actor_mode, critic_mode, adapt_heads_only, seed):
    path = _ckpt_path(base_dir, task_idx, actor_mode, critic_mode, adapt_heads_only, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No checkpoint at {path}. Check that a previous run used the '
            f'same config (actor_mode={actor_mode}, critic_mode={critic_mode}, '
            f'adapt_heads_only={adapt_heads_only}, seed={seed}).')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # Convert back to JAX arrays
    data_jax = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, data)
    print(f'  [ckpt] Loaded <- {path}', flush=True)
    return data_jax


def auto_resume(base_dir, num_tasks, actor_mode, critic_mode, adapt_heads_only, seed):
    """Find the highest completed task index, or -1 if none."""
    for probe_idx in range(num_tasks - 1, -1, -1):
        path = _ckpt_path(base_dir, probe_idx, actor_mode, critic_mode,
                          adapt_heads_only, seed)
        if os.path.exists(path):
            print(f'  [auto-resume] Found checkpoint for task {probe_idx} '
                  f'-> resuming from task {probe_idx + 1}.', flush=True)
            return probe_idx
    print(f'  [auto-resume] No existing checkpoints. Starting from task 0.',
          flush=True)
    return -1


# ---------------------------------------------------------------------------
# Cross-task evaluation
# ---------------------------------------------------------------------------

def _parse_num_cubes(task_id: str) -> int:
    """Extract cube count from task ID like 'cube-2-task1'."""
    m = re.search(r"cube-(\d+)", task_id)
    assert m is not None, f'Cannot parse cube count from task_id={task_id!r}'
    return int(m.group(1))


def evaluate_on_task(task_id, args, actor_params, critic_params,
                     actor, g_encoder, key):
    """Run evaluation on a single task and return metrics dict.

    ``actor_params`` must be the *composed* (full) policy params, not v_k.
    """
    orig_env_id = args.env_id
    args.env_id = task_id
    num_cubes = _parse_num_cubes(task_id)

    env_class, config = make_env(args)
    raw_eval_env = env_class(num_envs=args.num_eval_envs,
                             num_threads=args.num_threads,
                             config=config)
    eval_env = PaddedEnvWrapper(wrap_env(raw_eval_env, config.episode_length),
                                actual_cubes=num_cubes)

    make_policy = make_inference_fn(actor, g_encoder)
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=config.episode_length,
        key=key,
    )

    policy_params = {
        "actor": actor_params,
        "g_encoder": critic_params["g_encoder"],
    }
    metrics = evaluator.run_evaluation(
        policy_params=policy_params,
        training_metrics={},
    )
    rollout.shutdown_persistent_pool()

    args.env_id = orig_env_id
    return metrics


# ---------------------------------------------------------------------------
# CKA composition helpers
# ---------------------------------------------------------------------------

def _compose_params(base, pool_c, vk):
    """θ' = θ_base + pool_contribution + v_k."""
    return jax.tree_util.tree_map(lambda b, p, v: b + p + v, base, pool_c, vk)


def _split_head_body(base_params, vk_params):
    """Split v_k into head (for pool) and body (fold into base).

    For adapt_heads_only mode:
      - Head params (Dense_4, Dense_5): base unchanged, v_k goes to pool
      - Body params (everything else): fold v_k into base, zero out v_k

    Returns (new_base, head_only_vk).
    """
    out_base_leaves, out_vk_leaves = [], []
    flat_base, treedef = jax.tree_util.tree_flatten_with_path(base_params)
    flat_vk, _ = jax.tree_util.tree_flatten_with_path(vk_params)

    n_head, n_body = 0, 0
    for (path, b), (_, v) in zip(flat_base, flat_vk):
        if _is_actor_head_leaf(path):
            # Head: base unchanged, v_k goes to pool
            out_base_leaves.append(b)
            out_vk_leaves.append(v)
            n_head += 1
        else:
            # Body: fold v_k into base, zero out
            out_base_leaves.append(b + v)
            out_vk_leaves.append(jnp.zeros_like(v))
            n_body += 1

    new_base = treedef.unflatten(out_base_leaves)
    head_only_vk = treedef.unflatten(out_vk_leaves)
    print(f'  [pool] head params: {n_head}, body params (folded): {n_body}',
          flush=True)
    return new_base, head_only_vk


def _mask_body_grads(grads):
    """Zero out body gradients, keeping only head gradients.

    Used when encoder_from_base=True to freeze the encoder body.
    """
    def _mask_leaf(path, g):
        if _is_actor_head_leaf(path):
            return g  # head: keep gradient
        return jnp.zeros_like(g)  # body: zero gradient
    return jax.tree_util.tree_map_with_path(_mask_leaf, grads)


# ---------------------------------------------------------------------------
# Single-task training loop
# ---------------------------------------------------------------------------

def train_single_task(
    args: Args,
    task_idx: int,
    task_id: str,
    actor_state: TrainState,
    critic_state: TrainState,
    actor: Actor,
    sa_encoder: SA_encoder,
    g_encoder: G_encoder,
    key: jax.Array,
    # CKA-specific (None when not using CKA for that component)
    actor_base_params=None,
    actor_pool_c=None,
    critic_base_params=None,
    critic_pool_c=None,
):
    """Run the inner CRL training loop for one task.

    When ``actor_base_params`` is not None, the actor is in CKA mode:
    ``actor_state.params`` represents v_k and the effective policy is
    θ_base + pool_c + v_k.

    Same logic applies to critic with ``critic_base_params``.

    Returns:
        (actor_state, critic_state)
    """
    args.env_id = task_id
    use_actor_cka = actor_base_params is not None
    use_critic_cka = critic_base_params is not None
    # Determine whether to mask body gradients
    freeze_actor_body = (use_actor_cka and args.encoder_from_base
                         and task_idx > 0)

    # Use base_steps for task 0, steps_per_task for rest
    task_steps = args.base_steps if task_idx == 0 else args.steps_per_task

    # Derive training step counts
    num_training_step = task_steps // (args.num_envs * args.rollout_length)
    num_training_steps_per_eval = max(num_training_step // args.num_eval_steps, 1)

    # Metrics logging intervals (in training steps)
    metrics_every = max(num_training_step // args.num_eval_steps, 1)

    print(f'  Training steps = {num_training_step}', flush=True)
    print(f'  Env steps = {task_steps}', flush=True)
    print(f'  Gradient steps per training step = '
          f'{(args.sequence_length * args.num_envs) // args.batch_size}',
          flush=True)
    print(f'  Env steps per training step = '
          f'{args.num_envs * args.rollout_length}', flush=True)
    if use_actor_cka:
        print(f'  Actor CKA: training v_k (base frozen)', flush=True)
        if freeze_actor_body:
            print(f'  Actor: body gradients MASKED (encoder_from_base=True)',
                  flush=True)
        else:
            print(f'  Actor: body receives gradients (encoder evolves)',
                  flush=True)
    if use_critic_cka:
        print(f'  Critic CKA: training w_k (base frozen)', flush=True)

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
    for k in ("obj_reached_once", "obj_lifted", "obj_moved"):
        if k in env_state.metrics.keys():
            log_data_metric_keys.append(k)
    log_data_metric_keys = tuple(log_data_metric_keys)

    # JIT network applies
    actor_apply = jax.jit(actor.apply)
    sa_encoder_apply = jax.jit(sa_encoder.apply)
    g_encoder_apply = jax.jit(g_encoder.apply)

    # ---- training state ----------------------------------------------------
    training_state = CRLTrainingState(
        env_steps=np.zeros((), dtype=np.float64),
        gradient_steps=np.zeros((), dtype=np.float64),
        actor_state=actor_state,
        critic_state=critic_state,
    )

    # ---- replay buffer -----------------------------------------------------
    dummy_obs = jnp.zeros((obs_size,))
    dummy_goal = jnp.zeros((goal_size,))
    dummy_action = jnp.zeros((action_size,))

    dummy_transition = Transition(
        observation=dummy_obs,
        achieved_goal=dummy_goal,
        action=dummy_action,
        extras={
            "state_extras": {
                "traj_id": 0.0,
            }
        },
    )

    def jit_wrap(buffer):
        buffer.insert = jax.jit(buffer.insert)
        buffer.sample = jax.jit(buffer.sample)
        return buffer

    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=args.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=args.batch_size,
            num_envs=args.num_envs,
            sequence_length=args.sequence_length + 1,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    # ---- evaluator ---------------------------------------------------------
    make_policy = make_inference_fn(actor, g_encoder)
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=episode_length,
        key=eval_key,
    )

    # ---- helpers to get composed params from training state ----------------
    # These closures capture base/pool_c so the JIT-compiled functions can use
    # them.  When not in CKA mode the "compose" is identity.

    def _get_actor_params(ts):
        """Return the effective (composed) actor params."""
        if use_actor_cka:
            return _compose_params(actor_base_params, actor_pool_c,
                                   ts.actor_state.params)
        return ts.actor_state.params

    def _get_critic_params(ts):
        """Return the effective (composed) critic params."""
        if use_critic_cka:
            return _compose_params(critic_base_params, critic_pool_c,
                                   ts.critic_state.params)
        return ts.critic_state.params

    # ---- JIT-compiled training functions -----------------------------------

    def actor_step(training_state, env, env_state, key, extra_fields, metrics_fields):
        effective_critic = _get_critic_params(training_state)
        effective_actor = _get_actor_params(training_state)

        g_repr = g_encoder.apply(effective_critic["g_encoder"],
                                 env_state.info['target_goal'])
        means, log_stds = actor.apply(effective_actor, env_state.obs, g_repr)
        stds = jnp.exp(log_stds)
        actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))

        nstate = env.pre_step(env_state, actions)
        physics_state, sensor_data = env.step(nstate, actions)
        nstate = env.post_step(nstate, physics_state, sensor_data)

        state_extras = {x: nstate.info[x] for x in extra_fields}
        metrics = {x: nstate.metrics[x] for x in metrics_fields}

        return training_state, nstate, Transition(
            observation=env_state.obs,
            achieved_goal=env_state.info['achieved_goal'],
            action=actions,
            extras={"state_extras": state_extras},
        ), metrics

    @jax.jit
    def data_collect_step(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            training_state, env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            training_state, env_state, transition, metrics = actor_step(
                training_state, env, env_state, current_key,
                extra_fields=("traj_id",),
                metrics_fields=log_data_metric_keys,
            )
            return (training_state, env_state, next_key), (transition, metrics)

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
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            training_state, env_state, buffer_state, _ = data_collect_step(
                training_state, env_state, buffer_state, key,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f, (training_state, env_state, buffer_state, key), (),
            length=int(np.ceil(args.min_replay_size / args.rollout_length)))[0]

    # -- actor update --------------------------------------------------------
    # When CKA: actor_state.params = v_k.  Gradients of the actor loss w.r.t.
    # the composed policy equal gradients w.r.t. v_k (base and pool_c are
    # constant additive terms) so we can differentiate w.r.t. v_k directly.
    if use_actor_cka:
        @jax.jit
        def update_actor_and_alpha(transitions, training_state, key):
            def actor_loss(vk_params, critic_params, transitions, key):
                composed = _compose_params(actor_base_params, actor_pool_c,
                                           vk_params)
                state = transitions.observation
                goal = transitions.extras['future_goal']
                sa_encoder_params = jax.lax.stop_gradient(critic_params["sa_encoder"])
                g_encoder_params = jax.lax.stop_gradient(critic_params["g_encoder"])

                g_repr = g_encoder.apply(g_encoder_params, goal)
                means, log_stds = actor.apply(composed, state, g_repr)
                stds = jnp.exp(log_stds)
                x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
                action = nn.tanh(x_ts)
                log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
                log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
                log_prob = log_prob.sum(-1)

                sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
                qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

                return jnp.mean(args.entropy_cost * log_prob - qf_pi), log_prob

            effective_critic = _get_critic_params(training_state)
            (actorloss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
                training_state.actor_state.params, effective_critic,
                transitions, key)

            # When encoder_from_base=True, mask out body gradients so the
            # encoder stays at base-task values.
            if freeze_actor_body:
                actor_grad = _mask_body_grads(actor_grad)

            new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)
            training_state = training_state.replace(actor_state=new_actor_state)
            return training_state, {"sample_entropy": -log_prob, "actor_loss": actorloss}
    else:
        @jax.jit
        def update_actor_and_alpha(transitions, training_state, key):
            def actor_loss(actor_params, critic_params, transitions, key):
                state = transitions.observation
                goal = transitions.extras['future_goal']
                sa_encoder_params = jax.lax.stop_gradient(critic_params["sa_encoder"])
                g_encoder_params = jax.lax.stop_gradient(critic_params["g_encoder"])

                g_repr = g_encoder.apply(g_encoder_params, goal)
                means, log_stds = actor.apply(actor_params, state, g_repr)
                stds = jnp.exp(log_stds)
                x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
                action = nn.tanh(x_ts)
                log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
                log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
                log_prob = log_prob.sum(-1)

                sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
                qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

                return jnp.mean(args.entropy_cost * log_prob - qf_pi), log_prob

            (actorloss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
                training_state.actor_state.params, training_state.critic_state.params,
                transitions, key)
            new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)
            training_state = training_state.replace(actor_state=new_actor_state)
            return training_state, {"sample_entropy": -log_prob, "actor_loss": actorloss}

    # -- critic update -------------------------------------------------------
    if use_critic_cka:
        @jax.jit
        def update_critic(transitions, training_state, key):
            def critic_loss(wk_params, transitions, key):
                composed = _compose_params(critic_base_params, critic_pool_c,
                                           wk_params)
                sa_encoder_params = composed["sa_encoder"]
                g_encoder_params = composed["g_encoder"]

                state = transitions.observation
                action = transitions.action
                goal = transitions.extras['future_goal']

                sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
                g_repr = g_encoder.apply(g_encoder_params, goal)

                logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
                c_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                c_loss += args.logsumexp_cost * jnp.mean(logsumexp ** 2)

                I = jnp.eye(logits.shape[0])
                correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
                logits_pos = jnp.sum(logits * I) / jnp.sum(I)
                logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
                return c_loss, (logsumexp, correct, logits_pos, logits_neg)

            (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
                critic_loss, has_aux=True)(training_state.critic_state.params, transitions, key)
            new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
            training_state = training_state.replace(critic_state=new_critic_state)
            return training_state, {
                "categorical_accuracy": jnp.mean(correct),
                "logits_pos": logits_pos,
                "logits_neg": logits_neg,
                "logsumexp": logsumexp.mean(),
                "critic_loss": loss,
            }
    else:
        @jax.jit
        def update_critic(transitions, training_state, key):
            def critic_loss(critic_params, transitions, key):
                sa_encoder_params = critic_params["sa_encoder"]
                g_encoder_params = critic_params["g_encoder"]

                state = transitions.observation
                action = transitions.action
                goal = transitions.extras['future_goal']

                sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
                g_repr = g_encoder.apply(g_encoder_params, goal)

                logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
                c_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                c_loss += args.logsumexp_cost * jnp.mean(logsumexp ** 2)

                I = jnp.eye(logits.shape[0])
                correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
                logits_pos = jnp.sum(logits * I) / jnp.sum(I)
                logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
                return c_loss, (logsumexp, correct, logits_pos, logits_neg)

            (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
                critic_loss, has_aux=True)(training_state.critic_state.params, transitions, key)
            new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
            training_state = training_state.replace(critic_state=new_critic_state)
            return training_state, {
                "categorical_accuracy": jnp.mean(correct),
                "logits_pos": logits_pos,
                "logits_neg": logits_neg,
                "logsumexp": logsumexp.mean(),
                "critic_loss": loss,
            }

    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)
        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)
        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)
        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        return (training_state, key), metrics

    @jax.jit
    def learn_step(training_state, buffer_state, key):
        experience_key, sampling_key, training_key = jax.random.split(key, 3)

        buffer_state, transitions = replay_buffer.sample(buffer_state)

        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(
            TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0)
        )((args.discount,), transitions, batch_keys)

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        permutation = jax.random.permutation(experience_key, len(transitions.action))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

        (training_state, _), metrics = jax.lax.scan(
            sgd_step, (training_state, training_key), transitions)
        return training_state, buffer_state, metrics

    # ---- prefill -----------------------------------------------------------
    print(f'  Prefilling replay buffer...', flush=True)
    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key)

    # ---- RL metrics setup --------------------------------------------------
    next_metrics_frequent = metrics_every if args.log_rl_metrics else float('inf')
    next_metrics_occasional = 5 * metrics_every if args.log_rl_metrics else float('inf')

    # ---- training loop -----------------------------------------------------
    training_walltime = 0
    data_collect_step_time = 0
    learn_step_time = 0
    xt = time.time()
    metrics = None

    # Cache for last batch (used by rl_metrics)
    last_transitions = None

    # Per-task checkpoint save path
    if args.save_checkpoint:
        task_save_path = (Path(args.wandb_dir) /
                          f"checkpoints/{args.exp_name}/task{task_idx}_{task_id}/")
        os.makedirs(task_save_path, exist_ok=True)

    print(f'  Training for {num_training_step} steps...', flush=True)
    for ts in range(1, num_training_step + 1):
        key, key_sgd, key_generate_rollout = jax.random.split(key, 3)

        data_collect_start = time.time()
        training_state, env_state, buffer_state, data_metrics = data_collect_step(
            training_state, env_state, buffer_state, key_generate_rollout)
        data_collect_step_time += time.time() - data_collect_start

        learn_step_start = time.time()
        training_state, buffer_state, training_metrics = learn_step(
            training_state, buffer_state, key_sgd)
        learn_step_time += time.time() - learn_step_start

        if metrics is None:
            metrics = data_metrics | training_metrics
        else:
            metrics = jax.tree_util.tree_map(
                lambda x, y: x + y, metrics, (data_metrics | training_metrics))

        # ---- RL representation metrics ------------------------------------
        if args.log_rl_metrics and ts >= next_metrics_frequent:
            if ts >= next_metrics_occasional:
                rl_level = 'occasional'
                next_metrics_occasional = ts + 5 * metrics_every
                next_metrics_frequent = ts + metrics_every
            else:
                rl_level = 'frequent'
                next_metrics_frequent = ts + metrics_every

            try:
                current_actor_p = _get_actor_params(training_state)
                current_critic_p = _get_critic_params(training_state)

                # Use a fresh sample from the buffer for features
                key, metrics_key = jax.random.split(key)
                _bs, sample_transitions = replay_buffer.sample(buffer_state)
                # Take first batch_size samples, flatten
                obs_sample = sample_transitions.observation[0, :args.batch_size]
                act_sample = sample_transitions.action[0, :args.batch_size]
                goal_sample = sample_transitions.achieved_goal[0, :args.batch_size]

                m = rl_metrics.compute_all_metrics(
                    sa_encoder, g_encoder,
                    current_actor_p, current_critic_p,
                    obs_sample, act_sample, goal_sample,
                    action_dim=action_size,
                    level=rl_level)
                if args.track:
                    wandb_m = {f'rl_metrics/{k}': v for k, v in m.items()}
                    wandb_m['rl_metrics/env_steps'] = float(training_state.env_steps)
                    wandb.log(wandb_m)
            except Exception as e:
                print(f'  [rl_metrics] Warning: {e}', flush=True)

        if ts % num_training_steps_per_eval == 0:
            es = ts // num_training_steps_per_eval

            metrics = jax.tree_util.tree_map(
                lambda x: x / num_training_steps_per_eval, metrics)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            training_step_time = time.time() - xt
            training_walltime += training_step_time

            sps = (
                num_training_steps_per_eval
                * args.num_envs * args.rollout_length
            ) / training_step_time

            # Compose actor params for eval logging
            eval_actor_p = _get_actor_params(training_state)
            eval_critic_p = _get_critic_params(training_state)

            metrics = {
                'training/sps': sps,
                'training/walltime': training_walltime,
                'training/data_collection_time_fraction': data_collect_step_time / training_step_time,
                'training/learning_time_fraction': learn_step_time / training_step_time,
                'training/env_steps': training_state.env_steps,
                'training/task_idx': task_idx,
                **{f'training/{name}': value for name, value in metrics.items()},
                'buffer_current_size': replay_buffer.size(buffer_state),
            }

            rollout.shutdown_persistent_pool()
            metrics = evaluator.run_evaluation(
                policy_params={
                    "actor": eval_actor_p,
                    "g_encoder": eval_critic_p["g_encoder"],
                },
                training_metrics=metrics,
            )
            rollout.shutdown_persistent_pool()

            pprint.pprint(metrics)
            if args.track:
                wandb.log(metrics)
            metrics = None

            if args.save_checkpoint:
                save_params(
                    f"{task_save_path}/params_{es}.pkl",
                    params=(eval_actor_p, eval_critic_p),
                )

            xt, data_collect_step_time, learn_step_time = time.time(), 0, 0

    print(f'  Task {task_idx} [{task_id}] training complete.', flush=True)

    return training_state.actor_state, training_state.critic_state


# ---------------------------------------------------------------------------
# Main continual training loop
# ---------------------------------------------------------------------------

def main(args: Args):
    tasks = [t.strip() for t in args.task_sequence.split(',')]
    num_tasks = len(tasks)

    print(f'\n{"=" * 60}')
    print(f'Continual CRL — {num_tasks} tasks')
    print(f'Sequence: {tasks}')
    print(f'Actor mode: {args.actor_mode} | Critic mode: {args.critic_mode}')
    print(f'Steps per task: {args.steps_per_task} | Base steps: {args.base_steps}')
    print(f'adapt_heads_only: {args.adapt_heads_only} | '
          f'encoder_from_base: {args.encoder_from_base}')
    if args.actor_mode == 'cka' or args.critic_mode == 'cka':
        print(f'k_max: {args.k_max} | alpha_scale: {args.alpha_scale}')
    print(f'RL metrics: {args.log_rl_metrics}')
    print(f'{"=" * 60}\n')

    # Validate modes
    assert args.actor_mode in ('reset', 'persistent', 'cka'), \
        f'Unknown actor_mode: {args.actor_mode}'
    assert args.critic_mode in ('reset', 'persistent', 'cka'), \
        f'Unknown critic_mode: {args.critic_mode}'

    args.exp_name = (
        f"{args.wandb_name_tag + '__' if args.wandb_name_tag else ''}"
        f"continual__{args.actor_mode}_{args.critic_mode}"
        f"__{args.seed}__{int(time.time())}"
    )

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # ---- auto-resume -------------------------------------------------------
    last_completed = auto_resume(
        args.checkpoint_dir, num_tasks, args.actor_mode, args.critic_mode,
        args.adapt_heads_only, args.seed)
    start_task = last_completed + 1

    if start_task >= num_tasks:
        print(f'  All {num_tasks} tasks already completed. Nothing to do.')
        return

    # ---- initialise network modules ----------------------------------------
    args.env_id = tasks[0]
    env_class, default_config = make_env(args)
    probe_env = wrap_env(
        env_class(num_envs=1, num_threads=1, config=default_config),
        default_config.episode_length)

    key, actor_key, sa_key, g_key = jax.random.split(key, 4)
    obs_size = UNIFIED_OBS_DIM
    action_size = probe_env.action_size
    goal_size = UNIFIED_GOAL_DIM

    actor = Actor(action_size=action_size)
    sa_encoder = SA_encoder(rep_size=args.rep_size)
    g_encoder = G_encoder(rep_size=args.rep_size)

    # Fresh parameter templates
    fresh_actor_params = actor.init(actor_key, np.ones([1, obs_size]), np.ones([1, args.rep_size]))
    fresh_sa_params = sa_encoder.init(sa_key, np.ones([1, obs_size]), np.ones([1, action_size]))
    fresh_g_params = g_encoder.init(g_key, np.ones([1, goal_size]))

    # ---- continual state ---------------------------------------------------
    prev_actor_params = None       # full (composed) actor params from prev task
    prev_critic_params = None      # full (composed) critic params from prev task
    actor_base_params = None       # frozen base (CKA actor)
    critic_base_params = None      # frozen base (CKA critic)
    actor_pool = KnowledgePool(k_max=args.k_max)
    critic_pool = KnowledgePool(k_max=args.k_max)

    # ---- restore state from checkpoint if resuming -------------------------
    if start_task > 0:
        ckpt = load_ckpt(
            args.checkpoint_dir, start_task - 1,
            args.actor_mode, args.critic_mode,
            args.adapt_heads_only, args.seed)
        prev_actor_params = ckpt['actor_params']
        prev_critic_params = ckpt['critic_params']
        if args.actor_mode == 'cka':
            actor_base_params = ckpt.get('actor_base_params')
            pool_state = ckpt.get('actor_pool')
            if pool_state is not None:
                actor_pool.load_state_dict(pool_state)
        if args.critic_mode == 'cka':
            critic_base_params = ckpt.get('critic_base_params')
            pool_state = ckpt.get('critic_pool')
            if pool_state is not None:
                critic_pool.load_state_dict(pool_state)

    # ---- task loop ---------------------------------------------------------
    for task_idx in range(start_task, num_tasks):
        task_id = tasks[task_idx]

        print(f'\n{"=" * 60}')
        print(f'Task {task_idx}/{num_tasks - 1}: {task_id}')
        print(f'Actor mode: {args.actor_mode} | Critic mode: {args.critic_mode}')
        if args.actor_mode == 'cka':
            print(f'Actor pool: {len(actor_pool)}/{args.k_max}')
        if args.critic_mode == 'cka':
            print(f'Critic pool: {len(critic_pool)}/{args.k_max}')
        print(f'{"=" * 60}\n')

        key, task_key = jax.random.split(key)

        # ---- determine actor params for this task --------------------------
        # For CKA: actor_state.params = v_k (zero-init), base/pool_c passed
        #          to train_single_task.
        # For others: actor_state.params = full policy params.
        cka_actor_base = None   # passed to train_single_task
        cka_actor_pool_c = None
        cka_critic_base = None
        cka_critic_pool_c = None

        if task_idx == 0:
            actor_params = fresh_actor_params
        elif args.actor_mode == 'reset':
            key, reinit_key = jax.random.split(key)
            actor_params = actor.init(reinit_key, np.ones([1, obs_size]), np.ones([1, args.rep_size]))
        elif args.actor_mode == 'persistent':
            assert prev_actor_params is not None
            actor_params = prev_actor_params
        elif args.actor_mode == 'cka':
            if task_idx == 0:
                # Task 0: train from scratch (base will be set after)
                actor_params = fresh_actor_params
            else:
                assert actor_base_params is not None
                cka_actor_base = actor_base_params
                # Compute pool contribution (uniform blending)
                cka_actor_pool_c = actor_pool.compute_contribution(
                    alpha_logits=None, alpha_scale=args.alpha_scale)
                if cka_actor_pool_c is None:
                    cka_actor_pool_c = pytree_zeros_like(actor_base_params)
                # v_k starts as zeros
                actor_params = pytree_zeros_like(actor_base_params)

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=args.actor_learning_rate),
        )

        # ---- determine critic params for this task ------------------------
        if task_idx == 0:
            critic_params = {
                "sa_encoder": fresh_sa_params,
                "g_encoder": fresh_g_params,
            }
        elif args.critic_mode == 'reset':
            key, sa_reinit_key, g_reinit_key = jax.random.split(key, 3)
            critic_params = {
                "sa_encoder": sa_encoder.init(sa_reinit_key, np.ones([1, obs_size]), np.ones([1, action_size])),
                "g_encoder": g_encoder.init(g_reinit_key, np.ones([1, goal_size])),
            }
        elif args.critic_mode == 'persistent':
            assert prev_critic_params is not None
            critic_params = prev_critic_params
        elif args.critic_mode == 'cka':
            if task_idx == 0:
                critic_params = {
                    "sa_encoder": fresh_sa_params,
                    "g_encoder": fresh_g_params,
                }
            else:
                assert critic_base_params is not None
                cka_critic_base = critic_base_params
                cka_critic_pool_c = critic_pool.compute_contribution(
                    alpha_logits=None, alpha_scale=args.alpha_scale)
                if cka_critic_pool_c is None:
                    cka_critic_pool_c = pytree_zeros_like(critic_base_params)
                # w_k starts as zeros
                critic_params = pytree_zeros_like(critic_base_params)

        critic_state = TrainState.create(
            apply_fn=None,
            params=critic_params,
            tx=optax.adam(learning_rate=args.critic_learning_rate),
        )

        # ---- W&B init per task ---------------------------------------------
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                dir=args.wandb_dir,
                group=args.wandb_group,
                name=f'task{task_idx}_{task_id}_s{args.seed}',
                config={
                    **vars(args),
                    'task_idx': task_idx,
                    'task_id': task_id,
                    'num_tasks': num_tasks,
                },
                save_code=True,
                reinit=True,
            )
            if args.wandb_mode == 'offline':
                wandb_osh.set_log_level("ERROR")
                trigger_sync = TriggerWandbSyncHook()

        # ---- train -----------------------------------------------------------
        actor_state, critic_state = train_single_task(
            args=args,
            task_idx=task_idx,
            task_id=task_id,
            actor_state=actor_state,
            critic_state=critic_state,
            actor=actor,
            sa_encoder=sa_encoder,
            g_encoder=g_encoder,
            key=task_key,
            actor_base_params=cka_actor_base,
            actor_pool_c=cka_actor_pool_c,
            critic_base_params=cka_critic_base,
            critic_pool_c=cka_critic_pool_c,
        )

        # ---- post-task: carry forward / update pools -----------------------
        # Snapshot composed policy BEFORE modifying pool/base
        composed_actor = (_compose_params(cka_actor_base, cka_actor_pool_c,
                                          actor_state.params)
                          if cka_actor_base is not None
                          else actor_state.params)

        if args.actor_mode == 'cka':
            if task_idx == 0:
                # Freeze base after task 0: base = trained policy
                actor_base_params = actor_state.params
                # Initialise pool with zero vector (matches SGCRL)
                actor_pool.append(pytree_zeros_like(actor_base_params))
                prev_actor_params = actor_state.params  # composed = base
            else:
                # actor_state.params IS v_k (the trained delta)
                v_k = actor_state.params

                if args.adapt_heads_only:
                    # Split v_k: fold body into base, store only head in pool
                    actor_base_params, v_k_head_only = _split_head_body(
                        actor_base_params, v_k)
                    actor_pool.append(v_k_head_only)
                else:
                    # Full-policy adaptation: base unchanged, full v_k to pool
                    actor_pool.append(v_k)

                # Compose full params for eval/checkpoint
                prev_actor_params = composed_actor
        elif args.actor_mode == 'persistent':
            # Carry forward composed policy for next task
            prev_actor_params = composed_actor
        else:
            # Reset: just save for checkpoint
            prev_actor_params = actor_state.params

        # Critic post-task
        composed_critic = (_compose_params(cka_critic_base, cka_critic_pool_c,
                                           critic_state.params)
                           if cka_critic_base is not None
                           else critic_state.params)

        if args.critic_mode == 'cka':
            if task_idx == 0:
                critic_base_params = critic_state.params
                critic_pool.append(pytree_zeros_like(critic_base_params))
                prev_critic_params = critic_state.params
            else:
                w_k = critic_state.params
                critic_pool.append(w_k)
                prev_critic_params = composed_critic
        else:
            prev_critic_params = composed_critic

        # ---- save continual checkpoint ---------------------------------------
        ckpt_data = {
            'actor_params': prev_actor_params,
            'critic_params': prev_critic_params,
            'task_idx': task_idx,
            'task_id': task_id,
        }
        if args.actor_mode == 'cka':
            ckpt_data['actor_base_params'] = actor_base_params
            ckpt_data['actor_pool'] = actor_pool.state_dict()
        if args.critic_mode == 'cka':
            ckpt_data['critic_base_params'] = critic_base_params
            ckpt_data['critic_pool'] = critic_pool.state_dict()
        save_ckpt(args.checkpoint_dir, task_idx,
                  args.actor_mode, args.critic_mode,
                  args.adapt_heads_only, args.seed, ckpt_data)

        # ---- cross-task evaluation -------------------------------------------
        print(f'\n  Evaluating on all tasks seen so far...', flush=True)
        eval_results = {}
        for eval_idx in range(task_idx + 1):
            eval_task = tasks[eval_idx]
            key, eval_key = jax.random.split(key)
            task_metrics = evaluate_on_task(
                eval_task, args, prev_actor_params, prev_critic_params,
                actor, g_encoder, eval_key)
            sr = task_metrics.get('eval/episode_success_rate',
                                  task_metrics.get('eval/episode_reward', 0.0))
            eval_results[eval_task] = float(sr)
            print(f'    Task {eval_idx} [{eval_task}]: {sr:.3f}', flush=True)

        if eval_results:
            mean_sr = np.mean(list(eval_results.values()))
            print(f'    Mean: {mean_sr:.3f}', flush=True)

            if args.track:
                wandb_eval = {f'cross_eval/{name}': sr
                              for name, sr in eval_results.items()}
                wandb_eval['cross_eval/mean'] = mean_sr
                wandb_eval['cross_eval/num_tasks_seen'] = task_idx + 1
                wandb.log(wandb_eval)

        rollout.shutdown_persistent_pool()

        # ---- close W&B run for this task -------------------------------------
        if args.track:
            wandb.finish()

    print(f'\nAll {num_tasks} tasks complete.', flush=True)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
