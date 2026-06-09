"""Decomposed-critic networks for BuilderBench (Flax port of sgcrl/contrastive/decomposed_networks.py).

The contrastive critic factors into:

  z_shared = h_phi(b_shared(s, a))
  z_task   = phi_task(s, a)            # reset every task
  z_sa     = combine(z_shared, z_task) # 'add' or 'concat'
  z_g      = psi(g)                    # goal encoder family
  score    = -|| z_sa - z_g ||_2       # (BuilderBench's existing L2 energy)

with a separate dynamics head trained on a masked next-state target:

  s'_M_pred = h_dyn(b_shared(s, a))
  L_dyn     = || s'_M_pred - next_cube_positions ||_2^2

Trainable groups (each its own optax optimiser state):

  b_shared, h_phi, h_dyn, phi_task, psi
  (+ optional psi_task / psi_proj for goal-encoder variants)

Continual-learning rule at task boundary k > 0:

  b_shared, h_phi, h_dyn, psi (shared portion)  -> carry forward (transfer)
  phi_task, psi_task (if any)                    -> reinitialise (task-specific)

Architecture choices mirror the existing BuilderBench ``SA_encoder`` /
``G_encoder`` so the inductive bias does not change.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import variance_scaling, zeros


# Number of "stable" target dims for the dynamics head. BuilderBench's
# natural choice is cube positions (3 coords x MAX_CUBES = 9), which is
# exactly the achieved-goal subspace and so survives the padding wrapper
# without per-task remapping.
def cube_position_indices(fixed_obs_prefix: int, per_cube_obs: int,
                          max_cubes: int) -> Tuple[int, ...]:
    """Return the indices into the padded observation that hold cube
    positions (the first 3 dims of each per-cube block).
    """
    idxs = []
    for i in range(max_cubes):
        start = fixed_obs_prefix + i * per_cube_obs
        idxs.extend([start, start + 1, start + 2])
    return tuple(idxs)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


def _lecun_uniform():
    return variance_scaling(1 / 3, 'fan_in', 'uniform')


class _MLPBody(nn.Module):
    """Width x depth pre-activation MLP (Swish + LayerNorm), like BuilderBench's
    SA_encoder body but without the final projection. Returns the hidden
    representation of width `hidden_width`.
    """
    hidden_width: int = 1024
    depth: int = 4
    norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.depth):
            x = nn.Dense(self.hidden_width,
                         kernel_init=_lecun_uniform(), bias_init=zeros)(x)
            if self.norm:
                x = nn.LayerNorm()(x)
            x = nn.swish(x)
        return x


class BSharedBody(nn.Module):
    """Shared backbone: produces the hidden representation that h_phi and
    h_dyn read from. Inputs are concatenated [s; a].
    """
    hidden_width: int = 1024
    depth: int = 4

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([s, a], axis=-1)
        return _MLPBody(hidden_width=self.hidden_width, depth=self.depth)(x)


class LinearHead(nn.Module):
    """A bare linear projection from `hidden_width` to `out_dim`."""
    out_dim: int

    @nn.compact
    def __call__(self, hidden: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.out_dim,
                        kernel_init=_lecun_uniform(), bias_init=zeros,
                        name='out')(hidden)


class PhiTask(nn.Module):
    """Smaller task-specific encoder over [s; a]; reset every task."""
    rep_size: int
    width: int = 256
    depth: int = 4   # one residual-equivalent block (4 dense layers)

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([s, a], axis=-1)
        x = _MLPBody(hidden_width=self.width, depth=self.depth)(x)
        return nn.Dense(self.rep_size,
                        kernel_init=_lecun_uniform(), bias_init=zeros,
                        name='out')(x)


class GoalEncoder(nn.Module):
    """Full goal encoder: g -> rep_size. Matches BuilderBench's `G_encoder`
    in width and depth."""
    rep_size: int
    width: int = 1024
    depth: int = 4

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        x = _MLPBody(hidden_width=self.width, depth=self.depth)(g)
        return nn.Dense(self.rep_size,
                        kernel_init=_lecun_uniform(), bias_init=zeros,
                        name='out')(x)


class GoalProjector(nn.Module):
    """Single Dense projection (used when goal_encoder_mode='projected' so
    z_g matches the dim of `concat(z_shared, z_task)`)."""
    out_dim: int

    @nn.compact
    def __call__(self, g_repr: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.out_dim,
                        kernel_init=_lecun_uniform(), bias_init=zeros,
                        name='proj')(g_repr)


# ---------------------------------------------------------------------------
# Bundle returned to the learner.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DecomposedCriticNetworks:
    # Module instances (used to call `.apply` with the right param tree).
    b_shared: nn.Module
    h_phi: nn.Module
    h_dyn: nn.Module
    phi_task: nn.Module
    psi_shared: nn.Module
    psi_task: Optional[nn.Module]      # only used for some goal modes
    psi_proj: Optional[nn.Module]      # only used when combine='concat'

    # Init functions: (key) -> params.
    init_b_shared: Callable
    init_h_phi: Callable
    init_h_dyn: Callable
    init_phi_task: Callable
    init_psi_shared: Callable
    init_psi_task: Optional[Callable]
    init_psi_proj: Optional[Callable]

    # Shape metadata.
    rep_size: int        # dim of z_shared (and z_task)
    z_sa_dim: int        # final dim of z_sa after combine; equals rep_size for 'add',
                         # 2*rep_size for 'concat'.
    z_g_dim: int         # final dim of z_g (must equal z_sa_dim for scoring)
    dyn_target_dim: int  # number of stable indices targeted by h_dyn

    # Config (echoed back for inspection / metrics keys).
    combine_mode: str            # 'add' or 'concat'
    goal_encoder_mode: str       # see Args.dcc_goal_encoder_mode
    use_dyn: bool                # whether the dyn loss is active

    # Convenience apply functions over the FULL param dict.
    apply_sa_repr: Callable      # (params_dict, obs, action) -> z_sa
    apply_g_repr: Callable       # (params_dict, goal) -> z_g
    apply_h_dyn: Callable        # (params_dict, obs, action) -> dynamics prediction


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def make_decomposed_networks(
    *,
    obs_size: int,
    action_size: int,
    goal_size: int,
    rep_size: int = 64,
    network_width: int = 1024,
    critic_depth: int = 4,
    phi_task_width: int = 256,
    phi_task_depth: int = 4,
    combine_mode: str = 'add',
    goal_encoder_mode: str = 'shared',
    use_dyn: bool = True,
    dyn_target_dim: int = 9,
) -> DecomposedCriticNetworks:
    """Build the decomposed critic networks.

    Args:
      obs_size: padded obs dim (UNIFIED_OBS_DIM).
      action_size: action dim (BuilderBench: 4).
      goal_size: padded goal dim (UNIFIED_GOAL_DIM = 9).
      rep_size: per-encoder embedding dim. Matches BuilderBench default.
      network_width / critic_depth: shared body geometry.
      phi_task_width / phi_task_depth: task-specific encoder geometry.
      combine_mode: 'add' (z_sa = z_shared + z_task) or 'concat'
          (z_sa = [z_shared; z_task]; goal side must produce 2*rep_size).
      goal_encoder_mode: see continual_crl.Args.dcc_goal_encoder_mode.
      use_dyn: whether to build the dynamics head h_dyn (still built when
          False so checkpoints are uniform; the learner just doesn't use it).
      dyn_target_dim: width of h_dyn output (9 cube-position dims by default).
    """
    assert combine_mode in ('add', 'concat'), combine_mode
    assert goal_encoder_mode in (
        'shared', 'task_specific', 'partial_shared',
        'decomposed', 'projected'), goal_encoder_mode

    z_sa_dim = rep_size if combine_mode == 'add' else 2 * rep_size

    # ---- Critic modules ----
    b_shared = BSharedBody(hidden_width=network_width, depth=critic_depth)
    h_phi = LinearHead(out_dim=rep_size)
    h_dyn = LinearHead(out_dim=dyn_target_dim)
    phi_task = PhiTask(rep_size=rep_size,
                       width=phi_task_width, depth=phi_task_depth)

    # ---- Goal modules ----
    psi_shared = GoalEncoder(rep_size=rep_size,
                             width=network_width, depth=critic_depth)
    psi_task: Optional[nn.Module] = None
    psi_proj: Optional[nn.Module] = None

    if goal_encoder_mode == 'shared':
        # Single full encoder. z_g_dim = rep_size; project to z_sa_dim only
        # when combine='concat' (else direct).
        if combine_mode == 'concat':
            psi_proj = GoalProjector(out_dim=z_sa_dim)
    elif goal_encoder_mode == 'task_specific':
        # The whole goal encoder is reset each task. We still keep a
        # zero-shaped ``psi_shared`` to keep the params dict uniform across
        # modes. Caller stores ONLY psi_task in the params dict.
        psi_task = GoalEncoder(rep_size=rep_size,
                               width=network_width, depth=critic_depth)
        if combine_mode == 'concat':
            psi_proj = GoalProjector(out_dim=z_sa_dim)
    elif goal_encoder_mode == 'partial_shared':
        # Shared body provides z_g_shared (rep_size). A small per-task
        # encoder produces z_g_task; the two are combined the same way as
        # z_sa to keep the symmetry with the state-action side.
        psi_task = GoalEncoder(rep_size=rep_size,
                               width=phi_task_width, depth=phi_task_depth)
        if combine_mode == 'concat':
            psi_proj = GoalProjector(out_dim=z_sa_dim)
    elif goal_encoder_mode == 'decomposed':
        # Symmetric decomposition on the goal side too: shared psi + task psi.
        # Always combined the same way as z_sa.
        psi_task = GoalEncoder(rep_size=rep_size,
                               width=network_width, depth=critic_depth)
        if combine_mode == 'concat':
            psi_proj = GoalProjector(out_dim=z_sa_dim)
    elif goal_encoder_mode == 'projected':
        # Single full goal encoder, then a learnable projection forces the
        # goal embedding to live in the same space as z_sa. Useful when
        # combine='concat' (and harmless when combine='add' since
        # projection from rep_size -> rep_size is just a refinement).
        psi_proj = GoalProjector(out_dim=z_sa_dim)

    # ---- Init helpers ----
    dummy_s = np.zeros([1, obs_size], dtype=np.float32)
    dummy_a = np.zeros([1, action_size], dtype=np.float32)
    dummy_g = np.zeros([1, goal_size], dtype=np.float32)

    def init_b_shared(key):
        return b_shared.init(key, dummy_s, dummy_a)

    # Probe hidden width by running b_shared once with the just-created
    # params; this lets us init heads even if width changes.
    _probe_params = init_b_shared(jax.random.PRNGKey(0))
    _probe_hidden = b_shared.apply(_probe_params, dummy_s, dummy_a)
    dummy_hidden = jnp.zeros((1, _probe_hidden.shape[-1]), dtype=np.float32)

    def init_h_phi(key):
        return h_phi.init(key, dummy_hidden)

    def init_h_dyn(key):
        return h_dyn.init(key, dummy_hidden)

    def init_phi_task(key):
        return phi_task.init(key, dummy_s, dummy_a)

    def init_psi_shared(key):
        return psi_shared.init(key, dummy_g)

    init_psi_task: Optional[Callable] = None
    if psi_task is not None:
        def _init_psi_task(key):
            return psi_task.init(key, dummy_g)
        init_psi_task = _init_psi_task

    init_psi_proj: Optional[Callable] = None
    if psi_proj is not None:
        # The projector input dim depends on whether we sum or concat psi
        # outputs.  We pass the largest shape the projector will ever see.
        proj_in = rep_size if goal_encoder_mode == 'shared' or \
            goal_encoder_mode == 'projected' else \
            (rep_size if combine_mode == 'add' else 2 * rep_size)
        dummy_proj_in = jnp.zeros((1, proj_in), dtype=np.float32)

        def _init_psi_proj(key):
            return psi_proj.init(key, dummy_proj_in)
        init_psi_proj = _init_psi_proj

    # ---- Apply functions over a single params dict ----

    def apply_sa_repr(params, obs, action):
        """Return z_sa with shape (B, z_sa_dim).

        `params` is a dict containing 'b_shared', 'h_phi', 'phi_task'.
        Any other entries are ignored.
        """
        hidden = b_shared.apply(params['b_shared'], obs, action)
        z_shared = h_phi.apply(params['h_phi'], hidden)
        z_task = phi_task.apply(params['phi_task'], obs, action)
        if combine_mode == 'add':
            return z_shared + z_task
        # 'concat'
        return jnp.concatenate([z_shared, z_task], axis=-1)

    def apply_g_repr(params, goal):
        """Return z_g with shape (B, z_g_dim).

        `params` may contain 'psi_shared', 'psi_task', 'psi_proj'
        depending on `goal_encoder_mode`.
        """
        if goal_encoder_mode == 'shared':
            z = psi_shared.apply(params['psi_shared'], goal)
        elif goal_encoder_mode == 'task_specific':
            # Only psi_task; psi_shared is kept zero-valued and ignored.
            z = psi_task.apply(params['psi_task'], goal)
        elif goal_encoder_mode in ('partial_shared', 'decomposed'):
            z_s = psi_shared.apply(params['psi_shared'], goal)
            z_t = psi_task.apply(params['psi_task'], goal)
            if combine_mode == 'add':
                z = z_s + z_t
            else:
                z = jnp.concatenate([z_s, z_t], axis=-1)
        elif goal_encoder_mode == 'projected':
            z = psi_shared.apply(params['psi_shared'], goal)
        else:
            raise ValueError(goal_encoder_mode)

        if psi_proj is not None and 'psi_proj' in params:
            z = psi_proj.apply(params['psi_proj'], z)
        return z

    def apply_h_dyn_fn(params, obs, action):
        hidden = b_shared.apply(params['b_shared'], obs, action)
        return h_dyn.apply(params['h_dyn'], hidden)

    return DecomposedCriticNetworks(
        b_shared=b_shared, h_phi=h_phi, h_dyn=h_dyn,
        phi_task=phi_task, psi_shared=psi_shared,
        psi_task=psi_task, psi_proj=psi_proj,
        init_b_shared=init_b_shared, init_h_phi=init_h_phi,
        init_h_dyn=init_h_dyn, init_phi_task=init_phi_task,
        init_psi_shared=init_psi_shared,
        init_psi_task=init_psi_task,
        init_psi_proj=init_psi_proj,
        rep_size=rep_size, z_sa_dim=z_sa_dim,
        z_g_dim=z_sa_dim,           # by construction
        dyn_target_dim=dyn_target_dim,
        combine_mode=combine_mode,
        goal_encoder_mode=goal_encoder_mode,
        use_dyn=use_dyn,
        apply_sa_repr=apply_sa_repr,
        apply_g_repr=apply_g_repr,
        apply_h_dyn=apply_h_dyn_fn,
    )
