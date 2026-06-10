"""RL representation metrics in JAX (Flax version).

Metrics organized by computational cost:

  FREQUENT (every eval_every steps):
    - weight_norm_l2: L2 norm of all parameters
    - final_layer_norm: L2 norm of the actor's policy head weights
    - feature_entropy: Shannon entropy of |feature| distributions
    - gini_sparsity: Gini coefficient measuring feature sparsity

  OCCASIONAL (every 5 * eval_every steps):
    - feature_rank: effective rank via SVD (tau=0.99)
    - nrc1 / nrc2: Neural Rank Collapse metrics
    - dormant_ratio: fraction of neurons with negligible activation

Ported from sgcrl/contrastive/rl_metrics.py, adapted for Flax param
conventions (nested dicts with 'Dense_N/kernel' keys instead of Haiku's
flat 'Normal/linear' keys).
"""
import jax
import jax.numpy as jnp
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Parameter-level metrics (no forward pass needed)
# ═══════════════════════════════════════════════════════════════════════

def weight_norm_l2(params) -> float:
    """L2 norm of a parameter pytree."""
    leaves = jax.tree_util.tree_leaves(params)
    total = sum(float(jnp.sum(p ** 2)) for p in leaves)
    return float(np.sqrt(total))


def final_layer_norm(actor_params) -> float:
    """L2 norm of the actor's policy head (mean output layer) kernel.

    In the Flax Actor, the head layers are the last two nn.Dense calls:
      Dense_4 = mean projection
      Dense_5 = log_std projection
    We report the norm of Dense_4 (the mean layer).

    Returns -1.0 if not found.
    """
    # Flax params: {'params': {'Dense_4': {'kernel': ..., 'bias': ...}, ...}}
    # Navigate into the 'params' sub-dict if present.
    p = actor_params.get('params', actor_params)

    # Look for Dense_4 (mean head layer)
    for key in p:
        if 'Dense_4' in str(key):
            node = p[key]
            if isinstance(node, dict) and 'kernel' in node:
                w = node['kernel']
                return float(jnp.sqrt(jnp.sum(w ** 2)))
    return -1.0


# ═══════════════════════════════════════════════════════════════════════
# Feature-level metrics (need forward pass)
# ═══════════════════════════════════════════════════════════════════════

def feature_entropy(features: jnp.ndarray, eps: float = 1e-8) -> float:
    """Shannon entropy of |feature| distribution. Higher = more uniform."""
    X = jnp.abs(features)
    Z = jnp.maximum(jnp.sum(X, axis=1, keepdims=True), eps)
    p = X / Z
    H = -jnp.sum(p * jnp.log(p + eps), axis=1)
    return float(jnp.mean(H))


def gini_sparsity(features: jnp.ndarray, eps: float = 1e-12) -> float:
    """Gini coefficient. Higher = sparser features."""
    X = jnp.abs(features)
    B, D = X.shape
    Xs = jnp.sort(X, axis=1)
    row_sums = jnp.maximum(jnp.sum(Xs, axis=1), eps)
    idx = jnp.arange(1, D + 1, dtype=X.dtype)
    weights = (D - idx + 0.5) / D
    numer = jnp.sum(Xs * weights[None, :], axis=1)
    G = 1 - 2 * numer / row_sums
    return float(jnp.mean(G))


def feature_rank(features: jnp.ndarray, tau: float = 0.99) -> int:
    """Effective rank: min k s.t. top-k singular values explain >= tau variance."""
    X = features - jnp.mean(features, axis=0, keepdims=True)
    _, s, _ = jnp.linalg.svd(X, full_matrices=False)
    s2 = s * s
    denom = jnp.maximum(jnp.sum(s2), 1e-12)
    cumsum = jnp.cumsum(s2) / denom
    k = int(jnp.argmax(cumsum >= tau) + 1)
    return k


def compute_nrc1(features: jnp.ndarray, target_dim: int) -> float:
    """NRC1: how much features lie in a target_dim-dimensional subspace."""
    H = features
    H_centered = H - jnp.mean(H, axis=0, keepdims=True)
    H_norm = jnp.maximum(jnp.linalg.norm(H_centered, axis=1, keepdims=True), 1e-8)
    H_normalized = H_centered / H_norm
    _, S, Vh = jnp.linalg.svd(H_centered, full_matrices=False)
    PCs = Vh[:target_dim, :]
    P = PCs.T @ PCs
    H_proj = H_normalized @ P
    nrc1 = jnp.sum((H_proj - H_normalized) ** 2) / H.shape[0]
    return float(nrc1)


def compute_nrc2(features: jnp.ndarray, final_weights: jnp.ndarray) -> float:
    """NRC2: alignment between features and the final layer's row space."""
    H = features
    H_centered = H - jnp.mean(H, axis=0, keepdims=True)
    H_norm = jnp.maximum(jnp.linalg.norm(H_centered, axis=1, keepdims=True), 1e-8)
    H_normalized = H_centered / H_norm
    _, _, Vh = jnp.linalg.svd(final_weights, full_matrices=False)
    P = Vh.T @ Vh
    H_proj = H_normalized @ P
    nrc2 = jnp.sum((H_proj - H_normalized) ** 2) / H.shape[0]
    return float(nrc2)


def dormant_ratio(features: jnp.ndarray, dormant_pct: float = 1e-5) -> float:
    """Fraction of neurons with negligible activation."""
    mean_act = jnp.mean(jnp.abs(features), axis=0)
    avg_neuron = jnp.mean(mean_act)
    normalized = mean_act / jnp.maximum(avg_neuron, 1e-9)
    n_dormant = jnp.sum(normalized <= dormant_pct)
    return float(n_dormant / features.shape[1])


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction (Flax)
# ═══════════════════════════════════════════════════════════════════════

def extract_critic_features(sa_encoder, g_encoder, critic_params, obs, actions, goals):
    """Extract repr_dim features from Flax critic encoders.

    Returns (sa_features, g_features) each of shape (batch, rep_size).
    """
    sa_feats = sa_encoder.apply(critic_params['sa_encoder'], obs, actions)
    g_feats = g_encoder.apply(critic_params['g_encoder'], goals)
    return sa_feats, g_feats


def _get_encoder_final_kernel(encoder_params, encoder_name='sa_encoder'):
    """Extract the final projection kernel from a Flax encoder param tree.

    Handles residual MLPs (``Dense_0`` … ``Dense_N``) and single-head
    modules that name their output layer ``out`` (DCC ``h_phi``, ``phi_task``).
    """
    p = encoder_params.get('params', encoder_params)
    if 'out' in p and isinstance(p['out'], dict) and 'kernel' in p['out']:
        return p['out']['kernel']

    best_w = None
    best_idx = -1
    for key in p:
        key_str = str(key)
        if 'Dense' in key_str:
            node = p[key]
            if isinstance(node, dict) and 'kernel' in node:
                parts = key_str.split('_')
                try:
                    idx = int(parts[-1])
                except (ValueError, IndexError):
                    idx = 0
                if idx > best_idx:
                    best_idx = idx
                    best_w = node['kernel']
    return best_w


def _append_feature_metrics(
    metrics, prefix, features, level, nrc1_target_dim,
    final_kernel=None,
):
    """Add frequent + occasional feature metrics under ``prefix``."""
    metrics[f'{prefix}/entropy'] = feature_entropy(features)
    metrics[f'{prefix}/gini'] = gini_sparsity(features)
    if level != 'occasional':
        return
    metrics[f'{prefix}/feature_rank'] = feature_rank(features, tau=0.99)
    metrics[f'{prefix}/nrc1'] = compute_nrc1(features, target_dim=nrc1_target_dim)
    if final_kernel is not None:
        metrics[f'{prefix}/nrc2'] = compute_nrc2(features, final_kernel)
    metrics[f'{prefix}/dormant_ratio'] = dormant_ratio(
        features, dormant_pct=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Main compute function
# ═══════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    sa_encoder, g_encoder,
    actor_params, critic_params,
    obs_batch, action_batch, goal_batch,
    action_dim,
    level='frequent',
):
    """Compute RL metrics at the specified frequency level.

    Args:
        level: 'frequent' or 'occasional'.
    Returns:
        dict of metric_name -> value.
    """
    metrics = {}

    # ---- FREQUENT (no forward pass for weight norms) ----
    metrics['actor/weight_norm'] = weight_norm_l2(actor_params)
    metrics['critic/weight_norm'] = weight_norm_l2(critic_params)
    metrics['actor/final_layer_norm'] = final_layer_norm(actor_params)

    # Feature extraction (forward pass through critic encoders)
    sa_feats, g_feats = extract_critic_features(
        sa_encoder, g_encoder, critic_params, obs_batch, action_batch, goal_batch)

    _append_feature_metrics(
        metrics, 'critic_sa', sa_feats, level, action_dim,
        final_kernel=_get_encoder_final_kernel(critic_params.get('sa_encoder', {})),
    )
    _append_feature_metrics(
        metrics, 'critic_g', g_feats, level, nrc1_target_dim=1,
        final_kernel=_get_encoder_final_kernel(critic_params.get('g_encoder', {})),
    )

    return metrics


def compute_all_metrics_dcc(
    decomp,
    actor_params,
    critic_params,
    obs_batch,
    action_batch,
    goal_batch,
    action_dim,
    level='frequent',
):
    """Representation metrics for the decomposed contrastive critic.

    Logs separate feature tracks for the two state-action branches in DCC:

      * ``critic_sa_shared`` — z_shared = h_phi(b_shared(s, a))  (carried)
      * ``critic_sa_task``   — z_task  = phi_task(s, a)         (reset/task)
      * ``critic_sa_combined`` — z_sa used for contrastive scoring
      * ``critic_g`` — z_g = psi(g)

    Per-module weight norms (``critic/b_shared``, ``critic/h_phi``, …) help
    track plasticity on each continual-transfer group.
    """
    metrics = {}

    metrics['actor/weight_norm'] = weight_norm_l2(actor_params)
    metrics['actor/final_layer_norm'] = final_layer_norm(actor_params)
    metrics['critic/weight_norm'] = weight_norm_l2(critic_params)

    for key in ('b_shared', 'h_phi', 'phi_task', 'h_dyn', 'psi_shared',
                'psi_task', 'psi_proj'):
        if key in critic_params:
            metrics[f'critic/{key}/weight_norm'] = weight_norm_l2(
                critic_params[key])

    z_shared = decomp.apply_sa_shared_repr(critic_params, obs_batch, action_batch)
    z_task = decomp.apply_sa_task_repr(critic_params, obs_batch, action_batch)
    z_sa = decomp.apply_sa_repr(critic_params, obs_batch, action_batch)
    z_g = decomp.apply_g_repr(critic_params, goal_batch)

    h_phi_w = _get_encoder_final_kernel(critic_params.get('h_phi', {}))
    phi_task_w = _get_encoder_final_kernel(critic_params.get('phi_task', {}))
    psi_w = _get_encoder_final_kernel(critic_params.get('psi_shared', {}))
    if decomp.goal_encoder_mode == 'task_specific' and 'psi_task' in critic_params:
        psi_w = _get_encoder_final_kernel(critic_params['psi_task'])
    elif decomp.goal_encoder_mode in ('partial_shared', 'decomposed'):
        # NRC2 on the shared goal trunk; task-specific psi tracked separately.
        pass

    _append_feature_metrics(
        metrics, 'critic_sa_shared', z_shared, level, action_dim,
        final_kernel=h_phi_w,
    )
    _append_feature_metrics(
        metrics, 'critic_sa_task', z_task, level, action_dim,
        final_kernel=phi_task_w,
    )
    _append_feature_metrics(
        metrics, 'critic_sa_combined', z_sa, level, action_dim,
        final_kernel=h_phi_w if decomp.combine_mode == 'add' else None,
    )
    _append_feature_metrics(
        metrics, 'critic_g', z_g, level, nrc1_target_dim=1,
        final_kernel=psi_w,
    )

    if level == 'occasional' and 'psi_task' in critic_params and \
            decomp.goal_encoder_mode in ('partial_shared', 'decomposed',
                                         'task_specific'):
        z_g_task = decomp.psi_task.apply(
            critic_params['psi_task'], goal_batch)
        _append_feature_metrics(
            metrics, 'critic_g_task', z_g_task, level, nrc1_target_dim=1,
            final_kernel=_get_encoder_final_kernel(critic_params['psi_task']),
        )

    return metrics
