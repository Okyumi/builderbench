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
    """Extract the final Dense layer's kernel from a Flax encoder.

    Flax keys: {'params': {'Dense_0': {'kernel': ...}, ..., 'Dense_4': {'kernel': ...}}}
    We want the highest-indexed Dense layer's kernel.
    """
    p = encoder_params.get('params', encoder_params)
    best_w = None
    best_idx = -1
    for key in p:
        key_str = str(key)
        if 'Dense' in key_str:
            node = p[key]
            if isinstance(node, dict) and 'kernel' in node:
                # Extract index: 'Dense_0' -> 0, 'Dense_4' -> 4
                parts = key_str.split('_')
                try:
                    idx = int(parts[-1])
                except (ValueError, IndexError):
                    idx = 0
                if idx > best_idx:
                    best_idx = idx
                    best_w = node['kernel']
    return best_w


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

    # Feature entropy and Gini
    metrics['critic_sa/entropy'] = feature_entropy(sa_feats)
    metrics['critic_g/entropy'] = feature_entropy(g_feats)
    metrics['critic_sa/gini'] = gini_sparsity(sa_feats)
    metrics['critic_g/gini'] = gini_sparsity(g_feats)

    if level == 'occasional':
        # ---- OCCASIONAL ----
        metrics['critic_sa/feature_rank'] = feature_rank(sa_feats, tau=0.99)
        metrics['critic_g/feature_rank'] = feature_rank(g_feats, tau=0.99)

        metrics['critic_sa/nrc1'] = compute_nrc1(sa_feats, target_dim=action_dim)
        metrics['critic_g/nrc1'] = compute_nrc1(g_feats, target_dim=1)

        sa_final_w = _get_encoder_final_kernel(critic_params.get('sa_encoder', {}))
        g_final_w = _get_encoder_final_kernel(critic_params.get('g_encoder', {}))
        if sa_final_w is not None:
            metrics['critic_sa/nrc2'] = compute_nrc2(sa_feats, sa_final_w)
        if g_final_w is not None:
            metrics['critic_g/nrc2'] = compute_nrc2(g_feats, g_final_w)

        metrics['critic_sa/dormant_ratio'] = dormant_ratio(sa_feats, dormant_pct=1e-5)
        metrics['critic_g/dormant_ratio'] = dormant_ratio(g_feats, dormant_pct=1e-5)

    return metrics
