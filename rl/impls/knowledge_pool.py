"""Knowledge pool for CKA-RL style continual learning.

The pool stores per-task knowledge vectors (parameter deltas) at fixed
shape so the whole structure can live inside a JAX ``training_state``
pytree. Each pool slot has the same pytree structure as the policy (or
critic) parameters; an inactive slot is zero-filled and masked out via
``mask``.

Two pieces of state form the pool:

  * ``vectors``: pytree where every leaf has a leading axis of size
    ``capacity`` (= ``k_max + 1``); slot ``j`` holds the j-th
    knowledge vector when active and zeros otherwise.
  * ``mask``: jnp.ndarray of shape ``(capacity,)`` with ones for
    active slots and zeros for inactive slots.

This module also exposes:

  * ``CKAState``: dataclass bundling the pool, alpha_logits,
    alpha_scale, and their optimiser states.
  * ``compute_contribution``: stop-gradient-free pool blend that runs
    inside JIT, suitable for differentiation w.r.t. alpha_logits and
    alpha_scale.
  * ``append_vector_host``: append a new knowledge vector at task
    boundaries (host-side; mutates a Python copy of the pool).
  * ``merge_pair_host``: merge the two most cosine-similar vectors
    when ``mask.sum() > k_max``.

All host-side helpers operate on ``CKAPool`` immutably and return new
copies so they are safe to use across JAX retracing boundaries.
"""
from __future__ import annotations

from typing import Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax


# ---------------------------------------------------------------------------
# Pool data structure
# ---------------------------------------------------------------------------

@flax.struct.dataclass
class CKAPool:
    """Fixed-capacity knowledge pool that lives inside a JAX pytree.

    Attributes:
      vectors: pytree of arrays with leading dim ``capacity``.
      mask: bool array of shape ``(capacity,)`` marking active slots.
    """
    vectors: jax.tree_util.PyTreeDef  # type: ignore[type-arg]
    mask: jnp.ndarray


def empty_pool_like(template, capacity: int) -> CKAPool:
    """Create an empty CKAPool whose slots match ``template``'s structure."""
    vectors = jax.tree.map(
        lambda x: jnp.zeros((capacity,) + x.shape, dtype=x.dtype),
        template,
    )
    mask = jnp.zeros((capacity,), dtype=jnp.bool_)
    return CKAPool(vectors=vectors, mask=mask)


def pool_size(pool: CKAPool) -> int:
    """Number of active slots, host-side."""
    return int(jnp.sum(pool.mask))


# ---------------------------------------------------------------------------
# Differentiable pool contribution (runs inside JIT)
# ---------------------------------------------------------------------------

def compute_contribution(
    pool: CKAPool,
    alpha_logits: jnp.ndarray,
    alpha_scale: jnp.ndarray,
) -> "jax.tree_util.PyTreeDef":  # type: ignore[type-arg]
    """Compute Σ α_j v_j with masked softmax over active pool slots.

    The softmax is taken over alpha_logits with inactive slots set to
    -inf so that masked slots contribute exactly zero, regardless of
    alpha_scale. The output is a pytree with the same structure as a
    single knowledge vector (i.e., with the leading capacity axis
    removed).
    """
    # Mask inactive slots to -inf so softmax assigns them weight 0.
    masked_logits = jnp.where(pool.mask, alpha_logits * alpha_scale,
                              -jnp.inf)
    # If no slots are active, fall back to a uniform-zero distribution
    # whose product with the (zero) vectors is zero anyway. We still
    # need a numerically valid softmax to avoid NaN gradients.
    any_active = jnp.any(pool.mask)
    safe_logits = jnp.where(any_active, masked_logits,
                            jnp.zeros_like(masked_logits))
    alphas = jax.nn.softmax(safe_logits, axis=0)
    alphas = jnp.where(any_active, alphas,
                       jnp.zeros_like(alphas))

    # Per-leaf weighted sum along the leading capacity axis.
    def _blend(v_stack):
        # v_stack: [capacity, ...]; alphas: [capacity]
        # Reshape alphas to broadcast against the trailing dims of v_stack.
        broadcast_shape = (alphas.shape[0],) + (1,) * (v_stack.ndim - 1)
        return jnp.sum(alphas.reshape(broadcast_shape) * v_stack, axis=0)

    return jax.tree.map(_blend, pool.vectors)


# ---------------------------------------------------------------------------
# Bundled CKA state (pool + learnable scalars + optimiser states)
# ---------------------------------------------------------------------------

@flax.struct.dataclass
class CKAState:
    """All state needed to drive one CKA-decomposed component.

    For an actor: ``base_params`` is the frozen θ_base from task 0,
    ``v_k`` is the current task's knowledge vector being trained,
    ``pool`` holds prior-task knowledge vectors, and
    ``(alpha_logits, alpha_scale)`` are learnable scalars.

    The actor optimiser jointly updates ``v_k``, ``alpha_logits``, and
    ``alpha_scale`` so they are stored together as the ``trainable``
    pytree below. ``base_params`` and ``pool`` are constants from the
    optimiser's perspective (they receive no gradient inside the inner
    loop; ``pool`` is rewritten only at task boundaries).

    Attributes:
      base_params: frozen θ_base pytree (matches policy/critic param shape).
      v_k: trainable per-task delta pytree (same shape as base_params).
      pool: CKAPool of past-task knowledge vectors.
      alpha_logits: trainable logits, shape ``(capacity,)``.
      alpha_scale:  trainable scalar, shape ``()``.
    """
    base_params: "jax.tree_util.PyTreeDef"  # type: ignore[type-arg]
    v_k: "jax.tree_util.PyTreeDef"          # type: ignore[type-arg]
    pool: CKAPool
    alpha_logits: jnp.ndarray
    alpha_scale: jnp.ndarray


def init_cka_state(
    base_params,
    capacity: int,
    alpha_logits_init_std: float = 0.01,
    alpha_scale_init: float = 1.0,
) -> CKAState:
    """Build a CKAState for a fresh task.

    ``base_params`` should already reflect any post-task-zero freezing
    (i.e., it is the θ_base for task k ≥ 1). v_k is zero-initialised so
    the composed policy at the first inner-loop step equals
    θ_base + pool_contribution.
    """
    v_k = jax.tree.map(jnp.zeros_like, base_params)
    pool = empty_pool_like(base_params, capacity)
    alpha_logits = jnp.zeros((capacity,), dtype=jnp.float32)
    alpha_scale = jnp.array(alpha_scale_init, dtype=jnp.float32)
    return CKAState(
        base_params=base_params,
        v_k=v_k,
        pool=pool,
        alpha_logits=alpha_logits,
        alpha_scale=alpha_scale,
    )


def reinit_for_new_task(
    cka: CKAState,
    new_base_params,
    rng_key: jax.Array,
    alpha_logits_init_std: float = 0.01,
    alpha_scale_init: float = 1.0,
) -> CKAState:
    """Refresh v_k, alpha_logits, and alpha_scale at a task boundary.

    Pool and base_params are taken from the caller (typically the
    pool inherited from the prior task and a possibly-updated base).
    """
    v_k = jax.tree.map(jnp.zeros_like, new_base_params)
    capacity = cka.pool.mask.shape[0]
    alpha_logits = (
        jax.random.normal(rng_key, (capacity,)) * alpha_logits_init_std
    )
    alpha_logits = alpha_logits.astype(jnp.float32)
    # Mask inactive slots' logits to zero (purely cosmetic; softmax
    # ignores them via the mask) for cleaner logging.
    alpha_logits = jnp.where(cka.pool.mask, alpha_logits, 0.0)
    alpha_scale = jnp.array(alpha_scale_init, dtype=jnp.float32)
    return cka.replace(
        base_params=new_base_params,
        v_k=v_k,
        alpha_logits=alpha_logits,
        alpha_scale=alpha_scale,
    )


# ---------------------------------------------------------------------------
# Composed parameters helper
# ---------------------------------------------------------------------------

def compose(cka: CKAState) -> "jax.tree_util.PyTreeDef":  # type: ignore[type-arg]
    """θ' = θ_base + Σ α_j v_j + v_k."""
    contribution = compute_contribution(cka.pool, cka.alpha_logits,
                                        cka.alpha_scale)
    return jax.tree.map(lambda b, p, v: b + p + v,
                        cka.base_params, contribution, cka.v_k)


# ---------------------------------------------------------------------------
# Host-side pool mutation (only at task boundaries)
# ---------------------------------------------------------------------------

def _flatten_for_sim(v) -> jnp.ndarray:
    return jnp.concatenate([x.reshape(-1)
                            for x in jax.tree_util.tree_leaves(v)])


def append_vector_host(
    pool: CKAPool,
    new_vector,
    k_max: int,
) -> CKAPool:
    """Append ``new_vector`` and merge if ``mask.sum() > k_max``.

    Operates entirely on host (no JIT). Returns a new ``CKAPool``.
    """
    capacity = pool.mask.shape[0]
    n_active = int(jnp.sum(pool.mask))
    if n_active >= capacity:
        # Pool is full; do an in-place merge first to free a slot.
        pool = _merge_most_similar_pair_host(pool)
        n_active = int(jnp.sum(pool.mask))

    # Insert into the first inactive slot.
    insert_idx = n_active
    new_vectors = jax.tree.map(
        lambda stack, leaf: stack.at[insert_idx].set(leaf),
        pool.vectors, new_vector,
    )
    new_mask = pool.mask.at[insert_idx].set(True)
    pool = CKAPool(vectors=new_vectors, mask=new_mask)

    # If the active count now exceeds k_max, merge the two most
    # similar vectors. (The pool's *capacity* is k_max + 1, so this
    # condition can only fire once.)
    if int(jnp.sum(pool.mask)) > k_max:
        pool = _merge_most_similar_pair_host(pool)

    return pool


def _merge_most_similar_pair_host(pool: CKAPool) -> CKAPool:
    """Merge the two most cosine-similar active slots into one (host)."""
    capacity = pool.mask.shape[0]
    active_indices = [i for i in range(capacity) if bool(pool.mask[i])]
    if len(active_indices) < 2:
        return pool

    # Materialise active vectors flat, vectorised once.
    actives = []
    for idx in active_indices:
        leaves = jax.tree_util.tree_leaves(
            jax.tree.map(lambda stack: stack[idx], pool.vectors)
        )
        actives.append(jnp.concatenate([l.reshape(-1) for l in leaves]))
    flat = jnp.stack(actives, axis=0)  # [n_active, D]
    norms = jnp.linalg.norm(flat, axis=1) + 1e-8
    sims = (flat @ flat.T) / (norms[:, None] * norms[None, :])
    # Mask out the diagonal and lower triangle.
    n = flat.shape[0]
    inf_mask = jnp.tril(jnp.ones((n, n), dtype=jnp.bool_), k=0)
    sims = jnp.where(inf_mask, -jnp.inf, sims)
    flat_argmax = int(jnp.argmax(sims))
    i, j = divmod(flat_argmax, n)
    src_a = active_indices[i]
    src_b = active_indices[j]

    # Average the two slots; write the result into src_a, free src_b.
    avg_vectors = jax.tree.map(
        lambda stack: stack.at[src_a].set((stack[src_a] + stack[src_b]) / 2.0),
        pool.vectors,
    )
    # Zero out the freed slot to keep the pool tidy.
    avg_vectors = jax.tree.map(
        lambda stack: stack.at[src_b].set(jnp.zeros_like(stack[src_b])),
        avg_vectors,
    )
    new_mask = pool.mask.at[src_b].set(False)

    # Compact the pool so active slots occupy the leading positions.
    return _compact_pool(CKAPool(vectors=avg_vectors, mask=new_mask))


def _compact_pool(pool: CKAPool) -> CKAPool:
    """Move all active slots to the leading positions, in-order.

    Implementation: build a permutation that lists active source
    indices first, then inactive ones, then index every leaf along
    its leading axis with that permutation. Inactive slots are
    overwritten to zero by a final masked select.
    """
    capacity = pool.mask.shape[0]
    active_indices = [i for i in range(capacity) if bool(pool.mask[i])]
    inactive_indices = [i for i in range(capacity)
                        if not bool(pool.mask[i])]
    perm = jnp.array(active_indices + inactive_indices, dtype=jnp.int32)

    permuted_vectors = jax.tree.map(
        lambda stack: stack[perm], pool.vectors,
    )
    new_mask = jnp.zeros_like(pool.mask)
    new_mask = new_mask.at[:len(active_indices)].set(True)
    # Zero out the trailing (inactive) entries for cleanliness.
    permuted_vectors = jax.tree.map(
        lambda stack: jnp.where(
            new_mask.reshape((-1,) + (1,) * (stack.ndim - 1)),
            stack,
            jnp.zeros_like(stack),
        ),
        permuted_vectors,
    )
    return CKAPool(vectors=permuted_vectors, mask=new_mask)


# ---------------------------------------------------------------------------
# Convenience: zeros_like for arbitrary pytrees
# ---------------------------------------------------------------------------

def pytree_zeros_like(pytree):
    return jax.tree.map(jnp.zeros_like, pytree)
