"""Knowledge pool for CKA-RL style continual learning.

Stores per-task knowledge vectors (parameter deltas) and computes
blending contributions via softmax-weighted sums.

Compatible with Flax parameter pytrees (nested dicts of JAX arrays).
"""
import jax
import jax.numpy as jnp
from typing import List


class KnowledgePool:
    """Pool of knowledge vectors with fixed capacity and cosine-similarity merging."""

    def __init__(self, k_max: int = 10):
        self.k_max = k_max
        self._vectors: List = []

    def __len__(self):
        return len(self._vectors)

    def append(self, v_k):
        """Add a knowledge vector to the pool."""
        self._vectors.append(v_k)
        self.merge_if_needed()

    def merge_if_needed(self):
        """If pool exceeds k_max, merge the two most cosine-similar vectors."""
        while len(self._vectors) > self.k_max:
            flat_vecs = [
                jnp.concatenate([x.flatten() for x in jax.tree_util.tree_leaves(v)])
                for v in self._vectors
            ]
            n = len(flat_vecs)
            best_sim = -float('inf')
            best_i, best_j = 0, 1
            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(jnp.dot(flat_vecs[i], flat_vecs[j]) / (
                        jnp.linalg.norm(flat_vecs[i]) * jnp.linalg.norm(flat_vecs[j]) + 1e-8
                    ))
                    if sim > best_sim:
                        best_sim = sim
                        best_i, best_j = i, j
            merged = jax.tree.map(
                lambda a, b: (a + b) / 2.0,
                self._vectors[best_i], self._vectors[best_j],
            )
            self._vectors = [
                v for idx, v in enumerate(self._vectors)
                if idx != best_i and idx != best_j
            ] + [merged]

    def compute_contribution(self, alpha_logits=None, alpha_scale=1.0):
        """Compute pool contribution: Σ α_j v_j.

        Args:
            alpha_logits: learnable logits of shape ``(len(pool),)``.
                If ``None``, uses uniform weights.
            alpha_scale: scaling factor applied before softmax.

        Returns:
            A pytree with the same structure as the stored vectors,
            or ``None`` if the pool is empty.
        """
        if not self._vectors:
            return None
        n = len(self._vectors)
        if alpha_logits is None:
            alpha_logits = jnp.zeros(n)
        alphas = jax.nn.softmax(alpha_logits * alpha_scale)
        result = jax.tree.map(
            lambda *vs: sum(a * v for a, v in zip(alphas, vs)),
            *self._vectors,
        )
        return result

    def get_vectors(self):
        return list(self._vectors)

    def state_dict(self):
        """Serialize pool for checkpointing."""
        return {
            'vectors': [jax.tree.map(lambda x: x, v) for v in self._vectors],
            'k_max': self.k_max,
        }

    def load_state_dict(self, state):
        """Load pool from checkpoint."""
        self._vectors = state['vectors']
        self.k_max = state['k_max']


def pytree_zeros_like(pytree):
    """Create a zero-valued pytree with the same structure."""
    return jax.tree.map(jnp.zeros_like, pytree)
