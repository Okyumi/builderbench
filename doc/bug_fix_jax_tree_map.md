# Bug Fix — `jax.tree_map` removed in JAX 0.6

**Date.** 2026-04-23
**Symptom.** Runs with either `actor_mode=cka` or `critic_mode=cka`
crashed at the first pool operation with

```
AttributeError: jax.tree_map was removed in JAX v0.6.0: use jax.tree.map
(jax v0.4.25 or newer) or jax.tree_util.tree_map (any JAX version).
```

## Diagnosis

`jax.tree_map` was deprecated in JAX 0.4.25 and removed in JAX 0.6.0.
All live call sites were in `rl/impls/knowledge_pool.py`, which backs
both the actor and critic CKA paths in `continual_crl.py`:

- `actor_pool = KnowledgePool(k_max=args.k_max)` (line 1122)
- `critic_pool = KnowledgePool(k_max=args.k_max)` (line 1123)

and their uses of `compute_contribution`, `append` (pool merge),
`state_dict`, and `pytree_zeros_like`. Both CKA modes therefore shared
the same failure.

The four call sites were:

| # | Location | Purpose |
|---|---|---|
| 1 | `_merge_pair` in pool-overflow handling | Average two pool vectors when `|V| > K_max`. |
| 2 | `compute_contribution` | Softmax-weighted sum `Σ α_j v_j`. |
| 3 | `state_dict` (identity copy) | Deep-copy vectors for checkpointing. |
| 4 | `pytree_zeros_like` | Build a zero-valued pytree with matching structure. |

## Fix

Replace every `jax.tree_map(...)` with `jax.tree.map(...)`. `jax.tree.map`
is the canonical replacement, available in any JAX ≥ 0.4.25 and
required in 0.6+. No semantic change.

No changes needed elsewhere: a repository-wide `grep` for `jax.tree_map`
across the live source tree (`rl/impls/` excluding `wandb/` run
snapshots) now returns zero matches.

## Sanity checks to run

```bash
cd /scratch/yd2247/builderbench

# Syntactic import check — no runtime needed.
python -c "from rl.impls.knowledge_pool import KnowledgePool, pytree_zeros_like"

# Quick CKA smoke test on the shortest possible continual sequence.
STEPS_PER_TASK=10000 BASE_STEPS=10000 \
TASK_SEQUENCE='cube-1-task1,cube-2-task1' \
TASKS_PER_GPU=1 \
sbatch --time=01:00:00 --array=0-0 rl/impls/draft_4.sh
# In experiment_configs.py, temporarily set
#   ACTOR_MODES = ['cka']; CRITIC_MODES = ['cka']; SEEDS = [1]
# to exercise both paths.
```

## Unrelated note

The same `.err` included the JAX cuSPARSE fallback warning documented
in `doc/bug_fix_obs_padding.md`; fix remains `pip install
nvidia-cusparse-cu12` inside the conda env.
