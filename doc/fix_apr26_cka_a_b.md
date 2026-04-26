# Fix A + B — Learnable α and JIT Trace Reuse for CKA

**Date.** 2026-04-26
**Symptoms addressed.** `actor_mode=cka` showed no improvement (and
sometimes a degradation) over `reset` and `persistent`, and ran 3–10×
slower per task than the other modes. Both observations are attributed
to the same family of issues identified in
`doc/audit_apr26_cka_correctness.md`.

This change implements **Fix A** (learnable `alpha_logits` and
`alpha_scale`) and **Fix B** (move pool vectors and CKA scalars into
the JAX `training_state` pytree so JIT-compiled inner functions trace
once and reuse the cache across tasks). Fixes C, D, E from the audit
remain pending.

## What changed

### `rl/impls/knowledge_pool.py` — new API

The old `KnowledgePool` Python class is replaced with a JAX-friendly
data flow:

- `CKAPool` (Flax struct): pool of knowledge vectors stored as a
  pytree with a leading axis of size `K_max + 1`, plus a boolean mask
  of the same length marking active slots.
- `CKAState` (Flax struct): bundle of `(base_params, v_k, pool,
  alpha_logits, alpha_scale)`. This entire bundle lives inside
  `training_state`, so every JAX function that touches it can be
  traced once and reused.
- `compute_contribution(pool, alpha_logits, alpha_scale)`: differentiable
  blend of the pool vectors. Inactive slots are masked to `-∞` before
  the softmax, so they contribute zero regardless of `alpha_scale`.
- `init_cka_state` and `reinit_for_new_task`: build / refresh CKA state
  at task boundaries.
- `append_vector_host(pool, v, k_max)`: host-side append + merge.
  Operates on Python copies of `CKAPool` so it is safe outside JIT
  and only runs at task boundaries.
- `_merge_most_similar_pair_host`: vectorised cosine-similarity
  computation in one matrix multiply, replacing the nested Python
  loop with `float(...)` host syncs.
- `_compact_pool`: rearranges active slots to the leading positions so
  that, for a pool of `n` actives, slots `[:n]` are dense and slots
  `[n:]` are zeros with `mask=False`. Permutation-based, no Python
  loops.

### `rl/impls/continual_crl.py` — wiring

- `CRLTrainingState` gains two optional slots: `actor_cka:
  Optional[CKAState]` and `critic_cka: Optional[CKAState]`. Both are
  `None` when the corresponding component is not in CKA mode, so the
  non-CKA paths are byte-identical to before.
- For CKA modes, `actor_state.params` becomes the trainable bundle
  ```python
  {'v_k': pytree, 'alpha_logits': [capacity], 'alpha_scale': scalar}
  ```
  The single Adam optimiser updates all three jointly. A single
  `value_and_grad` call w.r.t. the bundle gives gradients for all
  three; the gradient w.r.t. `(alpha_logits, alpha_scale)` flows
  through `compute_contribution`, exactly as the original CKA-RL
  intends.
- New helpers `_cka_trainable_init` and `_cka_compose_from_trainable`
  replace the old `_compose_params(base, pool_c, vk)` closure.
- The actor and critic update functions take `cka_state` and the
  trainable bundle as ordinary arguments, with no Python closures
  capturing per-task constants. As a result, the inner
  `update_actor_and_alpha` and `update_critic` `@jax.jit` closures
  are constructed once and the JIT trace is reused across all tasks.
- The driver no longer maintains separate `actor_pool` and
  `critic_pool` Python objects. Instead it holds two
  `Optional[CKAState]` references and does one host-side
  `append_vector_host(...)` call at the end of each CKA task.
- Checkpoints now store `actor_cka_state` and `critic_cka_state`
  directly (Flax structs are picklable). The old `actor_pool`,
  `actor_base_params`, `critic_pool`, `critic_base_params` keys are
  no longer written.
- Logged diagnostics now include `actor_alpha_max`,
  `actor_alpha_entropy`, `actor_alpha_scale`,
  `critic_alpha_max`, `critic_alpha_entropy`, `critic_alpha_scale`.
  These give us a direct view of whether α is doing anything.

### Removed

- The seeded zero vector in the pool at task 0 is dropped. With
  `alpha_logits` learnable, the placeholder is unnecessary; a
  genuinely empty pool at task 1 is cleaner. (This was a side effect
  of the algorithm-level fix; an explicit Fix C will revisit any
  remaining seeding decisions.)

## Verification

Two host-side smoke tests confirm the new pool and CKA state behave
correctly:

```
single-slot test:  gradient flows through v_k as expected;
                   alpha_logits/alpha_scale grad = 0 (correct: a
                   1-active softmax has zero logit gradient).

two-slot test:     gradient flows through v_k AND alpha_logits AND
                   alpha_scale, with the two active slots receiving
                   equal-and-opposite gradients that sum to zero
                   (the standard softmax property).
```

The full repo `ast.parse` succeeds for both `continual_crl.py` and
`knowledge_pool.py`. A grep for stale references to the old API
(`KnowledgePool(`, `actor_pool.`, `critic_pool.`, `cka_actor_base`,
`actor_base_params =`, etc.) returns zero matches in the current file.

## Acceptance criteria for the next run

- `actor_mode=cka` should match or exceed `actor_mode=persistent` on
  the 9-cell ablation by the end of task 1, averaged over seeds.
- Per-task wall-clock for `cka` should be within roughly 10–20% of
  `persistent`, not the previous 3–10× slower.
- The new W&B series `actor_alpha_max`, `actor_alpha_entropy`, and
  `actor_alpha_scale` should show non-uniform values that change over
  training. A flat curve at the uniform-distribution point would mean
  the optimiser is ignoring α and we should investigate further.

## Pending

- Fix C: clean up any remaining seeded-zero behaviour (most of it is
  already gone; this fix will sweep the codebase for remnants).
- Fix D: the host-side cosine merge is already vectorised; the
  remaining `_compact_pool` permutation can also be reviewed.
- Fix E: replace string-based head detection in `_is_actor_head_leaf`
  with an explicit head tag on the actor module.
- Negative-bank variant: deferred until the corrected CKA matches its
  acceptance criteria on a smoke run.
