# Implementation Tracking

## Status: Phase 6 — Batch Experiment Pipeline

### Fix A + B (Apr 26, 2026) — learnable α and JIT trace reuse

- [x] Replaced the old `KnowledgePool` Python class with a JAX-friendly
  pool data flow (`CKAPool` + `CKAState` Flax structs); pool vectors
  now live as a pytree with leading axis `K_max + 1` plus a boolean
  mask of active slots, so the entire pool can sit inside the JAX
  `training_state` and feed JIT-compiled inner functions.
- [x] Made `alpha_logits` (length `K_max + 1`) and `alpha_scale`
  (scalar) into per-task learnable parameters. They sit inside the
  trainable bundle `{'v_k', 'alpha_logits', 'alpha_scale'}` that the
  actor's (resp. critic's) Adam optimiser updates jointly. The
  gradient through `(alpha_logits, alpha_scale)` flows via the masked
  softmax in `compute_contribution`.
- [x] Removed the closure capture of `actor_base_params` /
  `actor_pool_c` / `critic_base_params` / `critic_pool_c` from the
  inner-loop `@jax.jit` functions; everything reaches them through
  `training_state.actor_cka` / `training_state.critic_cka`. The inner
  functions are now built once per process; the JIT cache is reused
  across all tasks.
- [x] Vectorised the cosine-similarity merge (`_merge_most_similar_pair_host`
  in `knowledge_pool.py`); replaced the previous nested Python loop
  with one matmul. The merge runs only at task boundaries.
- [x] Dropped the seeded zero vector at task 0 (the original code
  appended a zero-valued knowledge vector that, under uniform-α,
  permanently halved every later pool contribution).
- [x] Added W&B-logged α diagnostics: `actor_alpha_max`,
  `actor_alpha_entropy`, `actor_alpha_scale`, and the symmetric
  critic series. A flat curve here would indicate the optimiser is
  ignoring α.
- [x] Smoke tests on a toy actor confirm gradients flow through `v_k`
  alone (single-active-slot pool), and through `(v_k, alpha_logits,
  alpha_scale)` (two-active-slot pool). Two slots produce
  equal-and-opposite logit gradients summing to zero, the standard
  softmax property.
- [x] `doc/fix_apr26_cka_a_b.md`: full write-up.
- [x] Acceptance criteria for the next run: `cka` matches or exceeds
  `persistent` by end of task 1; per-task wall-clock within 10–20%
  of `persistent`; α series non-uniform and drifting.

### Audit (Apr 26, 2026) — CKA correctness and slowdown

- [x] Cross-checked BuilderBench's CKA implementation against the
  original CKA-RL reference (`sgcrl/cka-rl-meta-world`) and the SGCRL
  contrastive port (`sgcrl/contrastive/{knowledge_pool,
  continual_learning}.py`). See `doc/audit_apr26_cka_correctness.md`.
- [x] Root cause of "`actor_mode=cka` is no better than `persistent`":
  the learnable `alpha_logits` and `alpha_scale` are missing entirely.
  `compute_contribution(alpha_logits=None, ...)` falls back to a
  uniform average over the pool; `args.alpha_scale` is a Python
  hyperparameter captured into the JIT trace. The composition is
  therefore $\theta'_k = \theta_{\text{base}} + \frac{1}{|V|}\sum_j v_j + v_k$
  with no learning of the attention, which is strictly worse than
  `persistent` because the agent starts from a contaminated init
  pulled by every prior task in equal measure.
- [x] Root cause of slowdown: `actor_base_params` and `actor_pool_c`
  are closed-over outside the `training_state` pytree, so the inner
  `@jax.jit` functions are rebuilt and re-traced at every task
  boundary. `reset` and `persistent` reuse the cached trace.
- [ ] Fix A (priority 1, correctness): make `alpha_logits` and
  `alpha_scale` learnable parameters in `training_state` with their
  own optimisers (mirrors `sgcrl/contrastive/continual_learning.py:
  beta_k, alpha_scale`).
- [ ] Fix B (priority 1, performance): move pool vectors into
  `training_state` as fixed-size leading-axis arrays with a mask, so
  closures no longer capture per-task constants and the JIT trace is
  reused across tasks.
- [ ] Fix C: drop the seeded zero vector at task 0.
- [ ] Fix D: vectorise `_merge_pair` cosine similarity.
- [ ] Fix E: replace string-based head detection with explicit head
  tagging on the actor module.

### Bug fix (Apr 23, 2026) — `jax.tree_map` removed in JAX 0.6

- [x] Root-cause. Runs with `actor_mode=cka` or `critic_mode=cka`
  crashed with `AttributeError: jax.tree_map was removed in JAX
  v0.6.0`. All four live call sites were in
  `rl/impls/knowledge_pool.py`, which backs both the actor pool and
  the critic pool in `continual_crl.py`.
- [x] Fix. Replace `jax.tree_map` with `jax.tree.map` in all four
  locations: pool-merge averaging, softmax-weighted contribution,
  checkpoint deep-copy, and `pytree_zeros_like`. No semantic change.
- [x] Repository-wide `grep` for `jax.tree_map` across the live
  source tree now returns zero matches.
- [x] `doc/bug_fix_jax_tree_map.md`: full diagnosis, call-site table,
  and post-fix smoke-test instructions.

### Bug fix (Apr 21, 2026) — observation padding dim

- [x] Root-caused `flax.errors.ScopeParamShapeError (121, 1024) vs (115, 1024)` in `continual_crl.py::actor_step`: `pad_wrapper.FIXED_OBS_SUFFIX` was set to 8 but the real `finger_pos` dim is 2 (from `build_block.py:152-155`, indexing `['left_driver_joint', 'right_driver_joint']`). The actor was initialised at declared `UNIFIED_OBS_DIM = 57` but the runtime obs was 51. See `doc/bug_fix_obs_padding.md`.
- [x] `rl/impls/utils/pad_wrapper.py`: `FIXED_OBS_SUFFIX = 2` (was 8) → `UNIFIED_OBS_DIM = 51`. Module docstring rewritten to cite the real obs layout. `_pad_obs` gains a runtime assertion so any future layout drift fails loudly with a clear message instead of a cryptic Flax error deep in the stack.
- [x] No changes needed in `continual_crl.py`: both `obs_size = UNIFIED_OBS_DIM` sites (lines 577, 1106) pick up the corrected value automatically.
- [x] `doc/bug_fix_obs_padding.md`: full diagnosis and post-fix smoke-test instructions.
- [ ] Pending: re-run the 9-cell grid once cuSPARSE is installed on the conda env (the same `.err` contained a cuSPARSE load failure that forces JAX to CPU; documented in the bug fix doc but fix is environmental, not in-repo).

### Added in Phase 6 (Apr 21, 2026)

- [x] `rl/impls/experiment_configs.py` — enumerates the 3 × 3 × 3 actor/critic/seed grid (27 configs); emits shell-sourceable `KEY=VALUE` lines for each config. Mirrors `sgcrl/experiment_configs.py`.
- [x] `rl/impls/draft_4.sh` — SLURM job-array launcher. Runs `TASKS_PER_GPU` experiments in parallel on one GPU; adapts `XLA_PYTHON_CLIENT_MEM_FRACTION` to the parallelism; writes identical headers to both `.out` and `.err`. Mirrors `sgcrl/draft_4.sh`.
- [x] Negative-bank flags are exposed as placeholders. The launcher probes `continual_crl.py` at startup (`grep -q neg_bank_mode`) and emits `--neg_bank_*` flags only if the driver accepts them. When the bank is wired into the driver later, the launcher needs no changes.
- [x] `doc/batch_experiments.md` — implementation note for the batch pipeline.
- [x] `doc/hpc_commands.md` — cheatsheet of commands to run the project on the HPC terminal (setup, submission, monitoring, debug, W&B sync, negative-bank re-run once wired in).

### Completed

- [x] Study BuilderBench paper, blog post, and codebase
- [x] Analyze environment: state/action/goal spaces, reward, parallelization
- [x] Identify key challenge: variable observation dimensions across cube counts
- [x] Write comprehensive implementation plan (`doc/implementation_plan.md`)
- [x] Initial commit to repository
- [x] Create `rl/impls/continual_crl.py` — continual training driver with:
  - Task sequence support
  - Actor/critic transfer logic (reset / persistent / cka placeholder)
  - Checkpoint save/load with config-keyed paths
  - Auto-resume from latest checkpoint
  - Cross-task evaluation after each task
  - W&B logging integration
- [x] Design 12-task mixed-cube sequence (2 × cube-1, 5 × cube-2, 5 × cube-3)
- [x] Create `rl/impls/utils/pad_wrapper.py` — observation/goal padding wrapper:
  - Pads obs from variable dims to UNIFIED_OBS_DIM=57 (3-cube max)
  - Pads goals from variable dims to UNIFIED_GOAL_DIM=9
  - Wraps outside AutoResetWrapper so reset internals stay in raw shape
  - Enables network weight transfer across tasks with different cube counts
- [x] Update `continual_crl.py` to use PaddedEnvWrapper for all tasks
- [x] Update `continual_crl.py` to use UNIFIED_OBS_DIM/UNIFIED_GOAL_DIM for network creation

- [x] Phase 3: CKA actor decomposition in Flax:
  - Created `rl/impls/knowledge_pool.py` — pool with cosine-similarity merging
  - CKA composition: θ' = θ_base + pool_c + v_k inside JIT-compiled training step
  - Actor and critic CKA modes fully functional
  - Gradients flow through composition to v_k (base and pool_c are additive constants)
  - Uniform pool blending (TODO: learnable alpha)
- [x] Phase 4: CKA critic decomposition (implemented alongside actor)

- [x] Phase 5: Parity with SGCRL reference implementation:
  - **adapt_heads_only** (default `True`): head/body splitting for CKA actor
  - **encoder_from_base** (default `False`): gradient masking for body params
  - **Post-task head/body splitting**: body v_k folded into θ_base, only head deltas pooled
  - **rl_metrics module**: full Flax port (weight_norm, final_layer_norm, feature_rank, NRC1/2, dormant_ratio, entropy, Gini)
  - **base_steps**: separate env step budget for task 0
  - **Checkpoint robustness**: JAX→numpy conversion for pickling
  - **RL metrics integration**: frequent/occasional logging to W&B

### In Progress

- [ ] End-to-end testing on GPU cluster

### Next Steps

- [ ] Run baseline experiments (all 9 actor×critic configurations)
- [ ] Learnable alpha blending weights
- [ ] Video recording during cross-task evaluation

---

## Decision Log

### 2026-04-19: SGCRL Parity — adapt_heads_only and encoder_from_base

**Decision:** Port the `adapt_heads_only` and `encoder_from_base` flags from SGCRL to BuilderBench, matching the exact CKA-RL algorithm.

**Rationale:** These are core CKA-RL features that were missing from the initial BuilderBench implementation:

1. **adapt_heads_only** (default `True`): After each task, the v_k delta is split:
   - Body params (Dense_0..3, LayerNorm_0..3): folded into θ_base — the encoder evolves across tasks
   - Head params (Dense_4 mean, Dense_5 log_std): stored in the knowledge pool as vectors
   - This matches the CKA-RL paper's design: the representation encoder adapts freely while only the policy head is decomposed.

2. **encoder_from_base** (default `False`): When `True`, body gradients are zeroed during training, freezing the encoder at the base-task values. CKA-RL's default is `False` — the body receives gradients and evolves. This flag exists for ablation studies.

**Implementation detail (Flax vs Haiku):**

In SGCRL (Haiku), head detection uses `'Normal' in path_str` because Haiku's NormalTanhDistribution creates modules named `Normal/linear` and `Normal/linear_1`.

In BuilderBench (Flax), the Actor uses `@nn.compact` with auto-naming. The body is Dense_0..3 + LayerNorm_0..3, and the head is Dense_4 (mean) + Dense_5 (log_std). Detection uses `ACTOR_HEAD_LAYER_NAMES = ('Dense_4', 'Dense_5')`.

### 2026-04-19: RL Representation Metrics

**Decision:** Port the full `rl_metrics` module from SGCRL, adapted for Flax.

**Metrics implemented:**
| Metric | Level | Description |
|---|---|---|
| weight_norm_l2 | frequent | L2 norm of all params (actor and critic) |
| final_layer_norm | frequent | L2 norm of actor head (Dense_4) kernel |
| feature_entropy | frequent | Shannon entropy of |feature| distributions |
| gini_sparsity | frequent | Gini coefficient for feature sparsity |
| feature_rank | occasional | Effective rank via SVD (τ=0.99) |
| nrc1 | occasional | Neural Rank Collapse: subspace collapse |
| nrc2 | occasional | Feature-weight alignment |
| dormant_ratio | occasional | Fraction of dead neurons (threshold 1e-5) |

**Flax adaptation:** Feature extraction calls `sa_encoder.apply(...)` and `g_encoder.apply(...)` directly. Final layer kernel detection uses Flax's `Dense_N/kernel` naming instead of Haiku's `sa_encoder/linear_N/w`.

### 2026-04-19: Post-task Persistent Actor Fix

**Decision:** Fix persistent actor mode to match SGCRL behaviour.

**Previous behaviour:** `prev_actor_params = actor_state.params` — this is correct for reset and persistent modes but the composed policy was not properly extracted when CKA was NOT used. Now, the `composed_actor` is always computed before any pool updates, ensuring the snapshot is correct.

### 2026-04-19: base_steps Flag

**Decision:** Add `base_steps` flag (default 50M) allowing task 0 to have a different training budget.

**Rationale:** In SGCRL, `base_steps` is separate from `steps_per_task` because the base task may need more/fewer steps. This is important for experiments where the base representation needs thorough training before continual learning begins.

### 2026-04-19: Checkpoint Robustness

**Decision:** Convert JAX arrays to numpy before pickling, matching SGCRL.

**Rationale:** JAX arrays use device-specific memory layouts. Pickling JAX arrays directly can fail when loading on a different device or after JAX version changes. The SGCRL approach (`jax.tree_map(np.array, ...)` on save, `jax.tree_map(jnp.array, ...)` on load) is portable and robust.

### 2026-04-18: 12-Task Sequence (Revised)

**Decision:** Use 12 tasks spanning 1, 2, and 3 cubes with zero-padding to unify dimensions.

**Previous approach:** 5 same-cube-count tasks (Phase 1) then mixed (Phase 2).
**New approach:** Single unified sequence, padding handles dimension mismatch directly.

**12-task sequence:**

| # | Task | Cubes | Difficulty | Skill tested |
|---|---|---|---|---|
| 0 | cube-1-task1 | 1 | Easy | Place at target |
| 1 | cube-1-task2 | 1 | Easy | Pick and lift |
| 2 | cube-2-task1 | 2 | Easy | Stack two cubes |
| 3 | cube-2-task2 | 2 | Easy | Horizontally align |
| 4 | cube-2-task3 | 2 | Easy | Tilting (rotation insight) |
| 5 | cube-3-task1 | 3 | Easy | Stack three cubes |
| 6 | cube-3-task3 | 3 | Hard | Inverted T lift |
| 7 | cube-2-task4 | 2 | Hard | Double horizontal pick |
| 8 | cube-2-task5 | 2 | Hard | Double vertical pick |
| 9 | cube-3-task2 | 3 | Hard | T-block (rotate base 45°) |
| 10 | cube-3-task4 | 3 | Hard | 2D packing problem |
| 11 | cube-3-task5 | 3 | Hard | Support structure (scaffolding) |

**Design rationale:**
- Starts with simplest motor skills (1-cube), builds to multi-object (2-cube), then to complex (3-cube)
- Interleaves cube counts: easy 3-cube tasks appear before hard 2-cube tasks
- This tests whether learned reachability/gripper control transfers across different object counts
- Hard tasks are concentrated toward the end but not exclusively: tests whether skills from easy→medium tasks provide forward transfer for hard tasks
- The sequence includes tasks that require qualitatively different skills (placement, stacking, alignment, rotation, dual manipulation, scaffolding)

### 2026-04-18: Observation Padding Strategy

**Decision:** Use zero-padding to unify observation/goal dimensions across all tasks.

**Implementation:**
- MAX_CUBES = 3 → UNIFIED_OBS_DIM = 57, UNIFIED_GOAL_DIM = 9
- PaddedEnvWrapper sits outside AutoResetWrapper
- Extra cube slots are filled with zeros in obs (pos, quat, vel) and goal
- Networks always operate on 57-dim obs and 9-dim goals regardless of actual cube count

**Trade-off:** Padding introduces zeros that the network must learn to ignore. For 1-cube tasks, 26 of 57 obs dims and 6 of 9 goal dims are zero. This could slightly hurt single-task performance but enables weight transfer across cube counts — which is the entire point of the continual setting.

### 2026-04-18: Architecture Decision (unchanged)

**Decision:** Keep the BuilderBench CRL architecture (1024-width, LayerNorm, Swish, L2 energy) as-is. Do not change any GCCRL configuration.

---

## Change Log

### 2026-04-19 (session 3) — SGCRL Parity Update

**New file: `rl/impls/rl_metrics.py`**
- Full RL representation metrics module for Flax
- weight_norm_l2, final_layer_norm (Dense_4 detection), feature_entropy, gini_sparsity
- feature_rank (SVD), compute_nrc1, compute_nrc2, dormant_ratio
- extract_critic_features using Flax encoder `.apply()`
- _get_encoder_final_kernel with Flax Dense_N naming
- compute_all_metrics dispatcher (frequent / occasional levels)

**Updated: `rl/impls/continual_crl.py`**

Major additions:
1. **adapt_heads_only flag** (default `True`):
   - `ACTOR_HEAD_LAYER_NAMES = ('Dense_4', 'Dense_5')` — Flax head layer detection
   - `_is_actor_head_leaf(path)` — checks if a param path belongs to the head
   - `_split_head_body(base_params, vk_params)` — post-task v_k splitting
   - Post-task logic: when `adapt_heads_only=True`, body v_k folded into base, head v_k stored in pool

2. **encoder_from_base flag** (default `False`):
   - `_mask_body_grads(grads)` — zeros out non-head gradients using `tree_map_with_path`
   - `freeze_actor_body` computed once at start of `train_single_task()`
   - Applied inside CKA actor update after `value_and_grad`

3. **base_steps flag** (default 50M):
   - `task_steps = args.base_steps if task_idx == 0 else args.steps_per_task`
   - Allows task 0 to train for a different number of steps

4. **rl_metrics integration**:
   - Import `rl_metrics` module
   - `log_rl_metrics` flag (default `True`)
   - Frequent metrics every eval interval, occasional every 5× interval
   - Samples a batch from replay buffer for feature-level metrics
   - Logs under `rl_metrics/` prefix to W&B

5. **Checkpoint robustness**:
   - `save_ckpt()`: `jax.tree_map(np.array, ...)` before pickle
   - `load_ckpt()`: `jax.tree_map(jnp.array, ...)` after unpickle
   - Checkpoint path now includes `adapt_heads_only` in config key

6. **Persistent actor fix**:
   - `composed_actor` computed before pool/base updates
   - Persistent mode correctly carries composed policy

7. **Code documentation**:
   - Actor module annotated with layer indices (Dense_0..5)
   - Head detection documented with Flax naming convention
   - Module docstring updated with adapt_heads_only and encoder_from_base usage

### 2026-04-18 (session 2)

- Created `rl/impls/utils/pad_wrapper.py`:
  - `PaddedEnvWrapper` class with `_pad_obs()`, `_pad_goal()`, `_pad_state()`
  - Constants: MAX_CUBES=3, UNIFIED_OBS_DIM=57, UNIFIED_GOAL_DIM=9
- Updated `rl/impls/continual_crl.py`:
  - New default 12-task sequence spanning cube-1, cube-2, cube-3
  - Added `_parse_num_cubes()` helper
  - `train_single_task()` wraps envs with `PaddedEnvWrapper`
  - `evaluate_on_task()` wraps eval envs with `PaddedEnvWrapper`
  - `main()` uses UNIFIED dims for network initialization

### 2026-04-18 (session 1)

- Created `doc/implementation_plan.md`
- Created `doc/implementation_tracking.md`
- Created `rl/impls/continual_crl.py`

---

## File Inventory

| File | Purpose | Status |
|---|---|---|
| `rl/impls/continual_crl.py` | Main continual training driver | Complete |
| `rl/impls/knowledge_pool.py` | KnowledgePool with cosine-similarity merging | Complete |
| `rl/impls/rl_metrics.py` | RL representation metrics (Flax) | Complete |
| `rl/impls/utils/pad_wrapper.py` | Observation/goal padding wrapper | Complete |
| `rl/impls/crl.py` | Original single-task CRL (unchanged) | Reference |
| `doc/implementation_plan.md` | Design document | Complete |
| `doc/implementation_tracking.md` | This file | Maintained |
