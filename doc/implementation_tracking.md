# Implementation Tracking

## Status: Phase 3 In Progress — CKA Decomposition

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

### In Progress

- [ ] Phase 3: CKA actor decomposition in Flax (knowledge pool, base+vector composition)

### Next Steps

- [ ] Phase 4: CKA critic decomposition
- [ ] Phase 5: Comprehensive evaluation and continual RL metrics (forgetting, forward transfer)
- [ ] Run baseline experiments (reset+reset, reset+persistent, persistent+persistent)

---

## Decision Log

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
