# Implementation Tracking

## Status: Phase 1 — Baseline Verification

### Completed

- [x] Study BuilderBench paper, blog post, and codebase
- [x] Analyze environment: state/action/goal spaces, reward, parallelization
- [x] Identify key challenge: variable observation dimensions across cube counts
- [x] Design 10-task sequence with difficulty gradient
- [x] Write comprehensive implementation plan (`doc/implementation_plan.md`)
- [x] Initial commit to repository

- [x] Create `rl/impls/continual_crl.py` — continual training driver with:
  - Task sequence support
  - Actor/critic transfer logic (reset / persistent / cka placeholder)
  - Checkpoint save/load with config-keyed paths
  - Auto-resume from latest checkpoint
  - Cross-task evaluation after each task
  - W&B logging integration

### In Progress

- [ ] Phase 1: Run single-task CRL baselines on selected tasks (cube-2-task1 through cube-2-task5)
- [ ] Test `continual_crl.py` end-to-end on a quick 2-task run

### Next Steps

- [ ] Phase 3: CKA actor decomposition in Flax (knowledge pool, base+vector composition)
- [ ] Phase 4: CKA critic decomposition
- [ ] Phase 5: Comprehensive evaluation and continual RL metrics (forgetting, forward transfer)

---

## Decision Log

### 2026-04-18: Task Sequence Design

**Decision:** Start with 5 same-cube-count tasks (all cube-2) for Phase 1, extending to mixed cube counts later.

**Rationale:** Tasks with different cube counts have different observation/goal dimensions. Starting with same-count tasks avoids the padding complexity and lets us validate the continual learning logic first.

**Phase 1 sequence (all 2-cube):**
1. cube-2-task1 (stack)
2. cube-2-task2 (horizontally align)
3. cube-2-task3 (side by side)
4. cube-2-task4 (rotation/tilting — hard)
5. cube-2-task5 (double horizontal pick — hard)

**Phase 2 sequence (mixed, 10 tasks):**
1. cube-1-task1 → cube-1-task2 → cube-2-task1 → cube-2-task2 → cube-2-task3 → cube-3-task1 → cube-3-task3 → cube-2-task4 → cube-2-task5 → cube-3-task2

### 2026-04-18: Architecture Decision

**Decision:** Port continual learning logic into the existing BuilderBench Flax codebase rather than wrapping BuilderBench in our Acme/Haiku pipeline.

**Rationale:** The JAX-parallelized environment (2048 envs) provides a major speedup. Using Acme/Reverb would limit us to 1 env and dramatically slow training. The Flax training loop is clean and the CKA composition logic is framework-agnostic (operates on parameter pytrees).

### 2026-04-18: Network Architecture

**Decision:** Keep the BuilderBench CRL architecture (1024-width, LayerNorm, Swish, L2 energy) as-is.

**Rationale:** This architecture is already validated for BuilderBench tasks. It's larger than our Meta-World setup (256-width) but appropriate for the more complex block-building domain. The L2 energy function differs from our Meta-World inner-product setup, but matches the 1000-layer scaling paper.

---

## Change Log

### 2026-04-18

- Created `doc/implementation_plan.md` — comprehensive implementation plan
- Created `doc/implementation_tracking.md` — this file
- Initial commit to repository
