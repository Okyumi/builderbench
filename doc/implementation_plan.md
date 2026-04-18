# Implementation Plan: Continual Goal-Conditioned Contrastive RL on BuilderBench

## 1. Research Motivation

BuilderBench is a block-building benchmark where an agent learns to construct target structures from cubes. The existing CRL implementation in BuilderBench already uses contrastive RL (Eysenbach et al., 2022) for single-task learning. Our research project extends this to the **continual learning** setting: the agent sequentially learns to build different structures, and we study how knowledge (in the actor and/or the critic) transfers across tasks.

This is a natural fit because:
- BuilderBench tasks share the same robot, physics, and action space
- The contrastive critic learns reachability structure that may generalize across block-building tasks
- The complexity ladder (1-cube → 4-cube) provides a natural curriculum for continual learning
- The JAX-parallelized environment is fast enough for the large-scale experiments needed

## 2. Environment Analysis

### Existing Setup (rl/impls/crl.py)

| Component | Details |
|---|---|
| Simulator | MuJoCo gripper-only (no full arm), JAX-parallelized via `mujoco.rollout` |
| Action space | 5D: delta (x, y, z, yaw, gripper_strength) — **fixed across all tasks** |
| State space | Gripper state + per-cube (pos, vel, yaw) + targets — **variable per task** |
| Goal space | Target cube positions (R^{3k} where k = cubes in target) — **variable per task** |
| Reward | Dense, permutation-invariant: Hungarian-matched 1−tanh(distance) |
| Success | All target cubes within 2cm of their targets and stable |
| Episode length | 100 + num_cubes × 50 |
| Parallelism | 2048 envs, 12 threads |
| Training | 50M env steps, rollout_length=64, batch=4096 |

### Key Challenge: Variable Observation/Goal Dimensions

Unlike Meta-World where all tasks share the same obs_dim, BuilderBench tasks with different numbers of cubes have **different observation and goal sizes**:

| Task | Cubes | Obs dim (approx) | Goal dim |
|---|---|---|---|
| cube-1-task1 | 1 | ~17 | 3 |
| cube-2-task1 | 2 | ~31 | 6 |
| cube-3-task1 | 3 | ~45 | 9 |
| cube-4-task1 | 4 | ~59 | 12 |

This means we **cannot simply carry forward network weights** between tasks with different cube counts — the input/output dimensions would mismatch.

### Solution: Unified Observation Space

We must design a **unified observation space** that is consistent across all tasks in our 10-task sequence, similar to what we did for Meta-World. Two options:

**Option A (recommended): Fixed-cube-count tasks only.** Select all 10 tasks from the same cube count (e.g., all from cube-2 or cube-3), so the observation and goal dimensions are naturally identical. This is cleanest and avoids padding artifacts.

**Option B: Padding.** Pad smaller-cube observations/goals to the max cube count in the sequence, filling unused slots with zeros or a sentinel value. This adds engineering complexity and may confuse the contrastive critic (zeros as negative samples).

**We recommend Option A** for the initial experiments — select tasks within the same cube count, or with very similar counts.

## 3. Task Selection: 10-Task Sequence

From the paper's task suite and Figure 12 RL results, here is the proposed 10-task sequence:

### Selected Tasks (ordered roughly easy → hard)

| # | Task ID | Cubes | Difficulty | Description |
|---|---|---|---|---|
| 0 | cube-1-task1 | 1 | Easy | Place one cube at target position |
| 1 | cube-1-task2 | 1 | Easy | Pick and lift one cube to elevated target |
| 2 | cube-2-task1 | 2 | Easy | Stack two cubes |
| 3 | cube-2-task2 | 2 | Easy | Horizontally align two cubes |
| 4 | cube-2-task3 | 2 | Easy | Two cubes side by side |
| 5 | cube-3-task1 | 3 | Easy | Stack three cubes |
| 6 | cube-3-task3 | 3 | Easy | Three cubes in a row |
| 7 | cube-2-task4 | 2 | Hard | Requires rotation (tilting problem) |
| 8 | cube-2-task5 | 2 | Hard | Double horizontal pick |
| 9 | cube-3-task2 | 3 | Hard | T-block (requires insight: rotate base 45°) |

### Rationale

- Tasks 0–1 (1-cube): simplest motor skills (place, pick+lift) — warm up
- Tasks 2–4 (2-cube easy): basic composition (stacking, alignment)
- Tasks 5–6 (3-cube easy): extending composition to 3 cubes
- Tasks 7–8 (2-cube hard): require reasoning (rotation, dual manipulation)
- Task 9 (3-cube hard): the T-block — requires the key insight of rotating the base cube 45°

This provides a natural difficulty gradient. The 1-cube and easy 2-cube tasks should be solvable by baseline CRL; the hard tasks test whether transferred knowledge helps.

### Handling Variable Cube Counts

Since this sequence includes tasks with 1, 2, and 3 cubes, we need the padding approach (Option B). The maximum cube count is 3, so:

- **Unified obs_dim**: pad all observations to the 3-cube observation size
- **Unified goal_dim**: pad all goals to 9 (3 × 3D positions)
- **Mask**: indicate which cubes are active vs. padding

Alternatively, we can start with a **same-cube-count sequence** (all 2-cube or all 3-cube) for the first experiments, then extend to mixed counts later.

**Phase 1 (start here):** 5 tasks, all 2-cube:
```
cube-2-task1 (stack) → cube-2-task2 (align) → cube-2-task3 (side) →
cube-2-task4 (rotation) → cube-2-task5 (double pick)
```

**Phase 2 (extend later):** Mixed cube counts with padding.

## 4. Architecture Adaptation

### What to Keep from BuilderBench CRL

The existing `rl/impls/crl.py` uses:
- SA_encoder: 4 × Dense(1024) + LayerNorm + Swish → Dense(64)
- G_encoder: 4 × Dense(1024) + LayerNorm + Swish → Dense(64)
- Actor: 4 × Dense(1024) + LayerNorm + Swish → mean + log_std
- L2 distance energy: `−‖φ(s,a) − ψ(g)‖₂`
- InfoNCE loss with logsumexp regularization (coeff 0.1)
- LeCun uniform initialization
- Entropy regularization (coeff 0.1)

This architecture is larger and more aligned with the 1000-layer scaling paper than our Meta-World setup (which uses 256-width networks). We should keep this architecture since it's already validated for BuilderBench.

### What to Add: Continual Learning Logic

Port the continual learning infrastructure from `sgcrl/`:
1. **CKA actor decomposition**: θ' = θ_base + Σ α_j v_j + v_k
2. **Critic evolution modes**: persistent / reset / CKA
3. **Knowledge pool**: vector storage, merging, serialization
4. **Checkpoint system**: auto-resume, config-keyed paths
5. **Cross-task evaluation**: evaluate on all tasks seen so far

### Key Difference from Meta-World Implementation

| Aspect | Meta-World (sgcrl) | BuilderBench |
|---|---|---|
| Framework | Haiku + Acme + Reverb | Flax + custom training loop |
| Parallelism | 1 env (CPU) | 2048 envs (JAX-parallelized) |
| Replay buffer | Reverb server | Custom JAX TrajectoryBuffer |
| UTD ratio | 64:1 | ~32:1 (rollout 64, batch 4096, seq 512) |
| Network width | 256 | 1024 |
| Energy function | Inner product | L2 distance |
| Action space | Sawyer arm (4D) | Gripper (5D) |

The Flax-based training loop is very different from our Haiku/Acme setup. We have two options:

**Option A: Port continual logic into the existing BuilderBench Flax codebase.** This preserves the fast JAX-parallelized training. More engineering work but much faster training.

**Option B: Wrap BuilderBench env in our Acme/Haiku pipeline.** This reuses our existing continual code. Slower (single env) but less new code.

**Recommendation: Option A** — port the continual logic into the BuilderBench codebase. The JAX parallelization (2048 envs) is a major advantage that we should not give up. Also, Flax is a more standard framework than Haiku for new development.

## 5. Implementation Steps

### Phase 1: Single-task CRL baseline on selected tasks (verify environment works)

1. Run the existing `rl/impls/crl.py` on each of the 5 Phase 1 tasks (cube-2-task1 through cube-2-task5)
2. Record baseline performance per task
3. Verify environment setup, reward, and success metrics

### Phase 2: Implement continual CRL training loop

1. Create `rl/impls/continual_crl.py` — the main continual training driver
2. Implement:
   - Task sequence configuration
   - Per-task training loop (based on existing crl.py)
   - Checkpoint save/load between tasks
   - Actor/critic state transfer logic (persistent, reset, CKA)
3. Create `rl/impls/knowledge_pool.py` — port from sgcrl/contrastive/knowledge_pool.py

### Phase 3: CKA actor decomposition in Flax

1. Implement the CKA composition: θ' = θ_base + pool_c + v_k
   - In Flax, this operates on the parameter pytree (same concept as Haiku)
   - The "base" is frozen params, "v_k" is trainable params
2. Implement head/body split for adapt_heads_only mode
3. Implement knowledge pool contribution computation

### Phase 4: Critic evolution modes

1. Persistent: carry forward critic TrainState across tasks
2. Reset: reinitialize critic each task
3. CKA: knowledge-vector decomposition for the critic

### Phase 5: Evaluation and logging

1. Cross-task evaluation after each task
2. Forward transfer and forgetting metrics
3. W&B integration for experiment tracking

## 6. File Structure

```
rl/
├── impls/
│   ├── crl.py                     # Existing single-task CRL (unchanged)
│   ├── continual_crl.py           # NEW: continual training driver
│   ├── knowledge_pool.py          # NEW: ported from sgcrl
│   └── utils/
│       ├── buffer.py              # Existing replay buffer
│       ├── evaluation.py          # Existing evaluator
│       ├── networks.py            # Existing + CKA decomposition
│       └── wrapper.py             # Existing env wrapper
├── builderbench/
│   ├── build_block.py             # Existing environment
│   ├── constants.py               # Existing task configs
│   ├── create_task_data.py        # Existing task definitions
│   └── env_utils.py               # Existing + continual extensions
└── doc/
    ├── implementation_plan.md      # This document
    └── implementation_tracking.md  # Progress tracking
```

## 7. Timeline Estimate

- Phase 1 (baselines): 1–2 days
- Phase 2 (continual training loop): 2–3 days
- Phase 3 (CKA actor): 2–3 days
- Phase 4 (critic modes): 1–2 days
- Phase 5 (evaluation): 1–2 days
- Testing + debugging: 2–3 days

Total: ~2 weeks

## 8. References

- Eysenbach et al., "Contrastive Learning as Goal-Conditioned RL," NeurIPS 2022
- Ghugare et al., "BuilderBench: The Building Blocks of Intelligent Agents," arXiv:2510.06288
- Liu et al., "A Single Goal Is All You Need," ICLR 2025
- Hu et al., "Continual Knowledge Adaptation for Reinforcement Learning," NeurIPS 2025
- Wang et al., "1000 Layer Networks for Self-Supervised RL," 2025
