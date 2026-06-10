# Fix — DCC dynamics loss `next_observation` plumbing

**Date.** 2026-06-10

**Symptom.** Smoke runs of `continual_crl_dcc.py` crashed on the first SGD
step with

```
KeyError: 'next_observation'
```

at `update_critic` → `transitions.extras['next_observation']`, even with
`--dcc_use_dyn` left at its default (`true`).

## Paper requirement

Decomposed Contrastive Critics (DCC) adds a masked dynamics auxiliary on
top of the contrastive critic (paper §3.4, Algorithm 1):

1. Sample `(s, a, g, s′) ∼ D_k` from replay (HER supplies `g`).
2. `L_dyn = ‖ h_dyn(b_shared(s, a)) − s′_M ‖²₂`
3. `L_critic = L_InfoNCE + μ · L_dyn` (μ = 1.0 in the paper).

The dynamics head reads **only** the shared body `b_shared(s, a)`; gradients
from `L_dyn` update `b_shared` and `h_dyn`, anchoring reusable transition
structure across tasks while `φ_task` adapts per task via the contrastive path.

On BuilderBench, the masked subspace **M** is cube positions (9 dims =
`UNIFIED_GOAL_DIM`), not the Sawyer end-effector indices `{0,1,2,3}` from
the paper. See `cube_position_indices` in `decomposed_networks.py`.

## Diagnosis

The DCC driver correctly implemented `dyn_loss` and called it from
`update_critic`, but `learn_step` never delivered `s′` to the flattened
batch.

The original attempt in `continual_crl_dcc.py`:

1. **`add_next_obs`** — shift the observation tensor along the sequence
   axis and stash `next_observation` in `extras` *before* HER flatten.
2. **`flatten_crl_fn`** — rebuild `extras` from scratch, keeping only
   `future_goal` and `state_extras`; drop everything else.

So `next_observation` was added and then immediately stripped. The inline
comment claiming flatten “produces” `next_observation` was incorrect.

The baseline CRL driver (`continual_crl.py`) does not need `s′`; its
`flatten_crl_fn` in `utils/buffer.py` was left unchanged on purpose so
the CKA grid stays bit-identical.

## Fix (Solution 1): `flatten_crl_dcc_fn`

Add a DCC-specific flatten next to the baseline:

```python
# rl/impls/utils/buffer.py
def flatten_crl_dcc_fn(buffer_config, transition, sample_key):
    flat = TrajectoryUniformSamplingQueue.flatten_crl_fn(
        buffer_config, transition, sample_key)
    extras = dict(flat.extras)
    extras["next_observation"] = transition.observation[1:]
    return flat._replace(extras=extras)
```

**Alignment.** `flatten_crl_fn` sets `observation = seq[:-1]` (states at
times `0 … T−2`). The matching next state for timestep `t` is
`seq[t+1]`, i.e. `transition.observation[1:]`. Each flattened row is
then `(s, a, g, s′)` as in Algorithm 1.

**`learn_step` change.** Remove `add_next_obs`; vmap
`flatten_crl_dcc_fn` instead of `flatten_crl_fn`.

## Rationale for this design

| Alternative | Why not chosen |
|-------------|----------------|
| Patch shared `flatten_crl_fn` with a flag | Risks changing baseline CRL / CKA behaviour |
| `add_next_obs` before flatten | Stripped by flatten; wrong place in the pipeline |
| Store `s′` in `Transition` at rollout time | More faithful to “env.step output”, but wider change; can follow up if episode-boundary masking is needed |
| Post-flatten shift on collapsed batch | Awkward once time axis is gone |

Keeping a separate `flatten_crl_dcc_fn` matches the port philosophy
(duplicate driver, don't touch baseline paths) and keeps HER + dynamics
alignment in one JIT-friendly function.

## Files touched

- `rl/impls/utils/buffer.py` — new `flatten_crl_dcc_fn`
- `rl/impls/continual_crl_dcc.py` — use it in `learn_step`; fix comments

`continual_crl.py` is **not** modified.

## Verification

On a GPU node with the project venv activated:

```bash
source /scratch/yd2247/.venvs/builderbench/bin/activate
cd rl/impls
python continual_crl_dcc.py \
    --seed 1 \
    --task_sequence cube-1-task1,cube-2-task1 \
    --steps_per_task 1000000 \
    --base_steps 1000000 \
    --num_envs 256 \
    --no-track \
    --no-save_checkpoint
```

**Pass criteria:**

- No `KeyError: 'next_observation'`
- Training progresses past the first SGD step
- Logged `dyn_mse` is finite when `--dcc_use_dyn` (default)

**Control:** `--no-dcc_use_dyn` should run without needing `s′` (dyn branch
skipped in `update_critic`).

## Follow-ups (not in this fix)

- **Trajectory boundaries.** If a rollout chunk spans two episodes,
  `obs[t+1]` may belong to a new trajectory. HER already masks goals via
  `traj_id`; a future hardening could mask `L_dyn` where
  `traj_id[t] ≠ traj_id[t+1]`, or store `nstate.obs` at collection time.
- **Login-node CUDA 303.** Environmental; use `sbatch draft_dcc.sh` or
  `srun --gres=gpu:1` for smoke tests.
