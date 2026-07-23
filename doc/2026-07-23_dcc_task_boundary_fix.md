# Fix — DCC task-boundary checkpoint crash (unpicklable `TrainState`)

**Date.** 2026-07-23

**Symptom.** Every DCC ablation trained task 0 to completion and then crashed
at the *first task boundary*, right where the driver tries to persist a
checkpoint:

```
Task 0 [cube-1-task1] training complete.
...
  File "continual_crl_dcc.py", line 289, in save_ckpt
    pickle.dump(data, f)
AttributeError: Can't pickle local object 'chain.<locals>.init_fn'
```

Because task 0's checkpoint never wrote, the sweep could not progress past the
first task for any DCC row.

## W&B evidence

The failure is shared and identical across the six DCC task-boundary rows in
group `dcc_ablations__*` (project `buildstuff`):

| Ablation                | Behaviour at task boundary                                    |
|-------------------------|---------------------------------------------------------------|
| `dcc_add_shared`        | crash in `save_ckpt` → `chain.<locals>.init_fn`               |
| `dcc_concat_shared`     | crash in `save_ckpt` → `chain.<locals>.init_fn`               |
| `dcc_goal_task`         | crash in `save_ckpt` → `chain.<locals>.init_fn`               |
| `dcc_goal_partial`      | crash in `save_ckpt` → `chain.<locals>.init_fn`               |
| `dcc_goal_decomposed`   | crash in `save_ckpt` → `chain.<locals>.init_fn`               |
| `dcc_goal_projected`    | crash in `save_ckpt` → `chain.<locals>.init_fn`               |

All six log `Task 0 [...] training complete.` and then die in `save_ckpt`,
confirming the crash is at the checkpoint boundary and not in training.

**Not this bug:** `dcc_no_dyn` (run `7nr5eqad`) died earlier, during task-0
replay prefill, with `RESOURCE_EXHAUSTED` in `prefill_replay_buffer` (~5.4 GB
allocation). That is a *GPU-memory* problem, not the serialization crash — it
never reached `save_ckpt`. See `draft_dcc.sh` header (`TASKS_PER_GPU=1`
guidance) for the memory mitigation; it is deliberately kept separate from
this fix.

## Root cause

The old `save_ckpt` pickled the whole actor `TrainState`:

```python
# old continual_crl_dcc.py (paraphrased)
data = dict(actor_state=actor_state, critic_groups=groups)
with open(path, 'wb') as f:
    pickle.dump(data, f)          # <-- crashes here
```

A Flax `TrainState` carries two objects `pickle` cannot serialise:

* **`tx`** — the Optax `GradientTransformation`. `optax.adam` is built with
  `optax.chain(...)`, whose `init_fn` is a *local closure*, so pickling raises
  `AttributeError: Can't pickle local object 'chain.<locals>.init_fn'`.
* **`apply_fn`** — a bound method of the Flax module (also unpicklable).

The `opt_state` itself (e.g. `ScaleByAdamState`) is fine — it is a NamedTuple
of arrays. Only the *transformation* and the *apply method* are the problem.

### Secondary bug (silent task-skip)

The old code opened the destination file **before** `pickle.dump` ran, so the
crash left a **zero-byte `task_00.pkl`** on disk. The old `auto_resume` used
`os.path.exists()` only, so on the next launch it would treat that empty file
as a completed task 0 and **skip straight to task 1** — training on a fresh
(untrained) actor/critic. A crash therefore silently corrupted the continual
schedule.

## Fix

Persist a **versioned, data-only payload** and reconstruct the `TrainState`
after loading, using the *current* Actor module and optimiser. The
optimiser/apply closures are rebuilt at load time and never touch disk. This
mirrors what the baseline `continual_crl.py` already does (it saves
`actor_params`/`critic_params` only).

New module: **`rl/impls/utils/dcc_checkpoint.py`** (environment/MuJoCo-free so
the boundary is unit-testable with tiny pytrees).

### On-disk schema (v2)

```python
{
  'format':  'builderbench-dcc-ckpt',
  'version': 2,
  'task_idx': <int>,                 # index of the task that just finished
  'task_id':  <str>,                 # e.g. 'cube-1-task1'
  'actor': {                         # data only — NO tx, NO apply_fn
      'step':      np.ndarray,
      'params':    pytree of np.ndarray,
      'opt_state': pytree of np.ndarray,
  },
  'critic_groups': {                 # canonical DCC groups
      'b_shared':  {'params': ..., 'opt_state': ...},
      'h_phi':     {...}, 'h_dyn': {...}, 'phi_task': {...},
      'psi_shared':{...},
      # optional groups absent / stored (None, None) when not built:
      # 'psi_task', 'psi_proj'
  },
  'config': {                        # structural fingerprint (validated on load)
      'combine_mode', 'goal_encoder_mode', 'use_dyn',
      'rep_size', 'phi_task_width', 'phi_task_depth', 'seed',
  },
}
```

All array leaves are stored as host `np.ndarray` (see `_to_numpy`) and
converted back to `jnp` on load (`_to_jax`), so no JAX device handles are
pickled either.

`version` 1 is reserved for the legacy full-`TrainState` payload. No valid v1
file can exist (it never wrote successfully), so it is listed only so the
loader can name it in an error message. `SUPPORTED_VERSIONS = (2,)`.

### Atomic writes

`atomic_pickle_dump` writes to a uniquely-named `.tmp_ckpt_*.pkl` in the *same*
directory, `flush()` + `os.fsync()`, then `os.replace()` onto the destination
(atomic on POSIX). On **any** exception the temp file is unlinked and the
destination is left untouched. Consequences:

* a serialisation failure or preemption can never leave a partial checkpoint;
* an existing good checkpoint is never clobbered by a failed rewrite;
* no `.tmp_ckpt_*` litter survives a crash.

### Validation & rejection

`load_payload` / `validate_payload` raise `CheckpointError` for: zero-byte
files, unpicklable/garbage bytes, wrong `format`, unsupported `version`,
missing required keys, `task_idx` mismatch, and structural-`config` mismatch
(`STRUCTURAL_CONFIG_KEYS = combine_mode, goal_encoder_mode, rep_size,
phi_task_width, phi_task_depth`). A legacy full-`TrainState` dict (no versioned
header, but with `actor_state`/`critic_groups` keys) gets a dedicated,
*actionable* error pointing here rather than being silently mis-resumed.

### Reconstruction

* `restore_actor_state(payload, actor_module, actor_optimizer)` →
  `TrainState.create(apply_fn=actor_module.apply, params=…, tx=actor_optimizer)`
  then `.replace(step=…, opt_state=…)`. Closures are supplied fresh; only
  data comes from disk.
* `restore_critic_groups(payload)` → `{name: (params, opt_state)}`, with
  absent/`None` optional groups returned as `(None, None)` so the structure
  matches a freshly-initialised store.

## Resume behaviour (migration)

`auto_resume` is now a thin wrapper over
`dcc_checkpoint.last_valid_contiguous_task`: it returns the index of the last
**valid, contiguous** completed task, stopping at the first task whose
checkpoint is missing *or* fails validation. So:

* a zero-byte / partial / corrupt / config-mismatched checkpoint → that task is
  **re-run**, not skipped (fixes the silent task-skip above);
* a valid checkpoint sitting *after* a gap is ignored;
* on resume the driver reloads task `start_task-1`, rebuilds the actor
  `TrainState` with the current module/optimiser, and restores the critic
  groups. If `auto_resume` said a task was complete but its file is now missing,
  the driver raises `CheckpointError` instead of restarting at the wrong task.

**Migration for the crashed sweep:** the crashed runs left only zero-byte
`task_00.pkl` files, which the new loader rejects automatically — no manual
cleanup is required. Pointing `--checkpoint_dir` at the old directory and
re-launching will simply re-run task 0 cleanly. (If you prefer a pristine
start, delete the per-run `dcc__*` subdirectory.)

## Files touched

- `rl/impls/utils/dcc_checkpoint.py` — **new** versioned, data-only checkpoint
  module (schema, atomic IO, validation, contiguous-resume scan,
  reconstruction). No env/MuJoCo imports.
- `rl/impls/continual_crl_dcc.py` — removed `import pickle`; added
  `from utils import dcc_checkpoint`; `save_ckpt`/`load_ckpt`/`auto_resume`
  now delegate to the module; added `_ckpt_config` structural fingerprint;
  `main()` resume path reconstructs the actor/critic from the data-only
  payload. Path layout (`dcc__{combine}__{goal}__dyn-{0/1}__seed{seed}/
  task_{idx:02d}.pkl`) is unchanged.
- `rl/impls/tests/test_dcc_checkpoint.py` — **new** boundary tests (tiny
  Flax/Optax pytrees, no MuJoCo).
- `rl/impls/tests/test_dcc_driver_checkpoint.py` — **new** driver-wiring test
  (stubs the heavy imports, drives the driver's real `save_ckpt`/`load_ckpt`/
  `auto_resume` against its own `Actor`).
- `rl/impls/draft_dcc.sh` — header note: use `TASKS_PER_GPU=1` if a run OOMs
  during prefill (memory only, independent of this fix).

`continual_crl.py` is **not** modified.

## Testing

`continual_crl_dcc.py` pulls in tyro / wandb / MuJoCo / the BuilderBench env,
so a two-task GPU smoke run is impractical on a headless CPU box. The
checkpoint boundary is therefore exercised directly on tiny pytrees, plus a
stubbed driver-wiring test.

```bash
cd rl/impls
JAX_PLATFORMS=cpu python3 -m pytest tests/ -v
# 22 passed
```

Coverage:

* **Data-only roundtrip** — the raw `TrainState` is proven unpicklable
  (`init_fn` / local-object), while the payload pickles cleanly.
* **Reconstruction** — params / opt_state / step come back equal after a real
  optimiser step, and the rebuilt state is functional (`apply_gradients`
  advances `step`). Critic groups round-trip, incl. `(None, None)` optionals.
* **Atomic save** — no `.tmp_ckpt_*` litter; a failed write leaves an existing
  checkpoint byte-identical; a failed *first* write leaves no file so
  `auto_resume` re-runs the task.
* **Rejection** — zero-byte, garbage, wrong format, unsupported version,
  incomplete payload, `task_idx` mismatch, legacy full-`TrainState`, and
  structural config mismatch all raise `CheckpointError`.
* **auto_resume** — none / all-valid / stops-at-gap / stops-at-corrupt, and the
  exact historical mode (zero-byte `task_00.pkl` is **not** complete).
* **Driver wiring** — the driver's own `save_ckpt`→`load_ckpt`→`auto_resume`
  round-trip reconstructs its real `Actor`, and a zero-byte task-0 file makes
  `auto_resume` return `-1`.

## HPC rerun commands

Minimal two-task smoke (one DCC row, one seed) on a GPU node:

```bash
source /scratch/yd2247/.venvs/builderbench/bin/activate
cd /scratch/yd2247/builderbench/rl/impls
python continual_crl_dcc.py \
    --seed 1 \
    --use_dcc --dcc_combine_mode add --dcc_goal_encoder_mode shared --dcc_use_dyn \
    --task_sequence cube-1-task1,cube-2-task1 \
    --steps_per_task 1000000 --base_steps 1000000 \
    --num_envs 256 \
    --checkpoint_dir /scratch/yd2247/builderbench/logs/dcc_smoke_ckpt \
    --no-track --save_checkpoint
```

Pass criteria:
* task 0 finishes and writes `.../dcc__add__shared__dyn-1__seed1/task_00.pkl`
  (non-empty, no `.tmp_ckpt_*` left behind);
* task 1 starts **without** the `chain.<locals>.init_fn` crash;
* task 1 writes `task_01.pkl`.

Resume check: re-run the exact command; it should log
`Resumed from task 1 (actor step=…)` (or `All … tasks already completed`) and
not retrain completed tasks.

Full sweep via SLURM (unchanged launcher; add `TASKS_PER_GPU=1` if a row OOMs
in prefill):

```bash
sbatch draft_dcc.sh                       # whole grid
TASKS_PER_GPU=1 sbatch --array=0-0 draft_dcc.sh   # single row, full GPU memory
```
