# Bug Fix — Observation Padding Dim Mismatch

**Date.** 2026-04-21
**Symptom.** SLURM array run `14940495_0` crashed at the very first JIT-compile of the actor forward pass with

```
flax.errors.ScopeParamShapeError: Initializer expected to generate shape
(121, 1024) but got shape (115, 1024) instead for parameter "kernel"
in "/Dense_0".
```

from `rl/impls/continual_crl.py::actor_step` when task 0 (`cube-1-task1`) was running.

## Diagnosis

Two observations explain the 6-dimension gap:

1. The actor's first Dense layer sees `concat(obs, g_repr)`. With `rep_size = 64`, the layer's input dim is `obs_size + 64`. The declared input dim was `57 + 64 = 121`, and the runtime input dim was `51 + 64 = 115`. So the mismatch is in `obs_size` alone: **57 declared vs 51 at runtime**.

2. The obs layout produced by `build_block.CreativeCube.get_obs` is
   ```
   gripper_pos(3) + gripper_quat(4) + gripper_linvel(3)  -> 10 (prefix)
   obj_pos(3N)   + obj_quat(4N)    + obj_linvel(3N) + obj_angvel(3N) -> 13N
   finger_pos(?)                                         -> suffix
   ```
   `finger_pos` is indexed from `self._fingers_qposadr`, defined at `build_block.py:152–155` as `['left_driver_joint', 'right_driver_joint']` — **two driver joints**. So `finger_pos` has dim **2**, not 8 as the pad wrapper assumed.

   Real per-task dims:
   - `cube-1-task*`: `10 + 13·1 + 2 = 25`
   - `cube-2-task*`: `10 + 13·2 + 2 = 38`
   - `cube-3-task*`: `10 + 13·3 + 2 = 51`

   After padding to `MAX_CUBES = 3`, every task yields a unified obs of dim **51**.

`utils/pad_wrapper.py` had the constant `FIXED_OBS_SUFFIX = 8`, which gave `UNIFIED_OBS_DIM = 10 + 13·3 + 8 = 57`. `continual_crl.py` initialised the actor's first Dense layer with `obs_size = UNIFIED_OBS_DIM = 57` (two sites: lines 577 and 1106). At runtime the wrapper produced an obs of dim 51 (the real suffix came through; no "8-zero pad" was ever inserted because `_pad_obs` does `suffix = obs[..., 10 + 13N:]`, which is whatever the env actually produces). Hence `(121, 1024)` was expected but `(115, 1024)` was received.

## Why the padding path itself was still correct

`PaddedEnvWrapper._pad_obs` slices with `suffix = obs[..., FIXED_OBS_PREFIX + actual_obj_dim:]`, which grabs the true trailing bytes regardless of what `FIXED_OBS_SUFFIX` was set to. So the padding did produce a consistent output dim across tasks. The only bug was the *declared* `UNIFIED_OBS_DIM` that the networks were sized against.

## Fix

`rl/impls/utils/pad_wrapper.py`:

- `FIXED_OBS_SUFFIX`: **8 → 2** (actual `finger_pos` dim).
- `UNIFIED_OBS_DIM`: recomputes to `10 + 13·3 + 2 = 51`.
- Module-level docstring rewritten to match the real obs layout and to cite the precise lines in `build_block.py` that define `finger_pos`.
- `_pad_obs` gains a runtime assertion that compares the incoming obs dim to `prefix + per_cube·N + suffix`, so any future drift between the env layout and these constants fails loudly with an explicit message instead of a cryptic Flax shape error six function calls deep.

No changes needed in `continual_crl.py`: both initialisation sites already do `obs_size = UNIFIED_OBS_DIM` and will pick up the corrected value.

## Sanity checks to run

```bash
# Quick single-task smoke test (one cube, 10k steps, 1 process, 1 GPU).
cd /scratch/yd2247/builderbench

STEPS_PER_TASK=10000 BASE_STEPS=10000 \
TASK_SEQUENCE='cube-1-task1' \
TASKS_PER_GPU=1 \
sbatch --time=01:00:00 --array=0-0 rl/impls/draft_4.sh

# After that passes, a 3-cube mixed-task smoke test that exercises the
# padding path on every N:
STEPS_PER_TASK=10000 BASE_STEPS=10000 \
TASK_SEQUENCE='cube-1-task1,cube-2-task1,cube-3-task1' \
TASKS_PER_GPU=1 \
sbatch --time=01:00:00 --array=0-0 rl/impls/draft_4.sh
```

If either run raises the new assertion from `_pad_obs`, the obs layout in `build_block.get_obs` has changed — update the three constants at the top of `pad_wrapper.py` to match.

## Unrelated warning visible in the same .err

```
ERROR jax._src.xla_bridge: Jax plugin configuration error ...
RuntimeError: Unable to load cuSPARSE. Is it installed?
WARNING jax._src.xla_bridge: Falling back to cpu.
```

This is an environment issue with the JAX CUDA 12 plugin on the current conda env — cuSPARSE is not on `LD_LIBRARY_PATH`. It is unrelated to the padding fix and will cause runs to silently use CPU (which is 10-100x slower but correct). To fix:

```bash
# Inside the conda env:
pip install nvidia-cusparse-cu12
# Or load a cuSPARSE-providing module, e.g. cuda/12.x with full toolkit.
```

Add the resolved `nvidia/cusparse/lib` path to `LD_LIBRARY_PATH` in `draft_4.sh` if necessary. This is not needed to verify the padding fix on CPU.
