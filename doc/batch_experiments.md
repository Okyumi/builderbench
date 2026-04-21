# BuilderBench Batch Experiments — Implementation Note

This document describes the SLURM batch pipeline for the BuilderBench continual CRL experiments. The pipeline mirrors the sgcrl one-to-one so the two codebases can be swept the same way.

## Files

| Path | Purpose |
|---|---|
| `rl/impls/experiment_configs.py` | Enumerate the actor × critic × seed grid; emit a single config as shell-sourceable `KEY=VALUE` lines. |
| `rl/impls/draft_4.sh`            | SLURM job-array launcher. Each array task runs `TASKS_PER_GPU` experiments in parallel on one GPU. |
| `doc/batch_experiments.md`       | This file. |
| `doc/hpc_commands.md`            | Cheatsheet of commands to run the project on the HPC terminal. |

The single-task driver is `rl/impls/continual_crl.py` (unchanged).

## Experiment grid

Default grid defined in `experiment_configs.py`:

- `actor_mode ∈ {reset, persistent, cka}`
- `critic_mode ∈ {reset, persistent, cka}`
- `seed ∈ {1, 2, 3}`

Total: 3 × 3 × 3 = 27 configurations.

To change the grid, edit the three lists at the top of `experiment_configs.py`. The launcher picks up the new total automatically (it calls `python experiment_configs.py --total`). Remember to update `#SBATCH --array=0-N` so that `N = ceil(total / TASKS_PER_GPU) - 1`.

## Negative bank placeholder

The BuilderBench `continual_crl.py` driver does not currently accept negative-bank flags. The launcher still exposes them as environment variables (`NEG_BANK_ENABLED`, `NEG_BANK_MODE`, …) and `experiment_configs.py` has a `NEG_BANK_ENABLED` switch that toggles per-config overrides. At startup, the launcher probes `continual_crl.py` with:

```bash
grep -q "neg_bank_mode" rl/impls/continual_crl.py
```

and emits `--neg_bank_*` flags only if **both** `NEG_BANK_ENABLED=true` **and** the driver contains the flag. Once the bank is wired into `continual_crl.py` (mirroring `sgcrl/contrastive/continual_learning.py`), nothing in this launcher needs to change — set `NEG_BANK_ENABLED=true` and the flags start flowing.

Priority ordering stays consistent with the paper plan: first run the 9 actor/critic cells with the bank off; once those land, re-run the headline cell (decomposed actor + persistent critic) with `NEG_BANK_MODE=hard_weighted`.

## SLURM array arithmetic

Each array task runs `TASKS_PER_GPU` configs. The `i`-th slot inside array task `a` runs configuration index

```
CONFIG_IDX = TASKS_PER_GPU * a + i
```

Configurations past `TOTAL_CONFIGS - 1` are silently skipped, so you can be safe with a slightly too-large array range.

Default arithmetic:

- `TOTAL_CONFIGS = 27` (3 × 3 × 3).
- `TASKS_PER_GPU = 6` (BuilderBench is lighter on memory than sgcrl; on an A100-80G 4–8 in parallel is comfortable).
- `ceil(27 / 6) - 1 = 4` → `#SBATCH --array=0-4`.

If you move to 5 seeds instead of 3:

- `TOTAL_CONFIGS = 45`, with `TASKS_PER_GPU = 6` → `ceil(45/6) - 1 = 7` → `#SBATCH --array=0-7`.

## Parallelism on the GPU

Each parallel process inside an array task preallocates `XLA_PYTHON_CLIENT_MEM_FRACTION` of the GPU. The launcher auto-sets it based on `TASKS_PER_GPU`:

| `TASKS_PER_GPU` | `XLA_PYTHON_CLIENT_MEM_FRACTION` |
|---|---|
| 1 | 0.85 |
| 2 | 0.45 |
| 3 | 0.30 |
| 4 | 0.22 |
| 6 | 0.15 |
| 8 | 0.11 |

Override with `TASKS_PER_GPU=N sbatch draft_4.sh` (the mem fraction adapts automatically; unusual values are computed as `0.9 / N` capped at 0.85).

If you hit OOM in `.err`, drop `TASKS_PER_GPU` and re-submit only the missing configs with `--array=<range>`.

## Logging

The launcher writes two per-experiment files, both containing the full context (config index, actor/critic/seed, command line, and the training stream):

```
$LOG_DIR/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${CONFIG_IDX}.out
$LOG_DIR/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${CONFIG_IDX}.err
```

Both files carry a header preamble so that a missing `.out` can be reconstructed from `.err` and vice versa. This matches the sgcrl convention.

W&B logging is on by default (`TRACK=true`) with group `continual_crl_grid` and per-run tag `actor_{MODE}_critic_{MODE}_seed_{SEED}`.

## Environment defaults

Everything the launcher sets can be overridden by exporting the variable before `sbatch`. Common overrides:

```bash
# Shorter runs while debugging:
STEPS_PER_TASK=5000000 BASE_STEPS=5000000 sbatch rl/impls/draft_4.sh

# Switch to the 20-task / 5-seed sweep:
# (first edit experiment_configs.py -> SEEDS = [1, 2, 3, 4, 5]; change --array=0-7)

# Single-config rerun:
sbatch --array=0-0 rl/impls/draft_4.sh   # runs TASKS_PER_GPU configs starting at idx 0
```

## Reproducibility

- Checkpoints: `{CHECKPOINT_DIR}/continual_crl__{actor_mode}_{critic_mode}/heads_{adapt_heads_only}/seed_{seed}/task_{k}.pkl` (structure from `continual_crl.py::_ckpt_dir`). A mismatched config raises a clear `FileNotFoundError` when resuming.
- Auto-resume: `continual_crl.py` probes for the latest checkpoint under the matching config and seed and resumes from there.
- `TASK_SEQUENCE` and `STEPS_PER_TASK` are keyed into the W&B run name via `wandb_name_tag`, so ad-hoc sweeps don't collide.

## Known differences from the sgcrl pipeline

- BuilderBench has no `adaptive_entropy` flag — the built-in `entropy_cost=0.1` is the default. The paper plan flags this as appendix-only.
- BuilderBench has no `actor_auto_reset` flag. Plasticity diagnostics are still logged via `log_rl_metrics=true`.
- BuilderBench has no negative bank yet (see above).

These gaps are intentional; the BuilderBench runs serve as a portability check in the appendix rather than the headline results.
