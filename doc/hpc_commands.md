# HPC Run Commands — BuilderBench Continual CRL

All commands assume the NYUAD HPC (Jubail / Dalma-style SLURM). Adjust the partition name and path prefixes if your cluster uses different conventions.

## One-time setup (run once per account)

```bash
# 0. Log in to the login node
ssh yd2247@jubail.abudhabi.nyu.edu     # replace with your own login host

# 1. Put the repo on /scratch (home quotas are tiny; scratch is big and fast).
mkdir -p /scratch/yd2247
cd /scratch/yd2247
git clone git@github.com:Okyumi/builderbench.git
cd builderbench

# 2. Scratch-based caches so pip / HF / W&B don't fill home
mkdir -p /scratch/yd2247/.cache /scratch/yd2247/tmp
export XDG_CACHE_HOME=/scratch/yd2247/.cache
export PIP_CACHE_DIR=/scratch/yd2247/.cache/pip
export TMPDIR=/scratch/yd2247/tmp

# 3. Create the conda env (same name the launcher activates)
module purge
module load cuda/11.8.0
module load conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda create -n builderbench python=3.10 -y
conda activate builderbench

# 4. Install BuilderBench in editable mode
pip install -e .
pip install -e rl          # installs rl subpackage if separate
pip install wandb wandb-osh tyro

# 5. Sign in to W&B (once)
wandb login
```

## Prepare the experiment grid

```bash
cd /scratch/yd2247/builderbench

# Inspect what will be run
python rl/impls/experiment_configs.py --list
python rl/impls/experiment_configs.py --total   # should print 27

# Make sure the SLURM launcher is executable
chmod +x rl/impls/draft_4.sh

# Dry run: print the flags that would be sent for config index 0 without submitting.
# (Useful for spotting typos before burning GPU hours.)
(
  set -euo pipefail
  TASKS_PER_GPU=1
  eval "$(python rl/impls/experiment_configs.py --setting 0)"
  echo "Would run: actor=$ACTOR_MODE critic=$CRITIC_MODE seed=$SEED"
)
```

## Submit the full 27-config sweep

```bash
cd /scratch/yd2247/builderbench

sbatch rl/impls/draft_4.sh
# -> submits a 5-task array: 0..4, each running 6 configs in parallel on one GPU.
# Total: 27 configs, 5 GPUs concurrent for ~steps_per_task seconds each.
```

## Submit a subset

```bash
# Only the first array task (configs 0-5):
sbatch --array=0-0 rl/impls/draft_4.sh

# Only configs 6-11 (array task 1):
sbatch --array=1-1 rl/impls/draft_4.sh

# Re-run just the reset/reset seeds (configs 0,1,2):
#   Option A: shrink TASKS_PER_GPU and submit one array task
TASKS_PER_GPU=3 sbatch --array=0-0 rl/impls/draft_4.sh
```

## Quick-debug submission

```bash
# 3-task sequence, 1M steps per task, single parallel process, 1-hour walltime.
STEPS_PER_TASK=1000000 BASE_STEPS=1000000 \
TASK_SEQUENCE='cube-1-task1,cube-2-task1,cube-3-task1' \
TASKS_PER_GPU=1 \
sbatch --time=01:00:00 --array=0-0 rl/impls/draft_4.sh
```

## Monitor progress

```bash
# Job queue
squeue -u yd2247
squeue --me

# Tail a specific experiment's logs (both .out and .err carry the header)
tail -f /scratch/yd2247/builderbench/logs/continual/<JOBID>_<ARRAYID>_<CONFIGIDX>.out

# List all outputs from one sweep
ls -lt /scratch/yd2247/builderbench/logs/continual/ | head

# Cancel a running array job
scancel <JOBID>
scancel <JOBID>_<ARRAYID>   # cancel just one array task
```

## W&B sync on a node with no internet

If compute nodes can't reach W&B directly, the launcher already sets `wandb_mode=online` with `wandb_osh` as a fallback. To sync runs manually after the job finishes:

```bash
cd /scratch/yd2247/builderbench
wandb sync --sync-all wandb/
```

## Re-run with the negative bank (once wired in)

```bash
# 1. Re-enable the placeholder in experiment_configs.py
#    (edit NEG_BANK_ENABLED = True near the top).
# 2. Submit only the headline cell (decomposed actor + persistent critic):
#    Edit experiment_configs.py: ACTOR_MODES=['cka']; CRITIC_MODES=['persistent'].
# 3. Submit.
NEG_BANK_ENABLED=true NEG_BANK_MODE=hard_weighted \
  sbatch --array=0-0 rl/impls/draft_4.sh
```

The launcher probes `continual_crl.py` at startup and emits the `--neg_bank_*` flags only if the driver accepts them. Until the bank is wired in, the flags are silently dropped and the runs proceed without them.

## Useful one-liners

```bash
# Count successful vs. failed runs from one sweep
grep -l "Run complete" /scratch/yd2247/builderbench/logs/continual/*.out | wc -l
grep -l "Traceback"     /scratch/yd2247/builderbench/logs/continual/*.err | wc -l

# Checkpoint sanity
find /scratch/yd2247/builderbench/logs/continual_checkpoints -name "task_*.pkl" | wc -l

# Disk usage on scratch
du -sh /scratch/yd2247/builderbench/logs/
```

## Project paths

| Path | Use |
|---|---|
| `/scratch/yd2247/builderbench`                                | Repo (read/write) |
| `/scratch/yd2247/builderbench/logs/continual/`                | Per-run `.out` / `.err` |
| `/scratch/yd2247/builderbench/logs/continual_checkpoints/`    | Per-run checkpoints |
| `/scratch/yd2247/.cache/`                                     | pip / HF / W&B caches |
| `/scratch/yd2247/tmp/`                                        | Scratch tmpdir for MuJoCo / JAX |
