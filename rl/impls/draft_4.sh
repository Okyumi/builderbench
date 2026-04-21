#!/bin/bash
#SBATCH --job-name=bb_continual_crl
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --partition=nvidia
#SBATCH --output=/scratch/yd2247/builderbench/logs/continual/%A_%a.out
#SBATCH --error=/scratch/yd2247/builderbench/logs/continual/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-4

# ==========================================================================
# BuilderBench Continual CRL -- Batch SLURM Launcher
#
# Runs MULTIPLE experiments per GPU using a job array. Each array task
# launches TASKS_PER_GPU experiments in parallel on the same GPU.
#
# Configurations are defined in experiment_configs.py (in the same dir).
#   python experiment_configs.py --list      # show all configs
#   python experiment_configs.py --total     # count
#
# Array-range arithmetic:
#   total        = $(python experiment_configs.py --total)
#   max_array_id = ceil(total / TASKS_PER_GPU) - 1
#
#   For 3 actor x 3 critic x 3 seeds = 27 configs, TASKS_PER_GPU=6:
#     ceil(27/6) - 1 = 4  ->  --array=0-4
#
# Usage:
#   sbatch draft_4.sh                  # run all configs
#   sbatch --array=0-0 draft_4.sh      # one array task (TASKS_PER_GPU runs)
#   TASKS_PER_GPU=4 sbatch draft_4.sh  # override parallelism on the GPU
#
# JAX GPU memory:
#   Each process preallocates XLA_PYTHON_CLIENT_MEM_FRACTION of GPU RAM.
#   Default below assumes TASKS_PER_GPU=6 on an A100-80G.
# ==========================================================================

set -euo pipefail

# ---- number of parallel tasks per GPU ------------------------------------
# BuilderBench uses num_envs=2048 by default, which is lighter on memory
# than the sgcrl grid. On an 80GB GPU you can typically run 4-8 in parallel.
# Adjust if you see OOM.
TASKS_PER_GPU="${TASKS_PER_GPU:-4}"

# ---- shared defaults (every experiment in the batch sees these) -----------
# These mirror the `Args` dataclass defaults in continual_crl.py. Override
# by exporting the variable before `sbatch`, e.g. `STEPS_PER_TASK=10000000`.
AGENT="${AGENT:-continual_crl}"

# Task sequence: the 12-task mixed-cube Continual-BuilderBench sequence.
TASK_SEQUENCE="${TASK_SEQUENCE:-cube-1-task1,cube-1-task2,cube-2-task1,cube-2-task2,cube-2-task3,cube-3-task1,cube-3-task3,cube-2-task4,cube-2-task5,cube-3-task2,cube-3-task4,cube-3-task5}"

STEPS_PER_TASK="${STEPS_PER_TASK:-50000000}"
BASE_STEPS="${BASE_STEPS:-50000000}"
K_MAX="${K_MAX:-10}"
ALPHA_SCALE="${ALPHA_SCALE:-1.0}"

ADAPT_HEADS_ONLY="${ADAPT_HEADS_ONLY:-true}"
ENCODER_FROM_BASE="${ENCODER_FROM_BASE:-false}"

# Training hyperparameters (single-task defaults from crl.py / continual_crl.py).
NUM_ENVS="${NUM_ENVS:-2048}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
NUM_THREADS="${NUM_THREADS:-12}"
ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-64}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-1e-3}"
DISCOUNT="${DISCOUNT:-0.99}"
ENTROPY_COST="${ENTROPY_COST:-0.1}"
LOGSUMEXP_COST="${LOGSUMEXP_COST:-0.1}"
REP_SIZE="${REP_SIZE:-64}"
MAX_REPLAY_SIZE="${MAX_REPLAY_SIZE:-10000}"
MIN_REPLAY_SIZE="${MIN_REPLAY_SIZE:-1000}"
EVAL_EPISODES="${EVAL_EPISODES:-128}"
NUM_EVAL_STEPS="${NUM_EVAL_STEPS:-50}"
NUM_RESET_STEPS="${NUM_RESET_STEPS:-1}"

LOG_RL_METRICS="${LOG_RL_METRICS:-true}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
ENV_EARLY_TERMINATION="${ENV_EARLY_TERMINATION:-true}"
PERMUTATION_INVARIANT_REWARD="${PERMUTATION_INVARIANT_REWARD:-true}"

# W&B
TRACK="${TRACK:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-buildstuff}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-continual_crl_grid}"

# Negative bank (placeholders). These flags are emitted only when
# NEG_BANK_ENABLED=true AND the upstream continual_crl.py accepts them.
# The launcher probes the driver once at startup to decide.
NEG_BANK_ENABLED="${NEG_BANK_ENABLED:-false}"
NEG_BANK_MODE="${NEG_BANK_MODE:-off}"
NEG_BANK_PER_TASK_CAPACITY="${NEG_BANK_PER_TASK_CAPACITY:-10000}"
NEG_BANK_N_PER_STEP="${NEG_BANK_N_PER_STEP:-256}"
NEG_BANK_CANDIDATE_POOL="${NEG_BANK_CANDIDATE_POOL:-1024}"
NEG_BANK_WEIGHT="${NEG_BANK_WEIGHT:-0.3}"
NEG_BANK_MAX_TASKS="${NEG_BANK_MAX_TASKS:-20}"

# ---- Directories (scratch to avoid home quota issues) ---------------------
LOG_DIR="${LOG_DIR:-/scratch/yd2247/builderbench/logs/continual}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/yd2247/builderbench/logs/continual_checkpoints}"
REPO_DIR="${REPO_DIR:-/scratch/yd2247/builderbench}"

# ---- Environment setup ----------------------------------------------------
module purge
module load cuda/12.2.0
module load conda-gcc/11.2.0

# Ensure CUDA runtime libs are discoverable for JAX CUDA plugin.
if command -v nvcc >/dev/null 2>&1; then
  CUDA_ROOT="$(dirname "$(dirname "$(command -v nvcc)")")"
  export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
fi

# Use uv-managed environment instead of conda.
UV_ENV_PATH="${UV_ENV_PATH:-/scratch/yd2247/.venvs/builderbench}"
if [ ! -f "$UV_ENV_PATH/bin/activate" ]; then
  echo "ERROR: uv environment not found at $UV_ENV_PATH" >&2
  exit 1
fi
source "$UV_ENV_PATH/bin/activate"

# JAX CUDA wheels ship CUDA shared libs in site-packages; expose them.
PY_CUDA_LIBS="$(python - <<'PY'
import importlib, os
mods = [
    "nvidia.cusparse",
    "nvidia.cublas",
    "nvidia.cudnn",
    "nvidia.cusolver",
    "nvidia.cuda_runtime",
]
paths = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        libdir = os.path.join(os.path.dirname(mod.__file__), "lib")
        if os.path.isdir(libdir):
            paths.append(libdir)
    except Exception:
        pass
print(":".join(paths))
PY
)"
[ -n "$PY_CUDA_LIBS" ] && export LD_LIBRARY_PATH="${PY_CUDA_LIBS}:${LD_LIBRARY_PATH:-}"

# Keep job caches and temp files on scratch.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/yd2247/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export TMPDIR="${TMPDIR:-/scratch/yd2247/tmp}"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

# Essential runtime environment.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

# ---- JAX GPU memory allocation for multi-process sharing ------------------
# With TASKS_PER_GPU=6 on an 80GB GPU we give each process ~15% (~12GB).
# Recompute if you change TASKS_PER_GPU: target ~(90/TASKS_PER_GPU)%.
case "$TASKS_PER_GPU" in
  1)  MEM_FRAC="0.85" ;;
  2)  MEM_FRAC="0.45" ;;
  3)  MEM_FRAC="0.30" ;;
  4)  MEM_FRAC="0.22" ;;
  6)  MEM_FRAC="0.15" ;;
  8)  MEM_FRAC="0.11" ;;
  *)  MEM_FRAC="$(python -c "print(min(0.85, 0.9/$TASKS_PER_GPU))")" ;;
esac
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC"

# ---- Detect whether the driver supports negative-bank flags ---------------
DRIVER="$REPO_DIR/rl/impls/continual_crl.py"
if grep -q "neg_bank_mode" "$DRIVER"; then
  DRIVER_HAS_NEG_BANK=true
else
  DRIVER_HAS_NEG_BANK=false
fi

# ---- Create output directories --------------------------------------------
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# ---- Flag builder ---------------------------------------------------------
# tyro boolean flags: --foo / --no-foo
bool_flag() {
  local name="$1"; local val="$2"
  if [ "$val" = "true" ]; then echo "--$name"; else echo "--no-$name"; fi
}

build_flags() {
  # Arguments: ACTOR_MODE CRITIC_MODE SEED
  local _ACTOR_MODE="$1"
  local _CRITIC_MODE="$2"
  local _SEED="$3"

  local F=""
  F+=" --seed $_SEED"
  F+=" --exp_name continual_crl"
  F+=" --agent $AGENT"

  # W&B
  F+=" $(bool_flag track $TRACK)"
  F+=" --wandb_project_name $WANDB_PROJECT_NAME"
  [ -n "$WANDB_ENTITY" ] && F+=" --wandb_entity $WANDB_ENTITY"
  F+=" --wandb_mode $WANDB_MODE"
  F+=" --wandb_group $WANDB_GROUP"
  F+=" --wandb_name_tag actor_${_ACTOR_MODE}_critic_${_CRITIC_MODE}_seed_${_SEED}"

  # Evaluation
  F+=" --num_eval_steps $NUM_EVAL_STEPS"
  F+=" --num_reset_steps $NUM_RESET_STEPS"
  F+=" --eval_episodes $EVAL_EPISODES"

  # Checkpointing
  F+=" $(bool_flag save_checkpoint $SAVE_CHECKPOINT)"
  F+=" --checkpoint_dir $CHECKPOINT_DIR"

  # Environment
  F+=" --num_envs $NUM_ENVS"
  F+=" --num_eval_envs $NUM_EVAL_ENVS"
  F+=" --num_threads $NUM_THREADS"
  F+=" $(bool_flag env_early_termination $ENV_EARLY_TERMINATION)"
  F+=" $(bool_flag permutation_invariant_reward $PERMUTATION_INVARIANT_REWARD)"

  # Single-task hyperparameters
  F+=" --rollout_length $ROLLOUT_LENGTH"
  F+=" --batch_size $BATCH_SIZE"
  F+=" --sequence_length $SEQUENCE_LENGTH"
  F+=" --actor_learning_rate $ACTOR_LR"
  F+=" --critic_learning_rate $CRITIC_LR"
  F+=" --discount $DISCOUNT"
  F+=" --entropy_cost $ENTROPY_COST"
  F+=" --logsumexp_cost $LOGSUMEXP_COST"
  F+=" --rep_size $REP_SIZE"
  F+=" --max_replay_size $MAX_REPLAY_SIZE"
  F+=" --min_replay_size $MIN_REPLAY_SIZE"

  # Continual learning
  F+=" --task_sequence $TASK_SEQUENCE"
  F+=" --actor_mode $_ACTOR_MODE"
  F+=" --critic_mode $_CRITIC_MODE"
  F+=" --steps_per_task $STEPS_PER_TASK"
  F+=" --base_steps $BASE_STEPS"
  F+=" --k_max $K_MAX"
  F+=" --alpha_scale $ALPHA_SCALE"
  F+=" $(bool_flag adapt_heads_only $ADAPT_HEADS_ONLY)"
  F+=" $(bool_flag encoder_from_base $ENCODER_FROM_BASE)"

  # Representation metrics
  F+=" $(bool_flag log_rl_metrics $LOG_RL_METRICS)"

  # Negative bank -- only emit if both enabled AND driver supports it.
  if [ "$NEG_BANK_ENABLED" = "true" ] && [ "$DRIVER_HAS_NEG_BANK" = "true" ]; then
    F+=" --neg_bank_mode $NEG_BANK_MODE"
    F+=" --neg_bank_per_task_capacity $NEG_BANK_PER_TASK_CAPACITY"
    F+=" --neg_bank_n_per_step $NEG_BANK_N_PER_STEP"
    F+=" --neg_bank_candidate_pool $NEG_BANK_CANDIDATE_POOL"
    F+=" --neg_bank_weight $NEG_BANK_WEIGHT"
    F+=" --neg_bank_max_tasks $NEG_BANK_MAX_TASKS"
  fi

  echo "$F"
}

# ---- Total number of configurations ---------------------------------------
cd "$REPO_DIR/rl/impls"
TOTAL_CONFIGS=$(python experiment_configs.py --total)

# ---- Batch header ---------------------------------------------------------
echo "============================================================"
echo "BuilderBench Continual CRL -- Batch Launcher"
echo "============================================================"
echo "SLURM Array Job ID : ${SLURM_ARRAY_JOB_ID:-local}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node               : $(hostname)"
echo "GPU                : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Tasks per GPU      : $TASKS_PER_GPU"
echo "Total configs      : $TOTAL_CONFIGS"
echo "JAX mem fraction   : $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "Driver neg-bank?   : $DRIVER_HAS_NEG_BANK (requested: $NEG_BANK_ENABLED)"
echo "Log dir            : $LOG_DIR"
echo "Checkpoint dir     : $CHECKPOINT_DIR"
echo "============================================================"

# ---- Launch TASKS_PER_GPU experiments in parallel on this GPU -------------
PIDS=()

for ((i = 0; i < TASKS_PER_GPU; i++)); do
  CONFIG_IDX=$(( TASKS_PER_GPU * ${SLURM_ARRAY_TASK_ID:-0} + i ))

  if [ "$CONFIG_IDX" -ge "$TOTAL_CONFIGS" ]; then
    echo "[slot $i] Config index $CONFIG_IDX >= $TOTAL_CONFIGS -- skipping."
    continue
  fi

  # Source per-experiment overrides from experiment_configs.py.
  # This sets ACTOR_MODE, CRITIC_MODE, SEED (and any extras) in the env.
  eval "$(python experiment_configs.py --setting "$CONFIG_IDX")"

  FLAGS=$(build_flags "$ACTOR_MODE" "$CRITIC_MODE" "$SEED")

  # Per-experiment log files (mirrors sgcrl/draft_4.sh behaviour):
  # both .out and .err contain the SAME header + the per-experiment stream.
  EXP_LOG_PREFIX="${LOG_DIR}/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${CONFIG_IDX}"

  echo ""
  echo "------------------------------------------------------------"
  echo "[slot $i] Config #${CONFIG_IDX}: actor=$ACTOR_MODE critic=$CRITIC_MODE seed=$SEED"
  echo "[slot $i] Log: ${EXP_LOG_PREFIX}.{out,err}"
  echo "[slot $i] Running: python continual_crl.py $FLAGS"
  echo "------------------------------------------------------------"

  (
    echo "============================================================"
    echo "BuilderBench Continual CRL"
    echo "============================================================"
    echo "SLURM Array Job ID : ${SLURM_ARRAY_JOB_ID:-local}"
    echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
    echo "Config Index       : $CONFIG_IDX"
    echo "Node               : $(hostname)"
    echo "GPU                : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "------------------------------------------------------------"
    echo "Seed           : $SEED"
    echo "Actor mode     : $ACTOR_MODE"
    echo "Critic mode    : $CRITIC_MODE"
    echo "Task sequence  : $TASK_SEQUENCE"
    echo "Steps per task : $STEPS_PER_TASK"
    echo "Base steps     : $BASE_STEPS"
    echo "K_max          : $K_MAX"
    echo "Heads only     : $ADAPT_HEADS_ONLY"
    echo "Encoder base   : $ENCODER_FROM_BASE"
    echo "Num envs       : $NUM_ENVS"
    echo "Batch size     : $BATCH_SIZE"
    echo "Rollout length : $ROLLOUT_LENGTH"
    echo "Actor LR       : $ACTOR_LR"
    echo "Critic LR      : $CRITIC_LR"
    echo "Log rl_metrics : $LOG_RL_METRICS"
    echo "W&B            : track=$TRACK project=$WANDB_PROJECT_NAME group=$WANDB_GROUP"
    echo "Neg bank       : enabled=$NEG_BANK_ENABLED driver_has=$DRIVER_HAS_NEG_BANK mode=$NEG_BANK_MODE"
    echo "============================================================"
    echo ""
    echo "Running: python continual_crl.py $FLAGS"
    echo ""

    # Same information duplicated on stderr so both .out and .err carry
    # the full context of this experiment, matching the sgcrl pipeline.
    (
      echo "[stderr preamble] Config #${CONFIG_IDX} actor=$ACTOR_MODE critic=$CRITIC_MODE seed=$SEED"
      echo "[stderr preamble] Running: python continual_crl.py $FLAGS"
    ) 1>&2

    python continual_crl.py $FLAGS

    echo ""
    echo "============================================================"
    echo "Run complete. Checkpoints saved to: $CHECKPOINT_DIR"
    echo "============================================================"
  ) > "${EXP_LOG_PREFIX}.out" 2> "${EXP_LOG_PREFIX}.err" &

  PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} experiment(s). PIDs: ${PIDS[*]}"
echo "Waiting for all to finish..."

# Wait for all background processes
wait

echo ""
echo "============================================================"
echo "All experiments on this GPU complete."
echo "============================================================"
