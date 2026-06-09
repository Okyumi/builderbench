#!/bin/bash
#SBATCH --job-name=bb_dcc_ablations
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=nvidia
#SBATCH --output=/scratch/yd2247/builderbench/logs/dcc/%A_%a.out
#SBATCH --error=/scratch/yd2247/builderbench/logs/dcc/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-13

# =============================================================================
# BuilderBench DCC ablation launcher.
#
# Reads the ablation grid from experiment_configs_dcc.py (each row points to
# either continual_crl.py (baseline rows) or continual_crl_dcc.py (DCC rows)).
# Adjust SEEDS in experiment_configs_dcc.py to grow / shrink the array.
#
# Usage:
#   sbatch draft_dcc.sh                # whole sweep
#   sbatch --array=0-2 draft_dcc.sh    # first three ablations
#   TASKS_PER_GPU=2 sbatch draft_dcc.sh
# =============================================================================

set -euo pipefail

TASKS_PER_GPU="${TASKS_PER_GPU:-2}"

# Shared defaults (same as draft_4.sh).
TASK_SEQUENCE="${TASK_SEQUENCE:-cube-1-task1,cube-1-task2,cube-2-task1,cube-2-task2,cube-2-task3,cube-3-task1,cube-3-task3,cube-2-task4,cube-2-task5,cube-3-task2,cube-3-task4,cube-3-task5}"
STEPS_PER_TASK="${STEPS_PER_TASK:-50000000}"
BASE_STEPS="${BASE_STEPS:-50000000}"

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
NUM_EVAL_STEPS="${NUM_EVAL_STEPS:-50}"

LOG_RL_METRICS="${LOG_RL_METRICS:-true}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
TRACK="${TRACK:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-buildstuff}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP_DEFAULT="${WANDB_GROUP:-dcc_ablations}"

LOG_DIR="${LOG_DIR:-/scratch/yd2247/builderbench/logs/dcc}"
CHECKPOINT_DIR_DCC="${CHECKPOINT_DIR_DCC:-/scratch/yd2247/builderbench/logs/dcc_checkpoints}"
CHECKPOINT_DIR_BASE="${CHECKPOINT_DIR_BASE:-/scratch/yd2247/builderbench/logs/dcc_baseline_checkpoints}"
REPO_DIR="${REPO_DIR:-/scratch/yd2247/builderbench}"

# ---- env setup (identical to draft_4.sh) ---------------------------------
module purge
module load cuda/12.2.0
module load conda-gcc/11.2.0
if command -v nvcc >/dev/null 2>&1; then
  CUDA_ROOT="$(dirname "$(dirname "$(command -v nvcc)")")"
  export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
  [ -d "${CUDA_ROOT}/targets/x86_64-linux/lib" ] && \
    export LD_LIBRARY_PATH="${CUDA_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

UV_ENV_PATH="${UV_ENV_PATH:-/scratch/yd2247/.venvs/builderbench}"
[ -f "$UV_ENV_PATH/bin/activate" ] || { echo "uv env missing at $UV_ENV_PATH" >&2; exit 1; }
source "$UV_ENV_PATH/bin/activate"

PY_CUDA_LIBS="$(python - <<'PY'
import glob, os
paths = []
for libdir in glob.glob(os.path.join(os.environ.get("VIRTUAL_ENV", ""), "lib", "python*", "site-packages", "nvidia", "*", "lib")):
    if os.path.isdir(libdir): paths.append(libdir)
print(":".join(dict.fromkeys(paths)))
PY
)"
[ -n "$PY_CUDA_LIBS" ] && export LD_LIBRARY_PATH="${PY_CUDA_LIBS}:${LD_LIBRARY_PATH:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

case "$TASKS_PER_GPU" in
  1) MEM_FRAC="0.85" ;; 2) MEM_FRAC="0.45" ;; 3) MEM_FRAC="0.30" ;;
  4) MEM_FRAC="0.22" ;; 6) MEM_FRAC="0.15" ;; 8) MEM_FRAC="0.11" ;;
  *) MEM_FRAC="$(python -c "print(min(0.85, 0.9/$TASKS_PER_GPU))")" ;;
esac
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR_DCC" "$CHECKPOINT_DIR_BASE"

bool_flag() {
  local n="$1"; local v="$2"
  [ "$v" = "true" ] && echo "--$n" || echo "--no-$n"
}

cd "$REPO_DIR/rl/impls"
TOTAL=$(python experiment_configs_dcc.py --total)

echo "============================================================"
echo "BuilderBench DCC Ablation Launcher"
echo "ArrayJob=${SLURM_ARRAY_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} total_cfgs=$TOTAL"
echo "============================================================"

PIDS=()
for ((i = 0; i < TASKS_PER_GPU; i++)); do
  IDX=$(( TASKS_PER_GPU * ${SLURM_ARRAY_TASK_ID:-0} + i ))
  if [ "$IDX" -ge "$TOTAL" ]; then
    echo "[slot $i] index $IDX >= $TOTAL — skipping."
    continue
  fi

  eval "$(python experiment_configs_dcc.py --setting "$IDX")"
  # $NAME, $RUNNER, $SEED are now set; DCC keys when applicable are too.

  EXP_LOG_PREFIX="${LOG_DIR}/${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${IDX}"
  WANDB_GROUP="${WANDB_GROUP_DEFAULT}__${NAME}"

  COMMON=""
  COMMON+=" --seed $SEED"
  COMMON+=" $(bool_flag track $TRACK)"
  COMMON+=" --wandb_project_name $WANDB_PROJECT_NAME"
  [ -n "$WANDB_ENTITY" ] && COMMON+=" --wandb_entity $WANDB_ENTITY"
  COMMON+=" --wandb_mode $WANDB_MODE"
  COMMON+=" --wandb_group $WANDB_GROUP"
  COMMON+=" --wandb_name_tag ${NAME}_seed_${SEED}"
  COMMON+=" --num_eval_steps $NUM_EVAL_STEPS"
  COMMON+=" $(bool_flag save_checkpoint $SAVE_CHECKPOINT)"
  COMMON+=" --num_envs $NUM_ENVS --num_eval_envs $NUM_EVAL_ENVS --num_threads $NUM_THREADS"
  COMMON+=" --rollout_length $ROLLOUT_LENGTH --batch_size $BATCH_SIZE"
  COMMON+=" --sequence_length $SEQUENCE_LENGTH"
  COMMON+=" --actor_learning_rate $ACTOR_LR --critic_learning_rate $CRITIC_LR"
  COMMON+=" --discount $DISCOUNT --entropy_cost $ENTROPY_COST"
  COMMON+=" --logsumexp_cost $LOGSUMEXP_COST --rep_size $REP_SIZE"
  COMMON+=" --max_replay_size $MAX_REPLAY_SIZE --min_replay_size $MIN_REPLAY_SIZE"
  COMMON+=" --task_sequence $TASK_SEQUENCE"
  COMMON+=" --steps_per_task $STEPS_PER_TASK --base_steps $BASE_STEPS"
  COMMON+=" $(bool_flag log_rl_metrics $LOG_RL_METRICS)"

  if [ "$RUNNER" = "continual_crl.py" ]; then
    # Baseline rows: pass actor_mode/critic_mode + the existing CKA defaults.
    FLAGS="$COMMON --checkpoint_dir $CHECKPOINT_DIR_BASE"
    FLAGS+=" --actor_mode ${ACTOR_MODE:-reset} --critic_mode ${CRITIC_MODE:-persistent}"
    DRIVER=continual_crl.py
  else
    # DCC rows.
    FLAGS="$COMMON --checkpoint_dir $CHECKPOINT_DIR_DCC"
    FLAGS+=" $(bool_flag use_dcc ${USE_DCC:-true})"
    FLAGS+=" $(bool_flag dcc_use_dyn ${DCC_USE_DYN:-true})"
    FLAGS+=" --dcc_combine_mode ${DCC_COMBINE_MODE:-add}"
    FLAGS+=" --dcc_goal_encoder_mode ${DCC_GOAL_ENCODER_MODE:-shared}"
    DRIVER=continual_crl_dcc.py
  fi

  echo "[slot $i] #$IDX  $DRIVER  name=$NAME  seed=$SEED"
  echo "[slot $i] log $EXP_LOG_PREFIX.{out,err}"
  (
    python "$DRIVER" $FLAGS
  ) > "${EXP_LOG_PREFIX}.out" 2> "${EXP_LOG_PREFIX}.err" &
  PIDS+=($!)
done

echo "Launched ${#PIDS[@]} experiments (pids: ${PIDS[*]})"
wait
echo "All experiments on this GPU complete."
