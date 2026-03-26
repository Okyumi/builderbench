#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/.." || { echo "Failed to move to repo root"; exit 1; }

# Configuration
TASKS_FILE="scripts/all_tasks.txt"
CLIENT_NAME="openai"
MODEL_ID="gpt-5.2-2025-12-11"
AGENT_NAME="cot"
GENERATE_KWARGS='{"reasoning_effort": "high"}'

# Exactly 5 tasks to run
NUM_JOBS=1

# Set the log directory to the scripts folder (absolute path)
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to run a single task
run_task() {
    local task_id=$1
    local slot_id=$2

    # Define the log file path for this specific task
    local log_file="${LOG_DIR}/${MODEL_ID}_${AGENT_NAME}_${task_id}_output.log"

    echo "[Slot ${slot_id}] Starting ${task_id} (Logging to ${log_file})"

    python run.py \
        --client_name "${CLIENT_NAME}" \
        --model_id "${MODEL_ID}" \
        --generate_kwargs "${GENERATE_KWARGS}" \
        --level_id "${task_id}" \
        --agent_name "${AGENT_NAME}" \
        --seed 0 \
        --num_episodes 1 > "$log_file" 2>&1

    echo "[Slot ${slot_id}] Finished ${task_id}"
}

# IMPORTANT: You must export EVERY variable the function needs
export -f run_task
export CLIENT_NAME MODEL_ID AGENT_NAME GENERATE_KWARGS LOG_DIR

# Run GNU Parallel
parallel --jobs "$NUM_JOBS" \
         --delay 5 \
         run_task {1} {%} \
         :::: "$TASKS_FILE"