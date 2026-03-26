#!/usr/bin/env bash

# Move to the repository root (assuming this script is inside the scripts/ folder)
cd "$(dirname "${BASH_SOURCE[0]}")/.." || { echo "Failed to move to repo root"; exit 1; }

# Configuration
TASKS_FILE="scripts/all_tasks.txt" 
CLIENT_NAME="claude"
MODEL_ID="claude-opus-4-6"
AGENT_NAME="reflexion"
GENERATE_KWARGS='{"max_tokens": 16000, "thinking": {"type": "adaptive"}}'

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
    
    # Run the python script and redirect stdout and stderr to the log file
    python run.py \
        --client_name "${CLIENT_NAME}" \
        --model_id "${MODEL_ID}" \
        --generate_kwargs "${GENERATE_KWARGS}" \
        --level_id "${task_id}" \
        --agent_name "${AGENT_NAME}" \
        --seed 0 \
        --num_episodes 3 > "$log_file" 2>&1
        
    echo "[Slot ${slot_id}] Finished ${task_id}"
}

# Export variables and functions for GNU parallel
export -f run_task
export CLIENT_NAME MODEL_ID AGENT_NAME GENERATE_KWARGS LOG_DIR

# Run the 5 tasks in parallel
parallel --jobs "$NUM_JOBS" \
         --delay 5 \
         run_task {1} {%} \
         :::: "$TASKS_FILE"