#!/bin/bash

# Configuration Variables
ROOT_DIR="/file_system/vepfs/algorithm/dujun.nie/code/WMNav/"
CONDA_PATH="/file_system/vepfs/algorithm/dujun.nie/miniconda3/etc/profile.d/conda.sh"
NUM_GPU=5
INSTANCES=50
NUM_EPISODES_PER_INSTANCE=44
MAX_STEPS_PER_EPISODE=40
TASK="ObjectNav"
DATASET="mp3d"
CFG="Baseline_v8_mp3d"
NAME="baseline_v8_flash_2_mp3d_another"
PROJECT_NAME="WMNav"
SLEEP_INTERVAL=200
LOG_FREQ=1
PORT=2000
VENV_NAME="wmnav" # Name of the conda environment
CMD="python scripts/main.py --config ${CFG} -ms ${MAX_STEPS_PER_EPISODE} -ne ${NUM_EPISODES_PER_INSTANCE} --name ${NAME} --instances ${INSTANCES} --parallel -lf ${LOG_FREQ} --port ${PORT} --dataset ${DATASET}"

# Tmux Session Names
SESSION_NAMES=()
AGGREGATOR_SESSION="aggregator_${NAME}"

# List of GPU IDs to use (1, 2, 4, 5, 6)
GPU_LIST=(3 4 5 6 7)

# Start Aggregator Session
tmux new-session -d -s "$AGGREGATOR_SESSION"
tmux send-keys -t $AGGREGATOR_SESSION "source ${CONDA_PATH} && conda activate ${VENV_NAME} && cd ${ROOT_DIR} && python scripts/aggregator.py --name ${TASK}_${NAME} --project ${PROJECT_NAME} --sleep ${SLEEP_INTERVAL} --config ${CFG} --port ${PORT}" C-m
SESSION_NAMES+=("$AGGREGATOR_SESSION")

# Cleanup Function
cleanup() {
  echo "\nCaught interrupt signal. Cleaning up tmux sessions..."

  for session in "${SESSION_NAMES[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
      echo "Killed session: $session"
    fi
  done

}

# Trap SIGINT to Run Cleanup
trap cleanup SIGINT

# Start Tmux Sessions for Each Instance
for instance_id in $(seq 0 $((INSTANCES - 1))); do
  #GPU_ID=$((instance_id % NUM_GPU))
  GPU_ID=${GPU_LIST[$((instance_id % ${#GPU_LIST[@]}))]}
  SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"

  tmux new-session -d -s "$SESSION_NAME"
  tmux send-keys -t $SESSION_NAME "source ${CONDA_PATH} && conda activate ${VENV_NAME} && cd ${ROOT_DIR} && CUDA_VISIBLE_DEVICES=$GPU_ID $CMD --instance $instance_id" C-m
  SESSION_NAMES+=("$SESSION_NAME")
done

# Monitor Tmux Sessions
while true; do
  sleep $SLEEP_INTERVAL

  ALL_DONE=true

  for instance_id in $(seq 0 $((INSTANCES - 1))); do
    SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "$SESSION_NAME finished"
    else
      ALL_DONE=false
    fi
  done

  if $ALL_DONE; then
    echo "DONE"
    echo "$(date): Sending termination signal to aggregator."
    curl -X POST http://localhost:${port}/terminate
    if [ $? -eq 0 ]; then
      echo "$(date): Termination signal sent successfully."
    else
      echo "$(date): Failed to send termination signal."
    fi

    sleep 10
    if tmux has-session -t "$AGGREGATOR_SESSION" 2>/dev/null; then
      tmux kill-session -t "$AGGREGATOR_SESSION"
      echo "Killed session: $AGGREGATOR_SESSION"
    fi
    break
  fi

done