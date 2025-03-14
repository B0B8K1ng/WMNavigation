#!/bin/bash

# Configuration Variables
ROOT_DIR=YOUR_ROOT
CONDA_PATH=YOUR_CONDA
NUM_GPU=5
INSTANCES=50
NUM_EPISODES_PER_INSTANCE=44
MAX_STEPS_PER_EPISODE=40
TASK="ObjectNav"
DATASET="mp3d"
CFG="WMNav"
NAME="baseline_v8_pro_1_5_mp3d"
PROJECT_NAME="WMNav"
VENV_NAME="wmnav" # Name of the conda environment
GPU_LIST=(3 4 5 6 7) # List of GPU IDs to use
SLEEP_INTERVAL=200
LOG_FREQ=1
PORT=2000
CMD="python scripts/main.py --config ${CFG} -ms ${MAX_STEPS_PER_EPISODE} -ne ${NUM_EPISODES_PER_INSTANCE} --name ${NAME} --instances ${INSTANCES} --parallel -lf ${LOG_FREQ} --port ${PORT} --dataset ${DATASET}"

# Tmux Session Names
SESSION_NAMES=()
AGGREGATOR_SESSION="aggregator_${NAME}"

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