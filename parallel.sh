#!/bin/bash

# Configuration Variables
NUM_GPU=5
INSTANCES=50
NUM_EPISODES_PER_INSTANCE=20
MAX_STEPS_PER_EPISODE=40
TASK="ObjectNav"
CFG="ObjectNav"
NAME="ours"
SLEEP_INTERVAL=200
LOG_FREQ=200
PORT=2000
VENV_NAME="vlm_nav" # Name of the conda environment
CMD="python scripts/main.py --config ${CFG} -ms ${MAX_STEPS_PER_EPISODE} -ne ${NUM_EPISODES_PER_INSTANCE} --name ${NAME} --instances ${INSTANCES} --parallel -lf ${LOG_FREQ} --port ${PORT}"

# Tmux Session Names
SESSION_NAMES=()
AGGREGATOR_SESSION="aggregator_${NAME}"

# Start Aggregator Session
tmux new-session -d -s "$AGGREGATOR_SESSION" \
  "bash -i -c 'source activate ${VENV_NAME} && python scripts/aggregator.py --name ${TASK}_${NAME} --sleep ${SLEEP_INTERVAL} --port ${PORT}'"
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
  GPU_ID=$((instance_id % NUM_GPU))
  SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"

  tmux new-session -d -s "$SESSION_NAME" \
    "bash -i -c 'source activate ${VENV_NAME} && CUDA_VISIBLE_DEVICES=$GPU_ID $CMD --instance $instance_id'"
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