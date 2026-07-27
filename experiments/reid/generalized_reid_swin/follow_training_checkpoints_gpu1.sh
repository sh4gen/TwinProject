#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/generalized_reid_swin
TRAINING_CONTAINER=${TRAINING_CONTAINER:-tao_generalized_reid_swin_gpu0}
GPU_DEVICE=${GPU_DEVICE:-1}
INITIAL_PID_FILE=$EXP/evaluate/all_checkpoints_controller_gpu${GPU_DEVICE}.pid

if [[ -f "$INITIAL_PID_FILE" ]]; then
  initial_pid=$(cat "$INITIAL_PID_FILE")
  while kill -0 "$initial_pid" 2>/dev/null; do
    sleep 30
  done
fi

while true; do
  echo "[$(date '+%F %T')] Running resumable sweep for newly available checkpoints."
  GPU_DEVICE="$GPU_DEVICE" RE_RANKING=false \
    bash "$EXP/evaluate_all_checkpoints_gpu1.sh"

  if ! docker ps --format '{{.Names}}' | grep -qx "$TRAINING_CONTAINER"; then
    echo "[$(date '+%F %T')] Training container is no longer active. Final sweep complete."
    break
  fi

  echo "[$(date '+%F %T')] Training is still active. Waiting for later checkpoints."
  sleep 60
done
