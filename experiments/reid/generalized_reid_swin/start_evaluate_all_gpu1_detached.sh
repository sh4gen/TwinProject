#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/generalized_reid_swin
GPU_DEVICE=${GPU_DEVICE:-1}
RE_RANKING=${RE_RANKING:-false}
LOG=$EXP/evaluate/all_checkpoints_controller_gpu${GPU_DEVICE}.log
PID_FILE=$EXP/evaluate/all_checkpoints_controller_gpu${GPU_DEVICE}.pid

mkdir -p "$EXP/evaluate"
cd "$ROOT"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Generalized evaluator is already running on GPU$GPU_DEVICE with PID $(cat "$PID_FILE")." >&2
  exit 1
fi

nohup setsid env GPU_DEVICE="$GPU_DEVICE" RE_RANKING="$RE_RANKING" \
  bash "$EXP/evaluate_all_checkpoints_gpu1.sh" \
  < /dev/null > "$LOG" 2>&1 &
pid=$!
printf "%s\n" "$pid" > "$PID_FILE"

echo "Started generalized multi-target evaluator on host GPU$GPU_DEVICE."
echo "Re-ranking: $RE_RANKING"
echo "PID: $pid"
echo "Controller log: $LOG"
