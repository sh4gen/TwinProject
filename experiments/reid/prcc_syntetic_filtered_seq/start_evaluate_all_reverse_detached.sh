#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/prcc_syntetic_filtered_seq
GPU_DEVICES=${GPU_DEVICES:-0 1}

mkdir -p "$EXP/evaluate"
cd "$ROOT"

read -r -a gpu_devices <<< "$GPU_DEVICES"
worker_count=${#gpu_devices[@]}

for worker_id in "${!gpu_devices[@]}"; do
  gpu_device=${gpu_devices[$worker_id]}
  controller_log=$EXP/evaluate/reverse_evaluator_gpu${gpu_device}.log
  pid_file=$EXP/evaluate/reverse_evaluator_gpu${gpu_device}.pid

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Reverse evaluator is already running on GPU$gpu_device with PID $(cat "$pid_file")." >&2
    exit 1
  fi

  nohup setsid env \
    GPU_DEVICE="$gpu_device" \
    WORKER_ID="$worker_id" \
    WORKER_COUNT="$worker_count" \
    bash "$EXP/evaluate_all_reverse.sh" \
    < /dev/null > "$controller_log" 2>&1 &
  pid=$!
  printf "%s\n" "$pid" > "$pid_file"

  echo "Started reverse PRCC checkpoint evaluator on GPU$gpu_device."
  echo "PID: $pid"
  echo "Controller log: $controller_log"
done

nohup setsid bash "$EXP/finalize_evaluation_report.sh" \
  < /dev/null > "$EXP/evaluate/finalizer.log" 2>&1 &
printf "%s\n" "$!" > "$EXP/evaluate/finalizer.pid"
echo "Started report finalizer with PID $!."
