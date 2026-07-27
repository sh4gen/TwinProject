#!/usr/bin/env bash
set -euo pipefail

EXP=/mnt/2tb_ssd/TwinProject/experiments/reid/syntetic_only_filtered_30k
EVAL_DIR=$EXP/evaluate/all_targets_raw

echo "Evaluator PID:"
if [[ -f "$EVAL_DIR/evaluator.pid" ]]; then
  pid=$(cat "$EVAL_DIR/evaluator.pid")
  if kill -0 "$pid" 2>/dev/null; then
    echo "$pid running"
  else
    echo "$pid finished"
  fi
else
  echo "missing"
fi
echo
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "Summary tail:"
tail -n 12 "$EVAL_DIR/summary.tsv" 2>/dev/null || true
echo
echo "Controller tail:"
tail -n 20 "$EVAL_DIR/evaluator_controller.log" 2>/dev/null || true
