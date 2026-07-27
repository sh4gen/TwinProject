#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/syntetic_only_filtered_30k
EVAL_DIR=$EXP/evaluate/all_targets_raw
mkdir -p "$EVAL_DIR"

nohup "$EXP/evaluate_all_targets_gpu1.sh" \
  > "$EVAL_DIR/evaluator_controller.log" 2>&1 &
echo $! > "$EVAL_DIR/evaluator.pid"
echo "Started synthetic-only 30k all-target evaluator on GPU 1."
echo "PID: $(cat "$EVAL_DIR/evaluator.pid")"
echo "Log: $EVAL_DIR/evaluator_controller.log"
