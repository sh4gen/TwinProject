#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/generalized_reid_swin
GPU_DEVICE=${GPU_DEVICE:-1}
LOG=$EXP/evaluate/all_checkpoints_controller_gpu${GPU_DEVICE}.log
SUMMARY=$EXP/evaluate/all_checkpoints/summary.tsv

echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "Completed rows:"
if [[ -f "$SUMMARY" ]]; then
  awk -F '\t' 'NR > 1 {count++; statuses[$7]++} END {print "rows=" count; for (status in statuses) print status "=" statuses[status]}' "$SUMMARY"
else
  echo "rows=0"
fi
echo
echo "Latest evaluator output:"
tail -n 16 "$LOG" 2>/dev/null || true
