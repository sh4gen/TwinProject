#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
REID=$ROOT/experiments/reid
LOG_DIR=$REID/repository_training_results/automation

echo "Automation PID:"
if [[ -f "$LOG_DIR/synthetic_impact_pipeline.pid" ]]; then
  pid=$(cat "$LOG_DIR/synthetic_impact_pipeline.pid")
  if kill -0 "$pid" 2>/dev/null; then
    echo "$pid running"
  else
    echo "$pid finished"
  fi
else
  echo "missing"
fi
echo
echo "Containers:"
docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -E 'tao_syntetic_only(_filtered)?_(30k|100k)|tao_syntetic_only_30k_eval|tao_syntetic_only_100k_eval' || true
echo
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "30k passed rows:"
awk -F '\t' 'NR > 1 && $7 == "passed" { seen[$1 "\t" $2] = 1 } END { print length(seen)+0 }' "$REID/syntetic_only_filtered_30k/evaluate/all_targets_raw/summary.tsv" 2>/dev/null || true
echo
echo "100k passed rows:"
awk -F '\t' 'NR > 1 && $7 == "passed" { seen[$1 "\t" $2] = 1 } END { print length(seen)+0 }' "$REID/syntetic_only_filtered_100k/evaluate/all_targets_raw/summary.tsv" 2>/dev/null || true
echo
echo "Automation log tail:"
tail -n 30 "$LOG_DIR/synthetic_impact_pipeline.log" 2>/dev/null || true
