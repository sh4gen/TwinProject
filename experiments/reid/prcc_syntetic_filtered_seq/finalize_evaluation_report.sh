#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/prcc_syntetic_filtered_seq

for pid_file in "$EXP"/evaluate/reverse_evaluator_gpu*.pid; do
  [[ -f "$pid_file" ]] || continue
  pid=$(cat "$pid_file")
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
done

"$EXP/merge_reverse_summaries.sh" > "$EXP/evaluate/merged_summary.log"
python3 "$ROOT/experiments/reid/generate_repository_training_report.py" \
  > "$EXP/evaluate/report_generation.log" 2>&1

echo "PRCC evaluation summaries merged and repository report regenerated."
