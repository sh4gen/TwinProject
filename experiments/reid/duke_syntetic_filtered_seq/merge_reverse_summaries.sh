#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/duke_syntetic_filtered_seq
EVAL_DIR=$EXP/evaluate/duke_filtered_syntetic
OUTPUT=$EVAL_DIR/summary_reverse.tsv

printf "checkpoint\tmAP\tRank-1\tRank-5\tRank-10\tstatus\tlog\n" > "$OUTPUT"

for summary in "$EVAL_DIR"/summary_reverse_gpu*.tsv; do
  [[ -f "$summary" ]] || continue
  tail -n +2 "$summary"
done | sort -Vr >> "$OUTPUT"

cat "$OUTPUT"
