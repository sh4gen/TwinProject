#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/prcc_syntetic_filtered_seq
EVAL_DIR=$EXP/evaluate/prcc_real_split
OUTPUT=$EVAL_DIR/summary_reverse.tsv

printf "experiment\tcheckpoint\tmAP\tRank-1\tRank-5\tRank-10\tstatus\tlog\n" > "$OUTPUT"

for summary in "$EVAL_DIR"/summary_reverse_gpu*.tsv; do
  [[ -f "$summary" ]] || continue
  tail -n +2 "$summary"
done | sort -t $'\t' -k1,1 -k2,2Vr >> "$OUTPUT"

cat "$OUTPUT"
