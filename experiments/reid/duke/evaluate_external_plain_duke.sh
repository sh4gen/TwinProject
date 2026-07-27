#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/plain_duke_checkpoint.pth" >&2
  exit 2
fi

checkpoint=$1
name=$(basename "$checkpoint" .pth)
name=${name#sanitized_}
results_dir=/mnt/2tb_ssd/TwinProject/experiments/reid/duke/evaluation_results_swin_working_plain
log_file="$results_dir/${name}_evaluation.log"
summary="$results_dir/summary.tsv"

mkdir -p "$results_dir"

tao model re_identification evaluate \
  -e /mnt/2tb_ssd/TwinProject/experiments/reid/duke/duke_swin_working_plain_eval.yaml \
  evaluate.checkpoint="$checkpoint" \
  evaluate.query_dataset=/mnt/2tb_ssd/TwinProject/experiments/reid/duke/data/query \
  evaluate.test_dataset=/mnt/2tb_ssd/TwinProject/experiments/reid/duke/data/bounding_box_test \
  evaluate.results_dir=/mnt/2tb_ssd/TwinProject/experiments/reid/duke/evaluate_swin_working_plain \
  evaluate.output_sampled_matches_plot=/mnt/2tb_ssd/TwinProject/experiments/reid/duke/evaluate_swin_working_plain/${name}_sampled_matches.png \
  2>&1 | tee "$log_file"

if grep -q "Execution status: FAIL" "$log_file"; then
  echo "TAO reported failure. See log: $log_file" >&2
  exit 1
fi

map=$(sed -n 's/.*mAP[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)
rank1=$(sed -n 's/.*Rank-1[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)
rank5=$(sed -n 's/.*Rank-5[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)
rank10=$(sed -n 's/.*Rank-10[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)

printf "checkpoint\tmAP\tRank-1\tRank-5\tRank-10\n" > "$summary"
printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$map" "$rank1" "$rank5" "$rank10" >> "$summary"
cat "$summary"
