#!/usr/bin/env bash
set -euo pipefail

checkpoint_dir=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/results_swin_plain/train
results_dir=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/evaluation_results_swin_plain
mkdir -p "$results_dir"

summary="$results_dir/summary.tsv"
printf "checkpoint\tmAP\tRank-1\tRank-5\tRank-10\n" > "$summary"

for checkpoint in "$checkpoint_dir"/sanitized_model_epoch_*.pth; do
  name=$(basename "$checkpoint" .pth)
  name=${name#sanitized_}
  log_file="$results_dir/${name}_evaluation.log"

  tao model re_identification evaluate \
    -e /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/ltcc_swin_plain.yaml \
    evaluate.checkpoint="$checkpoint" \
    evaluate.query_dataset=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/data/query \
    evaluate.test_dataset=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/data/bounding_box_test \
    evaluate.results_dir=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/evaluate_swin_plain \
    evaluate.output_sampled_matches_plot=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/evaluate_swin_plain/${name}_sampled_matches.png \
    2>&1 | tee "$log_file"

  map=$(sed -n 's/.*mAP[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)
  rank1=$(sed -n 's/.*Rank-1[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)
  rank5=$(sed -n 's/.*Rank-5[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)
  rank10=$(sed -n 's/.*Rank-10[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$map" "$rank1" "$rank5" "$rank10" >> "$summary"
done

cat "$summary"
