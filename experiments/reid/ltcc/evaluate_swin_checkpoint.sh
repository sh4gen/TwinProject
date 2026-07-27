#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/checkpoint.pth" >&2
  exit 1
fi

checkpoint=$1
name=$(basename "$checkpoint" .pth)
name=${name#sanitized_}

tao model re_identification evaluate \
  -e /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/ltcc_swin_plain.yaml \
  evaluate.checkpoint="$checkpoint" \
  evaluate.query_dataset=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/data/query \
  evaluate.test_dataset=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/data/bounding_box_test \
  evaluate.results_dir=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/evaluate_swin_plain \
  evaluate.output_sampled_matches_plot=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc/evaluate_swin_plain/${name}_sampled_matches.png
