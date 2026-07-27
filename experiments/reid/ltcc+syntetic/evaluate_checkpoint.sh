#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/checkpoint.pth" >&2
  exit 2
fi

tao model re_identification evaluate \
  -e /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/ltcc_syntetic_transfer.yaml \
  evaluate.checkpoint="$1" \
  evaluate.query_dataset=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/data/query \
  evaluate.test_dataset=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/data/bounding_box_test \
  evaluate.results_dir=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/evaluate_transfer \
  evaluate.output_sampled_matches_plot=/mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/evaluate_transfer/sampled_matches.png
