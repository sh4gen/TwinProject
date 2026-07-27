#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "$0")" && pwd)"
config="$experiment_dir/duke_syntetic_swin_working_lowbatch.yaml"
log_dir="$experiment_dir/results_swin_working_lowbatch"
log_file="$log_dir/start_train.log"

mkdir -p "$log_dir"

echo "TAO ReID training: Duke + synthetic Swin"
echo "Config copied from transferred Duke experiment.yaml, with local-safe batch sizes."
echo "Config: $config"
echo "Train: Duke merged train. Eval: Duke query/gallery only."

tao model re_identification train \
  -e "$config" \
  train.num_gpus=1 \
  train.gpu_ids=[0] \
  2>&1 | tee "$log_file"

if grep -q "Execution status: FAIL" "$log_file"; then
  echo "TAO reported failure. See log: $log_file" >&2
  exit 1
fi

echo "Training complete. Checkpoints: $experiment_dir/results_swin_working_lowbatch/train"
