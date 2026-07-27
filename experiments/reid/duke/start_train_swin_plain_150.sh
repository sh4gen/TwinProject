#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "$0")" && pwd)"
config="$experiment_dir/duke_swin_plain_150.yaml"

echo "TAO ReID training: Duke plain Swin, 150 epochs"
echo "Config: $config"

tao model re_identification train \
  -e "$config" \
  train.num_gpus=1 \
  train.gpu_ids=[0]

echo "Training complete. Checkpoints: $experiment_dir/results_swin_plain/train"
