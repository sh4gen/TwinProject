#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "$0")" && pwd)"
config="$experiment_dir/duke_syntetic_swin_combined_150.yaml"

echo "TAO ReID training: Duke + synthetic Swin, 150 epochs"
echo "Config: $config"
echo "Evaluation during training uses Duke query/gallery only."

tao model re_identification train \
  -e "$config" \
  train.num_gpus=1 \
  train.gpu_ids=[0]

echo "Training complete. Checkpoints: $experiment_dir/results_swin_combined/train"
