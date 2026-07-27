#!/bin/bash
# Fine-tune Swin-Base on DukeMTMC-VideoReID + Synthetic data (merged training)
# Model: swin_base_market1501_aicity156_featuredim1024.tlt
# Dataset: Duke (702 IDs) + Synthetic (4 IDs, offset 800) = 706 classes
# Evaluation: Duke test/query only (synthetic eval intentionally excluded)

set -e

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$EXPERIMENT_DIR/duke_swin_merged.yaml"

echo "=========================================="
echo "  TAO ReID Training - Duke + Synthetic"
echo "  Config: $CONFIG"
echo "=========================================="

tao model re_identification train \
    -e "$CONFIG" \
    train.num_gpus=1 \
    train.gpu_ids=[0]

echo "Training complete. Checkpoints in: $EXPERIMENT_DIR/results_merged/train/"
