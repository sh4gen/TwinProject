#!/bin/bash
# Fine-tune Swin-Base on DukeMTMC-VideoReID (plain, no synthetic data)
# Model: swin_base_market1501_aicity156_featuredim1024.tlt
# Dataset: DukeMTMC-VideoReID (702 train IDs, 4 frames/tracklet sampled)

set -e

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$EXPERIMENT_DIR/duke_swin.yaml"

echo "=========================================="
echo "  TAO ReID Training - Duke (Plain)"
echo "  Config: $CONFIG"
echo "=========================================="

tao model re_identification train \
    -e "$CONFIG" \
    train.num_gpus=1 \
    train.gpu_ids=[0]

echo "Training complete. Checkpoints in: $EXPERIMENT_DIR/results_plain/train/"
