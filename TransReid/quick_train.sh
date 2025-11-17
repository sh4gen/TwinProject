#!/bin/bash
# Quick training script for TransReID on LTCC dataset

set -e  # Exit on error

echo "=================================="
echo "TransReID Training on LTCC Dataset"
echo "=================================="
echo ""

# Parse arguments
GPU=${1:-0}
BATCH_SIZE=${2:-64}
EPOCHS=${3:-120}

echo "Configuration:"
echo "  GPU: $GPU"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Epochs: $EPOCHS"
echo ""

# Change to TransReID directory
cd "$(dirname "$0")/TransReID"

# Check if pretrained model exists
PRETRAIN_PATH="$HOME/.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth"
if [ ! -f "$PRETRAIN_PATH" ]; then
    echo "⚠ Warning: Pre-trained model not found at $PRETRAIN_PATH"
    echo "Downloading pre-trained ViT-Base model..."
    mkdir -p "$HOME/.cache/torch/checkpoints"
    wget -P "$HOME/.cache/torch/checkpoints" \
        https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
    echo "✓ Download complete"
    echo ""
fi

echo "Starting training..."
echo ""

# Run training
python train.py \
    --config_file configs/LTCC/vit_transreid_stride.yml \
    MODEL.DEVICE_ID "('$GPU')" \
    SOLVER.IMS_PER_BATCH $BATCH_SIZE \
    SOLVER.MAX_EPOCHS $EPOCHS

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="
