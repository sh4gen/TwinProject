#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/syntetic_only_filtered_100k

echo "Synthetic-only filtered 100k Swin training started."
echo "Train: exactly 100,000 filtered synthetic crops."
echo "Real PRCC query/gallery are reserved for later evaluation only."

re_identification train \
  -e "$EXP/configs/syntetic_only_filtered_100k.yaml"

echo "Synthetic-only filtered 100k Swin training complete."
