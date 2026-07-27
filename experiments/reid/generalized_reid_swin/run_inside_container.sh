#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/generalized_reid_swin

echo "Generalized Duke + LTCC + PRCC + 50% filtered synthetic Swin training."
echo "Validation: identity-disjoint namespaced real holdout."

re_identification train \
  -e "$EXP/configs/generalized_swin.yaml" \
  'train.gpu_ids=[0]' \
  'evaluate.gpu_ids=[0]'

echo "Generalized Swin training complete."
