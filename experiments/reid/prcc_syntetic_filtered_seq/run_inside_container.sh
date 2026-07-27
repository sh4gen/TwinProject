#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/prcc_syntetic_filtered_seq

run_stage() {
  local stage=$1
  local config=$2

  echo "[$(date '+%F %T')] Starting PRCC stage: $stage"
  echo "Config: $config"
  re_identification train \
    -e "$config" \
    'train.gpu_ids=[0]' \
    'evaluate.gpu_ids=[0]'
  echo "[$(date '+%F %T')] Completed PRCC stage: $stage"
}

echo "Sequential PRCC TAO training container."
echo "Stage 1: plain PRCC Swin."
echo "Stage 2: PRCC + three-variant filtered synthetic Swin."
echo "Both stages start independently from the same pretrained Swin model."

run_stage plain "$EXP/configs/prcc_plain_swin.yaml"
run_stage filtered_syntetic "$EXP/configs/prcc_filtered_syntetic_swin.yaml"

echo "[$(date '+%F %T')] Sequential PRCC training completed."
