#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/prcc_syntetic_filtered_seq

cd "$ROOT"
GPU_ID="${GPU_ID:-0}" \
EPOCHS="${EPOCHS:-120}" \
BATCH_SIZE="${BATCH_SIZE:-48}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
REBUILD_DATASET="${REBUILD_DATASET:-0}" \
  "$EXP/run_plain_then_mixed_rtx3090.sh"

echo "Follow progress with:"
echo "docker logs -f tao_prcc_plain_then_filtered_gpu${GPU_ID:-0}"
