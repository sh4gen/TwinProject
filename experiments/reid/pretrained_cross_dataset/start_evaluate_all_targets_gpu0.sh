#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/pretrained_cross_dataset
IMAGE=${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt}
GPU_ID=${GPU_ID:-0}
CONTAINER_NAME=${CONTAINER_NAME:-tao_pretrained_cross_dataset_gpu${GPU_ID}}

mkdir -p "$EXP/evaluate/all_targets_raw"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Container already exists: $CONTAINER_NAME" >&2
  echo "Remove it after checking its status, or set CONTAINER_NAME to another value." >&2
  exit 1
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus "\"device=$GPU_ID\"" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}" \
  -e NUM_WORKERS="${NUM_WORKERS:-4}" \
  -e RE_RANKING="${RE_RANKING:-false}" \
  -v "$ROOT:$ROOT" \
  -w "$ROOT" \
  "$IMAGE" \
  bash "$EXP/run_evaluate_all_targets_inside_container.sh"

echo "Started pretrained cross-dataset evaluator on GPU $GPU_ID."
echo "Container: $CONTAINER_NAME"
echo "Summary: $EXP/evaluate/all_targets_raw/summary.tsv"
