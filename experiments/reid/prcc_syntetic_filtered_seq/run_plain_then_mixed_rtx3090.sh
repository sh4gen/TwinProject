#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/prcc_syntetic_filtered_seq
IMAGE=${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt}
GPU_ID=${GPU_ID:-0}
CONTAINER_NAME=${CONTAINER_NAME:-tao_prcc_plain_then_filtered_gpu${GPU_ID}}
EPOCHS=${EPOCHS:-120}
BATCH_SIZE=${BATCH_SIZE:-48}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
REBUILD_DATASET=${REBUILD_DATASET:-0}

prepare_args=(--epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --val-batch-size "$VAL_BATCH_SIZE" --num-workers "$NUM_WORKERS")
if [[ "$REBUILD_DATASET" == "1" ]]; then
  prepare_args+=(--rebuild)
fi

cd "$ROOT"
"$EXP/prepare_prcc_experiments.py" "${prepare_args[@]}"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Container already exists: $CONTAINER_NAME" >&2
  echo "Remove it manually after checking its status." >&2
  exit 1
fi

echo "Starting detached sequential PRCC experiment on host GPU $GPU_ID."
echo "Stage 1: plain PRCC Swin."
echo "Stage 2: PRCC + three-variant filtered synthetic Swin."
echo "Container: $CONTAINER_NAME"

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus "\"device=$GPU_ID\"" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$ROOT:$ROOT" \
  -w "$ROOT" \
  "$IMAGE" \
  bash "$EXP/run_inside_container.sh"
