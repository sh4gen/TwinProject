#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/duke_syntetic_filtered_seq
IMAGE=${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt}
CONTAINER_NAME=${CONTAINER_NAME:-tao_duke_filtered_syntetic_seq_gpu0}
EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
REBUILD_DATASET=${REBUILD_DATASET:-0}
DETACHED=${DETACHED:-0}

prepare_args=(--epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --val-batch-size "$VAL_BATCH_SIZE" --num-workers "$NUM_WORKERS")
if [[ "$REBUILD_DATASET" == "1" ]]; then
  prepare_args+=(--rebuild)
fi

cd "$ROOT"
"$EXP/prepare_filtered_dataset.py" "${prepare_args[@]}"

if docker ps -a --format '{{.Names}}' | rg -q "^${CONTAINER_NAME}$"; then
  echo "Container already exists: $CONTAINER_NAME" >&2
  echo "Remove it with: docker rm $CONTAINER_NAME" >&2
  exit 1
fi

cmd=(
  docker run
  --name "$CONTAINER_NAME"
  --gpus '"device=0"'
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "$ROOT:$ROOT"
  -w "$ROOT"
)

if [[ "$DETACHED" == "1" ]]; then
  cmd+=(-d)
fi

cmd+=(
  "$IMAGE"
  re_identification train
  -e "$EXP/configs/duke_filtered_syntetic.yaml"
  'train.gpu_ids=[0]'
  'evaluate.gpu_ids=[0]'
)

echo "Starting Duke + filtered synthetic sequential training on RTX 3090 / GPU0."
echo "Config: $EXP/configs/duke_filtered_syntetic.yaml"
echo "Container: $CONTAINER_NAME"
"${cmd[@]}"
