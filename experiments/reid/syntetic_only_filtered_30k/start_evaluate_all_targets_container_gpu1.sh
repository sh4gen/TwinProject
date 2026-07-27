#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=${EXP_DIR:-$ROOT/experiments/reid/syntetic_only_filtered_30k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-syntetic_only_filtered_30k}
IMAGE=${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt}
GPU_ID=${GPU_ID:-1}
CONTAINER_NAME=${CONTAINER_NAME:-tao_syntetic_only_30k_eval_gpu${GPU_ID}}
EVAL_DIR=$EXP/evaluate/all_targets_raw

mkdir -p "$EVAL_DIR"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Container already exists: $CONTAINER_NAME" >&2
  echo "Remove it manually after checking its status." >&2
  exit 1
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus "\"device=$GPU_ID\"" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e EXP_DIR="$EXP" \
  -e EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  -e RE_RANKING="${RE_RANKING:-false}" \
  -e VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}" \
  -e NUM_WORKERS="${NUM_WORKERS:-4}" \
  -v "$ROOT:$ROOT" \
  -w "$ROOT" \
  "$IMAGE" \
  bash "$EXP/run_evaluate_all_targets_inside_container.sh"

echo "Started synthetic-only 30k all-target evaluator container on GPU $GPU_ID."
echo "Container: $CONTAINER_NAME"
