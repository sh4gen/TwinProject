#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/ltcc_syntetic_filtered_seq
IMAGE=${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt}
GPU_DEVICE=${GPU_DEVICE:-0}
CONFIG=$EXP/configs/ltcc_filtered_syntetic.yaml
TRAIN_DIR=$EXP/results/ltcc_filtered_syntetic/train
EVAL_DIR=$EXP/evaluate/ltcc_filtered_syntetic

checkpoint=$(find "$TRAIN_DIR" -maxdepth 1 -name 'model_epoch_*.pth' | sort -V | tail -1)
if [[ -z "$checkpoint" ]]; then
  echo "No checkpoint found in $TRAIN_DIR" >&2
  exit 1
fi

sanitized=$(dirname "$checkpoint")/sanitized_$(basename "$checkpoint")
docker run --rm \
  -v "$ROOT:$ROOT" \
  -w "$ROOT" \
  "$IMAGE" \
  python3 "$EXP/../ltcc_syntetic_sweep/sanitize_checkpoint.py" "$checkpoint" "$sanitized" >/dev/null

name=$(basename "$checkpoint" .pth)
mkdir -p "$EVAL_DIR/logs"

docker run --rm \
  --gpus "\"device=$GPU_DEVICE\"" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$ROOT:$ROOT" \
  -w "$ROOT" \
  "$IMAGE" \
  re_identification evaluate \
  -e "$CONFIG" \
  'evaluate.gpu_ids=[0]' \
  evaluate.checkpoint="$sanitized" \
  evaluate.query_dataset="$ROOT/experiments/reid/ltcc/data/query" \
  evaluate.test_dataset="$ROOT/experiments/reid/ltcc/data/bounding_box_test" \
  evaluate.results_dir="$EVAL_DIR" \
  evaluate.output_sampled_matches_plot="$EVAL_DIR/${name}_sampled_matches.png" \
  evaluate.output_cmc_curve_plot="$EVAL_DIR/${name}_cmc_curve.png" \
  2>&1 | tee "$EVAL_DIR/logs/${name}.log"
