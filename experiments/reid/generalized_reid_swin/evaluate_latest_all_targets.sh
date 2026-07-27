#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/generalized_reid_swin
IMAGE=${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt}
GPU_DEVICE=${GPU_DEVICE:-1}
CONFIG=$EXP/configs/generalized_swin.yaml
TRAIN_DIR=$EXP/results/generalized_swin/train
EVAL_DIR=$EXP/evaluate/final_targets
SUMMARY=$EVAL_DIR/summary.tsv
LOG_DIR=$EVAL_DIR/logs

mkdir -p "$LOG_DIR"
checkpoint=$(find "$TRAIN_DIR" -maxdepth 1 -name 'model_epoch_*.pth' | sort -V | tail -1)
if [[ -z "$checkpoint" ]]; then
  echo "No stable checkpoint found in $TRAIN_DIR" >&2
  exit 1
fi
name=$(basename "$checkpoint" .pth)
sanitized=$(dirname "$checkpoint")/sanitized_$(basename "$checkpoint")

docker run --rm \
  -v "$ROOT:$ROOT" \
  -w "$ROOT" \
  "$IMAGE" \
  python3 "$ROOT/experiments/reid/ltcc_syntetic_sweep/sanitize_checkpoint.py" "$checkpoint" "$sanitized" \
  >/dev/null

printf "target\tcheckpoint\tmAP\tRank-1\tRank-5\tRank-10\tstatus\tlog\n" > "$SUMMARY"

extract_metric() {
  local label=$1
  local log_file=$2
  sed -n "s/.*│ ${label}[[:space:]]*│[[:space:]]*\\([0-9.]*%\\).*/\\1/p" "$log_file" | tail -1
}

evaluate_one() {
  local target=$1
  local query=$2
  local gallery=$3
  local num_query=$4
  local log_file=$LOG_DIR/${target}_${name}.log
  local exit_code map rank1 rank5 rank10 status

  echo "Evaluating $target: $name"
  set +e
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
    evaluate.query_dataset="$query" \
    evaluate.test_dataset="$gallery" \
    evaluate.results_dir="$EVAL_DIR/$target" \
    evaluate.output_sampled_matches_plot="$EVAL_DIR/$target/${name}_sampled_matches.png" \
    evaluate.output_cmc_curve_plot="$EVAL_DIR/$target/${name}_cmc_curve.png" \
    re_ranking.num_query="$num_query" \
    > "$log_file" 2>&1
  exit_code=$?
  set -e

  map=$(extract_metric "mAP" "$log_file")
  rank1=$(extract_metric "CMC curve, Rank-1" "$log_file")
  rank5=$(extract_metric "CMC curve, Rank-5" "$log_file")
  rank10=$(extract_metric "CMC curve, Rank-10" "$log_file")
  status=failed
  if [[ "$exit_code" -eq 0 ]] && rg -q "Execution status: PASS" "$log_file"; then
    status=passed
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$target" "$name" "${map:-NA}" "${rank1:-NA}" "${rank5:-NA}" "${rank10:-NA}" "$status" "$log_file" \
    >> "$SUMMARY"
}

evaluate_one duke "$ROOT/experiments/reid/duke/data/query" "$ROOT/experiments/reid/duke/data/bounding_box_test" 702
evaluate_one ltcc "$ROOT/experiments/reid/ltcc/data/query" "$ROOT/experiments/reid/ltcc/data/bounding_box_test" 493
evaluate_one prcc "$ROOT/experiments/reid/prcc/data/query" "$ROOT/experiments/reid/prcc/data/bounding_box_test" 71
evaluate_one combined_stress "$EXP/data/official_stress/query" "$EXP/data/official_stress/bounding_box_test" 1266

docker run --rm -v "$ROOT:$ROOT" "$IMAGE" rm -f "$sanitized" >/dev/null
cat "$SUMMARY"
