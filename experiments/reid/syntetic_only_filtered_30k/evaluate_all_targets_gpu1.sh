#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/syntetic_only_filtered_30k
IMAGE=${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt}
GPU_DEVICE=${GPU_DEVICE:-1}
RE_RANKING=${RE_RANKING:-false}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
CONFIG=$EXP/configs/syntetic_only_filtered_30k.yaml
TRAIN_DIR=$EXP/results/syntetic_only_filtered_30k/train
EVAL_DIR=$EXP/evaluate/all_targets_raw
SUMMARY=$EVAL_DIR/summary.tsv
LOG_DIR=$EVAL_DIR/logs
SANITIZE=$ROOT/experiments/reid/ltcc_syntetic_sweep/sanitize_checkpoint.py

mkdir -p "$LOG_DIR"
if [[ ! -f "$SUMMARY" ]]; then
  printf "target\tcheckpoint\tmAP\tRank-1\tRank-5\tRank-10\tstatus\tlog\n" > "$SUMMARY"
fi

extract_metric() {
  local label=$1
  local log_file=$2
  sed -n "s/.*│ ${label}[[:space:]]*│[[:space:]]*\\([0-9.]*%\\).*/\\1/p" "$log_file" | tail -1
}

already_passed() {
  local target=$1
  local checkpoint_name=$2
  awk -F '\t' -v target="$target" -v checkpoint="$checkpoint_name" \
    'NR > 1 && $1 == target && $2 == checkpoint && $7 == "passed" { found=1 } END { exit !found }' \
    "$SUMMARY"
}

record_result() {
  local target=$1
  local checkpoint_name=$2
  local log_file=$3
  local status=$4
  local map rank1 rank5 rank10

  map=$(extract_metric "mAP" "$log_file")
  rank1=$(extract_metric "CMC curve, Rank-1" "$log_file")
  rank5=$(extract_metric "CMC curve, Rank-5" "$log_file")
  rank10=$(extract_metric "CMC curve, Rank-10" "$log_file")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$target" "$checkpoint_name" "${map:-NA}" "${rank1:-NA}" "${rank5:-NA}" "${rank10:-NA}" "$status" "$log_file" \
    >> "$SUMMARY"
}

evaluate_one() {
  local target=$1
  local checkpoint=$2
  local query=$3
  local gallery=$4
  local num_query=$5
  local checkpoint_name sanitized log_file exit_code status

  checkpoint_name=$(basename "$checkpoint" .pth)
  sanitized=$(dirname "$checkpoint")/sanitized_$(basename "$checkpoint")
  log_file=$LOG_DIR/${target}_${checkpoint_name}.log

  if already_passed "$target" "$checkpoint_name"; then
    echo "Reusing passed row: $target / $checkpoint_name"
    return
  fi
  if [[ -f "$log_file" ]] && grep -q "Execution status: PASS" "$log_file"; then
    echo "Recovering passed log: $target / $checkpoint_name"
    record_result "$target" "$checkpoint_name" "$log_file" passed
    return
  fi

  if [[ ! -f "$sanitized" ]]; then
    echo "Sanitizing: $checkpoint_name"
    docker run --rm \
      -v "$ROOT:$ROOT" \
      -w "$ROOT" \
      "$IMAGE" \
      python3 "$SANITIZE" "$checkpoint" "$sanitized" \
      >/dev/null
  fi

  echo "Evaluating: $target / $checkpoint_name"
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
    evaluate.output_sampled_matches_plot=null \
    evaluate.output_cmc_curve_plot="$EVAL_DIR/$target/${checkpoint_name}_cmc_curve.png" \
    dataset.val_batch_size="$VAL_BATCH_SIZE" \
    dataset.num_workers="$NUM_WORKERS" \
    re_ranking.re_ranking="$RE_RANKING" \
    re_ranking.num_query="$num_query" \
    > "$log_file" 2>&1
  exit_code=$?
  set -e

  status=failed
  if [[ "$exit_code" -eq 0 ]] && grep -q "Execution status: PASS" "$log_file"; then
    status=passed
  fi
  record_result "$target" "$checkpoint_name" "$log_file" "$status"
  echo "Completed: $target / $checkpoint_name / $status"
}

cleanup_sanitized() {
  local checkpoint=$1
  local sanitized
  sanitized=$(dirname "$checkpoint")/sanitized_$(basename "$checkpoint")
  docker run --rm -v "$ROOT:$ROOT" "$IMAGE" rm -f "$sanitized" >/dev/null
}

evaluate_checkpoint() {
  local checkpoint=$1
  evaluate_one duke "$checkpoint" \
    "$ROOT/experiments/reid/duke/data/query" \
    "$ROOT/experiments/reid/duke/data/bounding_box_test" \
    702
  evaluate_one ltcc "$checkpoint" \
    "$ROOT/experiments/reid/ltcc/data/query" \
    "$ROOT/experiments/reid/ltcc/data/bounding_box_test" \
    493
  evaluate_one prcc "$checkpoint" \
    "$ROOT/experiments/reid/prcc/data/query" \
    "$ROOT/experiments/reid/prcc/data/bounding_box_test" \
    71
  cleanup_sanitized "$checkpoint"
}

mapfile -t checkpoints < <(find "$TRAIN_DIR" -maxdepth 1 -name 'model_epoch_*.pth' | sort -V)
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "No stable checkpoints found in $TRAIN_DIR" >&2
  exit 1
fi

echo "Synthetic-only filtered 30k all-target raw evaluation started."
echo "Host GPU: $GPU_DEVICE"
echo "Re-ranking: $RE_RANKING"
echo "Validation batch size: $VAL_BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Checkpoints: ${#checkpoints[@]}"
echo "Summary: $SUMMARY"

for checkpoint in "${checkpoints[@]}"; do
  evaluate_checkpoint "$checkpoint"
done

echo "Synthetic-only filtered 30k all-target raw evaluation complete."
cat "$SUMMARY"
