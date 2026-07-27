#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/prcc_syntetic_filtered_seq
IMAGE=${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt}
GPU_DEVICE=${GPU_DEVICE:-1}
WORKER_ID=${WORKER_ID:-0}
WORKER_COUNT=${WORKER_COUNT:-1}
EVAL_DIR=$EXP/evaluate/prcc_real_split
LOG_DIR=$EVAL_DIR/logs
SUMMARY=$EVAL_DIR/summary_reverse_gpu${GPU_DEVICE}.tsv

mkdir -p "$LOG_DIR"
printf "experiment\tcheckpoint\tmAP\tRank-1\tRank-5\tRank-10\tstatus\tlog\n" > "$SUMMARY"

extract_metric() {
  local label=$1
  local log_file=$2
  sed -n "s/.*│ ${label}[[:space:]]*│[[:space:]]*\\([0-9.]*%\\).*/\\1/p" "$log_file" | tail -1
}

record_result() {
  local experiment=$1
  local name=$2
  local log_file=$3
  local status=$4
  local map rank1 rank5 rank10

  map=$(extract_metric "mAP" "$log_file")
  rank1=$(extract_metric "CMC curve, Rank-1" "$log_file")
  rank5=$(extract_metric "CMC curve, Rank-5" "$log_file")
  rank10=$(extract_metric "CMC curve, Rank-10" "$log_file")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$experiment" "$name" "${map:-NA}" "${rank1:-NA}" "${rank5:-NA}" "${rank10:-NA}" "$status" "$log_file" \
    >> "$SUMMARY"
}

cleanup_sanitized() {
  local sanitized=$1
  docker run --rm \
    -v "$ROOT:$ROOT" \
    "$IMAGE" \
    rm -f "$sanitized" \
    >/dev/null
}

evaluate_one() {
  local experiment=$1
  local config=$2
  local checkpoint=$3
  local name sanitized log_file exit_code

  name=$(basename "$checkpoint" .pth)
  sanitized=$(dirname "$checkpoint")/sanitized_$(basename "$checkpoint")
  log_file=$LOG_DIR/${experiment}_${name}.log

  if [[ -f "$log_file" ]] && rg -q "Execution status: PASS" "$log_file"; then
    echo "Reusing passed evaluation: $experiment / $name"
    record_result "$experiment" "$name" "$log_file" "passed"
    cleanup_sanitized "$sanitized"
    return
  fi

  echo "Evaluating: $experiment / $name"
  docker run --rm \
    -v "$ROOT:$ROOT" \
    -w "$ROOT" \
    "$IMAGE" \
    python3 "$ROOT/experiments/reid/ltcc_syntetic_sweep/sanitize_checkpoint.py" "$checkpoint" "$sanitized" \
    >/dev/null

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
    -e "$config" \
    'evaluate.gpu_ids=[0]' \
    evaluate.checkpoint="$sanitized" \
    evaluate.query_dataset="$ROOT/experiments/reid/prcc/data/query" \
    evaluate.test_dataset="$ROOT/experiments/reid/prcc/data/bounding_box_test" \
    evaluate.results_dir="$EVAL_DIR" \
    evaluate.output_sampled_matches_plot="$EVAL_DIR/${experiment}_${name}_sampled_matches.png" \
    evaluate.output_cmc_curve_plot="$EVAL_DIR/${experiment}_${name}_cmc_curve.png" \
    > "$log_file" 2>&1
  exit_code=$?
  set -e

  cleanup_sanitized "$sanitized"
  if [[ "$exit_code" -eq 0 ]] && rg -q "Execution status: PASS" "$log_file"; then
    record_result "$experiment" "$name" "$log_file" "passed"
    echo "Completed: $experiment / $name"
  else
    record_result "$experiment" "$name" "$log_file" "failed"
    echo "Failed: $experiment / $name. See $log_file" >&2
  fi
}

specs=(
  "prcc_plain_swin|$EXP/configs/prcc_plain_swin.yaml|$EXP/results/prcc_plain_swin/train"
  "prcc_filtered_syntetic_swin|$EXP/configs/prcc_filtered_syntetic_swin.yaml|$EXP/results/prcc_filtered_syntetic_swin/train"
)

echo "Reverse PRCC checkpoint evaluation started."
echo "GPU: $GPU_DEVICE"
echo "Worker: $WORKER_ID / $WORKER_COUNT"
echo "Summary: $SUMMARY"

index=0
for spec in "${specs[@]}"; do
  IFS='|' read -r experiment config train_dir <<< "$spec"
  mapfile -t checkpoints < <(find "$train_dir" -maxdepth 1 -name 'model_epoch_*.pth' | sort -Vr)
  for checkpoint in "${checkpoints[@]}"; do
    if (( index % WORKER_COUNT == WORKER_ID )); then
      evaluate_one "$experiment" "$config" "$checkpoint"
    fi
    ((index += 1))
  done
done

echo "Reverse PRCC checkpoint evaluation finished."
cat "$SUMMARY"
