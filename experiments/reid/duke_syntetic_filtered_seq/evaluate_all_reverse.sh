#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/duke_syntetic_filtered_seq
IMAGE=${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt}
GPU_DEVICE=${GPU_DEVICE:-1}
WORKER_ID=${WORKER_ID:-0}
WORKER_COUNT=${WORKER_COUNT:-1}
CONFIG=$EXP/configs/duke_filtered_syntetic.yaml
TRAIN_DIR=$EXP/results/duke_filtered_syntetic/train
EVAL_DIR=$EXP/evaluate/duke_filtered_syntetic
LOG_DIR=$EVAL_DIR/logs
SUMMARY=$EVAL_DIR/summary_reverse_gpu${GPU_DEVICE}.tsv

mkdir -p "$LOG_DIR"
printf "checkpoint\tmAP\tRank-1\tRank-5\tRank-10\tstatus\tlog\n" > "$SUMMARY"

extract_metric() {
  local label=$1
  local log_file=$2
  sed -n "s/.*│ ${label}[[:space:]]*│[[:space:]]*\\([0-9.]*%\\).*/\\1/p" "$log_file" | tail -1
}

record_result() {
  local name=$1
  local log_file=$2
  local status=$3
  local map rank1 rank5 rank10

  map=$(extract_metric "mAP" "$log_file")
  rank1=$(extract_metric "CMC curve, Rank-1" "$log_file")
  rank5=$(extract_metric "CMC curve, Rank-5" "$log_file")
  rank10=$(extract_metric "CMC curve, Rank-10" "$log_file")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "${map:-NA}" "${rank1:-NA}" "${rank5:-NA}" "${rank10:-NA}" "$status" "$log_file" \
    >> "$SUMMARY"
}

mapfile -t checkpoints < <(find "$TRAIN_DIR" -maxdepth 1 -name 'model_epoch_*.pth' | sort -Vr)
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "No checkpoints found in $TRAIN_DIR" >&2
  exit 1
fi

echo "Reverse Duke checkpoint evaluation started."
echo "GPU: $GPU_DEVICE"
echo "Worker: $WORKER_ID / $WORKER_COUNT"
echo "Checkpoints: ${#checkpoints[@]}"
echo "Summary: $SUMMARY"

for index in "${!checkpoints[@]}"; do
  if (( index % WORKER_COUNT != WORKER_ID )); then
    continue
  fi

  checkpoint=${checkpoints[$index]}
  name=$(basename "$checkpoint" .pth)
  sanitized=$(dirname "$checkpoint")/sanitized_$(basename "$checkpoint")
  log_file=$LOG_DIR/${name}.log

  if [[ -f "$log_file" ]] && rg -q "Execution status: PASS" "$log_file"; then
    echo "Reusing passed evaluation: $name"
    record_result "$name" "$log_file" "passed"
    continue
  fi

  echo "Evaluating: $name"
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
    -e "$CONFIG" \
    'evaluate.gpu_ids=[0]' \
    evaluate.checkpoint="$sanitized" \
    evaluate.query_dataset="$ROOT/experiments/reid/duke/data/query" \
    evaluate.test_dataset="$ROOT/experiments/reid/duke/data/bounding_box_test" \
    evaluate.results_dir="$EVAL_DIR" \
    evaluate.output_sampled_matches_plot="$EVAL_DIR/${name}_sampled_matches.png" \
    evaluate.output_cmc_curve_plot="$EVAL_DIR/${name}_cmc_curve.png" \
    > "$log_file" 2>&1
  exit_code=$?
  set -e

  if [[ "$exit_code" -eq 0 ]] && rg -q "Execution status: PASS" "$log_file"; then
    record_result "$name" "$log_file" "passed"
    echo "Completed: $name"
  else
    record_result "$name" "$log_file" "failed"
    echo "Failed: $name. See $log_file" >&2
  fi
done

echo "Reverse Duke checkpoint evaluation finished."
cat "$SUMMARY"
