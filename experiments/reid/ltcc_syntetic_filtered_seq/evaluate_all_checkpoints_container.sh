#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/ltcc_syntetic_filtered_seq
SWEEP_EXP=$ROOT/experiments/reid/ltcc_syntetic_sweep
GPU_ID=${GPU_ID:-0}
CONFIG=$EXP/configs/ltcc_filtered_syntetic.yaml
TRAIN_DIR=$EXP/results/ltcc_filtered_syntetic/train
EVAL_DIR=$EXP/evaluate/ltcc_filtered_syntetic
LOG_DIR=$EVAL_DIR/logs
SUMMARY=$EVAL_DIR/summary.tsv

mkdir -p "$LOG_DIR"
if [[ ! -f "$SUMMARY" ]]; then
  printf "checkpoint\tmAP\tRank-1\tRank-5\tRank-10\tstatus\tlog\n" > "$SUMMARY"
fi

metric_from_log() {
  local metric=$1
  local log_file=$2

  sed -n "s/.*${metric}[[:space:]]*│[[:space:]]*\\([0-9.]*%\\).*/\\1/p" "$log_file" | tail -1
}

already_recorded() {
  local checkpoint_name=$1

  awk -F '\t' -v ckpt="$checkpoint_name" \
    '$1 == ckpt && $6 == "passed" {found=1} END {exit found ? 0 : 1}' "$SUMMARY"
}

record_from_log() {
  local checkpoint_name=$1
  local log_file=$2
  local status=$3
  local map rank1 rank5 rank10

  map=$(metric_from_log mAP "$log_file")
  rank1=$(metric_from_log Rank-1 "$log_file")
  rank5=$(metric_from_log Rank-5 "$log_file")
  rank10=$(metric_from_log Rank-10 "$log_file")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$checkpoint_name" "${map:-NA}" "${rank1:-NA}" "${rank5:-NA}" "${rank10:-NA}" "$status" "$log_file" \
    >> "$SUMMARY"
}

evaluate_one() {
  local raw_checkpoint=$1
  local checkpoint_name sanitized log_file status

  checkpoint_name=$(basename "$raw_checkpoint" .pth)
  log_file=$LOG_DIR/${checkpoint_name}.log

  if already_recorded "$checkpoint_name"; then
    echo "SKIP recorded: $checkpoint_name"
    return
  fi

  if [[ -f "$log_file" ]] && grep -q "Execution status: PASS" "$log_file"; then
    record_from_log "$checkpoint_name" "$log_file" passed
    echo "RECORDED existing log: $checkpoint_name"
    return
  fi

  sanitized=$(dirname "$raw_checkpoint")/sanitized_$(basename "$raw_checkpoint")
  if [[ ! -f "$sanitized" ]]; then
    python3 "$SWEEP_EXP/sanitize_checkpoint.py" "$raw_checkpoint" "$sanitized" >/dev/null
  fi

  echo "[$(date '+%F %T')] Evaluating $checkpoint_name on container GPU $GPU_ID"
  status=passed
  set +e
  re_identification evaluate \
    -e "$CONFIG" \
    "evaluate.gpu_ids=[$GPU_ID]" \
    evaluate.checkpoint="$sanitized" \
    evaluate.query_dataset="$ROOT/experiments/reid/ltcc/data/query" \
    evaluate.test_dataset="$ROOT/experiments/reid/ltcc/data/bounding_box_test" \
    evaluate.results_dir="$EVAL_DIR" \
    evaluate.output_sampled_matches_plot="$EVAL_DIR/${checkpoint_name}_sampled_matches.png" \
    evaluate.output_cmc_curve_plot="$EVAL_DIR/${checkpoint_name}_cmc_curve.png" \
    >"$log_file" 2>&1
  exit_status=$?
  set -e

  if [[ "$exit_status" -ne 0 ]] || ! grep -q "Execution status: PASS" "$log_file"; then
    status=failed
  fi

  record_from_log "$checkpoint_name" "$log_file" "$status"
  echo "[$(date '+%F %T')] $status: $checkpoint_name"
}

while IFS= read -r checkpoint; do
  evaluate_one "$checkpoint"
done < <(find "$TRAIN_DIR" -maxdepth 1 -name 'model_epoch_*.pth' | sort -Vr)

echo "Summary: $SUMMARY"
