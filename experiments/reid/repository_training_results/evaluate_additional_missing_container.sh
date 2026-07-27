#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
REID=$ROOT/experiments/reid
OUT=$REID/repository_training_results/additional_validation
GPU_ID=${GPU_ID:-0}
WORKER_ID=${WORKER_ID:-0}
WORKER_COUNT=${WORKER_COUNT:-1}
SUMMARY=$OUT/summary_worker${WORKER_ID}.tsv
LOG_DIR=$OUT/logs

mkdir -p "$LOG_DIR"
printf "experiment\tcheckpoint\tmAP\tRank-1\tRank-5\tRank-10\tstatus\tlog\n" > "$SUMMARY"

extract_metric() {
  local label=$1
  local log_file=$2

  sed -n "s/.*${label}[[:space:]]*│[[:space:]]*\\([0-9.]*%\\).*/\\1/p" "$log_file" | tail -1
}

record_result() {
  local experiment=$1
  local checkpoint_name=$2
  local log_file=$3
  local status=$4
  local map rank1 rank5 rank10

  map=$(extract_metric "mAP" "$log_file")
  rank1=$(extract_metric "Rank-1" "$log_file")
  rank5=$(extract_metric "Rank-5" "$log_file")
  rank10=$(extract_metric "Rank-10" "$log_file")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$experiment" "$checkpoint_name" "${map:-NA}" "${rank1:-NA}" "${rank5:-NA}" "${rank10:-NA}" "$status" "$log_file" \
    >> "$SUMMARY"
}

evaluate_one() {
  local index=$1
  local experiment=$2
  local config=$3
  local raw_checkpoint=$4
  local query_dir=$5
  local gallery_dir=$6
  local class_override=${7:-}
  local checkpoint_name sanitized log_file result_dir status exit_status
  local -a tao_cmd

  if (( index % WORKER_COUNT != WORKER_ID )); then
    return
  fi

  checkpoint_name=$(basename "$raw_checkpoint" .pth)
  sanitized=$(dirname "$raw_checkpoint")/sanitized_$(basename "$raw_checkpoint")
  log_file=$LOG_DIR/${experiment}_${checkpoint_name}.log
  result_dir=$OUT/$experiment
  mkdir -p "$result_dir"

  if [[ -f "$log_file" ]] && grep -q "Execution status: PASS" "$log_file"; then
    record_result "$experiment" "$checkpoint_name" "$log_file" passed
    echo "RECORDED existing log: $experiment $checkpoint_name"
    return
  fi

  if [[ ! -f "$sanitized" ]]; then
    if ! python3 "$REID/ltcc_syntetic_sweep/sanitize_checkpoint.py" "$raw_checkpoint" "$sanitized" >"$log_file" 2>&1; then
      printf "%s\t%s\tNA\tNA\tNA\tNA\tfailed\t%s\n" \
        "$experiment" "$checkpoint_name" "$log_file" >> "$SUMMARY"
      echo "[$(date '+%F %T')] failed to sanitize: $experiment $checkpoint_name"
      return
    fi
  fi

  echo "[$(date '+%F %T')] Evaluating $experiment $checkpoint_name on container GPU $GPU_ID"
  status=passed
  set +e
  tao_cmd=(re_identification evaluate \
    -e "$config" \
    "evaluate.gpu_ids=[$GPU_ID]" \
    evaluate.checkpoint="$sanitized" \
    evaluate.query_dataset="$query_dir" \
    evaluate.test_dataset="$gallery_dir" \
    evaluate.results_dir="$result_dir" \
    evaluate.output_sampled_matches_plot="$result_dir/${checkpoint_name}_sampled_matches.png" \
    evaluate.output_cmc_curve_plot="$result_dir/${checkpoint_name}_cmc_curve.png")
  if [[ -n "$class_override" ]]; then
    tao_cmd+=("dataset.num_classes=$class_override")
  fi
  "${tao_cmd[@]}" >"$log_file" 2>&1
  exit_status=$?
  set -e

  if [[ "$exit_status" -ne 0 ]] || ! grep -q "Execution status: PASS" "$log_file"; then
    status=failed
  fi
  record_result "$experiment" "$checkpoint_name" "$log_file" "$status"
  echo "[$(date '+%F %T')] $status: $experiment $checkpoint_name"
}

index=0
evaluate_one \
  "$index" \
  ltcc_syntetic_10 \
  "$REID/ltcc_syntetic_sweep/configs/ltcc_syntetic_10.yaml" \
  "$REID/ltcc_syntetic_sweep/results/ltcc_syntetic_10/train/model_epoch_139_step_276159.pth" \
  "$REID/ltcc/data/query" \
  "$REID/ltcc/data/bounding_box_test"
index=$((index + 1))

while IFS= read -r checkpoint; do
  evaluate_one \
    "$index" \
    ltcc_swin_1.0.1 \
    "$REID/ltcc/ltcc_swin_plain.yaml" \
    "$checkpoint" \
    "$REID/ltcc/data/query" \
    "$REID/ltcc/data/bounding_box_test"
  index=$((index + 1))
done < <(find "$REID/ltcc/results_1.0.1/train" -maxdepth 1 -name 'model_epoch_*.pth' | sort -V | tail -n +7)

evaluate_one \
  "$index" \
  ltcc_swin_misfiled_0.1.4_epoch0 \
  "$REID/ltcc/ltcc_swin_plain.yaml" \
  "$REID/ltcc/results_0.1.4/train/model_epoch_000_step_00000.pth" \
  "$REID/ltcc/data/query" \
  "$REID/ltcc/data/bounding_box_test"
index=$((index + 1))

evaluate_one \
  "$index" \
  duke_swin_plain_local_partial \
  "$REID/duke/duke_swin_plain_150.yaml" \
  "$REID/duke/results_plain/train/model_epoch_004_step_02206.pth" \
  "$REID/duke/data/query" \
  "$REID/duke/data/bounding_box_test"
index=$((index + 1))

evaluate_one \
  "$index" \
  uliri_resnet_0.0.1_current_split_epoch13 \
  "$REID/uliri/uliri.yaml" \
  "$REID/uliri/results_0.0.1/train/model_epoch_013_step_38333.pth" \
  "$REID/uliri/data/query" \
  "$REID/uliri/data/bounding_box_test" \
  92
index=$((index + 1))

evaluate_one \
  "$index" \
  ltcc_sweep_smoke \
  "$REID/ltcc_syntetic_sweep/configs/ltcc_syntetic_10.yaml" \
  "$REID/ltcc_syntetic_sweep/results/ltcc_syntetic_10_smoke_20260522_1845/train/model_epoch_001_step_02621.pth" \
  "$REID/ltcc/data/query" \
  "$REID/ltcc/data/bounding_box_test"

echo "Summary: $SUMMARY"
