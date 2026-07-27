#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/ltcc_syntetic_sweep
PYTHON_BIN=${PYTHON_BIN:-python3}
GPU_ID=${GPU_ID:-0}
SORT_FLAGS=${SORT_FLAGS:--V}

RESULTS_DIR=$EXP/evaluation_full_gpu1
LOG_DIR=$RESULTS_DIR/logs
SUMMARY=$RESULTS_DIR/summary.tsv
REPORT=$EXP/LTCC_SYNTHETIC_SWEEP_REPORT.md

mkdir -p "$LOG_DIR"
if [[ ! -f "$SUMMARY" ]]; then
  printf "experiment\tcheckpoint\tmAP\tRank-1\tRank-5\tRank-10\tstatus\tlog\n" > "$SUMMARY"
fi

metric_from_log() {
  local metric=$1
  local log_file=$2

  case "$metric" in
    mAP)
      sed -n 's/.*mAP[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1
      ;;
    Rank-1)
      sed -n 's/.*Rank-1[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1
      ;;
    Rank-5)
      sed -n 's/.*Rank-5[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1
      ;;
    Rank-10)
      sed -n 's/.*Rank-10[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1
      ;;
  esac
}

already_passed() {
  local experiment=$1
  local checkpoint_name=$2
  awk -F '\t' -v experiment_name="$experiment" -v ckpt="$checkpoint_name" \
    '$1 == experiment_name && $2 == ckpt && $7 == "passed" {found=1} END {exit found ? 0 : 1}' "$SUMMARY"
}

append_summary_from_log() {
  local experiment=$1
  local checkpoint_name=$2
  local log_file=$3
  local status=$4
  local map rank1 rank5 rank10

  map=$(metric_from_log mAP "$log_file")
  rank1=$(metric_from_log Rank-1 "$log_file")
  rank5=$(metric_from_log Rank-5 "$log_file")
  rank10=$(metric_from_log Rank-10 "$log_file")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$experiment" "$checkpoint_name" "${map:-NA}" "${rank1:-NA}" "${rank5:-NA}" "${rank10:-NA}" "$status" "$log_file" \
    >> "$SUMMARY"
}

sanitize_checkpoint() {
  local raw_checkpoint=$1
  local sanitized_path

  sanitized_path=$(dirname "$raw_checkpoint")/sanitized_$(basename "$raw_checkpoint")
  if [[ ! -f "$sanitized_path" ]]; then
    "$PYTHON_BIN" "$EXP/sanitize_checkpoint.py" "$raw_checkpoint" "$sanitized_path" >/dev/null
  fi
  echo "$sanitized_path"
}

evaluate_one() {
  local experiment=$1
  local config=$2
  local raw_checkpoint=$3
  local class_override=${4:-}
  local checkpoint=$raw_checkpoint
  local checkpoint_name
  local log_file
  local eval_dir
  local status=passed
  local exit_status=0
  local -a tao_cmd

  if [[ "$raw_checkpoint" == *.pth ]]; then
    checkpoint=$(sanitize_checkpoint "$raw_checkpoint")
  fi

  checkpoint_name=$(basename "$checkpoint")
  checkpoint_name=${checkpoint_name%.*}
  checkpoint_name=${checkpoint_name#sanitized_}

  if already_passed "$experiment" "$checkpoint_name"; then
    echo "SKIP passed: $experiment $checkpoint_name"
    return
  fi

  eval_dir=$RESULTS_DIR/$experiment
  log_file=$LOG_DIR/${experiment}_${checkpoint_name}.log
  mkdir -p "$eval_dir"

  echo "[$(date '+%F %T')] Evaluating $experiment $checkpoint_name on container GPU $GPU_ID"

  tao_cmd=(re_identification evaluate \
    -e "$config" \
    "evaluate.gpu_ids=[$GPU_ID]" \
    evaluate.checkpoint="$checkpoint" \
    evaluate.query_dataset=$ROOT/experiments/reid/ltcc/data/query \
    evaluate.test_dataset=$ROOT/experiments/reid/ltcc/data/bounding_box_test \
    evaluate.results_dir="$eval_dir" \
    evaluate.output_sampled_matches_plot="$eval_dir/${checkpoint_name}_sampled_matches.png" \
    evaluate.output_cmc_curve_plot="$eval_dir/${checkpoint_name}_cmc_curve.png")

  if [[ -n "$class_override" ]]; then
    tao_cmd+=("dataset.num_classes=$class_override")
  fi

  set +e
  timeout --signal=TERM 360 "${tao_cmd[@]}" 2>&1 | tee "$log_file"
  exit_status=${PIPESTATUS[0]}
  set -e

  if [[ "$exit_status" -ne 0 ]] || ! grep -q "mAP" "$log_file"; then
    status=failed
  fi

  append_summary_from_log "$experiment" "$checkpoint_name" "$log_file" "$status"
  "$PYTHON_BIN" "$EXP/make_ltcc_syntetic_report.py" --output "$REPORT" || true

  echo "[$(date '+%F %T')] $status: $experiment $checkpoint_name"
}

evaluate_checkpoint_dir() {
  local experiment=$1
  local config=$2
  local train_dir=$3

  if [[ ! -d "$train_dir" ]]; then
    echo "Missing train dir, skipping: $train_dir"
    return
  fi

  while IFS= read -r checkpoint; do
    evaluate_one "$experiment" "$config" "$checkpoint"
  done < <(find "$train_dir" -maxdepth 1 -name 'model_epoch_*.pth' | sort "$SORT_FLAGS")
}

main() {
  cd "$ROOT"

  evaluate_one \
    pretrained_swin_market1501_aicity156 \
    "$ROOT/experiments/reid/ltcc/ltcc_swin_plain.yaml" \
    "$ROOT/models/reid/swin_base_market1501_aicity156_featuredim1024.tlt" \
    857

  evaluate_checkpoint_dir ltcc_syntetic_10 "$EXP/configs/ltcc_syntetic_10.yaml" "$EXP/results/ltcc_syntetic_10/train"
  evaluate_checkpoint_dir ltcc_syntetic_25 "$EXP/configs/ltcc_syntetic_25.yaml" "$EXP/results/ltcc_syntetic_25/train"
  evaluate_checkpoint_dir ltcc_syntetic_50 "$EXP/configs/ltcc_syntetic_50.yaml" "$EXP/results/ltcc_syntetic_50/train"
  evaluate_checkpoint_dir ltcc_syntetic_75 "$EXP/configs/ltcc_syntetic_75.yaml" "$EXP/results/ltcc_syntetic_75/train"
  evaluate_checkpoint_dir ltcc_syntetic_100 "$EXP/configs/ltcc_syntetic_100.yaml" "$EXP/results/ltcc_syntetic_100/train"
  evaluate_checkpoint_dir syntetic_only_100 "$EXP/configs/syntetic_only_100.yaml" "$EXP/results/syntetic_only_100_bs48_gpu0_detached/train"

  "$PYTHON_BIN" "$EXP/make_ltcc_syntetic_report.py" --output "$REPORT"
  echo "Summary: $SUMMARY"
  echo "Report: $REPORT"
}

main "$@"
