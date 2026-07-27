#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/ltcc_syntetic_sweep
TAO_BIN=${TAO_BIN:-/home/ika/miniconda3/bin/tao}
PYTHON_BIN=${PYTHON_BIN:-python3}
GPU_ID=${GPU_ID:-1}

RESULTS_DIR=$EXP/evaluation_progress_gpu${GPU_ID}
LOG_DIR=$RESULTS_DIR/logs
SUMMARY=$RESULTS_DIR/summary.tsv

mkdir -p "$LOG_DIR"
printf "experiment\tcheckpoint\tmAP\tRank-1\tRank-5\tRank-10\tstatus\n" > "$SUMMARY"

evaluate_one() {
  local experiment=$1
  local config=$2
  local raw_checkpoint=$3
  local checkpoint
  local name
  local log_file
  local eval_dir
  local map
  local rank1
  local rank5
  local rank10
  local status

  checkpoint=$("$PYTHON_BIN" "$EXP/sanitize_checkpoint.py" "$raw_checkpoint")
  name=$(basename "$checkpoint" .pth)
  name=${name#sanitized_}
  eval_dir=$RESULTS_DIR/$experiment
  log_file=$LOG_DIR/${experiment}_${name}.log
  mkdir -p "$eval_dir"

  echo "Evaluating $experiment: $checkpoint on GPU $GPU_ID"

  local cmd
  local quoted_cmd
  local -a tao_cmd

  tao_cmd=("$TAO_BIN" model re_identification evaluate \
    -e "$config" \
    "evaluate.gpu_ids=[$GPU_ID]" \
    evaluate.checkpoint="$checkpoint" \
    evaluate.query_dataset=$ROOT/experiments/reid/ltcc/data/query \
    evaluate.test_dataset=$ROOT/experiments/reid/ltcc/data/bounding_box_test \
    evaluate.results_dir="$eval_dir" \
    evaluate.output_sampled_matches_plot="$eval_dir/${name}_sampled_matches.png" \
    evaluate.output_cmc_curve_plot="$eval_dir/${name}_cmc_curve.png")

  printf -v quoted_cmd '%q ' "${tao_cmd[@]}"
  script -q -e -c "$quoted_cmd" "$log_file" || status=failed

  status=${status:-passed}
  if ! grep -q "Execution status: PASS" "$log_file"; then
    status=failed
  fi

  map=$(sed -n 's/.*mAP[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)
  rank1=$(sed -n 's/.*Rank-1[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)
  rank5=$(sed -n 's/.*Rank-5[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)
  rank10=$(sed -n 's/.*Rank-10[[:space:]]*│[[:space:]]*\([0-9.]*%\).*/\1/p' "$log_file" | tail -1)

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$experiment" "$name" "${map:-NA}" "${rank1:-NA}" "${rank5:-NA}" "${rank10:-NA}" "$status" \
    >> "$SUMMARY"

  if [[ "$status" == "failed" ]]; then
    echo "FAILED: $experiment $name. See $log_file"
  else
    echo "PASSED: $experiment $name"
  fi
}

latest_checkpoint() {
  local train_dir=$1
  find "$train_dir" -maxdepth 1 -name 'model_epoch_*.pth' | sort -V | tail -1
}

ckpt_10=$(latest_checkpoint "$EXP/results/ltcc_syntetic_10/train")
ckpt_25=$(latest_checkpoint "$EXP/results/ltcc_syntetic_25/train")
ckpt_50=$(latest_checkpoint "$EXP/results/ltcc_syntetic_50/train")

[[ -n "$ckpt_10" ]] && evaluate_one ltcc_syntetic_10 "$EXP/configs/ltcc_syntetic_10.yaml" "$ckpt_10"
[[ -n "$ckpt_25" ]] && evaluate_one ltcc_syntetic_25 "$EXP/configs/ltcc_syntetic_25.yaml" "$ckpt_25"
[[ -n "$ckpt_50" ]] && evaluate_one ltcc_syntetic_50 "$EXP/configs/ltcc_syntetic_50.yaml" "$ckpt_50"

echo
cat "$SUMMARY"
