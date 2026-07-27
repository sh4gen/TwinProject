#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/pretrained_cross_dataset
CONFIG=$EXP/configs/pretrained_swin_market1501_aicity156.yaml
CHECKPOINT=$ROOT/models/reid/swin_base_market1501_aicity156_featuredim1024.tlt
EVAL_DIR=$EXP/evaluate/all_targets_raw
SUMMARY=$EVAL_DIR/summary.tsv
LOG_DIR=$EVAL_DIR/logs
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
RE_RANKING=${RE_RANKING:-false}

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
  awk -F '\t' -v target="$target" \
    'NR > 1 && $1 == target && $2 == "pretrained_swin_market1501_aicity156" && $7 == "passed" { found=1 } END { exit !found }' \
    "$SUMMARY"
}

record_result() {
  local target=$1
  local log_file=$2
  local status=$3
  local map rank1 rank5 rank10

  map=$(extract_metric "mAP" "$log_file")
  rank1=$(extract_metric "CMC curve, Rank-1" "$log_file")
  rank5=$(extract_metric "CMC curve, Rank-5" "$log_file")
  rank10=$(extract_metric "CMC curve, Rank-10" "$log_file")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$target" "pretrained_swin_market1501_aicity156" "${map:-NA}" "${rank1:-NA}" "${rank5:-NA}" "${rank10:-NA}" "$status" "$log_file" \
    >> "$SUMMARY"
}

evaluate_one() {
  local target=$1
  local query=$2
  local gallery=$3
  local log_file=$LOG_DIR/${target}_pretrained_swin_market1501_aicity156.log
  local exit_code status

  if already_passed "$target"; then
    echo "Reusing passed row: $target"
    return
  fi
  if [[ -f "$log_file" ]] && grep -q "Execution status: PASS" "$log_file"; then
    echo "Recovering passed log: $target"
    record_result "$target" "$log_file" passed
    return
  fi

  echo "Evaluating pretrained Swin on $target"
  set +e
  re_identification evaluate \
    -e "$CONFIG" \
    'evaluate.gpu_ids=[0]' \
    evaluate.checkpoint="$CHECKPOINT" \
    evaluate.query_dataset="$query" \
    evaluate.test_dataset="$gallery" \
    evaluate.results_dir="$EVAL_DIR/$target" \
    evaluate.output_sampled_matches_plot=null \
    evaluate.output_cmc_curve_plot="$EVAL_DIR/$target/pretrained_swin_market1501_aicity156_cmc_curve.png" \
    dataset.val_batch_size="$VAL_BATCH_SIZE" \
    dataset.num_workers="$NUM_WORKERS" \
    re_ranking.re_ranking="$RE_RANKING" \
    > "$log_file" 2>&1
  exit_code=$?
  set -e

  status=failed
  if [[ "$exit_code" -eq 0 ]] && grep -q "Execution status: PASS" "$log_file"; then
    status=passed
  fi
  record_result "$target" "$log_file" "$status"
  echo "Completed: $target / $status"
}

echo "Pretrained Swin cross-dataset evaluation started."
echo "Checkpoint: $CHECKPOINT"
echo "Re-ranking: $RE_RANKING"
echo "Validation batch size: $VAL_BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Summary: $SUMMARY"

evaluate_one duke "$ROOT/experiments/reid/duke/data/query" "$ROOT/experiments/reid/duke/data/bounding_box_test"
evaluate_one ltcc "$ROOT/experiments/reid/ltcc/data/query" "$ROOT/experiments/reid/ltcc/data/bounding_box_test"
evaluate_one prcc "$ROOT/experiments/reid/prcc/data/query" "$ROOT/experiments/reid/prcc/data/bounding_box_test"
evaluate_one uliri "$ROOT/experiments/reid/uliri/data/query" "$ROOT/experiments/reid/uliri/data/bounding_box_test"
evaluate_one synthetic_market1501 "$ROOT/experiments/reid/syntetic/synthetic_market1501/query" "$ROOT/experiments/reid/syntetic/synthetic_market1501/bounding_box_test"
evaluate_one ccvid "$ROOT/experiments/reid/ccvid/data/query" "$ROOT/experiments/reid/ccvid/data/bounding_box_test"

echo "Pretrained Swin cross-dataset evaluation complete."
cat "$SUMMARY"
