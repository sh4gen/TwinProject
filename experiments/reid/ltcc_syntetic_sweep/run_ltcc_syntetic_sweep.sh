#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/2tb_ssd/TwinProject"
EXP_ROOT="$ROOT/experiments/reid/ltcc_syntetic_sweep"
CONDA_ENV="${CONDA_ENV:-tensorrt_blackwell}"
TAO_BIN="${TAO_BIN:-/home/ika/miniconda3/bin/tao}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REBUILD_DATASETS="${REBUILD_DATASETS:-0}"

EXPERIMENTS=(
  "ltcc_syntetic_10"
  "ltcc_syntetic_25"
  "ltcc_syntetic_50"
  "ltcc_syntetic_75"
  "ltcc_syntetic_100"
  "syntetic_only_100"
)

activate_conda() {
  local had_nounset=0
  case "$-" in
    *u*) had_nounset=1 ;;
  esac

  # Some conda activation/deactivation hooks in this env reference backup
  # variables that may not exist yet. Keep strict mode for our script, but not
  # while conda runs those shell fragments.
  set +u

  if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
      # shellcheck source=/dev/null
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
      # shellcheck source=/dev/null
      source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
      echo "conda was not found. Install conda or source it before running this script." >&2
      exit 1
    fi
  else
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
  fi

  conda activate "$CONDA_ENV"

  if [ "$had_nounset" = "1" ]; then
    set -u
  fi

  if [ ! -x "$TAO_BIN" ]; then
    echo "TAO executable was not found or is not executable: $TAO_BIN" >&2
    echo "Set TAO_BIN=/path/to/tao if your TAO launcher is elsewhere." >&2
    exit 1
  fi
}

prepare() {
  local rebuild_arg=()
  if [ "$REBUILD_DATASETS" = "1" ]; then
    rebuild_arg=(--rebuild)
  fi

  "$EXP_ROOT/prepare_ltcc_syntetic_sweep.py" \
    "${rebuild_arg[@]}" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
}

run_one() {
  local experiment="$1"
  local gpu_id="$2"
  local config="$EXP_ROOT/configs/${experiment}.yaml"
  local log_dir="$EXP_ROOT/logs"
  local log_file="$log_dir/${experiment}_gpu${gpu_id}.log"

  mkdir -p "$log_dir"

  echo "[$(date '+%F %T')] START $experiment on GPU $gpu_id"
  echo "Config: $config"
  echo "Log: $log_file"

  local cmd=(
    "$TAO_BIN"
    model
    re_identification
    train
    -e
    "$config"
    "train.gpu_ids=[$gpu_id]"
    "evaluate.gpu_ids=[$gpu_id]"
  )

  if command -v script >/dev/null 2>&1; then
    local quoted_cmd
    printf -v quoted_cmd "%q " "${cmd[@]}"
    script -q -e -c "$quoted_cmd" "$log_file"
  else
    "${cmd[@]}" 2>&1 | tee "$log_file"
  fi

  echo "[$(date '+%F %T')] DONE $experiment on GPU $gpu_id"
}

run_pair() {
  local first="$1"
  local second="${2:-}"
  local first_pid=""
  local second_pid=""
  local first_status=0
  local second_status=0

  run_one "$first" 0 &
  first_pid="$!"

  if [ -n "$second" ]; then
    run_one "$second" 1 &
    second_pid="$!"
  fi

  wait "$first_pid" || first_status="$?"
  if [ -n "$second_pid" ]; then
    wait "$second_pid" || second_status="$?"
  fi

  if [ "$first_status" -ne 0 ] || [ "$second_status" -ne 0 ]; then
    echo "A training job failed in this pair: $first $second" >&2
    exit 1
  fi
}

main() {
  cd "$ROOT"
  activate_conda
  prepare

  echo "Starting LTCC synthetic sweep with two parallel jobs."
  echo "Environment: $CONDA_ENV"
  echo "Epochs: $EPOCHS"
  echo "Batch size: $BATCH_SIZE"

  local index=0
  while [ "$index" -lt "${#EXPERIMENTS[@]}" ]; do
    run_pair "${EXPERIMENTS[$index]}" "${EXPERIMENTS[$((index + 1))]:-}"
    index=$((index + 2))
  done

  echo "All LTCC synthetic sweep trainings finished."
}

main "$@"
