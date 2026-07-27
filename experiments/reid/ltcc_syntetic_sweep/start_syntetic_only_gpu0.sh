#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP_ROOT=$ROOT/experiments/reid/ltcc_syntetic_sweep
CONDA_ENV=${CONDA_ENV:-tensorrt_blackwell}
TAO_BIN=${TAO_BIN:-/home/ika/miniconda3/bin/tao}
CONFIG=$EXP_ROOT/configs/syntetic_only_100.yaml
LOG_DIR=$EXP_ROOT/logs
LOG_FILE=$LOG_DIR/syntetic_only_100_gpu0.log

mkdir -p "$LOG_DIR"

activate_conda() {
  local had_nounset=0
  case "$-" in
    *u*) had_nounset=1 ;;
  esac

  set +u
  if ! command -v conda >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
  fi
  conda activate "$CONDA_ENV"
  if [ "$had_nounset" = "1" ]; then
    set -u
  fi
}

main() {
  cd "$ROOT"
  activate_conda

  echo "TAO ReID training: synthetic-only Swin"
  echo "Config: $CONFIG"
  echo "Train: synthetic-only. Eval during training: LTCC query/gallery."
  echo "GPU: 0"
  echo "Log: $LOG_FILE"

  cmd=(
    "$TAO_BIN"
    model
    re_identification
    train
    -e
    "$CONFIG"
    "train.gpu_ids=[0]"
    "evaluate.gpu_ids=[0]"
  )

  printf -v quoted_cmd "%q " "${cmd[@]}"
  script -q -e -c "$quoted_cmd" "$LOG_FILE"
}

main "$@"
