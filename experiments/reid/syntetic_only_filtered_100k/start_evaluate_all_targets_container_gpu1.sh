#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/syntetic_only_filtered_100k
BASE=$ROOT/experiments/reid/syntetic_only_filtered_30k

EXP_DIR="$EXP" \
EXPERIMENT_NAME=syntetic_only_filtered_100k \
NUM_WORKERS="${NUM_WORKERS:-0}" \
CONTAINER_NAME="${CONTAINER_NAME:-tao_syntetic_only_100k_eval_gpu${GPU_ID:-1}}" \
"$BASE/start_evaluate_all_targets_container_gpu1.sh"
