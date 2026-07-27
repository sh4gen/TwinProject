#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
BASE=$ROOT/experiments/reid/syntetic_only_filtered_30k

exec "$BASE/run_evaluate_all_targets_inside_container.sh"
