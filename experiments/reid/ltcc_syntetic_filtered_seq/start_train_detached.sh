#!/usr/bin/env bash
set -euo pipefail

cd /mnt/2tb_ssd/TwinProject
DETACHED=1 REBUILD_DATASET=${REBUILD_DATASET:-0} \
  experiments/reid/ltcc_syntetic_filtered_seq/run_sequential_rtx3090.sh
