#!/usr/bin/env bash
set -euo pipefail

tao model re_identification train \
  -e /mnt/2tb_ssd/TwinProject/experiments/reid/ltcc+syntetic/ltcc_syntetic_transfer.yaml
