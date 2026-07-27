#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
EXP=$ROOT/experiments/reid/pretrained_cross_dataset
CONTAINER_NAME=${CONTAINER_NAME:-tao_pretrained_cross_dataset_gpu0}
SUMMARY=$EXP/evaluate/all_targets_raw/summary.tsv

echo "Container:"
docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}\t{{.Status}}' || true
echo
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "Passed rows:"
awk -F '\t' 'NR > 1 && $7 == "passed" { seen[$1] = 1 } END { print length(seen)+0 }' "$SUMMARY" 2>/dev/null || true
echo
echo "Summary:"
cat "$SUMMARY" 2>/dev/null || true
echo
echo "Log tail:"
docker logs --tail 50 "$CONTAINER_NAME" 2>/dev/null || true
