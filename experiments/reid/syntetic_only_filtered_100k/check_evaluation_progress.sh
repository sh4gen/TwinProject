#!/usr/bin/env bash
set -euo pipefail

EXP=/mnt/2tb_ssd/TwinProject/experiments/reid/syntetic_only_filtered_100k
CONTAINER=${CONTAINER:-tao_syntetic_only_100k_eval_gpu1}

echo "Container:"
docker ps -a --filter "name=^/${CONTAINER}$" --format '{{.Names}}\t{{.Status}}'
echo
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "Summary tail:"
tail -n 12 "$EXP/evaluate/all_targets_raw/summary.tsv" 2>/dev/null || true
echo
echo "Container log tail:"
docker logs --tail 30 "$CONTAINER" 2>&1 || true
