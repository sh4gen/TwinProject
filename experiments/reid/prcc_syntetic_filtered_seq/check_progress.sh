#!/usr/bin/env bash
set -euo pipefail

CONTAINER=${CONTAINER:-tao_prcc_plain_then_filtered_gpu0}

echo "Container:"
docker ps -a --filter "name=^/${CONTAINER}$" --format '{{.Names}}\t{{.Status}}'
echo
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "Latest PRCC training output:"
docker logs --tail 12 "$CONTAINER" 2>&1 | tr '\r' '\n' | tail -n 12
