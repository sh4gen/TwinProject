#!/usr/bin/env bash
set -euo pipefail

CONTAINER=${CONTAINER:-tao_syntetic_only_filtered_30k_gpu0}

echo "Container:"
docker ps -a --filter "name=^/${CONTAINER}$" --format '{{.Names}}\t{{.Status}}'
echo
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "Latest training output:"
docker logs --tail 12 "$CONTAINER" 2>&1 | tr '\r' '\n' | tail -n 12
