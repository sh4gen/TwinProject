#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
REID=$ROOT/experiments/reid
OUT=$REID/repository_training_results
IMAGE=${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt}

container_running() {
  local name=$1

  docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -q true
}

echo "[$(date '+%F %T')] Waiting for the filtered-LTCC validator."
while container_running tao-report-ltcc-filtered; do
  sleep 30
done

echo "[$(date '+%F %T')] Launching additional checkpoint validator shard 0."
docker run -d --rm \
  --name tao-report-additional-gpu0 \
  --gpus '"device=0"' \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$ROOT:$ROOT" \
  -w "$ROOT" \
  -e GPU_ID=0 \
  -e WORKER_ID=0 \
  -e WORKER_COUNT=2 \
  "$IMAGE" \
  bash "$OUT/evaluate_additional_missing_container.sh"

echo "[$(date '+%F %T')] Waiting for the LTCC percentage-sweep validator."
while container_running tao-report-ltcc-sweep; do
  sleep 30
done

echo "[$(date '+%F %T')] Launching additional checkpoint validator shard 1."
docker run -d --rm \
  --name tao-report-additional-gpu1 \
  --gpus '"device=1"' \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$ROOT:$ROOT" \
  -w "$ROOT" \
  -e GPU_ID=0 \
  -e WORKER_ID=1 \
  -e WORKER_COUNT=2 \
  "$IMAGE" \
  bash "$OUT/evaluate_additional_missing_container.sh"

while container_running tao-report-additional-gpu0 || container_running tao-report-additional-gpu1; do
  sleep 30
done

echo "[$(date '+%F %T')] Regenerating repository training report."
cd "$ROOT"
python3 "$REID/generate_repository_training_report.py"
echo "[$(date '+%F %T')] Complete."
