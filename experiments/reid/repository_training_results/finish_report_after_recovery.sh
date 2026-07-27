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

echo "[$(date '+%F %T')] Waiting for GPU1 historical validation."
while container_running tao-report-additional-gpu1; do
  sleep 30
done

echo "[$(date '+%F %T')] Launching reverse sweep recovery on RTX 5070."
docker run -d --rm \
  --name tao-report-ltcc-sweep-recovery-gpu1 \
  --gpus '"device=1"' \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$ROOT:$ROOT" \
  -w "$ROOT" \
  -e GPU_ID=0 \
  -e SORT_FLAGS=-Vr \
  "$IMAGE" \
  bash "$REID/ltcc_syntetic_sweep/evaluate_all_available_container.sh"

echo "[$(date '+%F %T')] Waiting for both sweep recovery queues."
while container_running tao-report-ltcc-sweep-recovery || container_running tao-report-ltcc-sweep-recovery-gpu1; do
  sleep 30
done

echo "[$(date '+%F %T')] Retrying corrected shard 0 validations on RTX 3090."
docker run -d --rm \
  --name tao-report-additional-retry-shard0-gpu0 \
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

while container_running tao-report-additional-retry-shard0-gpu0; do
  sleep 30
done

echo "[$(date '+%F %T')] Retrying GPU1 historical-validation failures on RTX 3090."
docker run -d --rm \
  --name tao-report-additional-retry-shard1-gpu0 \
  --gpus '"device=0"' \
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

while container_running tao-report-additional-retry-shard1-gpu0; do
  sleep 30
done

echo "[$(date '+%F %T')] Regenerating final repository training report."
cd "$ROOT"
python3 "$REID/generate_repository_training_report.py"
echo "[$(date '+%F %T')] Complete."
