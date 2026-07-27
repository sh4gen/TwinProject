#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
REID=$ROOT/experiments/reid
LOG_DIR=$REID/repository_training_results/automation
LOG_FILE=$LOG_DIR/synthetic_impact_pipeline.log
NGINX_CONTAINER=${NGINX_CONTAINER:-rezzan-architect-portfolio-hero-3d}
EXPECTED_EVAL_ROWS=${EXPECTED_EVAL_ROWS:-36}

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %z"
}

log() {
  echo "[$(timestamp)] $*"
}

container_status() {
  local name=$1
  docker ps -a --filter "name=^/${name}$" --format '{{.Status}}' | head -1
}

container_exists() {
  local name=$1
  docker ps -a --filter "name=^/${name}$" --format '{{.Names}}' | grep -qx "$name"
}

container_running() {
  local name=$1
  docker ps --filter "name=^/${name}$" --format '{{.Names}}' | grep -qx "$name"
}

passed_eval_rows() {
  local summary=$1
  [[ -f "$summary" ]] || {
    echo 0
    return
  }
  awk -F '\t' 'NR > 1 && $7 == "passed" { seen[$1 "\t" $2] = 1 } END { print length(seen) }' "$summary"
}

start_or_restart_eval() {
  local container=$1
  local start_script=$2
  local val_batch=${3:-16}

  if container_exists "$container"; then
    if container_running "$container"; then
      log "Evaluator already running: $container"
      return
    fi
    log "Removing stopped evaluator container: $container ($(container_status "$container"))"
    docker rm "$container" >/dev/null
  fi

  log "Starting evaluator: $container with VAL_BATCH_SIZE=$val_batch"
  VAL_BATCH_SIZE="$val_batch" "$start_script"
}

wait_for_eval() {
  local label=$1
  local container=$2
  local summary=$3
  local start_script=$4
  local passed val_batch status

  val_batch=16
  start_or_restart_eval "$container" "$start_script" "$val_batch"

  while true; do
    passed=$(passed_eval_rows "$summary")
    status=$(container_status "$container")
    log "$label evaluation: $passed/$EXPECTED_EVAL_ROWS passed rows; container status: ${status:-missing}"
    if [[ "$passed" -ge "$EXPECTED_EVAL_ROWS" ]]; then
      log "$label evaluation complete."
      return
    fi

    if ! container_running "$container"; then
      if [[ "$val_batch" -gt 8 ]]; then
        val_batch=8
        log "$label evaluator stopped before completion. Retrying with smaller VAL_BATCH_SIZE=$val_batch."
      else
        log "$label evaluator stopped before completion at VAL_BATCH_SIZE=$val_batch. Restarting with same batch."
      fi
      start_or_restart_eval "$container" "$start_script" "$val_batch"
    fi
    sleep 120
  done
}

wait_for_training() {
  local label=$1
  local container=$2
  local start_script=$3
  local status

  if ! container_exists "$container"; then
    log "$label training container missing. Starting it."
    "$start_script"
  fi

  while true; do
    status=$(container_status "$container")
    log "$label training status: ${status:-missing}"
    if [[ "$status" == Exited\ \(0\)* ]]; then
      log "$label training complete."
      return
    fi
    if [[ "$status" == Exited* ]]; then
      log "$label training exited unexpectedly: $status"
      return 1
    fi
    sleep 300
  done
}

publish_dashboard() {
  log "Regenerating repository training report and dashboard data."
  python3 "$REID/generate_repository_training_report.py"

  if ! container_running "$NGINX_CONTAINER"; then
    log "Starting dashboard container: $NGINX_CONTAINER"
    docker start "$NGINX_CONTAINER" >/dev/null
  fi

  log "Publishing dashboard and report assets to $NGINX_CONTAINER."
  docker cp "$REID/dashboard/index.html" "$NGINX_CONTAINER:/usr/share/nginx/html/reid-dashboard/index.html"
  docker cp "$REID/repository_training_results/." "$NGINX_CONTAINER:/usr/share/nginx/html/repository_training_results/"
  curl -fsS "http://127.0.0.1:25565/reid-dashboard/" >/dev/null
  log "Dashboard published: http://127.0.0.1:25565/reid-dashboard/"
}

main() {
  log "Synthetic impact automation started."

  wait_for_eval \
    "synthetic-only 30k" \
    "tao_syntetic_only_30k_eval_gpu1" \
    "$REID/syntetic_only_filtered_30k/evaluate/all_targets_raw/summary.tsv" \
    "$REID/syntetic_only_filtered_30k/start_evaluate_all_targets_container_gpu1.sh"

  wait_for_training \
    "synthetic-only 100k" \
    "tao_syntetic_only_filtered_100k_gpu0" \
    "$REID/syntetic_only_filtered_100k/start_train_detached.sh"

  wait_for_eval \
    "synthetic-only 100k" \
    "tao_syntetic_only_100k_eval_gpu1" \
    "$REID/syntetic_only_filtered_100k/evaluate/all_targets_raw/summary.tsv" \
    "$REID/syntetic_only_filtered_100k/start_evaluate_all_targets_container_gpu1.sh"

  publish_dashboard
  log "Synthetic impact automation finished."
}

main "$@"
