#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/2tb_ssd/TwinProject
SCRIPT=$ROOT/experiments/reid/automate_synthetic_impact_pipeline.sh
LOG_DIR=$ROOT/experiments/reid/repository_training_results/automation
mkdir -p "$LOG_DIR"

setsid bash "$SCRIPT" > "$LOG_DIR/synthetic_impact_launcher.log" 2>&1 < /dev/null &
echo $! > "$LOG_DIR/synthetic_impact_pipeline.pid"
echo "Started synthetic impact automation."
echo "PID: $(cat "$LOG_DIR/synthetic_impact_pipeline.pid")"
echo "Log: $LOG_DIR/synthetic_impact_pipeline.log"
