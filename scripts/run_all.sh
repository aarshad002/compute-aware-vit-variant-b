#!/bin/bash
# Run all training jobs sequentially. Each job is launched with nohup and
# the script waits for it to finish before starting the next one.

set -e
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-b/scripts/logs
cd /home/arooba/compute-aware-vit-variant-b

run_job() {
  local config=$1
  local logfile=$2
  echo "=========================================="
  echo "Starting: $config"
  echo "Log:      $logfile"
  echo "=========================================="
  nohup conda run -n ai_assisted_env python -u src/train.py \
    --config "$config" \
    > "$logfile" 2>&1 &
  local PID=$!
  echo "PID: $PID"
  wait $PID
  echo "Finished: $config (exit $?)"
}

run_job configs/dense.yaml        scripts/logs/dense.out
run_job configs/static_25.yaml    scripts/logs/static_25.out
run_job configs/static_50.yaml    scripts/logs/static_50.out
run_job configs/static_75.yaml    scripts/logs/static_75.out
run_job configs/dynamic_25.yaml   scripts/logs/dynamic_25.out
run_job configs/dynamic_50.yaml   scripts/logs/dynamic_50.out
run_job configs/dynamic_75.yaml   scripts/logs/dynamic_75.out
run_job configs/controller.yaml   scripts/logs/controller.out
run_job configs/cascade.yaml      scripts/logs/cascade.out

echo "=========================================="
echo "All jobs complete."
echo "=========================================="
