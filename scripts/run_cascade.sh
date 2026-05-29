#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-b/scripts/logs
cd /home/arooba/compute-aware-vit-variant-b
nohup conda run -n ai_assisted_env python -u src/train.py \
  --config configs/cascade.yaml \
  > scripts/logs/cascade.out 2>&1 &
echo "started PID $!"
