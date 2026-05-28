# Project Context

This is a Master's thesis on compute-aware adaptive inference in Vision 
Transformers. The goal is to reduce inference cost by dynamically pruning 
visual tokens based on image difficulty, while maintaining competitive 
accuracy on CIFAR-100.

Conda environment name: ai_assisted_env

# Environment Notes

Follow these exactly — do not deviate:

- Server is a Linux GPU server. Do NOT use sbatch or SLURM.
- GPU: always set export CUDA_VISIBLE_DEVICES=1 before any run
- Dataset: CIFAR-100 already downloaded at:
  /home/arooba/compute-aware-vit-thesis/data/
- Use python -u for unbuffered output so logs appear in real time
- Hyperparameters — use exactly:
  batch_size: 32
  epochs: 20
  learning_rate: 0.0001
  weight_decay: 0.0001
  seed: 42

- nohup script pattern for each training job:
  #!/bin/bash
  export CUDA_VISIBLE_DEVICES=1
  mkdir -p /home/arooba/compute-aware-vit-variant-b/scripts/logs
  cd /home/arooba/compute-aware-vit-variant-b
  nohup conda run -n ai_assisted_env python -u src/train.py \
    --config configs/xxx.yaml \
    > scripts/logs/xxx.out 2>&1 &
  echo "started PID $!"

# Your Task

Design and implement a clean modular pipeline with strict separation 
of concerns. Every module must be independently importable and testable.
Clean interfaces between components matter more than anything else.

## Required modules — each in its own file

src/
├── datasets/
│   └── cifar.py          — CIFAR-100 loader, train/val splits, 224x224 resize
├── models/
│   ├── vit_dense.py      — Dense DeiT-Tiny baseline using timm
│   ├── vit_static.py     — Static token pruning (L2-norm scoring, keep_ratio)
│   ├── vit_dynamic.py    — Dynamic fixed-budget model (keep_ratio, controller disabled)
│   ├── controller.py     — Confidence-based token budget controller
│   └── cascade.py        — Cascade inference system
├── training/
│   ├── trainer.py        — Training loop, checkpointing, best model saving
│   └── evaluator.py      — Evaluation loop, FLOPs measurement, metrics reporting
├── utils/
│   ├── flops.py          — FLOPs computation using fvcore
│   └── metrics.py        — metrics.json saving and loading
└── train.py              — Main entry point, config-driven, supports all model types

## Required configs — each in configs/

- dense.yaml
- static_25.yaml, static_50.yaml, static_75.yaml
- dynamic_25.yaml, dynamic_50.yaml, dynamic_75.yaml
- controller.yaml
- cascade.yaml

All configs must use:
  data_dir: /home/arooba/compute-aware-vit-thesis/data/

## Required scripts — each in scripts/

- run_dense.sh
- run_static_25.sh, run_static_50.sh, run_static_75.sh
- run_dynamic_25.sh, run_dynamic_50.sh, run_dynamic_75.sh
- run_controller.sh
- run_cascade.sh
- run_all.sh — runs all training jobs sequentially using nohup

All scripts must follow the nohup pattern in Environment Notes.
All scripts must set export CUDA_VISIBLE_DEVICES=1.
Logs go to scripts/logs/

## Cascade specification

- Stage order: 25% → 50% → 75% → dense
- Each stage has its own independent confidence threshold:
  threshold_25, threshold_50, threshold_75
- Grid search over [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] for each stage
- Total combinations: 7 × 7 × 7 = 343
- For each combination report:
  accuracy, average FLOPs, budget distribution [25/50/75/dense]
- Dense model checkpoint path must be configurable in cascade.yaml

## Controller specification

- Prune after layer 6
- Controller head trained as auxiliary classifier on intermediate CLS token
- Training uses fixed keep_ratio=0.50 with auxiliary classification loss
- Total loss = main_CE + 0.5 × auxiliary_CE
- At inference: route per image based on confidence thresholds
  - confidence > high_thresh → 25% budget
  - confidence < low_thresh → 75% budget  
  - otherwise → 50% budget
- Evaluate 5 threshold pairs post-training:
  (0.9,0.7), (0.8,0.6), (0.7,0.5), (0.6,0.4), (0.5,0.3)

## Code requirements

- Every function must have a docstring and type hints
- Every module must be independently importable with no side effects
- Fixed random seed: 42 everywhere
- Save best_model.pt and metrics.json per run
- metrics.json must contain: model_name, parameters, flops_giga, 
  best_val_acc, epoch_history, and any pruning metadata

## What I will do

I will run training on the GPU server using the scripts you create.
Do not run training yourself.
Implement everything and verify code structure is correct.
Stop after full implementation and wait for my confirmation.