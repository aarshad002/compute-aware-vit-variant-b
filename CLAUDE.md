# Project Context

This is a Master's thesis on compute-aware adaptive inference in Vision 
Transformers. The goal is to reduce inference cost by dynamically pruning 
visual tokens based on image difficulty, while maintaining competitive 
accuracy on CIFAR-100.

The environment is already set up (see SETUP.md in the root directory).
Conda environment name: ai_assisted_env

# Your Task

Design and implement a clean modular pipeline. I care about separation 
of concerns, reusability, and clean interfaces between components.

## Required modules — each in its own file

- `src/datasets/cifar.py` — CIFAR-100 loader, augmentation, debug mode
- `src/models/vit_dense.py` — Dense DeiT-Tiny baseline using timm
- `src/models/vit_static.py` — Static token pruning wrapper (L2-norm scoring)
- `src/models/vit_dynamic.py` — Dynamic fixed-budget model (keep_ratio based)
- `src/models/controller.py` — Confidence-based token budget controller
- `src/models/cascade.py` — Cascade inference system
- `src/training/trainer.py` — Training loop with checkpointing
- `src/training/evaluator.py` — Evaluation, FLOPs measurement, metrics
- `src/utils/flops.py` — FLOPs computation using fvcore
- `src/utils/metrics.py` — metrics.json saving and loading
- `src/train.py` — Main entry point, config-driven

## Required configs — each in configs/

- `dense.yaml`
- `static_75.yaml`, `static_50.yaml`, `static_25.yaml`
- `dynamic_75.yaml`, `dynamic_50.yaml`, `dynamic_25.yaml`
- `controller.yaml`
- `cascade.yaml`

## Technical constraints

- Model: deit_tiny_patch16_224 from timm
- Dataset: CIFAR-100, images resized to 224x224
- Training: 20 epochs, Adam, lr=0.0001, weight decay=0.0001
- FLOPs: fvcore
- Seed: 42
- Every function must have a docstring and type hints
- Each module must be independently importable and testable

## What I will do

I will run training on a GPU cluster via SLURM.
Do not run training. Only implement and verify the code structure is correct.