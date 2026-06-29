# Cascade (clean split) — summary

Split: train=45000 val=5000 test=10000 (seed 42)

Thresholds/gate selected on **validation**; test used **once** for selected points. FLOPs are cumulative.


## Clean baselines (test acc @ GFLOPs)

| model | test_acc | GFLOPs |
|---|---|---|
| dense | 0.795 | 1.0794 |
| static_25 | 0.7283 | 0.4915 |
| static_50 | 0.7686 | 0.6875 |
| static_75 | 0.7891 | 0.8834 |
| controller | 0.7789 | 0.6847 |

## Selected threshold operating points

| selection | thresholds | val_acc | val_GFLOPs | test_acc | test_GFLOPs |
|---|---|---|---|---|---|
| highest_val_acc | (0.95,0.95,0.8) | 0.8226 | 1.3433 | 0.8175 | 1.3563 |
| pareto_knee | (0.3,0.3,0.3) | 0.7402 | 0.5212 | 0.739 | 0.5215 |
| best_under_static75_flops | (0.7,0.8,0.6) | 0.7992 | 0.8785 | 0.7973 | 0.8862 |
| best_under_static50_flops | (0.6,0.4,0.7) | 0.7744 | 0.685 | 0.7748 | 0.687 |

## Learned exit gate (test of val-selected settings)

| gate selection | gate_thresh | test_acc | test_GFLOPs |
|---|---|---|---|
| highest_val_acc | 0.7 | 0.8128 | 1.4129 |
| best_under_static75_flops | 0.3 | 0.7914 | 0.8737 |
