"""Progressive token-budget widening evaluation for the multi-budget ViT.

Inference: run the shared prefix ONCE, run the 25% tail; if confidence < threshold,
reuse the cached prefix and run the 50% tail; then 75%; then full. Honest cumulative
FLOPs = prefix once + every tail actually executed. Thresholds selected on validation;
test evaluated once for the selected operating points.

This is NOT the multi-pass cascade: the prefix is computed a single time and the same
weights serve every budget.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.flops import compute_flops
from src.training import cascade_eval as ce

BUDGET_ORDER = [0.25, 0.50, 0.75, 1.00]
EXIT_KEYS = ['25', '50', '75', 'dense']
GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


class _Prefix(nn.Module):
    """Wrapper to measure prefix-only FLOPs."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m.forward_prefix(x)


def budget_costs(model: nn.Module, device: torch.device) -> Dict[str, Any]:
    """Compute prefix, per-budget tail, and cumulative shared-prefix exit FLOPs."""
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    prefix = compute_flops(_Prefix(model), dummy)
    # full budget cost = compute_flops of forward_budget at b
    full = {}
    for b in BUDGET_ORDER:
        class _FB(nn.Module):
            def __init__(s, m, bb): super().__init__(); s.m, s.b = m, bb
            def forward(s, x): return s.m.forward_budget(x, s.b)
        full[b] = compute_flops(_FB(model, b), dummy)
    tail = {b: full[b] - prefix for b in BUDGET_ORDER}
    exit_cost = {
        '25':    full[0.25],
        '50':    full[0.25] + tail[0.50],
        '75':    full[0.25] + tail[0.50] + tail[0.75],
        'dense': full[0.25] + tail[0.50] + tail[0.75] + tail[1.00],
    }
    return {'prefix': prefix, 'full': full, 'tail': tail, 'exit_cost': exit_cost}


@torch.no_grad()
def cache_budgets(model: nn.Module, loader: Any, device: torch.device) -> Dict[str, Any]:
    """Cache per-budget confidence + correctness (shared prefix per batch)."""
    model.eval()
    conf = {b: [] for b in BUDGET_ORDER}
    corr = {b: [] for b in BUDGET_ORDER}
    labels = []
    for images, lbl in loader:
        images = images.to(device)
        labels.append(lbl.numpy())
        prefix = model.forward_prefix(images)
        for b in BUDGET_ORDER:
            logits = model.forward_tail(prefix, b)
            p = F.softmax(logits, dim=-1)
            conf[b].append(p.max(-1).values.cpu().numpy())
            corr[b].append((logits.argmax(-1).cpu().numpy() == lbl.numpy()))
    return {
        'labels': np.concatenate(labels),
        'conf': {b: np.concatenate(conf[b]) for b in BUDGET_ORDER},
        'corr': {b: np.concatenate(corr[b]) for b in BUDGET_ORDER},
    }


def eval_threshold(t: float, cache: Dict[str, Any], exit_cost: Dict[str, float]) -> Dict[str, Any]:
    """Progressive widening with a single confidence threshold across stages."""
    conf, corr = cache['conf'], cache['corr']
    n = len(cache['labels'])
    m25 = conf[0.25] >= t
    m50 = (~m25) & (conf[0.50] >= t)
    m75 = (~m25) & (~m50) & (conf[0.75] >= t)
    mdn = (~m25) & (~m50) & (~m75)

    correct = np.where(m25, corr[0.25],
              np.where(m50, corr[0.50],
              np.where(m75, corr[0.75], corr[1.00])))
    flops = (m25 * exit_cost['25'] + m50 * exit_cost['50'] +
             m75 * exit_cost['75'] + mdn * exit_cost['dense'])

    def exit_acc(mask, b):
        return round(float(corr[b][mask].mean()), 6) if mask.any() else None

    return {
        'threshold': t,
        'accuracy': round(float(correct.mean()), 6),
        'avg_flops_giga': round(float(flops.mean()), 6),
        'exit_25_rate': round(float(m25.mean()), 4),
        'exit_50_rate': round(float(m50.mean()), 4),
        'exit_75_rate': round(float(m75.mean()), 4),
        'dense_rate': round(float(mdn.mean()), 4),
        'exit_25_acc': exit_acc(m25, 0.25),
        'exit_50_acc': exit_acc(m50, 0.50),
        'exit_75_acc': exit_acc(m75, 0.75),
        'dense_exit_acc': exit_acc(mdn, 1.00),
    }


def _load_baseline(name: str) -> Optional[Dict[str, float]]:
    p = f'checkpoints/{name}_clean_split/metrics.json'
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return {'test': m.get('final_test_acc'),
            'flops': m.get('flops_giga', m.get('avg_flops_giga'))}


def run_progressive_eval(
    model: nn.Module, val_loader: Any, test_loader: Any, device: torch.device,
    output_dir: str, results_dir: str, split_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Sweep thresholds on val, select operating points, evaluate once on test."""
    costs = budget_costs(model, device)
    exit_cost = costs['exit_cost']
    print(f"  prefix={costs['prefix']:.4f}  exit_cost={ {k: round(v,4) for k,v in exit_cost.items()} }")

    val_cache = cache_budgets(model, val_loader, device)
    test_cache = cache_budgets(model, test_loader, device)

    # fixed-budget accuracy of the multibudget model
    fixed_val = {EXIT_KEYS[i]: round(float(val_cache['corr'][b].mean()), 4)
                 for i, b in enumerate(BUDGET_ORDER)}
    fixed_test = {EXIT_KEYS[i]: round(float(test_cache['corr'][b].mean()), 4)
                  for i, b in enumerate(BUDGET_ORDER)}

    # confidence AUROC at 25%
    from sklearn.metrics import roc_auc_score
    c25, k25 = val_cache['conf'][0.25], val_cache['corr'][0.25]
    auroc25 = round(float(roc_auc_score(k25, c25)), 4) if len(set(k25)) > 1 else None

    # threshold sweep on validation
    val_results = [eval_threshold(t, val_cache, exit_cost)
                   for t in [1.01] + GRID + [-0.01]]
    frontier = ce.mark_pareto(val_results)

    # operating-point selection (validation only)
    f75, f50, fdense = 0.88342848, 0.687450432, 1.079406528
    def best_under(cap):
        feas = [r for r in val_results if r['avg_flops_giga'] <= cap + 1e-9]
        return max(feas, key=lambda r: r['accuracy']) if feas else None
    selected = {
        'highest_val_acc':           max(val_results, key=lambda r: r['accuracy']),
        'best_under_static75_flops': best_under(f75),
        'best_under_static50_flops': best_under(f50),
        'pareto_knee': (max([r for r in val_results if r.get('pareto_optimal')],
                            key=lambda r: r['accuracy'] - r['avg_flops_giga'] / fdense)
                        if any(r.get('pareto_optimal') for r in val_results) else None),
    }

    # evaluate selected once on test
    final_test = {}
    for name, r in selected.items():
        if r is None:
            final_test[name] = None
            continue
        tr = eval_threshold(r['threshold'], test_cache, exit_cost)
        final_test[name] = {'val': r, 'test': tr}
        print(f"  [{name}] val {r['accuracy']:.4f}@{r['avg_flops_giga']:.4f} "
              f"-> test {tr['accuracy']:.4f}@{tr['avg_flops_giga']:.4f}")

    baselines = {n: _load_baseline(n) for n in
                 ['static_25', 'static_50', 'static_75', 'dense', 'controller']}
    baselines = {n: b for n, b in baselines.items() if b}
    cp = 'checkpoints/cascade_clean_split/metrics.json'
    if os.path.exists(cp):
        cm = json.load(open(cp))
        for k, v in cm.get('final_test_results_for_selected_points', {}).items():
            if v:
                baselines[f'cascade[{k}]'] = {'test': v['accuracy'], 'flops': v['avg_flops_giga']}

    metrics = {
        'model_name': 'multibudget_progressive',
        'cost_model': 'shared-prefix cumulative (prefix once + tails executed)',
        'flops_breakdown': {'prefix': costs['prefix'],
                            'full_per_budget': {str(b): costs['full'][b] for b in BUDGET_ORDER},
                            'tail_per_budget': {str(b): costs['tail'][b] for b in BUDGET_ORDER},
                            'shared_prefix_exit_cost': exit_cost},
        'fixed_budget_val_acc': fixed_val,
        'fixed_budget_test_acc': fixed_test,
        'auroc_conf25_predicts_correct25_val': auroc25,
        'threshold_grid': GRID,
        'all_threshold_results_val': val_results,
        'pareto_frontier_val': frontier,
        'selected_operating_points': {k: (v['val'] if v else None) for k, v in final_test.items()},
        'final_test_results': {k: (v['test'] if v else None) for k, v in final_test.items()},
        'baselines_test': baselines,
    }
    metrics.update(split_info)

    os.makedirs(output_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(output_dir, 'progressive_metrics.json'), 'w'), indent=2)
    os.makedirs(results_dir, exist_ok=True)
    ce.write_threshold_csv(os.path.join(results_dir, 'multibudget_threshold_results.csv'), val_results)
    ce.write_pareto_csv(os.path.join(results_dir, 'multibudget_pareto.csv'), frontier)
    ce.maybe_plot(os.path.join(results_dir, 'multibudget_pareto.png'),
                  val_results, frontier, {n: {'acc': b['test'], 'flops': b['flops']}
                                          for n, b in baselines.items() if b.get('test')})
    return metrics
