"""Oracle-ceiling diagnostic for compute-aware ViT (READ-ONLY analysis).

Uses only the existing clean-split checkpoints (dense, static_25/50/75) to obtain
per-image correctness on the clean validation and test splits, then computes the
theoretical best *single-pass* adaptive budget curve.

Cost model: each image is charged the SINGLE-MODEL FLOPs of its chosen budget
(static_25/50/75 or dense) — NOT cumulative cascade cost — because the oracle
represents the upper bound of a single-pass per-image budget selector.

No training, no new model, no MLP, no Gumbel, no cascade. The true labels are used
only to define the (unattainable) oracle ceiling; this is a feasibility bound, not
a deployable policy.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.cifar import get_dataloaders
from src.models.vit_dense import VitDense
from src.models.vit_static import VitStaticPruning
from src.utils.flops import compute_flops

DATA_DIR   = '/home/arooba/compute-aware-vit-thesis/data/'
CKPT = {
    '25':    'checkpoints/static_25_clean_split/best_model.pt',
    '50':    'checkpoints/static_50_clean_split/best_model.pt',
    '75':    'checkpoints/static_75_clean_split/best_model.pt',
    'dense': 'checkpoints/dense_clean_split/best_model.pt',
}
BUDGETS = ['25', '50', '75', 'dense']


def build_models(device):
    """Load the four clean-split models in eval mode."""
    models = {
        '25':    VitStaticPruning(keep_ratio=0.25, prune_layer=3),
        '50':    VitStaticPruning(keep_ratio=0.50, prune_layer=3),
        '75':    VitStaticPruning(keep_ratio=0.75, prune_layer=3),
        'dense': VitDense(),
    }
    for k, m in models.items():
        sd = torch.load(CKPT[k], map_location=device)
        m.load_state_dict(sd)
        m.to(device).eval()
    return models


@torch.no_grad()
def cache_correct(models, loader, device):
    """Return dict: per-budget boolean correctness arrays + labels (numpy)."""
    correct = {b: [] for b in BUDGETS}
    labels = []
    for images, lbl in loader:
        images = images.to(device)
        labels.append(lbl.numpy())
        for b in BUDGETS:
            pred = models[b](images).argmax(dim=-1).cpu().numpy()
            correct[b].append(pred == lbl.numpy())
    return {b: np.concatenate(correct[b]) for b in BUDGETS}, np.concatenate(labels)


def oracle_smallest_correct(correct, costs):
    """Smallest-correct-budget oracle: cheapest budget that is right per image.

    Images correct nowhere are charged the cheapest budget ('25') and counted wrong.

    Returns dict with accuracy, avg_flops, budget distribution (fractions), counts.
    """
    n = len(correct['25'])
    chosen = np.full(n, 'na', dtype=object)
    cost = np.zeros(n)
    got = np.zeros(n, dtype=bool)
    # assign in ascending cost order; first correct budget wins
    remaining = np.ones(n, dtype=bool)
    for b in BUDGETS:  # already cheap->expensive
        take = remaining & correct[b]
        chosen[take] = b
        cost[take] = costs[b]
        got[take] = True
        remaining = remaining & ~correct[b]
    # images correct nowhere -> cheapest budget, wrong
    chosen[remaining] = '25'
    cost[remaining] = costs['25']
    dist = {b: float(np.mean(chosen == b)) for b in BUDGETS}
    counts = {b: int(np.sum(chosen == b)) for b in BUDGETS}
    return {
        'accuracy':   float(got.mean()),       # == coverage (any model correct)
        'avg_flops':  float(cost.mean()),
        'budget_distribution': dist,
        'budget_counts': counts,
    }


def oracle_frontier(correct, costs):
    """Full oracle accuracy-vs-FLOPs frontier (max accuracy at each avg cost).

    Baseline: every image at the cheapest budget ('25'). Each image not already
    correct at '25' but correct at some budget is an 'upgrade' candidate with
    incremental cost (cheapest-correct cost - cost['25']) and gain +1 correct.
    Adding candidates in ascending incremental-cost order traces the optimal
    frontier (unit-gain knapsack).

    Returns sorted list of (avg_flops, accuracy) points from cheap to expensive.
    """
    n = len(correct['25'])
    base_correct = int(correct['25'].sum())
    base_cost = costs['25'] * n

    deltas = []
    for i in range(n):
        if correct['25'][i]:
            continue
        # cheapest correct budget among 50/75/dense
        for b in ['50', '75', 'dense']:
            if correct[b][i]:
                deltas.append(costs[b] - costs['25'])
                break
    deltas.sort()

    points = [(base_cost / n, base_correct / n)]
    cum_cost = base_cost
    cum_correct = base_correct
    for d in deltas:
        cum_cost += d
        cum_correct += 1
        points.append((cum_cost / n, cum_correct / n))
    return points


def best_under(frontier, cap):
    """Max-accuracy frontier point with avg_flops <= cap (or cheapest if none)."""
    feasible = [p for p in frontier if p[0] <= cap + 1e-9]
    if not feasible:
        return frontier[0]
    return max(feasible, key=lambda p: p[1])


def load_baselines():
    """Load clean-split baseline test/val acc + FLOPs from saved metrics."""
    out = {}
    for name in ['static_25', 'static_50', 'static_75', 'dense', 'controller']:
        p = f'checkpoints/{name}_clean_split/metrics.json'
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        out[name] = {
            'val':  m.get('best_val_acc'),
            'test': m.get('final_test_acc'),
            'flops': m.get('flops_giga', m.get('avg_flops_giga')),
        }
    # cascade selected points (cumulative FLOPs)
    cp = 'checkpoints/cascade_clean_split/metrics.json'
    if os.path.exists(cp):
        cm = json.load(open(cp))
        for k, v in cm.get('final_test_results_for_selected_points', {}).items():
            if v:
                out[f'cascade[{k}]'] = {'val': None, 'test': v['accuracy'],
                                        'flops': v['avg_flops_giga']}
    return out


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    _, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR, batch_size=64, num_workers=4, val_size=5000, split_seed=42)

    models = build_models(device)
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    costs = {b: compute_flops(models[b], dummy) for b in BUDGETS}
    print('Single-model GFLOPs:', {k: round(v, 4) for k, v in costs.items()})

    report = {
        'note': 'Oracle ceiling for SINGLE-PASS adaptive budget selection. Each image '
                'charged single-model FLOPs of its chosen budget (not cumulative). True '
                'labels used only to define the unattainable ceiling.',
        'cost_model': 'single-pass (one model per image)',
        'single_model_flops_giga': costs,
        'split_seed': 42,
    }

    for split, loader in [('val', val_loader), ('test', test_loader)]:
        correct, labels = cache_correct(models, loader, device)
        n = len(labels)
        per_model = {b: float(correct[b].mean()) for b in BUDGETS}
        coverage = float(np.any([correct[b] for b in BUDGETS], axis=0).mean())

        osc = oracle_smallest_correct(correct, costs)
        front = oracle_frontier(correct, costs)
        caps = {
            'under_static50_flops': costs['50'],
            'under_static75_flops': costs['75'],
            'under_dense_flops':    costs['dense'],
        }
        constrained = {}
        for name, cap in caps.items():
            f, a = best_under(front, cap)
            constrained[name] = {'avg_flops': f, 'accuracy': a, 'cap': cap}
        highest = {'avg_flops': front[-1][0], 'accuracy': front[-1][1]}

        report[split] = {
            'n': n,
            'per_model_accuracy': per_model,
            'coverage_any_correct': coverage,
            'oracle_smallest_correct': osc,
            'oracle_highest_accuracy': highest,
            'oracle_constrained': constrained,
            'frontier_sample': [
                {'avg_flops': round(front[i][0], 5), 'accuracy': round(front[i][1], 5)}
                for i in range(0, len(front), max(1, len(front) // 25))
            ],
        }

        print(f'\n========== {split.upper()} (n={n}) ==========')
        print('per-model acc:', {k: round(v, 4) for k, v in per_model.items()})
        print(f'coverage (any model correct): {coverage:.4f}')
        print(f'oracle smallest-correct: acc={osc["accuracy"]:.4f} '
              f'avg_flops={osc["avg_flops"]:.4f} dist='
              f'{ {k: round(v,3) for k,v in osc["budget_distribution"].items()} }')
        for name, c in constrained.items():
            print(f'oracle {name:22s}: acc={c["accuracy"]:.4f} @ {c["avg_flops"]:.4f} '
                  f'(cap {c["cap"]:.4f})')
        print(f'oracle highest accuracy : acc={highest["accuracy"]:.4f} @ '
              f'{highest["avg_flops"]:.4f}')

    report['baselines'] = load_baselines()

    os.makedirs('results', exist_ok=True)
    out_path = 'results/oracle_ceiling_report.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport saved -> {out_path}')


if __name__ == '__main__':
    main()
