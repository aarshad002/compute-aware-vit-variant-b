"""Shared-prefix progressive-widening feasibility diagnostic (READ-ONLY, no training).

Phase 1 evidence:
 - confirm static_25/50/75 are different weights (cannot share a prefix across them)
 - compute prefix-once FLOPs and per-budget tail FLOPs

Phase 2 diagnostic (single shared checkpoint, same weights at all budgets):
 - accuracy of each checkpoint evaluated at keep ratios {0.25, 0.50, 0.75, 1.00}
 - for the best shared checkpoint: AUROC of 25%-confidence for "25% correct",
   calibration, and a progressive-widening accuracy/FLOPs curve under HONEST
   shared-prefix cumulative FLOPs (prefix once + repeated tails only).

No new model is trained. No budget MLP. No Gumbel. Not the old multi-pass cascade
(prefix is counted once). Validation used for curves; test reported once per budget.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.cifar import get_dataloaders
from src.models.vit_static import VitStaticPruning
from src.utils.flops import compute_flops
from sklearn.metrics import roc_auc_score, brier_score_loss

DATA_DIR = '/home/arooba/compute-aware-vit-thesis/data/'
CKPT = {
    'static_25': ('checkpoints/static_25_clean_split/best_model.pt', False),
    'static_50': ('checkpoints/static_50_clean_split/best_model.pt', False),
    'static_75': ('checkpoints/static_75_clean_split/best_model.pt', False),
    'dense':     ('checkpoints/dense_clean_split/best_model.pt', True),
}
BUDGETS = [0.25, 0.50, 0.75, 1.00]
PRUNE_LAYER = 3


def load_into_static(path, is_dense, device):
    """Load any clean checkpoint into a VitStaticPruning backbone (shared arch)."""
    m = VitStaticPruning(keep_ratio=0.25, prune_layer=PRUNE_LAYER)
    sd = torch.load(path, map_location=device)
    if is_dense:  # VitDense stores timm model under 'model.'
        sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    return m, missing, unexpected


class Prefix(nn.Module):
    """Runs patch-embed + pos + the first PRUNE_LAYER blocks (all tokens)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        b = x.shape[0]
        x = self.m.patch_embed(x)
        cls = self.m.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.m.pos_embed
        x = self.m.pos_drop(x)
        for i in range(self.m.prune_layer):
            x = self.m.blocks[i](x)
        return x


@torch.no_grad()
def eval_budget(model, loader, keep_ratio, device):
    """Return (confidence array, correctness array) for the model at keep_ratio."""
    model.keep_ratio = keep_ratio
    conf, corr = [], []
    for images, lbl in loader:
        images = images.to(device)
        logits = model(images)
        p = F.softmax(logits, dim=-1)
        conf.append(p.max(-1).values.cpu().numpy())
        corr.append((logits.argmax(-1).cpu().numpy() == lbl.numpy()))
    return np.concatenate(conf), np.concatenate(corr)


def ece(conf, corr, bins=15):
    """Expected calibration error."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def progressive_curve(cache, costs_cum, grid):
    """Progressive widening with a single confidence threshold across stages.

    cache: dict budget-> (conf, corr) arrays (same shared checkpoint).
    costs_cum: dict exit-stage -> honest shared-prefix cumulative GFLOPs.
    """
    c25, k25 = cache[0.25]
    c50, k50 = cache[0.50]
    c75, k75 = cache[0.75]
    cdn, kdn = cache[1.00]
    n = len(k25)
    pts = []
    for t in [1.01] + grid + [-0.01]:
        m25 = c25 >= t
        m50 = (~m25) & (c50 >= t)
        m75 = (~m25) & (~m50) & (c75 >= t)
        mdn = (~m25) & (~m50) & (~m75)
        correct = np.where(m25, k25, np.where(m50, k50, np.where(m75, k75, kdn)))
        flops = (m25 * costs_cum['25'] + m50 * costs_cum['50'] +
                 m75 * costs_cum['75'] + mdn * costs_cum['dense'])
        pts.append({
            'threshold': round(t, 3),
            'accuracy': round(float(correct.mean()), 5),
            'avg_flops_giga': round(float(flops.mean()), 5),
            'exit_rates': {'25': round(float(m25.mean()), 4), '50': round(float(m50.mean()), 4),
                           '75': round(float(m75.mean()), 4), 'dense': round(float(mdn.mean()), 4)},
        })
    return pts


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    _, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR, batch_size=64, num_workers=4, val_size=5000, split_seed=42)

    # ---------- Phase 1: weights differ? ----------
    sds = {}
    for name, (path, is_dense) in CKPT.items():
        sd = torch.load(path, map_location='cpu')
        if is_dense:
            sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
        sds[name] = sd
    key = 'blocks.0.attn.qkv.weight'
    print('\n=== Phase 1: pairwise weight difference (blocks.0.attn.qkv.weight) ===')
    names = list(sds)
    diffs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = (sds[names[i]][key] - sds[names[j]][key]).abs().mean().item()
            diffs[f'{names[i]} vs {names[j]}'] = round(d, 6)
            print(f'  {names[i]:10s} vs {names[j]:10s}: mean|Δ| = {d:.6f}')

    # ---------- Phase 1: prefix / tail FLOPs ----------
    m_ref, miss, unexp = load_into_static(*CKPT['static_75'], device)
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    prefix_flops = compute_flops(Prefix(m_ref), dummy)
    full = {}
    for b in BUDGETS:
        m_ref.keep_ratio = b
        full[b] = compute_flops(m_ref, dummy)
    tail = {b: full[b] - prefix_flops for b in BUDGETS}
    costs_cum = {
        '25':    full[0.25],
        '50':    full[0.25] + tail[0.50],
        '75':    full[0.25] + tail[0.50] + tail[0.75],
        'dense': full[0.25] + tail[0.50] + tail[0.75] + tail[1.00],
    }
    print(f'\n=== Phase 1: FLOPs (GFLOPs) ===')
    print(f'  prefix once = {prefix_flops:.4f}')
    print(f'  full per budget = {({b: round(full[b],4) for b in BUDGETS})}')
    print(f'  tail per budget = {({b: round(tail[b],4) for b in BUDGETS})}')
    print(f'  shared-prefix cumulative exit cost = '
          f'{ {k: round(v,4) for k,v in costs_cum.items()} }')
    print(f'  (compare full multi-pass cascade exit-dense = '
          f'{round(sum(full[b] for b in BUDGETS),4)})')

    # ---------- Phase 2: each checkpoint at every budget (val) ----------
    print('\n=== Phase 2: accuracy of each checkpoint at each budget (VALIDATION) ===')
    table = {}
    caches = {}
    for name, (path, is_dense) in CKPT.items():
        m, _, _ = load_into_static(path, is_dense, device)
        caches[name] = {}
        row = {}
        for b in BUDGETS:
            conf, corr = eval_budget(m, val_loader, b, device)
            caches[name][b] = (conf, corr)
            row[b] = round(float(corr.mean()), 4)
        table[name] = row
        print(f'  {name:10s}: ' + '  '.join(f'{b:.2f}={row[b]:.4f}' for b in BUDGETS))

    # choose most budget-robust shared checkpoint: max mean accuracy across budgets
    best_name = max(table, key=lambda nm: np.mean(list(table[nm].values())))
    print(f'\nMost budget-robust single checkpoint (mean acc across budgets): {best_name}')

    report = {
        'note': 'Shared-prefix progressive widening feasibility. Prefix counted once; '
                'tails accumulate. Single checkpoint reused at all budgets. No training.',
        'phase1_weight_diffs': diffs,
        'prefix_flops_giga': prefix_flops,
        'full_flops_giga': {str(b): full[b] for b in BUDGETS},
        'tail_flops_giga': {str(b): tail[b] for b in BUDGETS},
        'shared_prefix_cumulative_exit_flops': costs_cum,
        'full_cascade_exit_dense_flops': sum(full[b] for b in BUDGETS),
        'val_accuracy_table': {k: {str(b): v[b] for b in BUDGETS} for k, v in table.items()},
        'baselines_val': {'static_25': 0.7304, 'static_50': 0.7738,
                          'static_75': 0.7884, 'dense': 0.8074, 'controller': 0.787},
    }

    # ---------- Phase 2: confidence signal + widening curve for top candidates ----------
    grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    cand = sorted(table, key=lambda nm: np.mean(list(table[nm].values())), reverse=True)[:2]
    report['candidates'] = {}
    for name in cand:
        c25, k25 = caches[name][0.25]
        auc = roc_auc_score(k25, c25) if len(set(k25)) > 1 else float('nan')
        report['candidates'][name] = {
            'auroc_conf25_predicts_correct25': round(float(auc), 4),
            'ece_at_25': round(ece(c25, k25.astype(float)), 4),
            'brier_at_25': round(brier_score_loss(k25.astype(int), c25), 4),
            'acc25': round(float(k25.mean()), 4),
            'progressive_widening_curve_val':
                progressive_curve(caches[name], costs_cum, grid),
        }
        print(f'\n=== Phase 2: shared checkpoint = {name} ===')
        print(f'  AUROC(conf25 -> correct25) = {report["candidates"][name]["auroc_conf25_predicts_correct25"]}'
              f'   ECE={report["candidates"][name]["ece_at_25"]}  acc25={report["candidates"][name]["acc25"]}')
        print('  progressive widening (val, honest shared-prefix cumulative FLOPs):')
        print(f'    baselines: static_50=0.7738@0.6875  static_75=0.7884@0.8834  dense=0.8074@1.0794')
        for p in report['candidates'][name]['progressive_widening_curve_val']:
            er = p['exit_rates']
            print(f"    t={p['threshold']:>5}: acc={p['accuracy']:.4f} flops={p['avg_flops_giga']:.4f} "
                  f"exit25={er['25']:.2f} dense={er['dense']:.2f}")

    os.makedirs('results', exist_ok=True)
    json.dump(report, open('results/shared_prefix_report.json', 'w'), indent=2)
    print('\nReport saved -> results/shared_prefix_report.json')


if __name__ == '__main__':
    main()
