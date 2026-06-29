"""3-seed confirmation + latency/throughput for the multi-budget ViT (READ-ONLY eval).

Loads the trained multibudget checkpoints (seeds 42/123/7), computes fixed-budget
val/test accuracy at 0.25/0.50/0.75/1.00 (no widening), aggregates mean±std and the
key head-to-head gaps, runs the load-bearing @25 stability check, and measures
wall-clock throughput/latency at each budget vs the specialist static models.

Test set used once per seed for fixed-budget numbers only; all selection already done.
True FLOPs via fvcore. No exit-only accounting. No progressive widening.
"""

import json
import os
import statistics as st
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.cifar import get_dataloaders
from src.models.vit_multibudget import VitMultiBudget
from src.models.vit_static import VitStaticPruning
from src.models.vit_dense import VitDense
from src.utils.flops import compute_flops

DATA_DIR = '/home/arooba/compute-aware-vit-thesis/data/'
BUDGETS = [0.25, 0.50, 0.75, 1.00]
BKEY = {0.25: '25', 0.50: '50', 0.75: '75', 1.00: 'dense'}
SEED_DIRS = {
    42:  'checkpoints/multibudget_clean_split',
    123: 'checkpoints/multibudget_clean_split_seed123',
    7:   'checkpoints/multibudget_clean_split_seed7',
}
SPECIALIST = {
    '25':    ('checkpoints/static_25_clean_split/best_model.pt', 0.25, 0.7283),
    '50':    ('checkpoints/static_50_clean_split/best_model.pt', 0.50, 0.7686),
    '75':    ('checkpoints/static_75_clean_split/best_model.pt', 0.75, 0.7891),
    'dense': ('checkpoints/dense_clean_split/best_model.pt',     1.00, 0.7950),
}
CASCADE_BEST_TEST = 0.7973  # cascade best_under_static75 test acc (single point)


@torch.no_grad()
def acc_at_budgets(model, loader, device):
    """Fixed-budget accuracy at each budget over a loader (shared prefix per batch)."""
    model.eval()
    correct = {b: 0 for b in BUDGETS}
    total = 0
    for images, lbl in loader:
        images = images.to(device)
        prefix = model.forward_prefix(images)
        for b in BUDGETS:
            logits = model.forward_tail(prefix, b)
            correct[b] += (logits.argmax(-1).cpu().numpy() == lbl.numpy()).sum()
        total += len(lbl)
    return {b: round(float(correct[b] / total), 4) for b in BUDGETS}


@torch.no_grad()
def measure_latency(forward_fn, device, batch_size=128, warmup=10, runs=30):
    """Median wall-clock per-batch time -> images/sec and per-image latency (ms)."""
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(warmup):
        forward_fn(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        forward_fn(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = st.median(times)
    return {'images_per_sec': round(batch_size / med, 1),
            'latency_ms_per_image': round(med / batch_size * 1000, 4)}


def load_multibudget(path, device):
    m = VitMultiBudget(num_classes=100, pretrained=False, prune_layer=3)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    print(f'Device: {device} ({dev_name})')
    _, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR, batch_size=64, num_workers=4, val_size=5000, split_seed=42)

    # FLOPs per budget (architecture-determined, same for all checkpoints)
    ref = load_multibudget(os.path.join(SEED_DIRS[42], 'best_model.pt'), device)
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    flops = {}
    for b in BUDGETS:
        flops[BKEY[b]] = round(compute_flops(
            type('FB', (torch.nn.Module,), {
                'forward': (lambda s, x, bb=b: ref.forward_budget(x, bb))})(), dummy), 4)

    # ---------- per-seed fixed-budget accuracy ----------
    per_seed = {}
    for seed, d in SEED_DIRS.items():
        ckpt = os.path.join(d, 'best_model.pt')
        if not os.path.exists(ckpt):
            print(f'  !! seed {seed}: checkpoint missing at {ckpt} — skipping')
            continue
        m = load_multibudget(ckpt, device)
        be = None
        mp = os.path.join(d, 'metrics.json')
        if os.path.exists(mp):
            be = json.load(open(mp)).get('best_epoch')
        va = acc_at_budgets(m, val_loader, device)
        te = acc_at_budgets(m, test_loader, device)
        per_seed[seed] = {'best_epoch': be,
                          'val':  {BKEY[b]: va[b] for b in BUDGETS},
                          'test': {BKEY[b]: te[b] for b in BUDGETS}}
        print(f'\nseed {seed} (best_epoch {be}):')
        print('  val :', per_seed[seed]['val'])
        print('  test:', per_seed[seed]['test'])

    seeds = sorted(per_seed)
    def col(split, k):
        return [per_seed[s][split][k] for s in seeds]

    def ms(vals):
        return (round(st.mean(vals), 4),
                round(st.pstdev(vals) if len(vals) > 1 else 0.0, 4))

    # ---------- aggregate test mean±std per budget ----------
    agg = {}
    for b in BUDGETS:
        k = BKEY[b]
        mean, std = ms(col('test', k))
        agg[k] = {'mean': mean, 'std': std, 'per_seed': col('test', k),
                  'flops_giga': flops[k]}

    # ---------- head-to-head gaps (mean±std) ----------
    gap_50_vs_static75 = ms([per_seed[s]['test']['50'] - SPECIALIST['75'][2] for s in seeds])
    gap_75_vs_cascade  = ms([per_seed[s]['test']['75'] - CASCADE_BEST_TEST for s in seeds])
    gap_dense_vs_dense = ms([per_seed[s]['test']['dense'] - SPECIALIST['dense'][2] for s in seeds])

    # ---------- load-bearing @25 check ----------
    at25 = col('test', '25')
    at25_ok = all(v > SPECIALIST['25'][2] for v in at25)

    # ---------- latency / throughput ----------
    print('\nMeasuring latency/throughput (batch=128, fp32, warmup=10, median of 30)…')
    lat = {'multibudget': {}, 'specialist': {}}
    for b in BUDGETS:
        lat['multibudget'][BKEY[b]] = measure_latency(
            lambda x, bb=b: ref.forward_budget(x, bb), device)
    for k, (path, kr, _) in SPECIALIST.items():
        if k == 'dense':
            sm = VitDense(num_classes=100, pretrained=False)
            sm.load_state_dict(torch.load(path, map_location=device))
            sm = sm.to(device).eval()
            fn = lambda x: sm(x)
        else:
            sm = VitStaticPruning(keep_ratio=kr, prune_layer=3)
            sm.load_state_dict(torch.load(path, map_location=device))
            sm = sm.to(device).eval()
            fn = lambda x, _sm=sm: _sm(x)
        with torch.no_grad():
            lat['specialist'][k] = measure_latency(fn, device)

    report = {
        'hardware': dev_name,
        'timing': {'batch_size': 128, 'precision': 'fp32', 'warmup': 10,
                   'runs': 30, 'metric': 'median per-batch, cuda-synchronized',
                   'note': 'Winning result is FIXED-budget: every image runs the '
                           'identical path, so it batches normally — no dynamic-'
                           'batching penalty (unlike the abandoned widening path).'},
        'split': {'split_seed': 42, 'train': 45000, 'val': 5000, 'test': 10000},
        'flops_giga_per_budget': flops,
        'per_seed': per_seed,
        'seeds_used': seeds,
        'aggregate_test_mean_std': agg,
        'head_to_head_gaps': {
            'multibudget@50_minus_static75': {'mean': gap_50_vs_static75[0], 'std': gap_50_vs_static75[1],
                'flops_multibudget@50': flops['50'], 'flops_static75': SPECIALIST['75'] and 0.8834},
            'multibudget@75_minus_cascade_best': {'mean': gap_75_vs_cascade[0], 'std': gap_75_vs_cascade[1]},
            'multibudget@dense_minus_dense': {'mean': gap_dense_vs_dense[0], 'std': gap_dense_vs_dense[1]},
        },
        'load_bearing_at25': {'per_seed_test_at25': at25, 'static_25_specialist': SPECIALIST['25'][2],
                              'all_seeds_above_specialist': bool(at25_ok)},
        'latency': lat,
    }
    os.makedirs('results', exist_ok=True)
    json.dump(report, open('results/multibudget_seed_confirmation.json', 'w'), indent=2)

    # ---------- console summary ----------
    print('\n================ AGGREGATE (test, mean±std over', len(seeds), 'seeds) ================')
    print(f"{'budget':8s} {'GFLOPs':>8} {'mean':>8} {'std':>7}   per-seed")
    for b in BUDGETS:
        k = BKEY[b]
        a = agg[k]
        print(f"{k:8s} {a['flops_giga']:>8} {a['mean']:>8} {a['std']:>7}   {a['per_seed']}")
    print('\nHead-to-head gaps (test, mean±std):')
    print(f"  multibudget@50 − static_75 (0.7891): {gap_50_vs_static75[0]:+.4f} ± {gap_50_vs_static75[1]:.4f}"
          f"   [FLOPs 0.687 vs 0.883]")
    print(f"  multibudget@75 − cascade_best (0.7973): {gap_75_vs_cascade[0]:+.4f} ± {gap_75_vs_cascade[1]:.4f}")
    print(f"  multibudget@dense − dense (0.795): {gap_dense_vs_dense[0]:+.4f} ± {gap_dense_vs_dense[1]:.4f}")
    print(f"\nLoad-bearing @25 vs static_25 (0.7283): per-seed {at25}  ->  all above? {at25_ok}")
    print('\nLatency (images/sec | ms/img):')
    for b in BUDGETS:
        k = BKEY[b]
        mb = lat['multibudget'][k]; sp = lat['specialist'][k]
        print(f"  {k:6s} multibudget {mb['images_per_sec']:>7}/{mb['latency_ms_per_image']:.3f}ms | "
              f"specialist {sp['images_per_sec']:>7}/{sp['latency_ms_per_image']:.3f}ms")
    print('\nReport saved -> results/multibudget_seed_confirmation.json')


if __name__ == '__main__':
    main()
