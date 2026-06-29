"""Early-signal separability diagnostic (READ-ONLY analysis, no deployed model).

Question: can CHEAP EARLY information (raw image stats + DeiT-Tiny blocks 1-3
features) predict the binary target "static_25 will be WRONG" (= needs upgrade)?

Rules honored:
- No new adaptive model is built or deployed.
- No final budget-controller MLP; the logistic regression here is a *separability
  measurement* only (5-fold cross-validated, out-of-fold scores), on EARLY
  (layer <= 3) features — explicitly NOT the failed layer-6 controller.
- No Gumbel, no cascade.
- Probe/curve use VALIDATION only (5-fold OOF). Test used once for reporting
  per-model correctness and the oracle-headroom decomposition.

It also decomposes oracle headroom into:
  (1) budget-routing headroom (FLOPs savings at <= dense accuracy), and
  (2) model-complementarity / ensemble headroom (accuracy above dense that a
      single backbone cannot capture).
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.cifar import get_dataloaders
from src.models.vit_dense import VitDense
from src.models.vit_static import VitStaticPruning
from src.utils.flops import compute_flops

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, brier_score_loss)

DATA_DIR = '/home/arooba/compute-aware-vit-thesis/data/'
CKPT = {
    '25':    'checkpoints/static_25_clean_split/best_model.pt',
    '50':    'checkpoints/static_50_clean_split/best_model.pt',
    '75':    'checkpoints/static_75_clean_split/best_model.pt',
    'dense': 'checkpoints/dense_clean_split/best_model.pt',
}
BUDGETS = ['25', '50', '75', 'dense']


def build_models(device):
    models = {
        '25':    VitStaticPruning(keep_ratio=0.25, prune_layer=3),
        '50':    VitStaticPruning(keep_ratio=0.50, prune_layer=3),
        '75':    VitStaticPruning(keep_ratio=0.75, prune_layer=3),
        'dense': VitDense(),
    }
    for k, m in models.items():
        m.load_state_dict(torch.load(CKPT[k], map_location=device))
        m.to(device).eval()
    return models


def _raw_features(images):
    """Cheap raw-image features (B, k) from a normalized image batch."""
    g = images.mean(dim=1)                                  # (B,224,224) grayscale
    B = g.shape[0]
    p = g.unfold(1, 16, 16).unfold(2, 16, 16)               # (B,14,14,16,16)
    p = p.reshape(B, 14 * 14, -1)                           # (B,196,256)
    patch_means = p.mean(-1)                                # (B,196)
    patch_contrast = patch_means.var(dim=1)                 # variance of patch means
    patch_texture = p.var(dim=-1).mean(dim=1)               # mean within-patch var
    dx = (g[:, :, 1:] - g[:, :, :-1]).abs().mean(dim=(1, 2))
    dy = (g[:, 1:, :] - g[:, :-1, :]).abs().mean(dim=(1, 2))
    edge_density = dx + dy
    return torch.stack([patch_contrast, patch_texture, edge_density], dim=1)


def _layer_features(x):
    """Per-layer token features from a sequence x (B, N, C)."""
    cls = x[:, 0]
    patches = x[:, 1:]
    cls_norm = cls.norm(dim=-1)
    pn = patches.norm(dim=-1)                               # (B, N-1)
    mean_pn = pn.mean(dim=1)
    std_pn = pn.std(dim=1)
    prob = pn / (pn.sum(dim=1, keepdim=True) + 1e-9)
    sal_entropy = -(prob * (prob + 1e-12).log()).sum(dim=1)  # low = concentrated
    return cls, torch.stack([cls_norm, mean_pn, std_pn, sal_entropy], dim=1)


@torch.no_grad()
def cache_all(models, loader, device):
    """Return per-model correctness, dense confidence/loss, and early features."""
    feats_hook = {}
    h1 = models['dense'].model.blocks[0].register_forward_hook(
        lambda m, i, o: feats_hook.__setitem__(1, o.detach()))
    h2 = models['dense'].model.blocks[1].register_forward_hook(
        lambda m, i, o: feats_hook.__setitem__(2, o.detach()))
    h3 = models['dense'].model.blocks[2].register_forward_hook(
        lambda m, i, o: feats_hook.__setitem__(3, o.detach()))

    correct = {b: [] for b in BUDGETS}
    labels, dense_conf, dense_loss, feat_rows = [], [], [], []
    for images, lbl in loader:
        images = images.to(device)
        lbl_np = lbl.numpy()
        labels.append(lbl_np)
        # dense forward (also fires hooks for early features)
        dlogits = models['dense'](images)
        dprob = F.softmax(dlogits, dim=-1)
        dense_conf.append(dprob.max(-1).values.cpu().numpy())
        dense_loss.append(F.cross_entropy(dlogits, lbl.to(device),
                                          reduction='none').cpu().numpy())
        correct['dense'].append(dlogits.argmax(-1).cpu().numpy() == lbl_np)
        for b in ['25', '50', '75']:
            correct[b].append(models[b](images).argmax(-1).cpu().numpy() == lbl_np)
        # features
        raw = _raw_features(images)
        cls1, l1 = _layer_features(feats_hook[1])
        cls2, l2 = _layer_features(feats_hook[2])
        cls3, l3 = _layer_features(feats_hook[3])
        cos13 = F.cosine_similarity(cls1, cls3, dim=-1).unsqueeze(1)
        cos23 = F.cosine_similarity(cls2, cls3, dim=-1).unsqueeze(1)
        drift13 = (cls3 - cls1).norm(dim=-1).unsqueeze(1)
        row = torch.cat([raw, l1, l2, l3, cos13, cos23, drift13], dim=1)
        feat_rows.append(row.cpu().numpy())

    h1.remove(); h2.remove(); h3.remove()
    out = {b: np.concatenate(correct[b]) for b in BUDGETS}
    return (out, np.concatenate(labels), np.concatenate(dense_conf),
            np.concatenate(dense_loss), np.concatenate(feat_rows))


FEATURE_NAMES = [
    'raw_patch_contrast', 'raw_patch_texture', 'raw_edge_density',
    'L1_cls_norm', 'L1_patch_norm_mean', 'L1_patch_norm_std', 'L1_sal_entropy',
    'L2_cls_norm', 'L2_patch_norm_mean', 'L2_patch_norm_std', 'L2_sal_entropy',
    'L3_cls_norm', 'L3_patch_norm_mean', 'L3_patch_norm_std', 'L3_sal_entropy',
    'cls_cos_1_3', 'cls_cos_2_3', 'cls_drift_1_3',
]


def headroom_decomposition(correct, costs):
    """Split oracle headroom into budget-routing vs ensemble complementarity."""
    c25, c50, c75, cd = (correct['25'], correct['50'], correct['75'], correct['dense'])
    n = len(cd)
    coverage = float((c25 | c50 | c75 | cd).mean())
    dense_acc = float(cd.mean())

    # Routing-only oracle (single-backbone-fair): never exceed dense accuracy.
    #  - dense-correct images: route to smallest correct budget (genuine easy).
    #  - dense-wrong images: route to dense (max effort), accept the error.
    cost = np.full(n, costs['dense'])
    for b in ['25', '50', '75']:  # cheapest first, only where dense also correct
        take = cd & correct[b] & (cost == costs['dense'])
        cost[take] = costs[b]
    routing_only_acc = dense_acc
    routing_only_flops = float(cost.mean())

    return {
        'coverage_any_correct': coverage,
        'dense_accuracy': dense_acc,
        'ensemble_headroom_acc': coverage - dense_acc,   # unreachable by 1 backbone
        'budget_routing_only_oracle': {
            'accuracy': routing_only_acc,               # == dense level
            'avg_flops': routing_only_flops,            # FLOPs saved vs dense
            'flops_saving_vs_dense': costs['dense'] - routing_only_flops,
        },
        'frac_25correct_but_dense_wrong': float((c25 & ~cd).mean()),  # ensemble luck
        'frac_25wrong_but_bigger_fixes': float((~c25 & (c50 | c75 | cd)).mean()),
        'frac_wrong_everywhere': float((~c25 & ~c50 & ~c75 & ~cd).mean()),
    }


def single_feature_auroc(feats, target):
    """AUROC of each single feature vs target (direction-agnostic)."""
    out = {}
    for j, name in enumerate(FEATURE_NAMES):
        try:
            auc = roc_auc_score(target, feats[:, j])
        except ValueError:
            auc = float('nan')
        out[name] = round(max(auc, 1 - auc), 4)  # report separability magnitude
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def probe_separability(feats, target, seed=42):
    """5-fold cross-validated logistic probe (MEASUREMENT only). Returns metrics + OOF scores."""
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight='balanced'))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(clf, feats, target, cv=skf, method='predict_proba')[:, 1]
    pred = (oof >= 0.5).astype(int)
    metrics = {
        'auroc':             round(roc_auc_score(target, oof), 4),
        'balanced_accuracy': round(balanced_accuracy_score(target, pred), 4),
        'macro_f1':          round(f1_score(target, pred, average='macro'), 4),
        'precision_needs_upgrade': round(precision_score(target, pred, zero_division=0), 4),
        'recall_needs_upgrade':    round(recall_score(target, pred, zero_division=0), 4),
        'brier':             round(brier_score_loss(target, oof), 4),
        'base_rate':         round(float(target.mean()), 4),
    }
    return metrics, oof


def routing_curve(oof_scores, correct, costs, hard_budget='75'):
    """Simulated 2-budget routing using OOF probe scores (no leakage).

    predicted-easy -> static_25, predicted-hard -> static_<hard_budget>.
    """
    c25 = correct['25']
    chard = correct[hard_budget]
    n = len(c25)
    pts = []
    for t in [1.01, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -0.01]:
        hard = oof_scores >= t
        acc = float(np.where(hard, chard, c25).mean())
        flops = float(np.where(hard, costs[hard_budget], costs['25']).mean())
        pts.append({'threshold': t, 'frac_hard': round(float(hard.mean()), 4),
                    'accuracy': round(acc, 5), 'avg_flops': round(flops, 5)})
    return pts


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    _, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR, batch_size=64, num_workers=4, val_size=5000, split_seed=42)
    models = build_models(device)
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    costs = {b: compute_flops(models[b], dummy) for b in BUDGETS}
    print('Single-model GFLOPs:', {k: round(v, 4) for k, v in costs.items()})

    print('Caching validation (correctness + early features)…')
    vcorr, vlab, vconf, vloss, vfeat = cache_all(models, val_loader, device)
    print('Caching test (correctness only used for headroom decomposition)…')
    tcorr, tlab, tconf, tloss, tfeat = cache_all(models, test_loader, device)

    report = {'note': 'Early-signal separability diagnostic. Probe = 5-fold CV '
                      'logistic measurement on layer<=3 features (NOT a deployed '
                      'model, NOT layer-6). Routing curve from out-of-fold scores '
                      'on validation. Decision happens at layer 3 (during the '
                      'prefix) so routing FLOPs are single-pass and honest.',
              'single_model_flops_giga': costs}

    # ---- headroom decomposition (val + test) ----
    report['headroom_decomposition'] = {
        'val':  headroom_decomposition(vcorr, costs),
        'test': headroom_decomposition(tcorr, costs),
    }

    # ---- targets (validation) ----
    t_static25_wrong = (~vcorr['25']).astype(int)                       # T1
    t_fixable_upgrade = ((~vcorr['25']) & (vcorr['50'] | vcorr['75'] | vcorr['dense'])).astype(int)
    # T4: hard by dense confidence (bottom-quartile confidence = hard)
    q = np.quantile(vconf, 0.25)
    t_hard_conf = (vconf <= q).astype(int)

    # ---- single-feature AUROC vs T1 ----
    report['single_feature_auroc_static25_wrong'] = single_feature_auroc(vfeat, t_static25_wrong)

    # ---- probe separability for each target ----
    probe = {}
    for name, tgt in [('static25_wrong', t_static25_wrong),
                      ('fixable_by_upgrade', t_fixable_upgrade),
                      ('hard_by_dense_confidence', t_hard_conf)]:
        m, oof = probe_separability(vfeat, tgt)
        probe[name] = m
        if name == 'static25_wrong':
            oof_main = oof
    report['probe_separability_val'] = probe

    # ---- estimated achievable routing curves (val, OOF, no leakage) ----
    report['routing_curve_25_to_75_val'] = routing_curve(oof_main, vcorr, costs, '75')
    report['routing_curve_25_to_dense_val'] = routing_curve(oof_main, vcorr, costs, 'dense')

    # ---- baselines (val) for comparison ----
    report['baselines_val'] = {b: round(float(vcorr[b].mean()), 4) for b in BUDGETS}

    os.makedirs('results', exist_ok=True)
    json.dump(report, open('results/early_signal_report.json', 'w'), indent=2)

    # ---------- console summary ----------
    hd = report['headroom_decomposition']['test']
    print('\n===== HEADROOM DECOMPOSITION (test) =====')
    print(f"coverage(any correct)      = {hd['coverage_any_correct']:.4f}")
    print(f"dense accuracy             = {hd['dense_accuracy']:.4f}")
    print(f"ENSEMBLE headroom (>dense) = {hd['ensemble_headroom_acc']:.4f}  "
          f"(NOT reachable by a single backbone)")
    bro = hd['budget_routing_only_oracle']
    print(f"budget-routing-only oracle = {bro['accuracy']:.4f} acc @ {bro['avg_flops']:.4f} "
          f"GFLOPs (saves {bro['flops_saving_vs_dense']:.4f} vs dense, same accuracy)")
    print(f"  25-correct-but-dense-wrong (ensemble luck) = {hd['frac_25correct_but_dense_wrong']:.4f}")
    print(f"  25-wrong-but-upgrade-fixes  (real routing) = {hd['frac_25wrong_but_bigger_fixes']:.4f}")
    print(f"  wrong-everywhere                            = {hd['frac_wrong_everywhere']:.4f}")

    print('\n===== TOP SINGLE-FEATURE AUROC vs (static_25 wrong) =====')
    for k, v in list(report['single_feature_auroc_static25_wrong'].items())[:6]:
        print(f'  {k:24s} {v}')

    print('\n===== PROBE (5-fold CV, layer<=3 features) =====')
    for name, m in probe.items():
        print(f"  {name:26s} AUROC={m['auroc']}  bal_acc={m['balanced_accuracy']}  "
              f"F1={m['macro_f1']}  P/R(upgrade)={m['precision_needs_upgrade']}/{m['recall_needs_upgrade']}  base={m['base_rate']}")

    print('\n===== ROUTING CURVE 25->75 (val, OOF) =====')
    print(f"  baselines val: 25={report['baselines_val']['25']} 50={report['baselines_val']['50']} "
          f"75={report['baselines_val']['75']} dense={report['baselines_val']['dense']}")
    for p in report['routing_curve_25_to_75_val']:
        print(f"  hard%={p['frac_hard']:.3f}  acc={p['accuracy']:.4f}  flops={p['avg_flops']:.4f}")

    print('\nReport saved -> results/early_signal_report.json')


if __name__ == '__main__':
    main()
