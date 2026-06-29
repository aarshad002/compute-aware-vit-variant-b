"""Main training entry point. Config-driven; supports all model types."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from src.datasets.cifar import get_dataloaders
from src.models.vit_dense import VitDense
from src.models.vit_static import VitStaticPruning
from src.models.vit_dynamic import VitDynamic
from src.models.controller import VitController
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.training import cascade_eval as ce
from src.training.cascade_eval import stage_entropy_margin
from src.utils.flops import compute_flops
from src.utils.metrics import save_metrics


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file.

    Args:
        config_path: Path to the .yaml config.

    Returns:
        Config dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict[str, Any]) -> Optional[nn.Module]:
    """Construct the model specified in cfg['model_type'].

    Args:
        cfg: Config dictionary.

    Returns:
        Initialised nn.Module, or None for the 'cascade' type.
    """
    model_type  = cfg['model_type']
    num_classes = cfg.get('num_classes', 100)
    pretrained  = cfg.get('pretrained', True)

    if model_type == 'dense':
        return VitDense(num_classes=num_classes, pretrained=pretrained)

    if model_type == 'static':
        return VitStaticPruning(
            keep_ratio=cfg['keep_ratio'],
            prune_layer=cfg.get('prune_layer', 3),
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if model_type == 'dynamic':
        return VitDynamic(
            keep_ratio=cfg['keep_ratio'],
            prune_layer=cfg.get('prune_layer', 6),
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if model_type == 'controller':
        return VitController(num_classes=num_classes, pretrained=pretrained)

    if model_type == 'cascade':
        return None

    raise ValueError(f"Unknown model_type: {model_type!r}")


def _cache_stage_probs(
    models: Dict[str, nn.Module], loader: Any, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Pre-compute per-stage confidences/predictions and labels for a loader.

    Args:
        models: Mapping {'25','50','75','dense'} → stage model (already on device).
        loader: DataLoader to cache over.
        device: Compute device.

    Returns:
        Dict with 'labels' and, per stage, '<stage>_conf', '<stage>_pred',
        '<stage>_entropy' and '<stage>_margin'.
    """
    labels: List[torch.Tensor] = []
    probs = {k: [] for k in models}
    with torch.no_grad():
        for images, lbl in loader:
            images = images.to(device)
            labels.append(lbl)
            for k, m in models.items():
                probs[k].append(F.softmax(m(images), dim=-1).cpu())

    cache: Dict[str, torch.Tensor] = {'labels': torch.cat(labels)}
    for k in models:
        p = torch.cat(probs[k])
        entropy, margin = stage_entropy_margin(p)
        cache[f'{k}_conf']    = p.max(dim=-1).values
        cache[f'{k}_pred']    = p.argmax(dim=-1)
        cache[f'{k}_entropy'] = entropy
        cache[f'{k}_margin']  = margin
    return cache


def _load_baseline(name: str) -> Optional[Dict[str, float]]:
    """Load a clean-split baseline's val/test accuracy and FLOPs from metrics.json.

    Args:
        name: Baseline directory name (e.g. 'dense', 'static_75', 'controller').

    Returns:
        Dict with 'val_acc', 'test_acc', 'flops', or None if unavailable.
    """
    path = os.path.join('checkpoints', f'{name}_clean_split', 'metrics.json')
    if not os.path.exists(path):
        return None
    m = json.load(open(path))
    flops = m.get('flops_giga', m.get('avg_flops_giga'))
    return {
        'val_acc':  m.get('best_val_acc'),
        'test_acc': m.get('final_test_acc'),
        'flops':    flops,
    }


def run_cascade(cfg: Dict[str, Any], device: torch.device) -> None:
    """Systematic cascade study: threshold grid + exit analysis + learned gate.

    All selection (thresholds and gate settings) is performed on validation; the
    official test split is evaluated only for the chosen operating points. FLOPs
    are cumulative along the cascade path. Results, CSVs and a Pareto plot are
    written under ``output_dir``.

    Args:
        cfg: Config dictionary (checkpoint paths, data_dir, split settings).
        device: Torch device to run on.
    """
    print("=== Cascade study (val-selected, test-evaluated) ===")

    _, val_loader, test_loader = get_dataloaders(
        data_dir=cfg['data_dir'],
        batch_size=cfg['batch_size'],
        seed=cfg.get('seed', 42),
        val_size=cfg.get('val_size', 5000),
        split_seed=cfg.get('split_seed', 42),
    )

    prune_layer = cfg.get('prune_layer', 3)
    models: Dict[str, nn.Module] = {
        '25':    VitStaticPruning(keep_ratio=0.25, prune_layer=prune_layer),
        '50':    VitStaticPruning(keep_ratio=0.50, prune_layer=prune_layer),
        '75':    VitStaticPruning(keep_ratio=0.75, prune_layer=prune_layer),
        'dense': VitDense(),
    }

    def _load(model: nn.Module, path: str) -> None:
        if path and os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"  loaded {path}")
        else:
            print(f"  WARNING: checkpoint not found at {path!r} — using random weights")

    _load(models['25'],    cfg.get('ckpt_25', ''))
    _load(models['50'],    cfg.get('ckpt_50', ''))
    _load(models['75'],    cfg.get('ckpt_75', ''))
    _load(models['dense'], cfg.get('ckpt_dense', ''))
    for m in models.values():
        m.to(device).eval()

    # Stage FLOPs (deterministic for fixed keep_ratio).
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    stage_flops = {k: compute_flops(m, dummy) for k, m in models.items()}
    print(f"Stage FLOPs (GFLOPs): {stage_flops}")

    print("Pre-caching stage statistics (val + test)…")
    val_cache  = _cache_stage_probs(models, val_loader,  device)
    test_cache = _cache_stage_probs(models, test_loader, device)
    val_n  = int(val_cache['labels'].shape[0])
    test_n = int(test_cache['labels'].shape[0])

    # Clean-split baselines for comparison (same split, honest numbers).
    baseline_names = ['dense', 'static_25', 'static_50', 'static_75', 'controller']
    raw_baselines = {n: _load_baseline(n) for n in baseline_names}
    raw_baselines = {n: b for n, b in raw_baselines.items() if b}
    val_baselines  = {n: {'acc': b['val_acc'],  'flops': b['flops']}
                      for n, b in raw_baselines.items() if b['val_acc'] is not None}
    test_baselines = {n: {'acc': b['test_acc'], 'flops': b['flops']}
                      for n, b in raw_baselines.items() if b['test_acc'] is not None}
    flops_per_model = {n: b['flops'] for n, b in raw_baselines.items()}
    dense_flops = flops_per_model.get('dense', stage_flops['dense'])
    f75 = flops_per_model.get('static_75', stage_flops['75'])
    f50 = flops_per_model.get('static_50', stage_flops['50'])

    # ---- Part B: systematic threshold grid on validation ----
    grid = ce.THRESHOLD_GRID
    print(f"Sweeping {len(grid)**3} threshold combinations on validation…")
    val_results = ce.sweep_threshold_grid(val_cache, stage_flops, grid)
    frontier = ce.mark_pareto(val_results)
    for r in val_results:
        ce.add_baseline_flags(r, val_baselines)
    val_results.sort(key=lambda r: r['accuracy'], reverse=True)

    # ---- Part B: select operating points on validation, evaluate once on test ----
    selected = ce.select_operating_points(val_results, dense_flops, f75, f50)
    final_test: Dict[str, Any] = {}
    for name, rec in selected.items():
        if rec is None:
            final_test[name] = None
            continue
        tr = ce.eval_threshold_triple(
            rec['threshold_25'], rec['threshold_50'], rec['threshold_75'],
            test_cache, stage_flops,
        )
        ce.add_baseline_flags(tr, test_baselines)
        final_test[name] = {'val': rec, 'test': tr}
        print(f"  [{name}] val_acc={rec['accuracy']:.4f}@{rec['avg_flops_giga']:.4f}  "
              f"-> test_acc={tr['accuracy']:.4f}@{tr['avg_flops_giga']:.4f}")

    # ---- Part D: learned exit gate ----
    gate_results = _run_learned_gate(
        val_cache, test_cache, stage_flops, f75, cfg.get('split_seed', 42)
    )

    # ---- Part F: artifacts ----
    output_dir = cfg.get('output_dir', 'checkpoints/cascade_clean_split')
    os.makedirs(output_dir, exist_ok=True)
    results_dir = cfg.get('results_dir', 'results')
    ce.write_threshold_csv(
        os.path.join(results_dir, 'cascade_clean_split_threshold_results.csv'), val_results)
    ce.write_pareto_csv(
        os.path.join(results_dir, 'cascade_clean_split_pareto.csv'), frontier)
    plotted = ce.maybe_plot(
        os.path.join(results_dir, 'cascade_clean_split_pareto.png'),
        val_results, frontier, val_baselines)

    # ---- Part E: assemble metrics ----
    metrics: Dict[str, Any] = {
        'model_name':        'cascade_clean_split',
        'notes': [
            'Threshold and gate selection done on VALIDATION only.',
            'TEST evaluated only for the selected operating points.',
            'FLOPs are cumulative cascade FLOPs (25->50->75->dense).',
            'Stages 25/50/75 are static (L2-norm, prune-layer 3) specialists; '
            'dense is the full model.',
            'Old checkpoints/cascade/metrics.json is the pre-clean-split, '
            'test-tuned (biased) result and is kept untouched.',
        ],
        'split_seed':        cfg.get('split_seed', 42),
        'train_size':        50000 - val_n,
        'val_size':          val_n,
        'test_size':         test_n,
        'flops_giga_per_model': {
            'dense':      flops_per_model.get('dense'),
            'static_25':  flops_per_model.get('static_25'),
            'static_50':  flops_per_model.get('static_50'),
            'static_75':  flops_per_model.get('static_75'),
            'controller': flops_per_model.get('controller'),
        },
        'stage_flops_giga':  stage_flops,
        'path_flops_giga':   ce.path_flops_giga(stage_flops),
        'threshold_grid':    grid,
        'total_combinations': len(val_results),
        'baselines_val':     val_baselines,
        'baselines_test':    test_baselines,
        'pareto_frontier_val': frontier,
        'selected_threshold_points': {
            k: (v['val'] if v else None) for k, v in final_test.items()},
        'final_test_results_for_selected_points': {
            k: (v['test'] if v else None) for k, v in final_test.items()},
        'exit_rate_analysis_note':
            'Per-combination exit rates, per-exit accuracy and per-path FLOPs are '
            'included in every entry of all_threshold_results_val.',
        'learned_gate_results': gate_results,
        'all_threshold_results_val': val_results,
    }
    out_path = os.path.join(output_dir, 'metrics.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    _write_cascade_summary_md(
        os.path.join(results_dir, 'cascade_clean_split_summary.md'),
        metrics, final_test, gate_results)

    hi = final_test.get('highest_val_acc')
    if hi:
        print(f"Highest-val-acc point  -> test acc={hi['test']['accuracy']:.4f} "
              f"@ {hi['test']['avg_flops_giga']:.4f} GFLOPs")
    if gate_results and gate_results.get('selected'):
        gt = gate_results['selected']['highest_val_acc']['test']
        print(f"Learned gate (best-val) -> test acc={gt['accuracy']:.4f} "
              f"@ {gt['avg_flops_giga']:.4f} GFLOPs")
    print(f"Plot written: {plotted}.  Metrics saved → {out_path}")


def _run_learned_gate(
    val_cache: Dict[str, Any],
    test_cache: Dict[str, Any],
    stage_flops: Dict[str, float],
    static75_flops: float,
    split_seed: int,
) -> Optional[Dict[str, Any]]:
    """Fit and evaluate a logistic-regression exit gate (Part D).

    The validation cache is split (deterministically) into a gate-fit portion
    (train the gate) and a gate-select portion (choose the gate threshold). The
    selected gate operating points are evaluated once on the test cache.

    Args:
        val_cache: Cached validation statistics.
        test_cache: Cached test statistics.
        stage_flops: Per-stage GFLOPs.
        static75_flops: static_75 GFLOPs ceiling for one operating point.
        split_seed: Seed for the gate-fit/select partition.

    Returns:
        Gate result dict, or None if sklearn is unavailable.
    """
    val_n = int(val_cache['labels'].shape[0])
    perm = torch.randperm(val_n, generator=torch.Generator().manual_seed(split_seed))
    n_fit = int(0.6 * val_n)
    fit_cache = ce.subcache(val_cache, perm[:n_fit])
    sel_cache = ce.subcache(val_cache, perm[n_fit:])

    x, y = ce.build_gate_training(fit_cache)
    gate = ce.fit_logistic_gate(x, y)
    if gate is None:
        print("  (sklearn unavailable — skipping learned gate)")
        return None

    sel_probs = ce.gate_stage_probs(sel_cache, gate)
    sel_results = ce.sweep_gate_thresholds(sel_cache, sel_probs, stage_flops)

    best_acc = max(sel_results, key=lambda r: r['accuracy'])
    under75 = [r for r in sel_results if r['avg_flops_giga'] <= static75_flops]
    best_u75 = max(under75, key=lambda r: r['accuracy']) if under75 else None

    test_probs = ce.gate_stage_probs(test_cache, gate)

    def _on_test(sel_rec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if sel_rec is None:
            return None
        tr = ce.eval_gate(test_cache, test_probs, sel_rec['gate_threshold'], stage_flops)
        return {'val_select': sel_rec, 'test': tr}

    return {
        'gate_type':           'logistic_regression (StandardScaler + balanced)',
        'gate_features':       ce.GATE_FEATURES,
        'gate_label':          'exit=1 if the stage prediction is correct',
        'gate_train_split':    f'validation gate-fit subset ({n_fit}/{val_n}, '
                               f'split_seed={split_seed}); threshold selected on the '
                               f'remaining {val_n - n_fit} validation samples',
        'gate_thresholds':     ce.GATE_THRESHOLDS,
        'val_select_results':  sel_results,
        'selected': {
            'highest_val_acc':           _on_test(best_acc),
            'best_under_static75_flops': _on_test(best_u75),
        },
    }


def _write_cascade_summary_md(
    path: str, metrics: Dict[str, Any],
    final_test: Dict[str, Any], gate_results: Optional[Dict[str, Any]],
) -> None:
    """Write a short Markdown summary of the cascade study (Part F).

    Args:
        path: Output Markdown path.
        metrics: Assembled metrics dict.
        final_test: Selected operating points with val+test records.
        gate_results: Learned-gate result dict (or None).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines: List[str] = []
    lines.append('# Cascade (clean split) — summary\n')
    lines.append(f"Split: train={metrics['train_size']} val={metrics['val_size']} "
                 f"test={metrics['test_size']} (seed {metrics['split_seed']})\n")
    lines.append('Thresholds/gate selected on **validation**; test used **once** '
                 'for selected points. FLOPs are cumulative.\n')

    lines.append('\n## Clean baselines (test acc @ GFLOPs)\n')
    lines.append('| model | test_acc | GFLOPs |\n|---|---|---|')
    for n, b in metrics['baselines_test'].items():
        lines.append(f"| {n} | {b['acc']} | {round(b['flops'], 4)} |")

    lines.append('\n## Selected threshold operating points\n')
    lines.append('| selection | thresholds | val_acc | val_GFLOPs | test_acc | test_GFLOPs |')
    lines.append('|---|---|---|---|---|---|')
    for name, v in final_test.items():
        if not v:
            lines.append(f"| {name} | — | — | — | — | — |")
            continue
        r, t = v['val'], v['test']
        thr = f"({r['threshold_25']},{r['threshold_50']},{r['threshold_75']})"
        lines.append(f"| {name} | {thr} | {r['accuracy']} | {round(r['avg_flops_giga'],4)} "
                     f"| {t['accuracy']} | {round(t['avg_flops_giga'],4)} |")

    if gate_results and gate_results.get('selected'):
        lines.append('\n## Learned exit gate (test of val-selected settings)\n')
        lines.append('| gate selection | gate_thresh | test_acc | test_GFLOPs |')
        lines.append('|---|---|---|---|')
        for name, v in gate_results['selected'].items():
            if not v:
                lines.append(f"| {name} | — | — | — |")
                continue
            t = v['test']
            lines.append(f"| {name} | {v['val_select']['gate_threshold']} | "
                         f"{t['accuracy']} | {round(t['avg_flops_giga'],4)} |")
    else:
        lines.append('\n## Learned exit gate\n\nNot available (sklearn missing).')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# Canonical per-budget FLOPs for the controller's keep ratios, taken from the
# clean-split static specialists (25/50/75% token budgets). Used as the cost
# basis for the controller's expected average FLOPs.
CONTROLLER_BUDGET_FLOPS: Dict[float, float] = {
    0.25: 0.491472384,
    0.50: 0.687450432,
    0.75: 0.883428480,
}


def _cache_controller(
    model: nn.Module, loader: Any, device: torch.device
) -> Dict[str, Any]:
    """Cache confidence, per-budget predictions and labels over a loader.

    One batched prefix pass + one batched tail per budget — no per-image loops.

    Args:
        model: Trained VitController.
        loader: DataLoader to cache over.
        device: Compute device.

    Returns:
        Dict with 'conf' (N,), 'labels' (N,) and 'pred' {budget: (N,)}.
    """
    confs:  List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    preds:  Dict[float, List[torch.Tensor]] = {b: [] for b in (0.25, 0.50, 0.75)}
    with torch.no_grad():
        for images, lbl in loader:
            images = images.to(device, non_blocking=True)
            conf, p = model.forward_cached(images)
            confs.append(conf.cpu())
            labels.append(lbl)
            for b in (0.25, 0.50, 0.75):
                preds[b].append(p[b].cpu())
    return {
        'conf':   torch.cat(confs),
        'labels': torch.cat(labels),
        'pred':   {b: torch.cat(preds[b]) for b in (0.25, 0.50, 0.75)},
    }


def _route_controller(
    cache: Dict[str, Any],
    high_thresh: float,
    low_thresh: float,
    budget_flops: Dict[float, float],
) -> Dict[str, Any]:
    """Route a cached split at a threshold pair; return acc/avg-FLOPs/distribution.

    Args:
        cache: Output of ``_cache_controller``.
        high_thresh: Confidence above which the 25% budget is used.
        low_thresh: Confidence below which the 75% budget is used.
        budget_flops: Per-budget GFLOPs basis.

    Returns:
        Dict with accuracy, avg_flops_giga and budget_distribution.
    """
    conf   = cache['conf']
    labels = cache['labels']
    n      = labels.shape[0]

    keep_ratios = torch.where(
        conf > high_thresh,
        torch.full_like(conf, 0.25),
        torch.where(conf < low_thresh, torch.full_like(conf, 0.75),
                    torch.full_like(conf, 0.50)),
    )

    preds = cache['pred'][0.50].clone()
    for b in (0.25, 0.75):
        mask = keep_ratios == b
        preds[mask] = cache['pred'][b][mask]

    counts = {b: int((keep_ratios == b).sum().item()) for b in (0.25, 0.50, 0.75)}
    avg_flops = sum(counts[b] * budget_flops[b] for b in (0.25, 0.50, 0.75)) / n

    return {
        'accuracy':       round((preds == labels).float().mean().item(), 6),
        'avg_flops_giga': round(avg_flops, 6),
        'budget_distribution': {
            '25pct': round(counts[0.25] / n, 4),
            '50pct': round(counts[0.50] / n, 4),
            '75pct': round(counts[0.75] / n, 4),
        },
    }


def run_controller_eval(
    model: nn.Module,
    device: torch.device,
    val_loader: Any,
    test_loader: Any,
    epoch_history: List[Dict[str, Any]],
    best_epoch: int,
    output_dir: str,
    num_params: int,
    split_info: Dict[str, Any],
) -> None:
    """Evaluate the controller: select thresholds on val, evaluate once on test.

    Thresholds are searched on the validation split only; the best pair is then
    evaluated exactly once on the official test split. Average FLOPs per pair is
    the expected cost from the realised budget distribution, using the static
    per-budget FLOPs basis — never a single dummy-image value.

    Args:
        model: Trained VitController.
        device: Evaluation device.
        val_loader: Validation DataLoader (threshold selection).
        test_loader: Test DataLoader (final evaluation only).
        epoch_history: Training epoch records.
        best_epoch: Epoch at which the best checkpoint was saved.
        output_dir: Directory to write metrics.json.
        num_params: Total trainable parameters.
        split_info: Dict with split_seed/train_size/val_size/test_size.
    """
    model.eval()
    budget_flops = CONTROLLER_BUDGET_FLOPS

    print("Caching controller predictions (val + test)…")
    val_cache  = _cache_controller(model, val_loader,  device)
    test_cache = _cache_controller(model, test_loader, device)

    threshold_pairs = [
        (0.9, 0.7), (0.8, 0.6), (0.7, 0.5), (0.6, 0.4), (0.5, 0.3),
    ]
    threshold_results: List[Dict[str, Any]] = []
    for high_thresh, low_thresh in threshold_pairs:
        r = _route_controller(val_cache, high_thresh, low_thresh, budget_flops)
        threshold_results.append({
            'high_thresh':         high_thresh,
            'low_thresh':          low_thresh,
            'val_acc':             r['accuracy'],
            'avg_flops_giga':      r['avg_flops_giga'],
            'budget_distribution': r['budget_distribution'],
        })
        bd = r['budget_distribution']
        print(
            f"  [{high_thresh},{low_thresh}]  val_acc={r['accuracy']:.4f}  "
            f"avg_flops={r['avg_flops_giga']:.4f}  "
            f"25%={bd['25pct']:.2%}  50%={bd['50pct']:.2%}  75%={bd['75pct']:.2%}"
        )

    # Select the best threshold pair on validation, evaluate once on test.
    best = max(threshold_results, key=lambda r: r['val_acc'])
    t = _route_controller(
        test_cache, best['high_thresh'], best['low_thresh'], budget_flops
    )
    test_result = {
        'high_thresh':         best['high_thresh'],
        'low_thresh':          best['low_thresh'],
        'test_acc':            t['accuracy'],
        'avg_flops_giga':      t['avg_flops_giga'],
        'budget_distribution': t['budget_distribution'],
    }
    print(
        f"Test eval [{best['high_thresh']},{best['low_thresh']}]: "
        f"test_acc={t['accuracy']:.4f}  avg_flops={t['avg_flops_giga']:.4f}"
    )

    metrics: Dict[str, Any] = {
        'model_name':            'vit_controller',
        'note':                  'Adaptive per-image token-budget routing '
                                 '(25/50/75% keep) from a learned confidence head, '
                                 'pruning after block 6. Thresholds selected on '
                                 'validation; test evaluated once. flops_giga_per_budget '
                                 'are the clean-split static-specialist FLOPs used as the '
                                 'cost basis; avg_flops_giga (== flops_giga) is the '
                                 'expected cost at the selected operating point.',
        'parameters':            num_params,
        'flops_giga_per_budget': {
            '25pct': budget_flops[0.25],
            '50pct': budget_flops[0.50],
            '75pct': budget_flops[0.75],
        },
        'best_val_acc':          best['val_acc'],
        'best_epoch':            best_epoch,
        'avg_flops_giga':        best['avg_flops_giga'],
        'flops_giga':            best['avg_flops_giga'],
        'final_test_acc':        test_result['test_acc'],
        'selected_thresholds':   {
            'high_thresh': best['high_thresh'],
            'low_thresh':  best['low_thresh'],
        },
        'test_result':           test_result,
        'threshold_results':     threshold_results,
        'epoch_history':         epoch_history,
    }
    metrics.update(split_info)
    os.makedirs(output_dir, exist_ok=True)
    save_metrics(metrics, os.path.join(output_dir, 'metrics.json'))


def run_multibudget(cfg: Dict[str, Any], device: torch.device) -> None:
    """Train the shared-weight multi-budget ViT and (optionally) progressive eval.

    Sandwich training of one backbone across token budgets; best checkpoint selected
    on mean validation accuracy. When ``eval_test`` is set, the progressive-widening
    evaluator selects confidence thresholds on validation and evaluates the chosen
    operating points once on test.

    Args:
        cfg: Config dictionary.
        device: Torch device.
    """
    from src.models.vit_multibudget import VitMultiBudget
    from src.training.multibudget_trainer import MultiBudgetTrainer
    from src.training.progressive_widening_eval import run_progressive_eval

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg['data_dir'], batch_size=cfg['batch_size'], seed=cfg.get('seed', 42),
        val_size=cfg.get('val_size', 5000), split_seed=cfg.get('split_seed', 42))
    split_info = {
        'split_seed': cfg.get('split_seed', 42),
        'train_size': len(train_loader.dataset),
        'val_size':   len(val_loader.dataset),
        'test_size':  len(test_loader.dataset),
    }
    print(f"Split   : {split_info}")

    eval_budgets = cfg.get('eval_budgets', [0.25, 0.50, 0.75, 1.00])
    model = VitMultiBudget(num_classes=cfg.get('num_classes', 100),
                           pretrained=cfg.get('pretrained', True),
                           prune_layer=cfg.get('prune_layer', 3),
                           budgets=tuple(eval_budgets)).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model   : {model.model_name}  ({num_params:,} params)")

    optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                            weight_decay=cfg['weight_decay'])
    output_dir = cfg.get('output_dir', 'checkpoints/multibudget_clean_split')
    results_dir = cfg.get('results_dir', 'results/multibudget_clean_split')

    trainer = MultiBudgetTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, device=device, output_dir=output_dir,
        eval_budgets=eval_budgets, middle_budgets=cfg.get('middle_budgets', [0.50, 0.75]),
        use_all_budgets=cfg.get('use_all_budgets', False),
        distill_weight=cfg.get('distill_weight', 0.5))
    history = trainer.train(epochs=cfg['epochs'])
    best_epoch = history['best_epoch']

    best_ckpt = os.path.join(output_dir, 'best_model.pt')
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"\nLoaded best checkpoint (epoch {best_epoch}, mean_val={history['best_mean_val']:.4f})")

    # Fixed-budget FLOPs + validation accuracy
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    flops = {}
    for b in eval_budgets:
        class _FB(nn.Module):
            def __init__(s, m, bb): super().__init__(); s.m, s.b = m, bb
            def forward(s, x): return s.m.forward_budget(x, s.b)
        flops[str(b)] = compute_flops(_FB(model, b), dummy)

    metrics: Dict[str, Any] = {
        'model_name': 'vit_multibudget',
        'parameters': num_params,
        'best_epoch': best_epoch,
        'best_mean_val_acc': history['best_mean_val'],
        'flops_giga_per_budget': flops,
        'epoch_history': history['epoch_history'],
    }
    metrics.update(split_info)

    if cfg.get('eval_test', False):
        print("\nProgressive widening evaluation (val-select, test-once):")
        prog = run_progressive_eval(model, val_loader, test_loader, device,
                                    output_dir, results_dir, split_info)
        metrics['fixed_budget_val_acc'] = prog['fixed_budget_val_acc']
        metrics['fixed_budget_test_acc'] = prog['fixed_budget_test_acc']

    os.makedirs(output_dir, exist_ok=True)
    save_metrics(metrics, os.path.join(output_dir, 'metrics.json'))
    print("\nDone.")


def main() -> None:
    """Parse CLI args, build model, train, evaluate, save metrics."""
    parser = argparse.ArgumentParser(description='Compute-Aware ViT Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Config  : {args.config}")
    print(f"Settings: {cfg}")

    model_type = cfg['model_type']

    if model_type == 'cascade':
        run_cascade(cfg, device)
        return

    if model_type == 'multibudget':
        run_multibudget(cfg, device)
        return

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg['data_dir'],
        batch_size=cfg['batch_size'],
        seed=cfg.get('seed', 42),
        val_size=cfg.get('val_size', 5000),
        split_seed=cfg.get('split_seed', 42),
    )

    split_info: Dict[str, Any] = {
        'split_seed': cfg.get('split_seed', 42),
        'train_size': len(train_loader.dataset),
        'val_size':   len(val_loader.dataset),
        'test_size':  len(test_loader.dataset),
    }
    print(f"Split   : {split_info}")

    model = build_model(cfg)
    model = model.to(device)

    model_name = getattr(model, 'model_name', model_type)
    num_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model   : {model_name}  ({num_params:,} params)")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
    )

    output_dir = cfg.get('output_dir', f"checkpoints/{model_type}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        output_dir=output_dir,
        model_type=model_type,
        aux_loss_weight=cfg.get('aux_loss_weight', 0.5),
    )
    history = trainer.train(epochs=cfg['epochs'])
    best_epoch = history.get('best_epoch', 0)

    # Reload best weights (selected on validation only)
    best_ckpt = os.path.join(output_dir, 'best_model.pt')
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"\nLoaded best checkpoint from {best_ckpt} (best epoch {best_epoch})")

    if model_type == 'controller':
        print(f"\nController threshold evaluation:")
        run_controller_eval(
            model=model,
            device=device,
            val_loader=val_loader,
            test_loader=test_loader,
            epoch_history=history['epoch_history'],
            best_epoch=best_epoch,
            output_dir=output_dir,
            num_params=num_params,
            split_info=split_info,
        )
    else:
        evaluator = Evaluator(
            model=model,
            val_loader=val_loader,
            device=device,
            output_dir=output_dir,
            model_type=model_type,
        )
        evaluator.evaluate(
            test_loader=test_loader,
            epoch_history=history['epoch_history'],
            best_epoch=best_epoch,
            split_info=split_info,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
