"""Cascade evaluation: systematic threshold search, exit-rate analysis, learned gate.

All functions operate on pre-cached per-stage statistics (confidence, prediction,
entropy, top1-top2 margin) so the full grid and the learned gate run on cached
tensors without re-running the backbones. Threshold/gate selection is always done
on validation data; the test split is only touched by the caller for the final
selected operating points.
"""

import csv
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Systematic threshold grid (supervisor-requested, includes 0.95).
THRESHOLD_GRID: List[float] = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
EXIT_STAGES: List[str] = ['25', '50', '75']   # stages with an exit decision
GATE_THRESHOLDS: List[float] = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


# ----------------------------------------------------------------------------- #
# Cache utilities
# ----------------------------------------------------------------------------- #
def stage_entropy_margin(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sample predictive entropy and top1-top2 margin.

    Args:
        probs: Softmax probabilities (N, num_classes).

    Returns:
        (entropy (N,), margin (N,)) where margin is top1 minus top2 probability.
    """
    eps = 1e-12
    entropy = -(probs * (probs + eps).log()).sum(dim=-1)
    top2 = probs.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    return entropy, margin


def subcache(cache: Dict[str, Any], idx: torch.Tensor) -> Dict[str, Any]:
    """Index every tensor in a stage cache by ``idx`` to form a sub-split view.

    Args:
        cache: Stage cache (tensors keyed by name).
        idx: Long tensor of indices to keep.

    Returns:
        New cache dict with each tensor restricted to ``idx``.
    """
    return {k: v[idx] for k, v in cache.items()}


def path_flops_giga(stage_flops: Dict[str, float]) -> Dict[str, float]:
    """Cumulative FLOPs along each exit path of the 25→50→75→dense cascade.

    Args:
        stage_flops: Per-stage GFLOPs with keys '25','50','75','dense'.

    Returns:
        Dict mapping exit stage → cumulative GFLOPs to reach and exit there.
    """
    f25 = stage_flops['25']
    f50 = f25 + stage_flops['50']
    f75 = f50 + stage_flops['75']
    fdn = f75 + stage_flops['dense']
    return {'25': f25, '50': f50, '75': f75, 'dense': fdn}


# ----------------------------------------------------------------------------- #
# Core cascade evaluation from per-stage exit decisions
# ----------------------------------------------------------------------------- #
def _eval_from_masks(
    exit_25: torch.Tensor,
    exit_50_raw: torch.Tensor,
    exit_75_raw: torch.Tensor,
    cache: Dict[str, Any],
    stage_flops: Dict[str, float],
) -> Dict[str, Any]:
    """Evaluate a cascade given raw per-stage exit decisions.

    Decisions are composed sequentially: a sample exits at the first stage whose
    raw decision is True; otherwise it falls through to the dense model.

    Args:
        exit_25: Raw exit decision at the 25% stage (N,).
        exit_50_raw: Raw exit decision at the 50% stage (N,).
        exit_75_raw: Raw exit decision at the 75% stage (N,).
        cache: Stage cache with '<stage>_pred' and 'labels'.
        stage_flops: Per-stage GFLOPs.

    Returns:
        Record with accuracy, cumulative avg FLOPs, exit counts/rates,
        per-exit accuracy and per-path FLOPs.
    """
    labels = cache['labels']
    n = labels.shape[0]

    m25 = exit_25
    m50 = (~m25) & exit_50_raw
    m75 = (~m25) & (~m50) & exit_75_raw
    mden = (~m25) & (~m50) & (~m75)

    preds = cache['dense_pred'].clone()
    preds[m25] = cache['25_pred'][m25]
    preds[m50] = cache['50_pred'][m50]
    preds[m75] = cache['75_pred'][m75]

    counts = {
        '25': int(m25.sum()), '50': int(m50.sum()),
        '75': int(m75.sum()), 'dense': int(mden.sum()),
    }
    pf = path_flops_giga(stage_flops)
    avg_flops = sum(counts[s] * pf[s] for s in ('25', '50', '75', 'dense')) / n

    def _exit_acc(mask: torch.Tensor, pred_key: str) -> Optional[float]:
        if not bool(mask.any()):
            return None
        return round(
            (cache[pred_key][mask] == labels[mask]).float().mean().item(), 6
        )

    return {
        'accuracy':       round((preds == labels).float().mean().item(), 6),
        'avg_flops_giga': round(avg_flops, 6),
        'exit_25_n':      counts['25'],
        'exit_50_n':      counts['50'],
        'exit_75_n':      counts['75'],
        'dense_n':        counts['dense'],
        'exit_25_rate':   round(counts['25'] / n, 4),
        'exit_50_rate':   round(counts['50'] / n, 4),
        'exit_75_rate':   round(counts['75'] / n, 4),
        'dense_rate':     round(counts['dense'] / n, 4),
        'exit_25_acc':    _exit_acc(m25, '25_pred'),
        'exit_50_acc':    _exit_acc(m50, '50_pred'),
        'exit_75_acc':    _exit_acc(m75, '75_pred'),
        'dense_exit_acc': _exit_acc(mden, 'dense_pred'),
        'avg_flops_exit_25':   round(pf['25'], 6),
        'avg_flops_exit_50':   round(pf['50'], 6),
        'avg_flops_exit_75':   round(pf['75'], 6),
        'avg_flops_dense_path': round(pf['dense'], 6),
    }


def eval_threshold_triple(
    t25: float, t50: float, t75: float,
    cache: Dict[str, Any], stage_flops: Dict[str, float],
) -> Dict[str, Any]:
    """Evaluate one confidence-threshold triple on a cached split.

    Args:
        t25/t50/t75: Confidence exit thresholds per stage.
        cache: Stage cache.
        stage_flops: Per-stage GFLOPs.

    Returns:
        Full record including the threshold values.
    """
    rec = _eval_from_masks(
        cache['25_conf'] >= t25,
        cache['50_conf'] >= t50,
        cache['75_conf'] >= t75,
        cache, stage_flops,
    )
    rec = {'threshold_25': t25, 'threshold_50': t50, 'threshold_75': t75, **rec}
    return rec


def sweep_threshold_grid(
    cache: Dict[str, Any], stage_flops: Dict[str, float],
    grid: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate every threshold triple in the grid on a cached split.

    Args:
        cache: Stage cache.
        stage_flops: Per-stage GFLOPs.
        grid: Threshold values (defaults to ``THRESHOLD_GRID``).

    Returns:
        List of records, one per triple.
    """
    grid = grid or THRESHOLD_GRID
    out: List[Dict[str, Any]] = []
    for t25 in grid:
        for t50 in grid:
            for t75 in grid:
                out.append(eval_threshold_triple(t25, t50, t75, cache, stage_flops))
    return out


# ----------------------------------------------------------------------------- #
# Pareto frontier, baseline comparison flags, operating-point selection
# ----------------------------------------------------------------------------- #
def mark_pareto(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tag each record with ``pareto_optimal`` and return the frontier.

    A record is Pareto-optimal if no other record has both ≥ accuracy and
    ≤ FLOPs (with at least one strict).

    Args:
        records: Records each carrying 'accuracy' and 'avg_flops_giga'.

    Returns:
        The frontier records sorted by ascending FLOPs.
    """
    order = sorted(range(len(records)),
                   key=lambda i: (records[i]['avg_flops_giga'],
                                  -records[i]['accuracy']))
    best_acc = -1.0
    frontier: List[Dict[str, Any]] = []
    for i in order:
        records[i]['pareto_optimal'] = False
    for i in order:
        acc = records[i]['accuracy']
        if acc > best_acc + 1e-12:
            records[i]['pareto_optimal'] = True
            frontier.append(records[i])
            best_acc = acc
    return frontier


def add_baseline_flags(
    record: Dict[str, Any], baselines: Dict[str, Dict[str, float]]
) -> None:
    """Annotate a record with whether it beats each baseline in acc and FLOPs.

    Args:
        record: A cascade record (mutated in place).
        baselines: Mapping name → {'acc': ..., 'flops': ...} on the same split.
    """
    record['beats_acc'] = {
        name: bool(record['accuracy'] > b['acc']) for name, b in baselines.items()
    }
    record['beats_flops'] = {
        name: bool(record['avg_flops_giga'] < b['flops']) for name, b in baselines.items()
    }


def select_operating_points(
    records: List[Dict[str, Any]],
    dense_flops: float,
    static75_flops: float,
    static50_flops: float,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Pick validation operating points under several criteria.

    Args:
        records: Validation grid records (with pareto flags set).
        dense_flops: Dense model GFLOPs (for the Pareto-knee trade-off).
        static75_flops: static_75 GFLOPs (FLOPs ceiling).
        static50_flops: static_50 GFLOPs (FLOPs ceiling).

    Returns:
        Mapping selection-name → chosen record (or None if no record qualifies).
    """
    highest = max(records, key=lambda r: r['accuracy'])

    frontier = [r for r in records if r.get('pareto_optimal')]
    # Knee: best accuracy-minus-normalised-cost trade-off on the frontier.
    knee = max(frontier, key=lambda r: r['accuracy'] - r['avg_flops_giga'] / dense_flops) \
        if frontier else None

    under75 = [r for r in records if r['avg_flops_giga'] <= static75_flops]
    under50 = [r for r in records if r['avg_flops_giga'] <= static50_flops]
    best_u75 = max(under75, key=lambda r: r['accuracy']) if under75 else None
    best_u50 = max(under50, key=lambda r: r['accuracy']) if under50 else None

    return {
        'highest_val_acc':            highest,
        'pareto_knee':                knee,
        'best_under_static75_flops':  best_u75,
        'best_under_static50_flops':  best_u50,
    }


# ----------------------------------------------------------------------------- #
# Learned exit gate (logistic regression on per-stage features)
# ----------------------------------------------------------------------------- #
GATE_FEATURES: List[str] = ['max_confidence', 'entropy', 'top1_top2_margin', 'stage_id']


def _stage_feature_matrix(cache: Dict[str, Any], stage_id: int, stage: str) -> np.ndarray:
    """Build the gate feature matrix for one stage over all cached samples.

    Args:
        cache: Stage cache.
        stage_id: Integer id of the stage (0/1/2 for 25/50/75).
        stage: Stage key ('25'/'50'/'75').

    Returns:
        Feature matrix (N, 4) with columns matching ``GATE_FEATURES``.
    """
    n = cache['labels'].shape[0]
    return np.stack([
        cache[f'{stage}_conf'].numpy(),
        cache[f'{stage}_entropy'].numpy(),
        cache[f'{stage}_margin'].numpy(),
        np.full(n, stage_id, dtype=np.float32),
    ], axis=1)


def build_gate_training(cache: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Build (features, labels) for the gate from a cached split.

    One training example per (sample, exit stage). Label = 1 if that stage's
    prediction is correct (the gate should exit), else 0.

    Args:
        cache: Stage cache (typically a sub-split of validation).

    Returns:
        (X (3N, 4), y (3N,)).
    """
    xs, ys = [], []
    labels = cache['labels']
    for sid, s in enumerate(EXIT_STAGES):
        xs.append(_stage_feature_matrix(cache, sid, s))
        ys.append((cache[f'{s}_pred'] == labels).numpy().astype(np.float32))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def fit_logistic_gate(x: np.ndarray, y: np.ndarray) -> Optional[Any]:
    """Fit a standardised logistic-regression gate; None if sklearn unavailable.

    Args:
        x: Feature matrix.
        y: Binary correctness labels.

    Returns:
        Fitted sklearn pipeline, or None.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    gate = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced'),
    )
    gate.fit(x, y)
    return gate


def gate_stage_probs(cache: Dict[str, Any], gate: Any) -> Dict[str, np.ndarray]:
    """Gate exit probability P(correct) at each exit stage for all samples.

    Args:
        cache: Stage cache.
        gate: Fitted gate pipeline.

    Returns:
        Mapping stage → probability array (N,).
    """
    probs: Dict[str, np.ndarray] = {}
    for sid, s in enumerate(EXIT_STAGES):
        probs[s] = gate.predict_proba(_stage_feature_matrix(cache, sid, s))[:, 1]
    return probs


def eval_gate(
    cache: Dict[str, Any], probs: Dict[str, np.ndarray],
    gate_threshold: float, stage_flops: Dict[str, float],
) -> Dict[str, Any]:
    """Evaluate the learned-gate cascade at a probability threshold.

    Args:
        cache: Stage cache.
        probs: Per-stage gate probabilities from ``gate_stage_probs``.
        gate_threshold: Exit if gate P(correct) ≥ this value.
        stage_flops: Per-stage GFLOPs.

    Returns:
        Full record including ``gate_threshold``.
    """
    to_t = lambda a: torch.from_numpy(a >= gate_threshold)
    rec = _eval_from_masks(
        to_t(probs['25']), to_t(probs['50']), to_t(probs['75']), cache, stage_flops
    )
    return {'gate_threshold': gate_threshold, **rec}


def sweep_gate_thresholds(
    cache: Dict[str, Any], probs: Dict[str, np.ndarray],
    stage_flops: Dict[str, float],
    gate_thresholds: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate the learned gate across probability thresholds on a split.

    Args:
        cache: Stage cache.
        probs: Per-stage gate probabilities.
        stage_flops: Per-stage GFLOPs.
        gate_thresholds: Thresholds to sweep (defaults to ``GATE_THRESHOLDS``).

    Returns:
        List of records, one per gate threshold.
    """
    gate_thresholds = gate_thresholds or GATE_THRESHOLDS
    return [eval_gate(cache, probs, g, stage_flops) for g in gate_thresholds]


# ----------------------------------------------------------------------------- #
# Artifact writers (CSV / Markdown / optional plot)
# ----------------------------------------------------------------------------- #
def write_threshold_csv(path: str, records: List[Dict[str, Any]]) -> None:
    """Write the full threshold-grid validation results to CSV.

    Args:
        path: Output CSV path.
        records: Threshold records.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = [
        'threshold_25', 'threshold_50', 'threshold_75', 'accuracy', 'avg_flops_giga',
        'exit_25_rate', 'exit_50_rate', 'exit_75_rate', 'dense_rate',
        'exit_25_acc', 'exit_50_acc', 'exit_75_acc', 'dense_exit_acc', 'pareto_optimal',
    ]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in records:
            w.writerow(r)


def write_pareto_csv(path: str, frontier: List[Dict[str, Any]]) -> None:
    """Write the validation Pareto frontier to CSV.

    Args:
        path: Output CSV path.
        frontier: Pareto-frontier records (ascending FLOPs).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = ['threshold_25', 'threshold_50', 'threshold_75',
            'avg_flops_giga', 'accuracy']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in frontier:
            w.writerow(r)


def maybe_plot(
    path: str, records: List[Dict[str, Any]], frontier: List[Dict[str, Any]],
    baselines: Dict[str, Dict[str, float]],
) -> bool:
    """Plot accuracy-vs-FLOPs with the Pareto frontier; no-op if matplotlib missing.

    Args:
        path: Output PNG path.
        records: All grid records (scatter).
        frontier: Pareto-frontier records (line).
        baselines: name → {'acc','flops'} markers.

    Returns:
        True if a plot was written, else False.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([r['avg_flops_giga'] for r in records],
               [r['accuracy'] for r in records],
               s=8, alpha=0.25, color='steelblue', label='threshold combos')
    fr = sorted(frontier, key=lambda r: r['avg_flops_giga'])
    ax.plot([r['avg_flops_giga'] for r in fr], [r['accuracy'] for r in fr],
            '-o', color='crimson', ms=4, label='Pareto frontier (val)')
    for name, b in baselines.items():
        ax.scatter([b['flops']], [b['acc']], marker='*', s=140, label=name, zorder=5)
    ax.set_xlabel('Average cumulative GFLOPs')
    ax.set_ylabel('Validation accuracy')
    ax.set_title('Cascade accuracy vs FLOPs (validation)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
