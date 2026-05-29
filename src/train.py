"""Main training entry point. Config-driven; supports all model types."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import itertools
import json
import random
from typing import Any, Dict, List, Optional

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


def run_cascade(cfg: Dict[str, Any], device: torch.device) -> None:
    """Grid-search cascade thresholds over the validation set.

    Loads pre-trained checkpoints for each stage, pre-caches all stage logits,
    then sweeps all 7×7×7=343 threshold combinations and saves results.

    Args:
        cfg: Config dictionary (must contain checkpoint paths and data_dir).
        device: Torch device to run on.
    """
    print("=== Cascade Grid Search ===")

    _, val_loader = get_dataloaders(
        data_dir=cfg['data_dir'],
        batch_size=cfg['batch_size'],
        seed=cfg.get('seed', 42),
    )

    prune_layer = cfg.get('prune_layer', 3)
    model_25    = VitStaticPruning(keep_ratio=0.25, prune_layer=prune_layer)
    model_50    = VitStaticPruning(keep_ratio=0.50, prune_layer=prune_layer)
    model_75    = VitStaticPruning(keep_ratio=0.75, prune_layer=prune_layer)
    model_dense = VitDense()

    def _load(model: nn.Module, path: str) -> None:
        if path and os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"  loaded {path}")
        else:
            print(f"  WARNING: checkpoint not found at {path!r} — using random weights")

    _load(model_25,    cfg.get('ckpt_25', ''))
    _load(model_50,    cfg.get('ckpt_50', ''))
    _load(model_75,    cfg.get('ckpt_75', ''))
    _load(model_dense, cfg.get('ckpt_dense', ''))

    stages = [(model_25, '25'), (model_50, '50'), (model_75, '75'), (model_dense, 'dense')]
    for m, _ in stages:
        m.to(device).eval()

    # Stage FLOPs (deterministic for fixed keep_ratio)
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    stage_flops = {
        '25':    compute_flops(model_25,    dummy),
        '50':    compute_flops(model_50,    dummy),
        '75':    compute_flops(model_75,    dummy),
        'dense': compute_flops(model_dense, dummy),
    }
    print(f"Stage FLOPs (GFLOPs): {stage_flops}")

    # Pre-cache all stage logits on the full val set
    print("Pre-caching stage logits…")
    all_labels:       List[torch.Tensor] = []
    all_probs_25:     List[torch.Tensor] = []
    all_probs_50:     List[torch.Tensor] = []
    all_probs_75:     List[torch.Tensor] = []
    all_probs_dense:  List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            all_labels.append(labels)
            all_probs_25.append(F.softmax(model_25(images),    dim=-1).cpu())
            all_probs_50.append(F.softmax(model_50(images),    dim=-1).cpu())
            all_probs_75.append(F.softmax(model_75(images),    dim=-1).cpu())
            all_probs_dense.append(F.softmax(model_dense(images), dim=-1).cpu())

    labels_t     = torch.cat(all_labels)
    probs_25_t   = torch.cat(all_probs_25)
    probs_50_t   = torch.cat(all_probs_50)
    probs_75_t   = torch.cat(all_probs_75)
    probs_dense_t = torch.cat(all_probs_dense)

    conf_25  = probs_25_t.max(dim=-1).values
    pred_25  = probs_25_t.argmax(dim=-1)
    conf_50  = probs_50_t.max(dim=-1).values
    pred_50  = probs_50_t.argmax(dim=-1)
    conf_75  = probs_75_t.max(dim=-1).values
    pred_75  = probs_75_t.argmax(dim=-1)
    pred_dense = probs_dense_t.argmax(dim=-1)

    N = labels_t.shape[0]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results: List[Dict[str, Any]] = []

    print(f"Sweeping {len(thresholds)**3} threshold combinations…")
    for t25, t50, t75 in itertools.product(thresholds, repeat=3):
        preds   = pred_dense.clone()
        stages  = ['dense'] * N
        f_total = 0.0

        mask_25   = conf_25 >= t25
        mask_50   = (~mask_25) & (conf_50 >= t50)
        mask_75   = (~mask_25) & (~mask_50) & (conf_75 >= t75)
        mask_dense = (~mask_25) & (~mask_50) & (~mask_75)

        preds[mask_25] = pred_25[mask_25]
        preds[mask_50] = pred_50[mask_50]
        preds[mask_75] = pred_75[mask_75]
        # preds[mask_dense] already = pred_dense

        n25    = mask_25.sum().item()
        n50    = mask_50.sum().item()
        n75    = mask_75.sum().item()
        ndense = mask_dense.sum().item()

        # Cascade FLOPs: each stage adds to cumulative cost
        f_total = (
            n25    * stage_flops['25'] +
            n50    * (stage_flops['25'] + stage_flops['50']) +
            n75    * (stage_flops['25'] + stage_flops['50'] + stage_flops['75']) +
            ndense * (stage_flops['25'] + stage_flops['50'] + stage_flops['75'] + stage_flops['dense'])
        )

        accuracy  = (preds == labels_t).float().mean().item()
        avg_flops = f_total / N

        results.append({
            'threshold_25':   t25,
            'threshold_50':   t50,
            'threshold_75':   t75,
            'accuracy':       round(accuracy, 6),
            'avg_flops_giga': round(avg_flops, 6),
            'budget_distribution': {
                '25pct':    round(n25    / N, 4),
                '50pct':    round(n50    / N, 4),
                '75pct':    round(n75    / N, 4),
                'dense_pct': round(ndense / N, 4),
            },
        })

    results.sort(key=lambda r: r['accuracy'], reverse=True)

    output_dir = cfg.get('output_dir', 'checkpoints/cascade')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'metrics.json')

    cascade_metrics: Dict[str, Any] = {
        'model_name':         'cascade',
        'stage_flops_giga':   stage_flops,
        'total_combinations': len(results),
        'best_result':        results[0],
        'top_10_results':     results[:10],
        'all_results':        results,
    }
    with open(out_path, 'w') as f:
        json.dump(cascade_metrics, f, indent=2)

    best = results[0]
    print(
        f"Best: acc={best['accuracy']:.4f}  avg_flops={best['avg_flops_giga']:.4f} GFLOPs  "
        f"t25={best['threshold_25']}  t50={best['threshold_50']}  t75={best['threshold_75']}"
    )
    print(f"Results saved → {out_path}")


def run_controller_eval(
    model: nn.Module,
    device: torch.device,
    val_loader: Any,
    epoch_history: List[Dict[str, Any]],
    output_dir: str,
    num_params: int,
    flops_giga: float,
) -> None:
    """Evaluate the controller across 5 predefined threshold pairs.

    Args:
        model: Trained VitController.
        device: Evaluation device.
        val_loader: Validation DataLoader.
        epoch_history: Training epoch records.
        output_dir: Directory to write metrics.json.
        num_params: Total trainable parameters.
        flops_giga: FLOPs at 50% budget (training budget).
    """
    threshold_pairs = [
        (0.9, 0.7), (0.8, 0.6), (0.7, 0.5), (0.6, 0.4), (0.5, 0.3),
    ]
    model.eval()
    threshold_results: List[Dict[str, Any]] = []

    for high_thresh, low_thresh in threshold_pairs:
        model.high_thresh = high_thresh
        model.low_thresh  = low_thresh

        correct         = 0
        total           = 0
        keep_ratios_all: List[float] = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits, krs = model.forward_inference(images)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total   += images.size(0)
                keep_ratios_all.extend(krs.cpu().tolist())

        val_acc = correct / total
        counts  = {0.25: 0, 0.50: 0, 0.75: 0}
        for kr in keep_ratios_all:
            kr_r = round(kr, 2)
            if kr_r in counts:
                counts[kr_r] += 1
        n = len(keep_ratios_all)

        result = {
            'high_thresh': high_thresh,
            'low_thresh':  low_thresh,
            'val_acc':     round(val_acc, 6),
            'budget_distribution': {
                '25pct': round(counts[0.25] / n, 4),
                '50pct': round(counts[0.50] / n, 4),
                '75pct': round(counts[0.75] / n, 4),
            },
        }
        threshold_results.append(result)
        print(
            f"  [{high_thresh},{low_thresh}]  acc={val_acc:.4f}  "
            f"25%={counts[0.25]/n:.2%}  50%={counts[0.50]/n:.2%}  75%={counts[0.75]/n:.2%}"
        )

    best_acc = max(r['val_acc'] for r in threshold_results)
    metrics: Dict[str, Any] = {
        'model_name':        'vit_controller',
        'parameters':        num_params,
        'flops_giga':        flops_giga,
        'best_val_acc':      best_acc,
        'epoch_history':     epoch_history,
        'threshold_results': threshold_results,
    }
    os.makedirs(output_dir, exist_ok=True)
    save_metrics(metrics, os.path.join(output_dir, 'metrics.json'))


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

    train_loader, val_loader = get_dataloaders(
        data_dir=cfg['data_dir'],
        batch_size=cfg['batch_size'],
        seed=cfg.get('seed', 42),
    )

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

    # Reload best weights
    best_ckpt = os.path.join(output_dir, 'best_model.pt')
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"\nLoaded best checkpoint from {best_ckpt}")

    if model_type == 'controller':
        dummy      = torch.zeros(1, 3, 224, 224, device=device)
        flops_giga = compute_flops(model, dummy)
        print(f"\nController threshold evaluation:")
        run_controller_eval(
            model=model,
            device=device,
            val_loader=val_loader,
            epoch_history=history['epoch_history'],
            output_dir=output_dir,
            num_params=num_params,
            flops_giga=flops_giga,
        )
    else:
        evaluator = Evaluator(
            model=model,
            val_loader=val_loader,
            device=device,
            output_dir=output_dir,
            model_type=model_type,
        )
        evaluator.evaluate(epoch_history=history['epoch_history'])

    print("\nDone.")


if __name__ == '__main__':
    main()
