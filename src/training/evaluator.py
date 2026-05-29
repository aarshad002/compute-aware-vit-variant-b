"""Evaluation loop with FLOPs measurement and metrics reporting."""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.flops import compute_flops
from src.utils.metrics import save_metrics


class Evaluator:
    """Runs a full evaluation pass, measures FLOPs, and writes metrics.json."""

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str,
        model_type: str = 'dense',
    ) -> None:
        """Initialize evaluator.

        Args:
            model: Trained model (should already have best weights loaded).
            val_loader: Validation DataLoader.
            device: Evaluation device.
            output_dir: Directory where metrics.json will be written.
            model_type: One of 'dense', 'static', 'dynamic', 'controller'.
        """
        self.model      = model
        self.val_loader = val_loader
        self.device     = device
        self.output_dir = output_dir
        self.model_type = model_type
        self.criterion  = nn.CrossEntropyLoss()

    def evaluate(
        self, epoch_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Evaluate model on val_loader and persist metrics.json.

        Args:
            epoch_history: Training epoch records to embed in metrics.json.

        Returns:
            The metrics dictionary that was saved.
        """
        self.model.eval()
        total_loss      = 0.0
        correct         = 0
        total           = 0
        keep_ratios_all: List[float] = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.model_type == 'controller':
                    logits, keep_ratios = self.model.forward_inference(images)
                    keep_ratios_all.extend(keep_ratios.cpu().tolist())
                else:
                    logits = self.model(images)

                loss        = self.criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                correct    += (logits.argmax(dim=1) == labels).sum().item()
                total      += images.size(0)

        val_acc  = correct / total
        val_loss = total_loss / total

        dummy      = torch.zeros(1, 3, 224, 224, device=self.device)
        flops_giga = compute_flops(self.model, dummy)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_name = getattr(self.model, 'model_name', self.model_type)

        metrics: Dict[str, Any] = {
            'model_name':    model_name,
            'parameters':    num_params,
            'flops_giga':    flops_giga,
            'best_val_acc':  round(val_acc,  6),
            'val_loss':      round(val_loss, 6),
            'epoch_history': epoch_history or [],
        }

        if hasattr(self.model, 'keep_ratio'):
            metrics['keep_ratio'] = self.model.keep_ratio
        if hasattr(self.model, 'prune_layer'):
            metrics['prune_layer'] = self.model.prune_layer

        if self.model_type == 'controller' and keep_ratios_all:
            counts = {0.25: 0, 0.50: 0, 0.75: 0}
            for kr in keep_ratios_all:
                kr_r = round(kr, 2)
                if kr_r in counts:
                    counts[kr_r] += 1
            n = len(keep_ratios_all)
            metrics['budget_distribution'] = {
                '25pct': round(counts[0.25] / n, 4),
                '50pct': round(counts[0.50] / n, 4),
                '75pct': round(counts[0.75] / n, 4),
            }

        os.makedirs(self.output_dir, exist_ok=True)
        save_metrics(metrics, os.path.join(self.output_dir, 'metrics.json'))
        print(f"val_acc={val_acc:.4f}  flops={flops_giga:.4f} GFLOPs  params={num_params:,}")
        return metrics
