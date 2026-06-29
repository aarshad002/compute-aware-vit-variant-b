"""Evaluation loop with FLOPs measurement and metrics reporting."""

import os
from typing import Any, Dict, List, Optional, Tuple

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

    def _run_loader(self, loader: DataLoader) -> Tuple[float, float]:
        """Run a single evaluation pass over ``loader``.

        Args:
            loader: DataLoader to evaluate.

        Returns:
            (accuracy, mean_loss) over the loader.
        """
        self.model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits      = self.model(images)
                loss        = self.criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                correct    += (logits.argmax(dim=1) == labels).sum().item()
                total      += images.size(0)

        return correct / total, total_loss / total

    def evaluate(
        self,
        test_loader: DataLoader,
        epoch_history: Optional[List[Dict[str, Any]]] = None,
        best_epoch: int = 0,
        split_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the best checkpoint on validation and test, then persist metrics.

        Validation accuracy is recomputed (matches the selection criterion);
        the official test split is evaluated exactly once here.

        Args:
            test_loader: Official CIFAR-100 test DataLoader (final evaluation only).
            epoch_history: Training epoch records to embed in metrics.json.
            best_epoch: Epoch at which the best checkpoint was saved.
            split_info: Dict with split_seed/train_size/val_size/test_size.

        Returns:
            The metrics dictionary that was saved.
        """
        val_acc,  val_loss  = self._run_loader(self.val_loader)
        test_acc, test_loss = self._run_loader(test_loader)

        dummy      = torch.zeros(1, 3, 224, 224, device=self.device)
        flops_giga = compute_flops(self.model, dummy)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_name = getattr(self.model, 'model_name', self.model_type)

        metrics: Dict[str, Any] = {
            'model_name':     model_name,
            'parameters':     num_params,
            'flops_giga':     flops_giga,
            'best_val_acc':   round(val_acc,  6),
            'best_epoch':     best_epoch,
            'final_test_acc': round(test_acc, 6),
            'val_loss':       round(val_loss,  6),
            'test_loss':      round(test_loss, 6),
            'epoch_history':  epoch_history or [],
        }
        metrics.update(split_info or {})

        if hasattr(self.model, 'keep_ratio'):
            metrics['keep_ratio'] = self.model.keep_ratio
        if hasattr(self.model, 'prune_layer'):
            metrics['prune_layer'] = self.model.prune_layer

        os.makedirs(self.output_dir, exist_ok=True)
        save_metrics(metrics, os.path.join(self.output_dir, 'metrics.json'))
        print(
            f"val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  "
            f"flops={flops_giga:.4f} GFLOPs  params={num_params:,}"
        )
        return metrics
