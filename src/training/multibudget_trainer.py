"""Sandwich-style trainer for the shared-weight multi-budget ViT.

Each step trains the smallest budget (0.25) and the largest (1.00) always, plus one
randomly sampled middle budget from {0.50, 0.75} (configurable to use all). The prefix
is computed once per step and reused across budget tails (efficient + consistent).
Optional in-place distillation from the full-budget output to the smaller budgets
(standard universally-slimmable trick) improves the small budgets.

No budget predictor, no routing, no Gumbel: the budgets are forced during training.
Best checkpoint is selected on the MEAN validation accuracy across budgets.
"""

import os
import random
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MultiBudgetTrainer:
    """Trains one shared backbone to operate at multiple token budgets."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str,
        eval_budgets: List[float],
        middle_budgets: List[float],
        use_all_budgets: bool = False,
        distill_weight: float = 0.5,
    ) -> None:
        """Initialize the multi-budget trainer.

        Args:
            model: VitMultiBudget instance.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            optimizer: Optimiser.
            device: Compute device.
            output_dir: Where best_model.pt is written.
            eval_budgets: Budgets to report val accuracy for (e.g. [0.25,0.5,0.75,1.0]).
            middle_budgets: Pool to sample one middle budget from each step.
            use_all_budgets: If True, train all eval_budgets every step (slower).
            distill_weight: Weight of KL distillation from the 1.0 budget to smaller ones.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.eval_budgets = eval_budgets
        self.middle_budgets = middle_budgets
        self.use_all_budgets = use_all_budgets
        self.distill_weight = distill_weight
        self.criterion = nn.CrossEntropyLoss()

        os.makedirs(output_dir, exist_ok=True)
        self.best_mean_val: float = 0.0
        self.best_epoch: int = 0
        self.epoch_history: List[Dict[str, Any]] = []

    def _step_budgets(self) -> List[float]:
        """Budgets trained this step (sandwich: smallest + largest + one middle)."""
        if self.use_all_budgets:
            return list(self.eval_budgets)
        return [0.25, 1.00, random.choice(self.middle_budgets)]

    def _train_epoch(self) -> Dict[str, float]:
        """One training epoch. Returns mean loss and per-budget train accuracy."""
        self.model.train()
        loss_sum, n = 0.0, 0
        correct = {b: 0 for b in self.eval_budgets}
        seen = {b: 0 for b in self.eval_budgets}

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            prefix = self.model.forward_prefix(images)         # shared prefix (once)
            budgets = self._step_budgets()

            # Full-budget logits first (teacher for distillation), if 1.0 is trained.
            logits_by_b = {}
            for b in budgets:
                logits_by_b[b] = self.model.forward_tail(prefix, b)

            loss = 0.0
            teacher = logits_by_b.get(1.00, None)
            for b in budgets:
                lg = logits_by_b[b]
                loss = loss + self.criterion(lg, labels)
                if teacher is not None and b < 1.0 and self.distill_weight > 0:
                    loss = loss + self.distill_weight * F.kl_div(
                        F.log_softmax(lg, dim=-1),
                        F.softmax(teacher.detach(), dim=-1),
                        reduction='batchmean')
            loss = loss / len(budgets)

            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item() * images.size(0)
            n += images.size(0)
            for b in budgets:
                correct[b] += (logits_by_b[b].argmax(-1) == labels).sum().item()
                seen[b] += images.size(0)

        train_acc = {b: (correct[b] / seen[b] if seen[b] else 0.0) for b in self.eval_budgets}
        return {'loss': loss_sum / n, 'train_acc': train_acc}

    @torch.no_grad()
    def _val_epoch(self) -> Dict[float, float]:
        """Validation accuracy at every eval budget (shared prefix per batch)."""
        self.model.eval()
        correct = {b: 0 for b in self.eval_budgets}
        total = 0
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            prefix = self.model.forward_prefix(images)
            for b in self.eval_budgets:
                logits = self.model.forward_tail(prefix, b)
                correct[b] += (logits.argmax(-1) == labels).sum().item()
            total += images.size(0)
        return {b: correct[b] / total for b in self.eval_budgets}

    def train(self, epochs: int) -> Dict[str, Any]:
        """Run sandwich multi-budget training.

        Args:
            epochs: Number of epochs.

        Returns:
            Dict with best_mean_val, best_epoch, epoch_history.
        """
        print(f"Multi-budget sandwich training for {epochs} epochs -> {self.output_dir}")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr = self._train_epoch()
            val_acc = self._val_epoch()
            elapsed = time.time() - t0
            mean_val = sum(val_acc.values()) / len(val_acc)
            min_val = min(val_acc.values())

            rec = {
                'epoch': epoch,
                'train_loss': round(tr['loss'], 6),
                'train_acc_per_budget': {str(b): round(v, 6) for b, v in tr['train_acc'].items()},
                'val_acc_per_budget': {str(b): round(v, 6) for b, v in val_acc.items()},
                'mean_val_acc': round(mean_val, 6),
                'min_val_acc': round(min_val, 6),
                'time_s': round(elapsed, 2),
            }
            self.epoch_history.append(rec)
            va = '  '.join(f'{b:.2f}={val_acc[b]:.4f}' for b in self.eval_budgets)
            print(f"Epoch {epoch:3d}/{epochs} | loss={tr['loss']:.4f} | val[{va}] "
                  f"| mean={mean_val:.4f} min={min_val:.4f} | {elapsed:.1f}s")

            if mean_val > self.best_mean_val:
                self.best_mean_val = mean_val
                self.best_epoch = epoch
                torch.save(self.model.state_dict(),
                           os.path.join(self.output_dir, 'best_model.pt'))
                print(f"  ✓ best (mean_val={mean_val:.4f}) saved")

        return {
            'best_mean_val': self.best_mean_val,
            'best_epoch': self.best_epoch,
            'epoch_history': self.epoch_history,
        }
