"""Training loop with checkpointing and best-model saving."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Manages the training loop, optimizer, LR scheduler, and checkpointing.

    Saves best_model.pt whenever val_acc improves.
    For model_type='controller' it calls forward_train() and computes the
    combined loss; all other types use the standard forward().
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object],
        device: torch.device,
        output_dir: str,
        model_type: str = 'dense',
        aux_loss_weight: float = 0.5,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Model to train.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            optimizer: Optimiser instance.
            scheduler: Optional LR scheduler (must implement .step()).
            device: Training device.
            output_dir: Directory for checkpoints and logs.
            model_type: One of 'dense', 'static', 'dynamic', 'controller'.
            aux_loss_weight: Weight applied to auxiliary CE loss (controller).
        """
        self.model           = model
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.optimizer       = optimizer
        self.scheduler       = scheduler
        self.device          = device
        self.output_dir      = output_dir
        self.model_type      = model_type
        self.aux_loss_weight = aux_loss_weight
        self.criterion       = nn.CrossEntropyLoss()

        os.makedirs(output_dir, exist_ok=True)
        self.best_val_acc: float = 0.0
        self.epoch_history: List[Dict[str, Any]] = []

    def _train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch over train_loader.

        Returns:
            (train_loss, train_acc) averaged over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.model_type == 'controller':
                main_logits, aux_logits = self.model.forward_train(images)
                main_loss = self.criterion(main_logits, labels)
                aux_loss  = self.criterion(aux_logits, labels)
                loss      = main_loss + self.aux_loss_weight * aux_loss
                logits    = main_logits
            else:
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += images.size(0)

        return total_loss / total, correct / total

    def _val_epoch(self) -> Tuple[float, float]:
        """Run one validation epoch over val_loader.

        Returns:
            (val_loss, val_acc) averaged over the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.model_type == 'controller':
                    logits, _ = self.model.forward_inference(images)
                else:
                    logits = self.model(images)

                loss = self.criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                correct    += (logits.argmax(dim=1) == labels).sum().item()
                total      += images.size(0)

        return total_loss / total, correct / total

    def train(self, epochs: int) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            epochs: Number of epochs to train.

        Returns:
            Dictionary with 'best_val_acc' and 'epoch_history'.
        """
        model_name = getattr(self.model, 'model_name', self.model_type)
        print(f"Training {model_name} for {epochs} epochs  |  output: {self.output_dir}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch()
            val_loss,   val_acc   = self._val_epoch()
            elapsed = time.time() - t0

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            record = {
                'epoch':      epoch,
                'train_loss': round(train_loss, 6),
                'train_acc':  round(train_acc,  6),
                'val_loss':   round(val_loss,   6),
                'val_acc':    round(val_acc,    6),
                'lr':         current_lr,
                'time_s':     round(elapsed, 2),
            }
            self.epoch_history.append(record)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                ckpt = os.path.join(self.output_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), ckpt)
                print(f"  ✓ best model saved → {ckpt}  (val_acc={val_acc:.4f})")

        return {
            'best_val_acc':  self.best_val_acc,
            'epoch_history': self.epoch_history,
        }
