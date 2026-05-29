"""Cascade inference system: 25% → 50% → 75% → dense."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadeInference(nn.Module):
    """Multi-stage cascade that exits early when a stage is sufficiently confident.

    Stage order: 25% budget → 50% → 75% → dense.
    Each stage has its own independent confidence threshold.
    If a stage's max-softmax confidence meets its threshold the image
    exits at that stage; otherwise it proceeds to the next stage.

    Note: this module is used programmatically during grid search in train.py;
    the grid search pre-caches all stage logits for efficiency.
    """

    STAGES: List[str] = ['25', '50', '75', 'dense']

    def __init__(
        self,
        model_25: nn.Module,
        model_50: nn.Module,
        model_75: nn.Module,
        model_dense: nn.Module,
        threshold_25: float = 0.7,
        threshold_50: float = 0.7,
        threshold_75: float = 0.7,
    ) -> None:
        """Initialize cascade.

        Args:
            model_25: Stage-1 model (25% token budget).
            model_50: Stage-2 model (50% token budget).
            model_75: Stage-3 model (75% token budget).
            model_dense: Stage-4 dense model.
            threshold_25: Confidence threshold for exiting at stage 1.
            threshold_50: Confidence threshold for exiting at stage 2.
            threshold_75: Confidence threshold for exiting at stage 3.
        """
        super().__init__()
        self.model_25    = model_25
        self.model_50    = model_50
        self.model_75    = model_75
        self.model_dense = model_dense
        self.threshold_25 = threshold_25
        self.threshold_50 = threshold_50
        self.threshold_75 = threshold_75

    def set_thresholds(
        self,
        threshold_25: float,
        threshold_50: float,
        threshold_75: float,
    ) -> None:
        """Update cascade exit thresholds.

        Args:
            threshold_25: Exit threshold for stage 1.
            threshold_50: Exit threshold for stage 2.
            threshold_75: Exit threshold for stage 3.
        """
        self.threshold_25 = threshold_25
        self.threshold_50 = threshold_50
        self.threshold_75 = threshold_75

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Run cascade inference on a batch, image-by-image.

        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            (predictions, stages_used) where predictions is a (B,) long tensor
            of predicted class indices and stages_used is a list of stage labels.
        """
        B = x.shape[0]
        predictions = torch.zeros(B, dtype=torch.long, device=x.device)
        stages_used: List[str] = ['dense'] * B

        for b in range(B):
            img = x[b : b + 1]

            with torch.no_grad():
                logits_25 = self.model_25(img)
            if F.softmax(logits_25, dim=-1).max().item() >= self.threshold_25:
                predictions[b] = logits_25.argmax(dim=-1)
                stages_used[b] = '25'
                continue

            with torch.no_grad():
                logits_50 = self.model_50(img)
            if F.softmax(logits_50, dim=-1).max().item() >= self.threshold_50:
                predictions[b] = logits_50.argmax(dim=-1)
                stages_used[b] = '50'
                continue

            with torch.no_grad():
                logits_75 = self.model_75(img)
            if F.softmax(logits_75, dim=-1).max().item() >= self.threshold_75:
                predictions[b] = logits_75.argmax(dim=-1)
                stages_used[b] = '75'
                continue

            with torch.no_grad():
                logits_dense = self.model_dense(img)
            predictions[b] = logits_dense.argmax(dim=-1)
            stages_used[b] = 'dense'

        return predictions, stages_used
