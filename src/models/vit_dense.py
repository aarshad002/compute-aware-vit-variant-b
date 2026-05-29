"""Dense DeiT-Tiny baseline using timm."""

import torch
import torch.nn as nn
import timm


class VitDense(nn.Module):
    """DeiT-Tiny without any token pruning.

    Serves as the accuracy upper-bound and FLOPs reference for pruning variants.
    """

    def __init__(self, num_classes: int = 100, pretrained: bool = False) -> None:
        """Initialize dense DeiT-Tiny.

        Args:
            num_classes: Number of output classes.
            pretrained: Load ImageNet pretrained weights from timm.
        """
        super().__init__()
        self.model = timm.create_model(
            'deit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.model_name = 'vit_dense'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            Logits (B, num_classes).
        """
        return self.model(x)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
