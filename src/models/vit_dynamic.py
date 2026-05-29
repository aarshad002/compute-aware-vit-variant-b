"""Dynamic fixed-budget ViT — controller architecture, controller disabled."""

import torch
import torch.nn as nn
import timm


class VitDynamic(nn.Module):
    """DeiT-Tiny pruned at layer 6 with a fixed keep_ratio (no adaptive routing).

    Shares the same pruning point as VitController (after block 6) so it acts
    as a direct ablation: same architecture, fixed budget instead of dynamic.
    """

    def __init__(
        self,
        keep_ratio: float = 0.50,
        prune_layer: int = 6,
        num_classes: int = 100,
        pretrained: bool = False,
    ) -> None:
        """Initialize dynamic fixed-budget model.

        Args:
            keep_ratio: Fraction of patch tokens to retain.
            prune_layer: Prune after this transformer block (1-indexed).
            num_classes: Number of output classes.
            pretrained: Load ImageNet pretrained weights from timm.
        """
        super().__init__()
        base = timm.create_model(
            'deit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.patch_embed = base.patch_embed
        self.cls_token   = base.cls_token
        self.pos_embed   = base.pos_embed
        self.pos_drop    = base.pos_drop
        self.blocks      = base.blocks
        self.norm        = base.norm
        self.head        = base.head

        self.keep_ratio  = keep_ratio
        self.prune_layer = prune_layer
        self.model_name  = f'vit_dynamic_{int(keep_ratio * 100)}'

    def _prune_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Retain top-k patch tokens by L2 norm; always keep CLS token.

        Args:
            x: Token sequence (B, N, C).

        Returns:
            Pruned sequence (B, 1 + num_keep, C).
        """
        B, N, C      = x.shape
        cls_token    = x[:, :1]
        patch_tokens = x[:, 1:]

        scores   = patch_tokens.norm(dim=-1)
        num_keep = max(1, int((N - 1) * self.keep_ratio))
        _, idx   = scores.topk(num_keep, dim=-1, sorted=False)
        idx      = idx.sort(dim=-1).values
        patch_tokens = patch_tokens.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, C)
        )
        return torch.cat([cls_token, patch_tokens], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fixed-budget pruning at prune_layer.

        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            Logits (B, num_classes).
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.prune_layer - 1:
                x = self._prune_tokens(x)

        x = self.norm(x)
        return self.head(x[:, 0])

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
