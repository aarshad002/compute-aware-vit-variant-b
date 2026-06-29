"""Shared-weight multi-budget DeiT-Tiny for progressive token-budget widening.

One backbone and one head are trained (sandwich-style) to operate at several token
keep-ratios. A prefix (first ``prune_layer`` blocks, all tokens) is shared; the tail
prunes to a chosen keep-ratio and finishes the network. At inference the prefix is
computed once and cached; tails at increasing budgets reuse it (progressive widening).

This is NOT a budget-prediction MLP, NOT Gumbel routing, and NOT a multi-pass cascade
(the prefix is computed once). It is a single set of weights robust to many budgets.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import timm


class VitMultiBudget(nn.Module):
    """DeiT-Tiny that shares weights across token budgets via a shared prefix.

    Pruning happens after block index ``prune_layer - 1`` using L2-norm token
    importance (CLS always kept), identical to the static models, but the keep-ratio
    is a runtime argument rather than a fixed attribute.
    """

    def __init__(
        self,
        num_classes: int = 100,
        pretrained: bool = False,
        prune_layer: int = 3,
        budgets: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
    ) -> None:
        """Initialize the multi-budget backbone.

        Args:
            num_classes: Number of output classes.
            pretrained: Load ImageNet-pretrained timm weights.
            prune_layer: Prune after this transformer block (1-indexed).
            budgets: Token keep-ratios the model is expected to serve.
        """
        super().__init__()
        base = timm.create_model(
            'deit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes)
        self.patch_embed = base.patch_embed
        self.cls_token   = base.cls_token
        self.pos_embed   = base.pos_embed
        self.pos_drop    = base.pos_drop
        self.blocks      = base.blocks
        self.norm        = base.norm
        self.head        = base.head

        self.prune_layer = prune_layer
        self.budgets     = tuple(budgets)
        self.model_name  = 'vit_multibudget'

    def _prune_tokens(self, x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """Keep top-k patch tokens by L2 norm (CLS always kept).

        Args:
            x: Token sequence (B, N, C).
            keep_ratio: Fraction of patch tokens to retain; >=1 keeps all.

        Returns:
            Pruned sequence (B, 1 + num_keep, C).
        """
        if keep_ratio >= 0.999:
            return x
        B, N, C = x.shape
        cls = x[:, :1]
        patches = x[:, 1:]
        scores = patches.norm(dim=-1)
        num_keep = max(1, int((N - 1) * keep_ratio))
        _, idx = scores.topk(num_keep, dim=-1, sorted=False)
        idx = idx.sort(dim=-1).values
        patches = patches.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
        return torch.cat([cls, patches], dim=1)

    def forward_prefix(self, x: torch.Tensor) -> torch.Tensor:
        """Run patch-embed + positional encoding + the shared prefix blocks.

        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            Full-token sequence after ``prune_layer`` blocks (B, N, C).
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for i in range(self.prune_layer):
            x = self.blocks[i](x)
        return x

    def forward_tail(self, prefix_tokens: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """Prune the cached prefix to ``keep_ratio`` and finish the network.

        Args:
            prefix_tokens: Output of ``forward_prefix`` (B, N, C).
            keep_ratio: Token keep-ratio for the tail.

        Returns:
            Logits (B, num_classes).
        """
        x = self._prune_tokens(prefix_tokens, keep_ratio)
        for block in self.blocks[self.prune_layer:]:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    def forward_budget(self, x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """Full forward at a single keep-ratio (prefix + tail).

        Args:
            x: Input images (B, 3, 224, 224).
            keep_ratio: Token keep-ratio.

        Returns:
            Logits (B, num_classes).
        """
        return self.forward_tail(self.forward_prefix(x), keep_ratio)

    def forward(self, x: torch.Tensor, keep_ratio: float = 1.0) -> torch.Tensor:
        """Default forward at ``keep_ratio`` (defaults to full budget).

        Args:
            x: Input images (B, 3, 224, 224).
            keep_ratio: Token keep-ratio.

        Returns:
            Logits (B, num_classes).
        """
        return self.forward_budget(x, keep_ratio)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
