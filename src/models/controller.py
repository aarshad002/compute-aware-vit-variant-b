"""Confidence-based token budget controller for adaptive per-image routing."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class VitController(nn.Module):
    """DeiT-Tiny with a learned confidence head that selects token budget per image.

    Architecture
    ------------
    Blocks 0-5  : standard transformer blocks (shared prefix).
    After block 5: auxiliary classifier head reads CLS token → confidence.
    Based on confidence, select keep_ratio ∈ {0.25, 0.50, 0.75}.
    Blocks 6-11 : run on the pruned token sequence.

    Training (forward_train)
    ------------------------
    Controller is disabled; fixed keep_ratio=0.50 is used.
    Auxiliary head is active and supervised with classification loss.
    Total loss = CE(main) + 0.5 × CE(aux).

    Inference (forward_inference)
    -----------------------------
    Controller is active; keep_ratio chosen per image from confidence:
        confidence > high_thresh  →  25% budget
        confidence < low_thresh   →  75% budget
        otherwise                 →  50% budget
    """

    PRUNE_LAYER: int = 6  # prune after this block index (0-based: block 5)
    BUDGETS: Tuple[float, float, float] = (0.25, 0.50, 0.75)  # supported keep ratios

    def __init__(
        self,
        num_classes: int = 100,
        pretrained: bool = False,
        high_thresh: float = 0.9,
        low_thresh: float = 0.7,
    ) -> None:
        """Initialize controller model.

        Args:
            num_classes: Number of output classes.
            pretrained: Load ImageNet pretrained weights from timm.
            high_thresh: Confidence above which 25% budget is selected.
            low_thresh: Confidence below which 75% budget is selected.
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
        self.num_classes = num_classes
        self.high_thresh = high_thresh
        self.low_thresh  = low_thresh
        self.model_name  = 'vit_controller'

        embed_dim     = base.embed_dim  # 192 for DeiT-Tiny
        self.aux_head = nn.Linear(embed_dim, num_classes)

    def _prune_tokens(self, x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """Retain top-k patch tokens by L2 norm; always keep the CLS token.

        Args:
            x: Token sequence (B, N, C).
            keep_ratio: Fraction of patch tokens to retain.

        Returns:
            Pruned token sequence (B, 1 + num_keep, C).
        """
        B, N, C      = x.shape
        cls_token    = x[:, :1]
        patch_tokens = x[:, 1:]

        scores   = patch_tokens.norm(dim=-1)
        num_keep = max(1, int((N - 1) * keep_ratio))
        _, idx   = scores.topk(num_keep, dim=-1, sorted=False)
        idx      = idx.sort(dim=-1).values
        patch_tokens = patch_tokens.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, C)
        )
        return torch.cat([cls_token, patch_tokens], dim=1)

    def forward_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward: fixed keep_ratio=0.50, auxiliary head active.

        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            (main_logits, aux_logits), each (B, num_classes).
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        aux_logits = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.PRUNE_LAYER - 1:          # after block index 5
                aux_logits = self.aux_head(x[:, 0])
                x = self._prune_tokens(x, keep_ratio=0.50)

        x = self.norm(x)
        main_logits = self.head(x[:, 0])
        return main_logits, aux_logits

    def _prefix(self, x: torch.Tensor) -> torch.Tensor:
        """Run patch embedding, positional encoding and the shared prefix blocks.

        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            Token sequence after the first ``PRUNE_LAYER`` blocks (B, N, C).
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i in range(self.PRUNE_LAYER):
            x = self.blocks[i](x)
        return x

    def _confidence(self, x_prefix: torch.Tensor) -> torch.Tensor:
        """Max-softmax confidence of the auxiliary head at the prefix output.

        Args:
            x_prefix: Prefix token sequence from ``_prefix`` (B, N, C).

        Returns:
            Per-image confidence (B,).
        """
        return F.softmax(self.aux_head(x_prefix[:, 0]), dim=-1).max(dim=-1).values

    def _tail_logits(self, x_prefix: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """Prune to ``keep_ratio`` then run the remaining blocks and head (batched).

        Args:
            x_prefix: Prefix token sequence from ``_prefix`` (B, N, C).
            keep_ratio: Fraction of patch tokens to retain.

        Returns:
            Logits (B, num_classes).
        """
        x = self._prune_tokens(x_prefix, keep_ratio=keep_ratio)
        for block in self.blocks[self.PRUNE_LAYER:]:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    def _route(self, confidences: torch.Tensor) -> torch.Tensor:
        """Map confidences to per-image keep ratios using the thresholds.

        Args:
            confidences: Per-image confidence (B,).

        Returns:
            Per-image keep ratio in {0.25, 0.50, 0.75} (B,).
        """
        return torch.where(
            confidences > self.high_thresh,
            torch.full_like(confidences, 0.25),
            torch.where(
                confidences < self.low_thresh,
                torch.full_like(confidences, 0.75),
                torch.full_like(confidences, 0.50),
            ),
        )

    def forward_fixed(self, x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """Batched forward at a single fixed keep_ratio (no per-image routing).

        Args:
            x: Input images (B, 3, 224, 224).
            keep_ratio: Fixed fraction of patch tokens to retain.

        Returns:
            Main logits (B, num_classes).
        """
        return self._tail_logits(self._prefix(x), keep_ratio)

    def forward_cached(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[float, torch.Tensor]]:
        """One prefix pass + one batched tail per budget; for fast evaluation.

        Lets the evaluator route any confidence-threshold pair without re-running
        the backbone, by caching the confidence and the prediction at each budget.

        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            (confidences (B,), {budget: predictions (B,)}) for budgets 0.25/0.50/0.75.
        """
        x_prefix = self._prefix(x)
        conf  = self._confidence(x_prefix)
        preds = {
            b: self._tail_logits(x_prefix, b).argmax(dim=-1) for b in self.BUDGETS
        }
        return conf, preds

    def forward_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference forward with batched per-image budget routing.

        Vectorised equivalent of processing each image at its routed budget: the
        shared prefix runs once, each budget runs as a single batched tail, and
        per-image logits are gathered by the routed keep ratio. Produces the same
        result as a per-image loop but far faster.

        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            (logits, keep_ratios) where keep_ratios is a float tensor (B,)
            indicating the budget fraction used per image.
        """
        x_prefix    = self._prefix(x)
        confidences = self._confidence(x_prefix)
        keep_ratios = self._route(confidences)

        logits_by_budget = {b: self._tail_logits(x_prefix, b) for b in self.BUDGETS}
        logits = torch.empty_like(logits_by_budget[0.50])
        for b in self.BUDGETS:
            mask = keep_ratios == b
            if mask.any():
                logits[mask] = logits_by_budget[b][mask]
        return logits, keep_ratios

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward uses inference routing.

        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            Logits (B, num_classes).
        """
        logits, _ = self.forward_inference(x)
        return logits

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
