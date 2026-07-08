from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        inter = torch.sum(prob * target, dim=dims)
        denom = torch.sum(prob + target, dim=dims)
        dice = 1.0 - torch.mean((2.0 * inter + self.smooth) / (denom + self.smooth))
        return bce + dice

