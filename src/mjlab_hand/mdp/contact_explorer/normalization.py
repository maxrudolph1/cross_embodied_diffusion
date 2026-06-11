"""Normalization helpers for ContactExplorer state features."""

from __future__ import annotations

import torch
from torch import nn


def finite_row_mask(x: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask for rows that contain only finite values."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return torch.isfinite(x.reshape(x.shape[0], -1)).all(dim=1)


class EmpiricalNormalization(nn.Module):
    """Online mean/variance normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(self.dim))
        self.register_buffer("running_var", torch.ones(self.dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, self.dim)
        x = x[finite_row_mask(x)]
        if x.shape[0] == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        alpha = min(0.05 * float(x.shape[0]), 0.25)
        count = int(self.count.item())
        if count == 0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var.clamp_min(self.eps))
        else:
            self.running_mean.mul_(1.0 - alpha).add_(batch_mean * alpha)
            self.running_var.mul_(1.0 - alpha).add_(batch_var * alpha)
        self.count.add_(x.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(self.count.item()) == 0:
            return x
        return (x - self.running_mean) / (self.running_var.sqrt() + self.eps)
