"""Linear min/max normalizer for obs/action tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LinearNormalizer:
    """Maps values in [min, max] to [-1, 1] (and back)."""

    low: torch.Tensor
    high: torch.Tensor
    eps: float = 1e-6

    @classmethod
    def fit(cls, data: np.ndarray | torch.Tensor, eps: float = 1e-6) -> "LinearNormalizer":
        if isinstance(data, np.ndarray):
            data_t = torch.from_numpy(data.astype(np.float32))
        else:
            data_t = data.detach().float().cpu()
        flat = data_t.reshape(-1, data_t.shape[-1])
        low = flat.min(dim=0).values
        high = flat.max(dim=0).values
        # Avoid zero-range dims.
        high = torch.maximum(high, low + eps)
        return cls(low=low, high=high, eps=eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low.to(device=x.device, dtype=x.dtype)
        high = self.high.to(device=x.device, dtype=x.dtype)
        return 2.0 * (x - low) / (high - low) - 1.0

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low.to(device=x.device, dtype=x.dtype)
        high = self.high.to(device=x.device, dtype=x.dtype)
        return (x + 1.0) * 0.5 * (high - low) + low

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"low": self.low, "high": self.high}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.low = state["low"].float()
        self.high = state["high"].float()
