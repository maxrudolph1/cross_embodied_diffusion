"""Linear (affine) normalizer for obs/action tensors.

One class, three ways to fit it (CHANGES.md item 63):
  - `fit`:              per-column [min, max] -> [-1, 1] over all rows (the
                        single-hand path, and --norm-mode shared, where the
                        padding zeros of a multi-hand store are included);
  - `fit_masked`:       per-column [min, max] over only the rows that use that
                        column (--norm-mode pad-aware), optionally percentiles;
  - `fit_standardized`: z-score written in the same affine form, i.e.
                        low = mean - std, high = mean + std (--norm-mode zscore);
                        values then routinely exceed +-1, so zscore needs a
                        larger --x0-clamp.
The frozen per-family artifacts (frozen_norm.py) are LinearNormalizers too.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

# A group is (row_start, row_end, valid_columns): rows [start, end) only use
# the columns where valid_columns is True (padding elsewhere).
Group = tuple[int, int, np.ndarray]


def _as_numpy(data) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.detach().float().cpu().numpy()
    data = np.asarray(data, dtype=np.float32)
    return data.reshape(-1, data.shape[-1])


@dataclass
class LinearNormalizer:
    """Maps values in [low, high] to [-1, 1] (and back), per column."""

    low: torch.Tensor
    high: torch.Tensor
    eps: float = 1e-6

    @classmethod
    def _from_bounds(cls, low: np.ndarray, high: np.ndarray, eps: float) -> "LinearNormalizer":
        low = torch.from_numpy(np.asarray(low, dtype=np.float32))
        high = torch.from_numpy(np.asarray(high, dtype=np.float32))
        high = torch.maximum(high, low + eps)  # avoid zero-range dims
        return cls(low=low, high=high, eps=eps)

    @classmethod
    def fit(cls, data, eps: float = 1e-6) -> "LinearNormalizer":
        flat = _as_numpy(data)
        return cls._from_bounds(flat.min(axis=0), flat.max(axis=0), eps)

    @classmethod
    def fit_masked(
        cls,
        data,
        groups: list[Group],
        clip_pct: float | None = None,
        include_zero: bool = False,
        eps: float = 1e-6,
    ) -> "LinearNormalizer":
        """Per column, min/max (or the clip_pct / 100-clip_pct percentiles) over
        only the rows whose group uses that column. A column no group uses
        gets [-1, 1]. include_zero widens each range to contain 0, so padding
        zeros stay in [-1, 1] after normalization."""
        flat = _as_numpy(data)
        d = flat.shape[1]
        low, high = np.full(d, -1.0, np.float32), np.full(d, 1.0, np.float32)
        for j in range(d):
            parts = [flat[s:e, j] for s, e, valid in groups if valid[j]]
            if not parts:
                continue
            col = np.concatenate(parts)
            if clip_pct:
                low[j], high[j] = np.percentile(col, [clip_pct, 100.0 - clip_pct])
            else:
                low[j], high[j] = col.min(), col.max()
        if include_zero:
            low, high = np.minimum(low, 0.0), np.maximum(high, 0.0)
        return cls._from_bounds(low, high, eps)

    @classmethod
    def fit_standardized(
        cls, data, groups: list[Group] | None = None, eps: float = 1e-6
    ) -> "LinearNormalizer":
        """z-score as an affine map: low = mean - std, high = mean + std, so
        normalize(x) = (x - mean) / std. With groups, mean/std per column use
        only the rows that use it."""
        flat = _as_numpy(data)
        d = flat.shape[1]
        if groups is None:
            mean, std = flat.mean(axis=0), flat.std(axis=0)
        else:
            mean, std = np.zeros(d, np.float32), np.ones(d, np.float32)
            for j in range(d):
                parts = [flat[s:e, j] for s, e, valid in groups if valid[j]]
                if parts:
                    col = np.concatenate(parts)
                    mean[j], std[j] = col.mean(), col.std()
        std = np.maximum(std, eps)
        return cls._from_bounds(mean - std, mean + std, eps)

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


def norm_digest(obs_norm: LinearNormalizer, act_norm: LinearNormalizer) -> str:
    """sha256 over the four bound arrays (float32, C order) -- recorded in
    train_config.json so runs can be checked to share a normalizer."""
    h = hashlib.sha256()
    for t in (obs_norm.low, obs_norm.high, act_norm.low, act_norm.high):
        h.update(np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return "sha256:" + h.hexdigest()
