"""Custom RL runner utilities for in-hand rotation tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from mjlab.rl import MjlabOnPolicyRunner


def _compute_explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """Compute explained variance: 1 - Var[y - y_hat] / Var[y]."""
    y = returns.detach().flatten()
    y_hat = values.detach().flatten()
    var_y = torch.var(y, unbiased=False)
    if torch.isclose(var_y, torch.tensor(0.0, device=var_y.device)):
        return 0.0
    residual_var = torch.var(y - y_hat, unbiased=False)
    explained_variance = 1.0 - residual_var / (var_y + 1e-8)
    return float(explained_variance.item())


class InHandRotationOnPolicyRunner(MjlabOnPolicyRunner):
    """Runner that logs explained variance for critic diagnostics."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        original_update: Callable[[], dict[str, Any]] = self.alg.update

        def update_with_explained_variance() -> dict[str, Any]:
            storage = self.alg.storage
            explained_variance = _compute_explained_variance(
                values=storage.values,
                returns=storage.returns,
            )
            loss_dict = original_update()
            loss_dict["explained_variance"] = explained_variance
            return loss_dict

        self.alg.update = update_with_explained_variance  # type: ignore[method-assign]
        try:
            return super().learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        finally:
            self.alg.update = original_update  # type: ignore[method-assign]
