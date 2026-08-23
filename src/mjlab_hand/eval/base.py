"""Base evaluator framework for task-agnostic evaluation loops."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import torch


class TaskEvaluator(ABC):
    """One per task family. Owns all per-env trackers and metric reporting."""

    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def matches(cls, env) -> bool:
        """Return True if this evaluator handles the given env."""
        ...

    @classmethod
    def add_cli_args(cls, parser) -> None:
        """Register task-specific CLI flags. Default: no-op."""

    @abstractmethod
    def __init__(self, env, args, device: torch.device) -> None: ...

    @abstractmethod
    def snapshot(self) -> None:
        """Capture pre-step env state. Called **before** every env.step()."""
        ...

    @abstractmethod
    def on_step(self, obs, rewards, dones, extras) -> None:
        """Called after every env.step().

        Metrics should be read from the most recent :meth:`snapshot`, not
        directly from the env, to avoid post-reset contamination for envs
        where ``dones==1``.

        ``dones`` is a long/int tensor (0/1) from RslRlVecEnvWrapper.
        """
        ...

    @abstractmethod
    def bootstrap(self, env_ids: torch.Tensor) -> None:
        """Called once after the initial env.reset() to set up trackers."""
        ...

    @abstractmethod
    def on_reset(self, done_mask: torch.Tensor) -> None:
        """Called when envs auto-reset (``done_mask`` indicates which envs finished)."""
        ...

    @abstractmethod
    def finalize(self) -> dict[str, float]:
        """Return final metrics dict. Called once at the end of evaluation."""
        ...

    @abstractmethod
    def report(self, metrics: dict[str, float]) -> None:
        """Print / log metrics."""
        ...


_EVALUATORS: list[type[TaskEvaluator]] = []


def register_evaluator(cls: type[TaskEvaluator]) -> type[TaskEvaluator]:
    _EVALUATORS.append(cls)
    return cls


def select_evaluator(env) -> type[TaskEvaluator]:
    for cls in _EVALUATORS:
        if cls.matches(env):
            return cls
    raise RuntimeError("No registered evaluator matches this env.")


def register_cli_args(parser) -> None:
    """Let every registered evaluator contribute its own CLI flags."""
    for cls in _EVALUATORS:
        cls.add_cli_args(parser)


def run_eval(
    env, policy, evaluator: TaskEvaluator, num_steps: int, device: torch.device
) -> dict[str, float]:
    """Generic evaluation loop.

    Calls :meth:`TaskEvaluator.snapshot` **before** each ``env.step()`` so
    that per-step metrics are captured from pre-step state.  This avoids the
    post-reset contamination bug: when an env terminates, ``env.step()``
    auto-resets it, so any env read after the call returns the reset state,
    not the terminal state.
    """

    obs, _ = env.reset()
    evaluator.bootstrap(torch.arange(env.num_envs, device=device))
    evaluator.snapshot()  # prime pre-step buffers for the first step

    t0 = time.time()
    for _ in range(num_steps):
        with torch.no_grad():
            actions = policy(_policy_input(obs, device))
        obs, rewards, dones, extras = env.step(actions)
        evaluator.on_step(obs, rewards, dones, extras)
        if dones.any():
            evaluator.on_reset(dones.bool())
        evaluator.snapshot()  # capture state for the next iteration

    metrics = evaluator.finalize()
    metrics["eval_time_s"] = time.time() - t0
    return metrics


def _policy_input(obs, device: torch.device) -> torch.Tensor:
    """Normalize policy input across wrapper variants."""
    # TensorDict is Mapping-like but not a dict subclass.
    if isinstance(obs, dict) or (hasattr(obs, "keys") and hasattr(obs, "__getitem__")):
        for key in ("actor", "policy", "obs"):
            if key in obs:
                return obs[key].to(device)
        raise KeyError(
            f"None of {('actor', 'policy', 'obs')} found in obs; got {list(obs.keys())}"
        )
    if isinstance(obs, (list, tuple)):
        return obs[0].to(device)
    return obs.to(device)


def write_metrics_json(metrics: dict, task: str, checkpoint: Path, output_path: Path) -> None:
    """Write metrics to a JSON file."""
    payload = {
        "task": task,
        "checkpoint": str(checkpoint),
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] Metrics written to {output_path}")
