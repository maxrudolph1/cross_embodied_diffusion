"""Checkpoint resolution and eval cache utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

_ITER_RE = re.compile(r"^model_(\d+)\.pt$")


def discover_checkpoints(log_dir: Path, pattern: str = "model_*.pt") -> list[tuple[int, Path]]:
    """Find checkpoints matching `pattern` under `log_dir`, sorted by iteration."""
    out: list[tuple[int, Path]] = []
    for ckpt in log_dir.glob(pattern):
        m = _ITER_RE.match(ckpt.name)
        if m:
            out.append((int(m.group(1)), ckpt))
    out.sort(key=lambda x: x[0])
    return out


def get_eval_cache_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / (checkpoint_path.stem + "_eval.json")


def load_cached_eval(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    with open(cache_path) as f:
        return json.load(f)


def write_eval_cache(
    metrics: dict[str, float],
    task: str,
    iteration: int,
    total_steps: int,
    ckpt_path: Path,
) -> Path:
    """Write eval metrics to JSON beside the checkpoint."""
    cache_path = get_eval_cache_path(ckpt_path)
    payload = {
        "task": task,
        "checkpoint": str(ckpt_path),
        "iteration": iteration,
        "total_interaction_steps": total_steps,
        "metrics": metrics,
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    return cache_path


def read_training_config(log_dir: Path) -> tuple[int | None, int | None]:
    """Read num_envs and num_steps_per_env from saved training config YAMLs.

    Returns (num_envs, num_steps_per_env) with None for missing files or keys.
    """
    num_envs: int | None = None
    num_steps_per_env: int | None = None

    agent_yaml = log_dir / "params" / "agent.yaml"
    if agent_yaml.exists():
        data = yaml.safe_load(agent_yaml.read_text())
        num_steps_per_env = data.get("num_steps_per_env")

    env_yaml = log_dir / "params" / "env.yaml"
    if env_yaml.exists():
        data = yaml.safe_load(env_yaml.read_text())
        num_envs = data.get("scene", {}).get("num_envs")

    return num_envs, num_steps_per_env


def resolve_checkpoint(
    checkpoint: str | Path | None,
    wandb_run_path: str | None,
    wandb_checkpoint_name: str | None,
    task: str,
    wandb_cache_root: Path,
) -> Path:
    """Resolve a checkpoint from a local path or a WandB run path."""
    if checkpoint is not None:
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    if wandb_run_path is None:
        raise ValueError("Either --checkpoint or --wandb-run-path must be provided.")

    from mjlab.utils.os import get_wandb_checkpoint_path

    log_root = (Path(wandb_cache_root) / task).resolve()
    ckpt, was_cached = get_wandb_checkpoint_path(
        log_root, Path(wandb_run_path), wandb_checkpoint_name
    )
    status = "cached" if was_cached else "downloaded"
    print(f"[INFO] WandB checkpoint resolved: {ckpt.name} ({status})")
    return ckpt
