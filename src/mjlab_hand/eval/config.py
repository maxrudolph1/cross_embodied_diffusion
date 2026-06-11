"""Configuration loading and validation for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalConfig:
    """Evaluation configuration shared across single, batch, and regression eval."""

    # Environment
    num_envs: int = 256
    device: str = "cuda:0"
    seed: int = 42

    # Evaluation
    num_steps: int | None = None  # auto-detect by task type if None
    num_steps_per_env: int | None = None  # for plot_eval_curve compute of total steps

    # Checkpoint discovery
    checkpoint_pattern: str = "model_*.pt"
    wandb_cache_root: Path = field(default_factory=lambda: Path("logs/rsl_rl"))

    # Object patching
    object: str | None = None
    scale: tuple[float, float, float] | None = None

    # Batch eval
    skip_existing: bool = True
    max_checkpoints: int | None = None

    # Task-specific evaluator params (passed through to evaluator __init__)
    grasp_success_tolerance: float = 0.03
    grasp_success_steps: int = 10
    rotation_success_tolerance: float = 0.1
    rotation_success_steps: int = 10

    def merge(self, overrides: dict[str, Any]) -> EvalConfig:
        """Return a copy with overrides applied (None values are skipped)."""
        valid = {f.name for f in fields(self)}
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        for k, v in overrides.items():
            if v is None or k not in valid:
                continue
            if k == "scale" and isinstance(v, str):
                v = _parse_scale_str(v)
            data[k] = v
        return EvalConfig(**data)


def _parse_scale_str(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--scale expects 3 comma-separated floats, got {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def load_config(path: Path | None) -> EvalConfig:
    """Load YAML config with optional 'extends' inheritance.

    If `path` is None, returns defaults. The 'extends' key is resolved relative
    to the file's directory.
    """
    if path is None:
        return EvalConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    data = _load_yaml_with_extends(path)
    return _config_from_dict(data)


def _load_yaml_with_extends(path: Path) -> dict[str, Any]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if "extends" in raw:
        parent = path.parent / raw.pop("extends")
        base = _load_yaml_with_extends(parent)
        base.update(raw)
        return base
    return raw


def _config_from_dict(d: dict[str, Any]) -> EvalConfig:
    valid = {f.name for f in fields(EvalConfig)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    if "scale" in d and isinstance(d["scale"], list):
        d["scale"] = tuple(d["scale"])
    if "wandb_cache_root" in d:
        d["wandb_cache_root"] = Path(d["wandb_cache_root"])
    return EvalConfig(**d)


def add_config_arg(parser) -> None:
    """Add --config flag to an argparse parser."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML eval config (CLI flags override config values)",
    )
