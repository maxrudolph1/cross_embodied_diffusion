"""Shared environment + policy setup for evaluation scripts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch


def setup_eval_env(task: str, eval_cfg, device: torch.device):
    """Build an evaluation env and return (env, rl_cfg).

    Reads `num_envs`, `seed`, `object`, `scale` from `eval_cfg` (an EvalConfig).
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.torch import configure_torch_backends

    from mjlab_hand.object.patch import patch_object

    configure_torch_backends()
    torch.manual_seed(eval_cfg.seed)

    env_cfg = load_env_cfg(task, play=True)
    rl_cfg = load_rl_cfg(task)

    env_cfg.scene.num_envs = eval_cfg.num_envs
    env_cfg.sim.device = str(device)
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = eval_cfg.seed

    if eval_cfg.object is not None:
        scale = eval_cfg.scale if eval_cfg.scale is not None else (1.0, 1.0, 1.0)
        patch_object(env_cfg, eval_cfg.object, scale)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))
    env = RslRlVecEnvWrapper(raw_env, clip_actions=rl_cfg.clip_actions)
    return env, rl_cfg


def make_runner(task: str, env, rl_cfg, device: torch.device):
    """Construct an RL runner appropriate for `task` (reusable across checkpoints)."""
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_runner_cls

    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    return runner_cls(env, asdict(rl_cfg), device=str(device))


def load_policy(runner, ckpt_path: Path, device: torch.device):
    """Load a checkpoint into an existing runner and return its inference policy."""
    runner.load(str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=str(device))
    return runner.get_inference_policy(device=str(device))


def default_eval_steps(task_type: str) -> int:
    """Default number of eval steps when not provided by user."""
    return 5000 if task_type == "rotation" else 2000
