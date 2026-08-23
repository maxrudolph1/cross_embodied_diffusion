"""Evaluate a trained diffusion policy in the mjlab env."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import torch

from mjlab_hand.diffusion.policy import DiffusionPolicy
from mjlab_hand.eval.base import _policy_input, run_eval, select_evaluator


class DiffusionActionChunkPolicy:
    """Open-loop action-chunk execution with receding-horizon replan."""

    def __init__(
        self, policy: DiffusionPolicy, device: torch.device, replan_every: int | None = None
    ):
        self.policy = policy
        self.device = device
        self.obs_horizon = policy.cfg.obs_horizon
        self.action_horizon = policy.cfg.action_horizon
        self.replan_every = replan_every or max(1, self.action_horizon // 2)
        self._obs_hist: deque[torch.Tensor] | None = None
        self._action_queue: torch.Tensor | None = None
        self._steps_since_replan = 0

    def reset(self, num_envs: int) -> None:
        del num_envs
        self._obs_hist = None
        self._action_queue = None
        self._steps_since_replan = 0

    def __call__(self, obs) -> torch.Tensor:
        obs_t = _policy_input(obs, self.device).to(dtype=torch.float32)
        if obs_t.ndim != 2:
            obs_t = obs_t.reshape(obs_t.shape[0], -1)
        if self._obs_hist is None:
            self._obs_hist = deque(
                [obs_t.clone() for _ in range(self.obs_horizon)],
                maxlen=self.obs_horizon,
            )
        else:
            self._obs_hist.append(obs_t.clone())

        need_replan = (
            self._action_queue is None
            or self._action_queue.shape[1] == 0
            or self._steps_since_replan >= self.replan_every
        )
        if need_replan:
            hist = torch.stack(list(self._obs_hist), dim=1)  # (B, To, Do)
            chunk = self.policy.predict_action(hist)
            self._action_queue = chunk
            self._steps_since_replan = 0

        action = self._action_queue[:, 0]
        self._action_queue = self._action_queue[:, 1:]
        self._steps_since_replan += 1
        return action


def evaluate_diffusion_policy(
    *,
    task: str,
    policy_path: Path,
    num_envs: int = 16,
    num_steps: int = 2000,
    device: str = "cuda:0",
    seed: int = 0,
) -> dict[str, float]:
    import mjlab_hand  # noqa: F401

    from mjlab_hand.eval.config import EvalConfig
    from mjlab_hand.eval.env_setup import setup_eval_env

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    policy = DiffusionPolicy.load(policy_path, device=device_t)
    policy.eval()

    eval_cfg = EvalConfig(num_envs=num_envs, seed=seed, device=str(device_t))
    env, _rl_cfg = setup_eval_env(task, eval_cfg, device_t)

    EvaluatorCls = select_evaluator(env)

    class _Args:
        grasp_success_tolerance = None
        grasp_success_steps = None
        rotation_success_tolerance = None
        rotation_success_steps = None

    evaluator = EvaluatorCls(env, _Args(), device_t)
    chunk_policy = DiffusionActionChunkPolicy(policy, device_t)
    chunk_policy.reset(num_envs)

    metrics = run_eval(env, chunk_policy, evaluator, num_steps, device_t)
    evaluator.report(metrics)
    env.close()
    return metrics
