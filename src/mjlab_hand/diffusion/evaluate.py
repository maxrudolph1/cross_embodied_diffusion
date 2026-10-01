"""Evaluate a trained diffusion policy in the mjlab env."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import torch

from mjlab_hand.diffusion.padding import build_plan, family_of
from mjlab_hand.diffusion.policy import DiffusionPolicy
from mjlab_hand.eval.base import _policy_input, run_eval, select_evaluator

# Cache of (env, rl_cfg) keyed by (task, num_envs, device) for reuse across
# in-training eval calls. Not closed by evaluate_diffusion_policy when
# reuse_env=True; call close_cached_envs() to tear down explicitly.
_ENV_CACHE: dict[tuple[str, int, str], tuple] = {}


def close_cached_envs() -> None:
    for env, _rl_cfg in _ENV_CACHE.values():
        env.close()
    _ENV_CACHE.clear()


class DiffusionActionChunkPolicy:
    """Open-loop action-chunk execution with receding-horizon replan.

    pad_task (term-aligned multi-hand policy, CHANGES.md item 63): the env's
    native obs is scattered into the padded layout with
    `padding.build_plan(family).obs_index(task)` (a native width that doesn't
    match the schema is a hard error) and the action prefix of the task's
    native width is executed. If the checkpoint was trained with
    mask_pad_loss, the padded action channels are held fixed while sampling.
    """

    def __init__(
        self,
        policy: DiffusionPolicy,
        device: torch.device,
        replan_every: int | None = None,
        onehot: list[float] | None = None,
        pad_task: str | None = None,
    ):
        self.policy = policy
        self.device = device
        self.obs_horizon = policy.cfg.obs_horizon
        self.action_horizon = policy.cfg.action_horizon
        self.replan_every = replan_every or max(1, self.action_horizon // 2)
        self._obs_hist: deque[torch.Tensor] | None = None
        self._action_queue: torch.Tensor | None = None
        self._steps_since_replan = 0
        self.onehot = (
            torch.tensor(onehot, dtype=torch.float32, device=device)
            if onehot is not None
            else None
        )
        self.pad_task = pad_task
        self._obs_index = None
        self._native_act = None
        self._action_valid = None
        if pad_task is not None:
            assert onehot is None, "onehot and pad are mutually exclusive"
            plan = build_plan(family_of(pad_task))
            if (plan.obs_dim, plan.action_dim) != (policy.cfg.obs_dim, policy.cfg.action_dim):
                raise ValueError(
                    f"policy dims {policy.cfg.obs_dim}/{policy.cfg.action_dim} != "
                    f"{plan.family} padded layout {plan.obs_dim}/{plan.action_dim}"
                )
            self._plan = plan
            self._obs_index = torch.from_numpy(plan.obs_index(pad_task)).to(device)
            self._native_obs = plan.native[pad_task]["obs_dim"]
            self._native_act = plan.native[pad_task]["action_dim"]
            if policy.cfg.mask_pad_loss:
                self._action_valid = torch.from_numpy(plan.action_valid(pad_task)).to(device)

    def reset(self, num_envs: int) -> None:
        del num_envs
        self._obs_hist = None
        self._action_queue = None
        self._steps_since_replan = 0

    def __call__(self, obs) -> torch.Tensor:
        obs_t = _policy_input(obs, self.device).to(dtype=torch.float32)
        if obs_t.ndim != 2:
            obs_t = obs_t.reshape(obs_t.shape[0], -1)
        if self.onehot is not None:
            obs_t = torch.cat([obs_t, self.onehot.expand(obs_t.shape[0], -1)], dim=1)
        elif self._obs_index is not None:
            if obs_t.shape[1] != self._native_obs:
                raise ValueError(
                    f"{self.pad_task}: env obs width {obs_t.shape[1]} != schema width {self._native_obs}"
                )
            padded = torch.zeros(obs_t.shape[0], self.policy.cfg.obs_dim, device=obs_t.device)
            padded[:, self._obs_index] = obs_t
            obs_t = padded
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
            if self._action_valid is not None:
                valid = self._action_valid.expand(hist.shape[0], -1)
                chunk = self.policy.predict_action(hist, action_valid=valid)
            else:
                chunk = self.policy.predict_action(hist)
            self._action_queue = chunk
            self._steps_since_replan = 0

        action = self._action_queue[:, 0]
        self._action_queue = self._action_queue[:, 1:]
        self._steps_since_replan += 1
        if self._native_act is not None:
            action = action[:, : self._native_act]
        return action


def _warn_obs_mismatch(policy: DiffusionPolicy, env_obs_dim: int, onehot: list[float] | None, pad: bool) -> None:
    if pad:
        return  # DiffusionActionChunkPolicy hard-checks the native width
    expected = policy.cfg.obs_dim - (len(onehot) if onehot else 0)
    if expected != env_obs_dim:
        print(
            f"[WARN] policy obs_dim={policy.cfg.obs_dim} "
            f"({'no onehot' if onehot is None else f'onehot dim={len(onehot)}'}) "
            f"does not match env obs_dim={env_obs_dim}; "
            "a padded multi-hand policy needs pad=True"
        )


def evaluate_diffusion_policy(
    *,
    task: str,
    policy_path: Path,
    num_envs: int = 16,
    num_steps: int = 2000,
    device: str = "cuda:0",
    seed: int = 0,
    onehot: list[float] | None = None,
    pad: bool = False,
    reuse_env: bool = False,
) -> dict[str, float]:
    """
    Args:
        pad: the policy was trained on a term-aligned padded multi-hand store;
            drive `task` through the padded layout (DiffusionActionChunkPolicy
            pad_task). Mutually exclusive with `onehot`.
    """
    import mjlab_hand  # noqa: F401
    from mjlab_hand.eval.config import EvalConfig
    from mjlab_hand.eval.env_setup import setup_eval_env

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is not available")
    device_t = torch.device(device)
    policy = DiffusionPolicy.load(policy_path, device=device_t)
    policy.eval()

    cache_key = (task, num_envs, str(device_t))
    if reuse_env and cache_key in _ENV_CACHE:
        env, _rl_cfg = _ENV_CACHE[cache_key]
    else:
        eval_cfg = EvalConfig(num_envs=num_envs, seed=seed, device=str(device_t))
        env, _rl_cfg = setup_eval_env(task, eval_cfg, device_t)
        if reuse_env:
            _ENV_CACHE[cache_key] = (env, _rl_cfg)

    env_obs_dim = int(env.num_obs) if hasattr(env, "num_obs") else policy.cfg.obs_dim
    _warn_obs_mismatch(policy, env_obs_dim, onehot, pad)

    EvaluatorCls = select_evaluator(env)

    class _Args:
        grasp_success_tolerance = None
        grasp_success_steps = None
        rotation_success_tolerance = None
        rotation_success_steps = None

    evaluator = EvaluatorCls(env, _Args(), device_t)
    chunk_policy = DiffusionActionChunkPolicy(
        policy, device_t, onehot=onehot, pad_task=task if pad else None
    )
    chunk_policy.reset(num_envs)

    metrics = run_eval(env, chunk_policy, evaluator, num_steps, device_t)
    evaluator.report(metrics)
    if not reuse_env:
        env.close()
    return metrics


def render_diffusion_rollout(
    *,
    task: str,
    policy_path: Path,
    output_dir: Path,
    num_steps: int = 400,
    num_envs: int = 1,
    device: str = "cuda:0",
    seed: int = 0,
    tag: str = "rollout",
    onehot: list[float] | None = None,
    pad: bool = False,
) -> Path | None:
    """Record an mp4 of a diffusion-policy rollout. Requires MUJOCO_GL=egl.

    Builds a fresh env per call: VideoRecorder keeps an internal step counter
    driving `step_trigger`, so reusing a cached env would need trigger
    bookkeeping to open a new file each time.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.wrappers import VideoRecorder

    import mjlab_hand  # noqa: F401

    configure_torch_backends()
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is not available")
    device_t = torch.device(device)
    torch.manual_seed(seed)

    policy = DiffusionPolicy.load(policy_path, device=device_t)
    policy.eval()

    env_cfg = load_env_cfg(task, play=True)
    rl_cfg = load_rl_cfg(task)
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = str(device_t)
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed

    output_dir.mkdir(parents=True, exist_ok=True)
    video_folder = output_dir / f"{task}_{tag}"
    video_folder.mkdir(parents=True, exist_ok=True)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device_t), render_mode="rgb_array")
    raw_env = VideoRecorder(
        raw_env,
        video_folder=video_folder,
        step_trigger=lambda step: step == 0,
        video_length=num_steps,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(raw_env, clip_actions=rl_cfg.clip_actions)

    env_obs_dim = int(env.num_obs) if hasattr(env, "num_obs") else policy.cfg.obs_dim
    _warn_obs_mismatch(policy, env_obs_dim, onehot, pad)

    chunk_policy = DiffusionActionChunkPolicy(
        policy, device_t, onehot=onehot, pad_task=task if pad else None
    )
    chunk_policy.reset(num_envs)

    obs, _ = env.reset()
    with torch.no_grad():
        for _ in range(num_steps):
            actions = chunk_policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()
    videos = sorted(video_folder.glob("*.mp4"))
    if not videos:
        print(f"[WARN] No mp4 written under {video_folder}")
        return None
    return videos[0]
