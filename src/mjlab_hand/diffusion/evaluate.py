"""Evaluate a trained diffusion policy in the mjlab env."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab_hand.diffusion.policy import DiffusionPolicy
from mjlab_hand.eval.base import _policy_input, run_eval, select_evaluator


@dataclass
class EmbodimentStats:
    """One source embodiment's static normalization stats + real dims, as
    written by `build_padded_dataset.py` / `train_diffusion.py` into
    `source_stats.json`. Used at eval time to normalize a live env's raw
    (real-dim) obs and un-normalize/un-pad the policy's predicted action --
    the exact inverse of what `build_padded_dataset.py` did at training
    time, so the same static per-embodiment stats are used in both places.
    """

    embodiment: str
    task: str
    obs_dim: int
    action_dim: int
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    @classmethod
    def load(cls, source_stats_path: Path, embodiment: str) -> "EmbodimentStats":
        payload = json.loads(Path(source_stats_path).read_text())
        for src in payload["sources"]:
            if src["embodiment"] == embodiment or src["task"] == embodiment:
                return cls(
                    embodiment=src["embodiment"],
                    task=src["task"],
                    obs_dim=int(src["obs_dim"]),
                    action_dim=int(src["action_dim"]),
                    obs_mean=torch.tensor(src["obs_mean"], dtype=torch.float32),
                    obs_std=torch.tensor(src["obs_std"], dtype=torch.float32),
                    action_mean=torch.tensor(src["action_mean"], dtype=torch.float32),
                    action_std=torch.tensor(src["action_std"], dtype=torch.float32),
                )
        available = [s["embodiment"] for s in payload["sources"]]
        raise KeyError(f"{embodiment!r} not found in {source_stats_path}; available: {available}")

# Cache of (env, rl_cfg) keyed by (task, num_envs, device) for reuse across
# in-training eval calls. Not closed by evaluate_diffusion_policy when
# reuse_env=True; call close_cached_envs() to tear down explicitly.
_ENV_CACHE: dict[tuple[str, int, str], tuple] = {}


def close_cached_envs() -> None:
    for env, _rl_cfg in _ENV_CACHE.values():
        env.close()
    _ENV_CACHE.clear()


class DiffusionActionChunkPolicy:
    """Open-loop action-chunk execution with receding-horizon replan."""

    def __init__(
        self,
        policy: DiffusionPolicy,
        device: torch.device,
        replan_every: int | None = None,
        onehot: list[float] | None = None,
        embodiment_stats: EmbodimentStats | None = None,
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
        # Cross-embodiment padded scheme: normalize this embodiment's raw obs
        # with its own static stats before padding to the policy's (larger)
        # padded width, and invert the same static stats -- un-normalize,
        # then slice off the padded tail -- on the policy's predicted action.
        # Mutually exclusive with `onehot` (that's the 2-source mixed
        # scheme's conditioning path; this is the N-source padded scheme's
        # normalization path).
        self.embodiment_stats = embodiment_stats
        if embodiment_stats is not None:
            assert onehot is None, "onehot and embodiment_stats are mutually exclusive"
            self._obs_mean = embodiment_stats.obs_mean.to(device)
            self._obs_std = embodiment_stats.obs_std.to(device)
            self._action_mean = embodiment_stats.action_mean.to(device)
            self._action_std = embodiment_stats.action_std.to(device)

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
        elif self.embodiment_stats is not None:
            obs_t = (obs_t - self._obs_mean) / self._obs_std
            pad = self.policy.cfg.obs_dim - obs_t.shape[1]
            if pad > 0:
                obs_t = torch.nn.functional.pad(obs_t, (0, pad))
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
        if self.embodiment_stats is not None:
            real_dim = self.embodiment_stats.action_dim
            action = action[:, :real_dim] * self._action_std + self._action_mean
        return action


def _warn_obs_mismatch(
    policy: DiffusionPolicy,
    env_obs_dim: int,
    onehot: list[float] | None,
    embodiment_stats: EmbodimentStats | None = None,
) -> None:
    if embodiment_stats is not None:
        if embodiment_stats.obs_dim != env_obs_dim:
            print(
                f"[WARN] embodiment_stats obs_dim={embodiment_stats.obs_dim} "
                f"does not match env obs_dim={env_obs_dim} for {embodiment_stats.embodiment!r}"
            )
        return
    expected = policy.cfg.obs_dim - (len(onehot) if onehot else 0)
    if expected != env_obs_dim:
        print(
            f"[WARN] policy obs_dim={policy.cfg.obs_dim} "
            f"({'no onehot' if onehot is None else f'onehot dim={len(onehot)}'}) "
            f"does not match env obs_dim={env_obs_dim}; "
            "did you forget --onehot for a mixed-embodiment policy?"
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
    embodiment: str | None = None,
    reuse_env: bool = False,
) -> dict[str, float]:
    """
    Args:
        embodiment: for a cross-embodiment padded policy (see
            `build_padded_dataset.py`), the source embodiment name to
            evaluate as (must match an entry in `source_stats.json` next to
            `policy_path`). Loads that embodiment's static normalization
            stats and drives the env through them -- mutually exclusive with
            `onehot` (the older 2-source mixed scheme's conditioning path).
    """
    import mjlab_hand  # noqa: F401
    from mjlab_hand.eval.config import EvalConfig
    from mjlab_hand.eval.env_setup import setup_eval_env

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    policy = DiffusionPolicy.load(policy_path, device=device_t)
    policy.eval()

    embodiment_stats = None
    if embodiment is not None:
        stats_path = Path(policy_path).parent / "source_stats.json"
        embodiment_stats = EmbodimentStats.load(stats_path, embodiment)

    cache_key = (task, num_envs, str(device_t))
    if reuse_env and cache_key in _ENV_CACHE:
        env, _rl_cfg = _ENV_CACHE[cache_key]
    else:
        eval_cfg = EvalConfig(num_envs=num_envs, seed=seed, device=str(device_t))
        env, _rl_cfg = setup_eval_env(task, eval_cfg, device_t)
        if reuse_env:
            _ENV_CACHE[cache_key] = (env, _rl_cfg)

    env_obs_dim = int(env.num_obs) if hasattr(env, "num_obs") else policy.cfg.obs_dim
    _warn_obs_mismatch(policy, env_obs_dim, onehot, embodiment_stats)

    EvaluatorCls = select_evaluator(env)

    class _Args:
        grasp_success_tolerance = None
        grasp_success_steps = None
        rotation_success_tolerance = None
        rotation_success_steps = None

    evaluator = EvaluatorCls(env, _Args(), device_t)
    chunk_policy = DiffusionActionChunkPolicy(
        policy, device_t, onehot=onehot, embodiment_stats=embodiment_stats
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
    embodiment: str | None = None,
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
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    policy = DiffusionPolicy.load(policy_path, device=device_t)
    policy.eval()

    embodiment_stats = None
    if embodiment is not None:
        stats_path = Path(policy_path).parent / "source_stats.json"
        embodiment_stats = EmbodimentStats.load(stats_path, embodiment)

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
    _warn_obs_mismatch(policy, env_obs_dim, onehot, embodiment_stats)

    chunk_policy = DiffusionActionChunkPolicy(
        policy, device_t, onehot=onehot, embodiment_stats=embodiment_stats
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
