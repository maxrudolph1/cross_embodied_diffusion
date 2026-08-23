"""Collect expert demonstration trajectories from an RL checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mjlab_hand.diffusion.dataset import TrajectoryStore
from mjlab_hand.eval.base import _policy_input


def _as_numpy(x) -> np.ndarray:
    """Convert policy tensors / TensorDict leaves to a dense numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy") and not isinstance(x, dict):
        try:
            return np.asarray(x.numpy(), dtype=np.float32)
        except Exception:
            pass
    if isinstance(x, dict) or (hasattr(x, "keys") and hasattr(x, "__getitem__")):
        for key in ("actor", "policy", "obs"):
            if key in x:
                return _as_numpy(x[key])
        raise KeyError(f"Could not extract array from keys={list(x.keys())}")
    return np.asarray(x, dtype=np.float32)


@torch.no_grad()
def collect_demos(
    *,
    task: str,
    checkpoint: Path,
    output: Path,
    num_envs: int = 64,
    num_episodes: int = 200,
    max_episode_steps: int = 500,
    device: str = "cuda:0",
    seed: int = 0,
    success_tolerance: float = 0.03,
    success_steps: int = 10,
    keep_failures: bool = False,
) -> dict:
    """Roll out an RL expert and write successful (optionally failed) episodes to zarr."""
    from mjlab_hand.eval.config import EvalConfig
    from mjlab_hand.eval.env_setup import load_policy, make_runner, setup_eval_env

    import mjlab_hand  # noqa: F401

    eval_cfg = EvalConfig(num_envs=num_envs, seed=seed, device=device)
    env, rl_cfg = setup_eval_env(task, eval_cfg, torch.device(device))
    runner = make_runner(task, env, rl_cfg, torch.device(device))
    policy = load_policy(runner, checkpoint, torch.device(device))

    store = TrajectoryStore(output, mode="w")
    # Probe dims from a reset.
    obs, _ = env.reset()
    obs_t = _as_numpy(_policy_input(obs, torch.device(device)))
    act_dim = env.num_actions
    store.initialize(
        obs_dim=int(obs_t.shape[-1]),
        action_dim=int(act_dim),
        task=task,
        checkpoint=str(checkpoint),
        extra_meta={
            "num_envs": num_envs,
            "max_episode_steps": max_episode_steps,
            "success_tolerance": success_tolerance,
            "success_steps": success_steps,
        },
    )

    # Per-env episode buffers.
    ep_obs = [[] for _ in range(num_envs)]
    ep_act = [[] for _ in range(num_envs)]
    ep_rew = [[] for _ in range(num_envs)]
    consecutive_near = torch.zeros(num_envs, dtype=torch.int64, device=device)
    success_flag = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def goal_pos():
        return env.unwrapped.command_manager.get_term("pose").goal_pos

    def object_pos():
        return env.unwrapped.scene["object"].data.root_link_pos_w

    has_pose = "pose" in env.unwrapped.command_manager.active_terms
    first_goal = goal_pos().clone() if has_pose else None

    written = 0
    successes = 0
    steps = 0
    max_total_steps = max(num_episodes * max_episode_steps // max(num_envs // 4, 1), 10_000)

    print(f"[INFO] Collecting up to {num_episodes} episodes from {checkpoint}")
    while written < num_episodes and steps < max_total_steps:
        # RL inference policy expects a TensorDict (with 'actor' / 'critic'), not a bare tensor.
        actions = policy(obs)
        next_obs, rewards, dones, _extras = env.step(actions)

        obs_np = _as_numpy(_policy_input(obs, torch.device(device)))
        act_np = _as_numpy(actions)
        rew_np = _as_numpy(rewards).reshape(-1)

        if has_pose:
            dist = torch.norm(object_pos() - first_goal, dim=-1)
            near = dist < success_tolerance
            consecutive_near = torch.where(near, consecutive_near + 1, torch.zeros_like(consecutive_near))
            success_flag = success_flag | (consecutive_near >= success_steps)

        for i in range(num_envs):
            ep_obs[i].append(obs_np[i])
            ep_act[i].append(act_np[i])
            ep_rew[i].append(rew_np[i])

        done_ids = torch.nonzero(dones.bool(), as_tuple=False).view(-1).tolist()
        for i in done_ids:
            if written >= num_episodes:
                break
            succ = bool(success_flag[i].item()) if has_pose else True
            if succ or keep_failures:
                store.append_episode(
                    np.stack(ep_obs[i], axis=0),
                    np.stack(ep_act[i], axis=0),
                    np.asarray(ep_rew[i], dtype=np.float32),
                    success=succ,
                )
                written += 1
                successes += int(succ)
                if written % 10 == 0 or written == num_episodes:
                    print(
                        f"[INFO] episodes={written}/{num_episodes} "
                        f"successes={successes} steps={steps}"
                    )
            ep_obs[i].clear()
            ep_act[i].clear()
            ep_rew[i].clear()
            consecutive_near[i] = 0
            success_flag[i] = False
            if has_pose:
                first_goal[i] = goal_pos()[i]

        # Cap runaway episodes without natural termination.
        for i in range(num_envs):
            if len(ep_obs[i]) >= max_episode_steps and written < num_episodes:
                succ = bool(success_flag[i].item()) if has_pose else True
                if succ or keep_failures:
                    store.append_episode(
                        np.stack(ep_obs[i], axis=0),
                        np.stack(ep_act[i], axis=0),
                        np.asarray(ep_rew[i], dtype=np.float32),
                        success=succ,
                    )
                    written += 1
                    successes += int(succ)
                ep_obs[i].clear()
                ep_act[i].clear()
                ep_rew[i].clear()
                consecutive_near[i] = 0
                success_flag[i] = False
                if has_pose:
                    first_goal[i] = goal_pos()[i]

        obs = next_obs
        steps += 1

    env.close()
    summary = store.summary()
    summary["requested_episodes"] = num_episodes
    summary["collection_steps"] = steps
    print(f"[INFO] Wrote dataset: {summary}")
    return summary
