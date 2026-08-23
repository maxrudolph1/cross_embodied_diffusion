#!/usr/bin/env python3
"""Headless rollout video from a checkpoint (no interactive viewer)."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch

# Must be set before mujoco/egl init.
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="e.g. Grasp-Allegro")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--video-length", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import mjlab.tasks  # noqa: F401
    import mjlab_hand  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.wrappers import VideoRecorder

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = args.checkpoint.resolve()
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_folder = args.output_dir / f"{args.task}_{ckpt.stem}"
    video_folder.mkdir(parents=True, exist_ok=True)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = VideoRecorder(
        env,
        video_folder=video_folder,
        step_trigger=lambda step: step == 0,
        video_length=args.video_length,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    print(f"[INFO] Recording {args.video_length} steps from {ckpt.name} -> {video_folder}")
    with torch.inference_mode():
        for step in range(args.video_length):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if (step + 1) % 50 == 0:
                print(f"[INFO] step {step + 1}/{args.video_length}")

    env.close()
    videos = sorted(video_folder.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No mp4 written under {video_folder}")
    for v in videos:
        print(f"Saved {v}")


if __name__ == "__main__":
    main()
