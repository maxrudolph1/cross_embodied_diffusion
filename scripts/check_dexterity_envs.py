#!/usr/bin/env python3
"""Sequential GPU smoke test of all ten dexterity tasks, without training/rendering.

Run on the allocated GPU from the repo root:
  .venv/bin/python scripts/check_dexterity_envs.py
Or narrow debugging with --task Baoding-Allegro --steps 50.
"""

from __future__ import annotations

import argparse
import gc
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 2:
        parser.error("--steps must be positive and --num-envs must be at least 2")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("WARP_CACHE_PATH", "/tmp/dexterity_smoke_warp")
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_hand  # noqa: F401

    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("This integration check requires an allocated local CUDA GPU")
    hands = ("Allegro", "LEAP", "Shadow", "Sharpa", "Wuji")
    tasks = [f"{family}-{hand}" for family in ("Baoding", "Pen-Spinning") for hand in hands]
    if args.task:
        if args.task not in tasks:
            parser.error(f"Unknown task: {args.task}")
        tasks = [args.task]
    results = []
    for task in tasks:
        cfg = load_env_cfg(task)
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.seed
        cfg.auto_reset = False
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        try:
            obs, _ = env.reset(seed=args.seed)
            command = env.command_manager.get_term("baoding" if task.startswith("Baoding-") else "pen_spin")
            assert torch.count_nonzero(command.metrics["episode_success"]) == 0, task
            object_names = ("ball1", "ball2") if task.startswith("Baoding-") else ("object",)
            # Command reset precedes the simulator's first forward pass. Verify
            # first-reset geometry explicitly, not merely finite observations:
            # stale palm coordinates used to spawn every object far from its hand.
            if task.startswith("Baoding-"):
                radius = command.cfg.orbit_radius
                orbit = torch.stack((radius * command.phase.cos(), radius * command.phase.sin(),
                                     torch.zeros_like(command.phase)), dim=-1)
                expected = torch.stack((orbit, -orbit), dim=1)
                expected[..., 2] += command.cfg.initial_ball_lift
                actual = command.positions - command.center[:, None]
                assert torch.allclose(actual, expected, atol=1e-5, rtol=0), (
                    task, "first reset spawned balls away from their reference orbit",
                    float((actual - expected).abs().max()),
                )
            else:
                offset = env.scene["object"].data.root_link_pos_w - command.center
                assert (offset[:, :2].abs() <= 0.003 + 1e-5).all(), (
                    task, "first reset pen XY is outside its reset jitter",
                )
                assert torch.allclose(offset[:, 2], torch.full_like(offset[:, 2], command.cfg.initial_lift),
                                      atol=1e-5, rtol=0), (task, "first reset pen height is wrong")
            first_poses = {name: env.scene[name].data.root_link_pose_w.clone() for name in object_names}
            for name in object_names:
                assert torch.allclose(env.scene[name].data.root_link_vel_w,
                                      torch.zeros_like(env.scene[name].data.root_link_vel_w), atol=1e-6), (
                    task, "reset injected object velocity", name,
                )
            obs, _ = env.reset(seed=args.seed)
            for name, value in first_poses.items():
                assert torch.allclose(env.scene[name].data.root_link_pose_w, value, atol=1e-5, rtol=0), (
                    task, "first and repeated seeded resets disagree", name,
                )
            dimensions = {name: value.shape[-1] for name, value in obs.items()}
            assert all(value.shape[0] == args.num_envs for value in obs.values())
            assert dimensions["actor"] > 0 and dimensions["critic"] > 0
            action_dim = env.action_manager.total_action_dim
            assert action_dim > 0
            terminations = 0
            for _ in range(args.steps):
                actions = torch.rand(args.num_envs, action_dim, device=args.device) * 2 - 1
                obs, reward, terminated, truncated, _ = env.step(actions)
                assert all(torch.isfinite(value).all() for value in obs.values()), (task, "observations")
                assert torch.isfinite(reward).all(), (task, "rewards")
                assert all(torch.isfinite(value).all() for value in command.metrics.values()), (task, "metrics")
                ended = terminated | truncated
                terminations += int(ended.sum())
                if ended.any():
                    env.reset(env_ids=ended.nonzero(as_tuple=False).flatten())

            # Reset only env 0 and ensure cumulative command state of others survives.
            others = slice(1, None)
            cumulative = ("revolutions", "episode_success", "mean_tracking_error") if task.startswith("Baoding-") else (
                "signed_revolutions", "completed_revolutions", "episode_success", "best_revolutions"
            )
            before = {name: command.metrics[name][others].clone() for name in cumulative}
            root_before = {name: env.scene[name].data.root_link_pose_w[others].clone()
                           for name in (("ball1", "ball2") if task.startswith("Baoding-") else ("object",))}
            env.reset(env_ids=torch.tensor([0], device=args.device))
            assert command.metrics["episode_success"][0] == 0
            for name, value in before.items():
                assert torch.equal(command.metrics[name][others], value), (task, "subset reset altered metric", name)
            for name, value in root_before.items():
                assert torch.allclose(env.scene[name].data.root_link_pose_w[others], value, atol=1e-6), (
                    task, "subset reset moved another environment", name
                )
            results.append({"task": task, "status": "passed", "steps": args.steps,
                            "num_envs": args.num_envs, "obs_dims": dimensions,
                            "action_dim": action_dim, "terminations": terminations})
            print(json.dumps(results[-1]), flush=True)
        finally:
            env.close()
        del env, command, obs
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"passed": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
