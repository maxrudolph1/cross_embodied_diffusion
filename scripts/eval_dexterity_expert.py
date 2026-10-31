#!/usr/bin/env python3
"""Score Baoding/Pen-Spinning RL experts and stream an honest first-episode MP4.

Run from the repository root, e.g.:
  .venv/bin/python scripts/eval_dexterity_expert.py --task Baoding-Allegro \
    --checkpoint logs/rsl_rl/baoding_allegro/RUN/model_1000.pt \
    --output-dir outputs/expert_eval/baoding --video

Scores exactly the first fresh-reset episode of every environment, including failures.
The video shows environment 0 from this same evaluation, without selecting a success.
No checkpoint means an explicitly labelled zero-action diagnostic, not an expert.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from dexterity_runtime import check_checkpoint_task, task_signature


def summarize(episodes: list[dict]) -> dict:
    """Keep censoring distinct from environment failures and time limits."""
    n = len(episodes)
    if not n:
        raise ValueError("No episodes to summarize")
    names = set.intersection(*(set(row["metrics"]) for row in episodes))
    result = {
        "episodes": n,
        "failure_rate": sum(row["terminated"] for row in episodes) / n,
        "timeout_rate": sum(row["truncated"] for row in episodes) / n,
        "censored_rate": sum(row["censored"] for row in episodes) / n,
        "mean_duration_s": sum(row["duration_s"] for row in episodes) / n,
        "mean_return": sum(row["return"] for row in episodes) / n,
        "mean_final_command_metrics": {
            name: sum(row["metrics"][name] for row in episodes) / n
            for name in sorted(names)
        },
    }
    if "episode_success" in names:
        result["success_rate"] = sum(
            row["metrics"]["episode_success"] >= 0.5 for row in episodes
        ) / n
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=None,
                        help="Default: one full task episode; shorter windows are censored.")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    if args.seed == 0:
        parser.error("Use a held-out evaluation seed, not demonstration seed 0")
    for name in ("num_envs", "video_steps", "width", "height"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if not args.task.startswith(("Baoding-", "Pen-Spinning-")):
        parser.error("This evaluator supports Baoding-* and Pen-Spinning-* tasks")
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    import mjlab_hand  # noqa: F401
    from mjlab_hand.eval.env_setup import load_policy, make_runner

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no silent CPU fallback. Use --device cpu explicitly.")
    configure_torch_backends()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.checkpoint:
        signature = task_signature(args.task, load_env_cfg(args.task))
        check_checkpoint_task(args.checkpoint, args.task, signature)
    # Keep the training reset distribution (not the play view's fixed Baoding
    # phase) while turning observation corruption off for policy evaluation.
    cfg = load_env_cfg(args.task)
    cfg.observations["actor"].enable_corruption = False
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.auto_reset = False  # Read terminal task metrics before any resets.
    cfg.viewer.width, cfg.viewer.height = args.width, args.height
    rl_cfg = load_rl_cfg(args.task)
    command_name = "baoding" if args.task.startswith("Baoding-") else "pen_spin"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode = "policy" if args.checkpoint else "zero_action_diagnostic"
    video_path = args.output_dir / f"{mode}_seed{args.seed}_env0.mp4"
    from dataclasses import asdict
    config_path = args.output_dir / f"{mode}_seed{args.seed}_env.yaml"
    dump_yaml(config_path, asdict(cfg))

    with ExitStack() as resources:
        raw = ManagerBasedRlEnv(cfg, device=args.device,
                                render_mode="rgb_array" if args.video else None)
        resources.callback(raw.close)
        env = RslRlVecEnvWrapper(raw, clip_actions=rl_cfg.clip_actions)
        if args.checkpoint:
            runner = make_runner(args.task, env, rl_cfg, torch.device(args.device))
            policy = load_policy(runner, args.checkpoint, torch.device(args.device))
        else:
            def policy(obs):
                return torch.zeros((env.num_envs, env.num_actions), device=args.device)
        # Reset after runner construction/loading to fix the initial RNG stream.
        raw.reset(seed=args.seed)
        obs = env.get_observations()
        command = raw.command_manager.get_term(command_name)
        steps = args.steps or raw.max_episode_length
        returns = torch.zeros(env.num_envs, device=args.device)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=args.device)
        episodes: list[dict] = []
        writer = None
        frames = 0

        def record_frame() -> None:
            nonlocal writer, frames
            frame = raw.render()
            if frame is None:
                raise RuntimeError("Renderer returned no RGB image")
            frame = np.asarray(frame)
            if frame.ndim == 4:
                frame = frame[0]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            if writer is None:
                import mediapy
                writer = resources.enter_context(mediapy.VideoWriter(
                    str(video_path), shape=frame.shape[:2], fps=1.0 / raw.step_dt
                ))
            writer.add_image(frame)
            frames += 1

        if args.video:
            record_frame()
        with torch.inference_mode():
            for step in range(steps):
                obs, reward, dones, _ = env.step(policy(obs))
                returns += reward * alive
                ended = dones.bool()
                newly_ended = ended & alive
                last = step == steps - 1
                collect = alive if last else newly_ended
                for i in collect.nonzero(as_tuple=False).flatten().tolist():
                    episodes.append({
                        "env_id": i,
                        "duration_s": (step + 1) * raw.step_dt,
                        "return": float(returns[i]),
                        "terminated": bool(raw.reset_terminated[i]),
                        "truncated": bool(raw.reset_time_outs[i]),
                        "censored": not bool(ended[i]),
                        "metrics": {name: float(value[i]) for name, value in command.metrics.items()},
                    })
                if args.video and bool(alive[0]) and frames <= args.video_steps:
                    record_frame()
                alive &= ~ended
                if not alive.any() or last:
                    break
                if ended.any():
                    # Finished environments still simulate, but never contribute a second episode.
                    raw.reset(env_ids=ended.nonzero(as_tuple=False).flatten())
                    obs = env.get_observations()
                    if args.checkpoint:
                        runner.alg.actor.reset(dones)
        episodes.sort(key=lambda row: row["env_id"])
        payload = {
            "task": args.task, "mode": mode,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed, "num_envs": args.num_envs, "max_steps": steps,
            "step_dt": raw.step_dt, "protocol": "first fresh-reset episode per environment",
            "env_config": str(config_path),
            "video": str(video_path) if args.video else None,
            "video_env_id": 0 if args.video else None, "video_frames": frames,
            "summary": summarize(episodes), "per_episode": episodes,
        }
    path = args.output_dir / f"{mode}_seed{args.seed}.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Metrics: {path}")
    if args.video:
        print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
