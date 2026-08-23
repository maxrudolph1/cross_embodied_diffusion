#!/usr/bin/env python
"""CLI: collect expert demos from an RL checkpoint into a zarr dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--success-tolerance", type=float, default=0.03)
    parser.add_argument("--success-steps", type=int, default=10)
    args = parser.parse_args()

    from mjlab_hand.diffusion.collect import collect_demos

    if args.output.exists():
        import shutil

        shutil.rmtree(args.output)

    summary = collect_demos(
        task=args.task,
        checkpoint=args.checkpoint,
        output=args.output,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        device=args.device,
        seed=args.seed,
        keep_failures=args.keep_failures,
        success_tolerance=args.success_tolerance,
        success_steps=args.success_steps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
