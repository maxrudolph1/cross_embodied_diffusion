#!/usr/bin/env python
"""CLI: evaluate a trained diffusion policy in simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from mjlab_hand.diffusion.evaluate import evaluate_diffusion_policy

    metrics = evaluate_diffusion_policy(
        task=args.task,
        policy_path=args.policy,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
