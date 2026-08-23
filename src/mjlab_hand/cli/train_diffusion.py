#!/usr/bin/env python
"""CLI: train a diffusion policy on a collected demo dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-task",
        type=str,
        default=None,
        help="If set, run env eval every --eval-every-epochs during training",
    )
    parser.add_argument("--eval-every-epochs", type=int, default=10)
    parser.add_argument("--eval-num-envs", type=int, default=32)
    parser.add_argument("--eval-num-steps", type=int, default=1500)
    args = parser.parse_args()

    from mjlab_hand.diffusion.train import TrainConfig, train_diffusion

    train_diffusion(
        TrainConfig(
            dataset=args.dataset,
            output_dir=args.output_dir,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=args.device,
            num_workers=args.num_workers,
            success_only=args.success_only,
            seed=args.seed,
            eval_task=args.eval_task,
            eval_every_epochs=args.eval_every_epochs,
            eval_num_envs=args.eval_num_envs,
            eval_num_steps=args.eval_num_steps,
        )
    )


if __name__ == "__main__":
    main()
