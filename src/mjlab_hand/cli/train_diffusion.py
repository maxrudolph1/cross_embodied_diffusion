#!/usr/bin/env python
"""CLI: train a diffusion policy on a collected demo dataset."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--save-every-epochs", type=int, default=10)
    parser.add_argument(
        "--latest-every-epochs",
        type=int,
        default=1,
        help="Cadence for writing policy_latest.pt / policy_best.pt (always saved on the final "
        "epoch and before any eval/render).",
    )
    parser.add_argument(
        "--eval-task",
        type=str,
        default=None,
        help="If set, run env eval every --eval-every-epochs during training",
    )
    parser.add_argument(
        "--eval-spec",
        type=str,
        default=None,
        help='JSON list of {"task": ..., "onehot": [..] | null} for multi-target eval '
        "(e.g. mixed-embodiment policies). Overrides --eval-task.",
    )
    parser.add_argument("--eval-every-epochs", type=int, default=10)
    parser.add_argument("--eval-num-envs", type=int, default=32)
    parser.add_argument("--eval-num-steps", type=int, default=1500)
    parser.add_argument("--render-every-epochs", type=int, default=0)
    parser.add_argument("--render-num-steps", type=int, default=400)
    parser.add_argument("--render-num-envs", type=int, default=1)
    parser.add_argument(
        "--ambient-tmin",
        type=int,
        nargs="+",
        default=None,
        help="One t_min per source (dataset order) for a mixed dataset, e.g. "
        "'--ambient-tmin 0 50' admits source 0 everywhere and source 1 only at t >= 50.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="If set, log this run to WandB under this project name.",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=None)
    args = parser.parse_args()

    from mjlab_hand.diffusion.train import TrainConfig, train_diffusion

    eval_specs = json.loads(args.eval_spec) if args.eval_spec is not None else None

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
            save_every_epochs=args.save_every_epochs,
            latest_every_epochs=args.latest_every_epochs,
            eval_task=args.eval_task,
            eval_specs=eval_specs,
            eval_every_epochs=args.eval_every_epochs,
            eval_num_envs=args.eval_num_envs,
            eval_num_steps=args.eval_num_steps,
            render_every_epochs=args.render_every_epochs,
            render_num_steps=args.render_num_steps,
            render_num_envs=args.render_num_envs,
            ambient_tmin=args.ambient_tmin,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_tags=args.wandb_tags,
        )
    )


if __name__ == "__main__":
    main()
