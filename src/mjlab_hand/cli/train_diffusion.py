#!/usr/bin/env python
"""CLI: train a diffusion policy on a collected demo dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        required=True,
        help="One dataset zarr, or several single-embodiment zarrs, which are pooled in memory "
        "into a padded cross-embodiment dataset (same result as build_padded_dataset.py).",
    )
    parser.add_argument(
        "--pool-sources",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force (or forbid) the in-memory padded pooling. Default: pool iff >1 dataset. "
        "--pool-sources with one dataset trains it through the padded path.",
    )
    parser.add_argument(
        "--source-norm",
        choices=["gaussian", "minmax"],
        default="gaussian",
        help="Per-source normalization for pooled runs: mean/std, or min/max to [-1, 1] "
        "(what the plain path's LinearNormalizer does).",
    )
    parser.add_argument(
        "--task-family",
        type=str,
        default=None,
        help="task_family recorded in source_stats.json for a pooled run (default: inferred).",
    )
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
        help="Cadence for writing policy_latest.pt / policy_best_val.pt (always saved on the final "
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
        "--val-fraction",
        type=float,
        default=0.0,
        help="Fraction of trajectories per source held out for a validation "
        "loss (denoised-action MSE on held-out episodes, not held-out "
        "individual states). 0 (default) disables validation.",
    )
    parser.add_argument("--val-seed", type=int, default=0)
    parser.add_argument(
        "--val-embodiment",
        type=str,
        default=None,
        help="Compute the val loss (and pick policy_best_val.pt) on this source of a pooled "
        "dataset only, e.g. the target hand of a scarce/ambient run.",
    )
    parser.add_argument("--val-every-epochs", type=int, default=1)
    parser.add_argument(
        "--val-max-batches",
        type=int,
        default=20,
        help="Cap on validation batches per check (DDIM sampling is far "
        "costlier per batch than a training step). Pass -1 to use the "
        "whole val set every time.",
    )
    parser.add_argument(
        "--source-sample-mode",
        choices=["uniform", "balanced"],
        default="uniform",
        help="Per-source sampling ratio for a padded/mixed dataset. 'uniform' "
        "(default): proportional to row count. 'balanced': equal expected "
        "representation per source per epoch regardless of size.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="If set, log this run to WandB under this project name.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default=None,
        help="torch.compile the denoising UNet with this mode (default: eager).",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=None)
    args = parser.parse_args()

    from mjlab_hand.diffusion.train import TrainConfig, train_diffusion

    eval_specs = json.loads(args.eval_spec) if args.eval_spec is not None else None

    train_diffusion(
        TrainConfig(
            dataset=args.dataset[0] if len(args.dataset) == 1 else args.dataset,
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
            val_fraction=args.val_fraction,
            val_seed=args.val_seed,
            val_embodiment=args.val_embodiment,
            val_every_epochs=args.val_every_epochs,
            val_max_batches=None if args.val_max_batches < 0 else args.val_max_batches,
            source_sample_mode=args.source_sample_mode,
            compile_mode=args.compile_mode,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_tags=args.wandb_tags,
            pool_sources=args.pool_sources,
            task_family=args.task_family,
            source_norm=args.source_norm,
        )
    )


if __name__ == "__main__":
    main()
