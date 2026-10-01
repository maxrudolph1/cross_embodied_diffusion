#!/usr/bin/env python
"""CLI: train a diffusion policy on a collected demo dataset (CHANGES.md item 63)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="One store: a single-hand demo zarr, or a term-aligned padded multi-hand zarr "
        "(scripts/build_padded_dataset.py / padded_grid.py).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers. The experiment recipe uses 0 (workers recreated every epoch "
        "deadlocked in Bundle; persistent workers would change the shuffle stream).",
    )
    parser.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=10,
        help="Accepted for compatibility; not used (snapshots are --keep-last rolling ones).",
    )
    parser.add_argument(
        "--latest-every-epochs",
        type=int,
        default=1,
        help="Cadence of policy_latest.pt and the rolling policy_last{k}_epoch{N}.pt snapshots "
        "(always also on the final epoch).",
    )
    parser.add_argument("--keep-last", type=int, default=3, help="Rolling snapshots to keep (last0 = newest).")
    parser.add_argument(
        "--eval-task",
        type=str,
        default=None,
        help="If set, run env eval every --eval-every-epochs (single-hand policy).",
    )
    parser.add_argument(
        "--eval-spec",
        type=str,
        default=None,
        help='JSON list of {"task": ..., "pad": true} (term-aligned multi-hand policy) or '
        '{"task": ..., "onehot": [..]} (2-hand onehot mixtures). Overrides --eval-task.',
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
        help="One t_min per source, in the store's source order (HANDS order for padded "
        "stores; padded_grid.target_ambient_tmin). Source i trains only at t >= t_min[i].",
    )
    parser.add_argument(
        "--ambient-sampler",
        choices=["data-first", "noise-first"],
        default="data-first",
        help="data-first: window then t ~ U[t_min, T) (starves low-noise steps); noise-first: "
        "t then a window admitted at t. Ambient experiments should pass noise-first.",
    )
    parser.add_argument(
        "--mask-pad-loss",
        action="store_true",
        help="Exclude padded action channels from the loss and hold them fixed when sampling.",
    )
    parser.add_argument(
        "--norm-mode",
        choices=["shared", "pad-aware", "zscore", "frozen"],
        default="shared",
        help="shared: min/max over the store incl. padding; pad-aware: per column over rows "
        "using it; zscore: mean/std (needs --x0-clamp > 1); frozen: --norm-artifact.",
    )
    parser.add_argument("--norm-clip-pct", type=float, default=None, help="pad-aware only: percentile clip.")
    parser.add_argument("--norm-artifact", type=Path, default=None, help="frozen only: norm_<family>_*.json.")
    parser.add_argument(
        "--x0-clamp",
        type=float,
        default=1.0,
        help="Sampler clamps its x0 estimate to +-this in normalized units (1.0 = min/max data range).",
    )
    parser.add_argument(
        "--val-dataset",
        type=Path,
        default=None,
        help="Separate term-aligned val store (scripts/build_val_split.py); enables policy_best_val.pt.",
    )
    parser.add_argument("--val-windows", type=int, default=2048)
    parser.add_argument("--wandb-project", type=str, default=None, help="Optional WandB logging.")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=None)
    args = parser.parse_args()

    from mjlab_hand.diffusion.train import TrainConfig, check_config, train_diffusion

    cfg = TrainConfig(
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
        keep_last=args.keep_last,
        eval_task=args.eval_task,
        eval_specs=json.loads(args.eval_spec) if args.eval_spec is not None else None,
        eval_every_epochs=args.eval_every_epochs,
        eval_num_envs=args.eval_num_envs,
        eval_num_steps=args.eval_num_steps,
        render_every_epochs=args.render_every_epochs,
        render_num_steps=args.render_num_steps,
        render_num_envs=args.render_num_envs,
        ambient_tmin=args.ambient_tmin,
        ambient_sampler=args.ambient_sampler,
        mask_pad_loss=args.mask_pad_loss,
        norm_mode=args.norm_mode,
        norm_clip_pct=args.norm_clip_pct,
        norm_artifact=args.norm_artifact,
        x0_clamp=args.x0_clamp,
        val_dataset=args.val_dataset,
        val_windows=args.val_windows,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )
    try:
        check_config(cfg)
    except ValueError as e:
        parser.error(str(e))
    train_diffusion(cfg)


if __name__ == "__main__":
    main()
