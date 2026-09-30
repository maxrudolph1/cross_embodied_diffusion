#!/usr/bin/env python3
"""Pool N single-embodiment demo datasets into one zero-padded,
per-source-normalized dataset for cross-embodiment BC, written to a zarr.

The pooling itself (per-source static Gaussian normalization, zero-padding
to the max dims, concatenation, `extra.sources` provenance) lives in
`mjlab_hand.diffusion.pooling` -- see its docstring for the scheme. That
module is shared with `train-diffusion --dataset A.zarr B.zarr ...`, which
builds the same pool in memory at load time, so a padded zarr on disk is
optional (CHANGES.md item 56). This script is kept for inspecting or sharing
a fixed pool.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mjlab_hand.diffusion.dataset import TrajectoryStore
from mjlab_hand.diffusion.pooling import embodiment_name, pool_padded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, nargs="+", required=True, help="N source .zarr dirs")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--task-family", type=str, required=True, help="e.g. Grasp, InHand-Rotation")
    ap.add_argument("--success-only", action="store_true", default=True)
    ap.add_argument("--no-success-only", dest="success_only", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output} already exists (pass --overwrite)")
        shutil.rmtree(args.output)

    pool = pool_padded(args.sources, task_family=args.task_family, success_only=args.success_only)
    extra = pool["extra"]

    dst = TrajectoryStore(args.output, mode="w")
    dst.initialize(
        obs_dim=extra["max_obs_dim"],
        action_dim=extra["max_action_dim"],
        task=args.task_family,
        checkpoint=pool["checkpoint"],
        extra_meta=extra,
    )
    for name in ("obs", "action", "reward", "success", "episode_ends"):
        arr = pool[name]
        ds = dst.data[name]
        ds.resize((arr.shape[0], *arr.shape[1:]))
        ds[:] = arr

    print(json.dumps(dst.summary(), indent=2))
    print(f"[INFO] embodiments: {[embodiment_name(p) for p in args.sources]}")


if __name__ == "__main__":
    main()
