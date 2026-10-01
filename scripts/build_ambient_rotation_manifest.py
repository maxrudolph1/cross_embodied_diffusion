#!/usr/bin/env python3
"""Manifests for ambient diffusion on term-aligned multi-hand InHand-Rotation
stores, in Bundle's experiment recipe (MIGRATION.md section 5.2; CHANGES.md
item 63). One run per manifest task, for
`slurm_jobs/vista_train_manifest.sbatch` (PACK = runs per node).

Store: data/mjlab_hand_demos/padded_ta/InHand-Rotation_pad5_scarce<Hand>_K50k.zarr
(target hand 50k, the other four 1M; build with
`scripts/padded_grid.py --family InHand-Rotation --config scarce<Hand>_K50k --build`).
`--ambient-tmin` is one entry per source in the store's HANDS order (Allegro,
LEAP, Shadow, Sharpa, Wuji): 0 for the target, sigma for the others
(`padded_grid.target_ambient_tmin`). sigma = 0 is plain co-training, sigma =
100 target-only. Recipe flags: noise-first sampler, frozen min/max family
normalizer, x0 clamp 1.0, held-out val store, keep-last 3, target-only padded
eval spec at 1500 steps, num_workers 0, ~784k steps.

--kind sweep: 5 targets x 16 sigmas x 4 seeds = 320 runs; a (target, sigma)'s
4 seeds are consecutive (PACK=4 -> one config per node).
--kind diagnostic: Allegro x sigma {0, 2, 100} x seeds {0, 1, 2} = 9 runs, the
points Bundle reports for a 50k rotation target (0.655 / 1.137 / 0.767,
MIGRATION section 7).

Report from `scripts/rescore_selected.py --which best_rollout best_val last0
--envs 100 --steps 1500 --eval-seed 1234` (Bundle: report last0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from padded_grid import (  # noqa: E402
    epochs_for,
    n_windows,
    sources,
    store_path,
    target_ambient_tmin,
)

FAMILY = "InHand-Rotation"
HANDS = ["Allegro", "LEAP", "Shadow", "Sharpa", "Wuji"]
SIGMAS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 25, 100]
SEEDS = [0, 1, 2, 3]
NORM = "configs/norm_rotation_minmax.json"
VAL = "data/mjlab_hand_demos/val/rotation_val_20k.zarr"


def task(hand: str) -> str:
    return f"{FAMILY}-{hand}"


def windows(hand: str) -> int:
    """Training windows of the target's store (from the sources if not built)."""
    cfg = f"scarce{hand}_K50k"
    p = store_path(FAMILY, cfg)
    return n_windows(p) if p.exists() else sum(n_windows(s) for s in sources(FAMILY, cfg))


def run(hand: str, sigma: int, seed: int, root: str, epochs: int) -> dict:
    return {
        "dataset": str(store_path(FAMILY, f"scarce{hand}_K50k")),
        "output-dir": f"{root}/{task(hand)}_sigma{sigma}_seed{seed}",
        "num-epochs": epochs,
        "batch-size": 256,
        "lr": 1e-4,
        "obs-horizon": 2,
        "action-horizon": 8,
        "num-workers": 0,
        "seed": seed,
        "ambient-tmin": target_ambient_tmin(hand, sigma),
        "ambient-sampler": "noise-first",
        "norm-mode": "frozen",
        "norm-artifact": NORM,
        "x0-clamp": 1.0,
        "val-dataset": VAL,
        "val-windows": 2048,
        "keep-last": 3,
        "save-every-epochs": epochs,
        "latest-every-epochs": max(1, epochs // 10),
        "eval-spec": json.dumps([{"task": task(hand), "pad": True}]),
        "eval-every-epochs": max(1, epochs // 10),
        "eval-num-envs": 32,
        "eval-num-steps": 1500,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["sweep", "diagnostic"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.kind == "sweep":
        grid, root = [(h, s, k) for h in HANDS for s in SIGMAS for k in SEEDS], "outputs/diffusion/ambient_rot_ta"
    else:
        grid, root = [("Allegro", s, k) for s in (0, 2, 100) for k in (0, 1, 2)], "outputs/diffusion/diag_rot_ta"
    epochs = {h: epochs_for(windows(h)) for h in sorted({g[0] for g in grid})}
    tasks = [[run(h, s, k, root, epochs[h])] for h, s, k in grid]
    args.out.write_text(json.dumps(tasks, indent=1) + "\n")
    print(f"[INFO] wrote {len(tasks)} runs -> {args.out}; epochs per target {epochs}")


if __name__ == "__main__":
    main()
