#!/usr/bin/env python3
"""Ambient-pipeline manifests for any family / targets / sigmas / seeds, in
Bundle's recipe (MIGRATION.md section 5.2; CHANGES.md item 65). Generalizes
build_ambient_rotation_manifest.py.

Each run trains on the target's term-aligned pooled store
<Family>_pad5_scarce<Hand>_K50k (target 50k + the other four hands at 1M) with
--ambient-tmin = 0 for the target and sigma for the others, in the store's
HANDS order. sigma 0 = full co-training; sigma 100 = target-only (the others
are never admitted, so every batch is target data) at the same step budget.
Noise-first sampler, frozen family min/max, x0 clamp 1.0, family val store,
keep-last 3, target-only padded eval 32 x 1500 every ~10%, num_workers 0,
~784k steps.

Tasks are ordered (family, target, sigma, seed): with -N K and PACK=1, a job of
K nodes holds K consecutive runs (e.g. K = len(sigmas) * len(seeds) -> one
target per job).

  python scripts/build_ambient_manifest.py --families Grasp InHand-Rotation \\
      --sigmas 0 100 --seeds 0 1 --out slurm_jobs/cotrain_vs_target_manifest.json

--leave-one-out (CHANGES.md item 77): pre-train WITHOUT the target, i.e. --ambient-tmin 100 for the target and
0 for the other four hands, so the noise-first sampler never draws a target window (same store, same step
budget). Output <Task>_loo_seed<k>; --sigmas is ignored. The run still validates and evals on the target
(zero-shot monitoring); fine-tune from policy_latest.pt (the final epoch), not best_val, so no target data
picks the pre-train checkpoint. E22:

  python scripts/build_ambient_manifest.py --families InHand-Rotation --leave-one-out --seeds 0 1 2 \\
      --size 50kr --root outputs/diffusion/loo_r --out slurm_jobs/loo_pretrain_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from padded_grid import (  # noqa: E402
    HANDS,
    epochs_for,
    n_windows,
    sources,
    store_path,
    target_ambient_tmin,
)

SHORT = {"Grasp": "grasp", "InHand-Rotation": "rotation"}


def run(family: str, hand: str, sigma: int, seed: int, root: str, epochs: int, size: str = "50k",
        loo: bool = False) -> dict:
    task = f"{family}-{hand}"
    tmin = [100 if h == hand else 0 for h in HANDS] if loo else target_ambient_tmin(hand, sigma)
    return {
        "dataset": str(store_path(family, f"scarce{hand}_K{size}")),
        "output-dir": f"{root}/{task}_loo_seed{seed}" if loo else f"{root}/{task}_sigma{sigma}_seed{seed}",
        "num-epochs": epochs,
        "batch-size": 256,
        "lr": 1e-4,
        "obs-horizon": 2,
        "action-horizon": 8,
        "num-workers": 0,
        "seed": seed,
        "ambient-tmin": tmin,
        "ambient-sampler": "noise-first",
        "norm-mode": "frozen",
        "norm-artifact": f"configs/norm_{SHORT[family]}_minmax.json",
        "x0-clamp": 1.0,
        "val-dataset": f"data/mjlab_hand_demos/val/{SHORT[family]}_val_20k.zarr",
        "val-windows": 2048,
        "keep-last": 3,
        "save-every-epochs": epochs,
        "latest-every-epochs": max(1, epochs // 10),
        "eval-spec": json.dumps([{"task": task, "pad": True}]),
        "eval-every-epochs": max(1, epochs // 10),
        "eval-num-envs": 32,
        "eval-num-steps": 1500,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["Grasp", "InHand-Rotation"])
    ap.add_argument("--hands", nargs="+", default=HANDS)
    ap.add_argument("--sigmas", type=int, nargs="+", default=[0])
    ap.add_argument("--leave-one-out", action="store_true", help="exclude the target (tmin 100), others at 0")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--root", default="outputs/diffusion/ambient_ta")
    ap.add_argument("--size", default="50k",
                    help="target-hand subset: 50k (front of the 1M store) or 50kr (random draw, item 70)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tasks = []
    for fam in args.families:
        for hand in args.hands:
            cfg = f"scarce{hand}_K{args.size}"
            p = store_path(fam, cfg)
            w = n_windows(p) if p.exists() else sum(n_windows(s) for s in sources(fam, cfg))
            ep = epochs_for(w)
            sigmas = [0] if args.leave_one_out else args.sigmas
            tasks += [[run(fam, hand, s, k, args.root, ep, args.size, args.leave_one_out)]
                      for s in sigmas for k in args.seeds]
    args.out.write_text(json.dumps(tasks, indent=1) + "\n")
    print(f"[INFO] wrote {len(tasks)} runs -> {args.out}")


if __name__ == "__main__":
    main()
