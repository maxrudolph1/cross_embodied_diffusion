#!/usr/bin/env python3
"""Co-training vs solo manifests (CHANGES.md item 64): does pooling the other
four hands help a data-scarce target hand?

For each target hand of a family, two arms at the same ~784k-step budget:
  cotrain  term-aligned store <Family>_pad5_scarce<Hand>_K50k (target 50k + the
           other four hands at 1M), no ambient gating, Bundle's recipe:
           frozen family min/max normalizer, x0 clamp 1.0, held-out val store,
           eval spec [{"task": target, "pad": true}]
  solo     the target's own 50k store (subsets_50k), the plain single-hand
           path: shared min/max normalizer fit on that store, --eval-task
Both: num_workers 0, keep-last 3, eval every ~10%, 32 envs x 1500 steps; the
same val store (solo scores the target's native dims of it). Report from
rescore_selected.py (--which last0 best_rollout best_val, 100 envs, 1500
steps, eval seed 1234).

Bundle (MIGRATION.md section 7): co-train minus solo = grasp +0.46 (50k),
rotation -0.27 (50k).

  python scripts/build_cotrain_solo_manifest.py --families Grasp InHand-Rotation \\
      --seeds 0 1 2 --out slurm_jobs/cotrain_solo_manifest.json
Tasks are ordered (family, target, arm, seed), so PACK consecutive tasks share a
node within one family/target where possible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from padded_grid import (  # noqa: E402
    HANDS,
    demo_path,
    epochs_for,
    n_windows,
    sources,
    store_path,
)

SHORT = {"Grasp": "grasp", "InHand-Rotation": "rotation"}
DEMOS = "data/mjlab_hand_demos"


def common(family: str, epochs: int, seed: int) -> dict:
    return {
        "num-epochs": epochs,
        "batch-size": 256,
        "lr": 1e-4,
        "obs-horizon": 2,
        "action-horizon": 8,
        "num-workers": 0,
        "seed": seed,
        "x0-clamp": 1.0,
        "val-dataset": f"{DEMOS}/val/{SHORT[family]}_val_20k.zarr",
        "val-windows": 2048,
        "keep-last": 3,
        "save-every-epochs": epochs,
        "latest-every-epochs": max(1, epochs // 10),
        "eval-every-epochs": max(1, epochs // 10),
        "eval-num-envs": 32,
        "eval-num-steps": 1500,
    }


def cotrain_run(family: str, hand: str, seed: int, root: str) -> dict:
    cfg = f"scarce{hand}_K50k"
    p = store_path(family, cfg)
    w = n_windows(p) if p.exists() else sum(n_windows(s) for s in sources(family, cfg))
    task = f"{family}-{hand}"
    return {
        "dataset": str(p),
        "output-dir": f"{root}/{task}_cotrain_K50k_seed{seed}",
        **common(family, epochs_for(w), seed),
        "norm-mode": "frozen",
        "norm-artifact": f"configs/norm_{SHORT[family]}_minmax.json",
        "eval-spec": json.dumps([{"task": task, "pad": True}]),
    }


def solo_run(family: str, hand: str, seed: int, root: str) -> dict:
    p = demo_path(family, hand, "50k")
    task = f"{family}-{hand}"
    return {
        "dataset": str(p),
        "output-dir": f"{root}/{task}_solo_50k_seed{seed}",
        **common(family, epochs_for(n_windows(p)), seed),
        "norm-mode": "shared",
        "eval-task": task,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["Grasp", "InHand-Rotation"])
    ap.add_argument("--hands", nargs="+", default=HANDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", choices=["cotrain", "solo"], default=["cotrain", "solo"])
    ap.add_argument("--root", default="outputs/diffusion/cotrain_solo")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    make = {"cotrain": cotrain_run, "solo": solo_run}
    tasks = [
        [make[arm](fam, hand, seed, args.root)]
        for fam in args.families
        for hand in args.hands
        for arm in args.arms
        for seed in args.seeds
    ]
    args.out.write_text(json.dumps(tasks, indent=1) + "\n")
    eps = sorted({(t[0]["output-dir"].split("/")[-1].rsplit("_seed", 1)[0], t[0]["num-epochs"]) for t in tasks})
    print(f"[INFO] wrote {len(tasks)} runs -> {args.out}")
    for name, e in eps:
        print(f"  {name}: {e} epochs")


if __name__ == "__main__":
    main()
