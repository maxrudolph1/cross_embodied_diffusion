#!/usr/bin/env python3
"""Fine-tune co-trained policies on their target hand only (CHANGES.md item 67).

Starts from each co-trained run's checkpoint (default policy_best_val.pt of
outputs/diffusion/ambient_ta/<Task>_sigma0_seed<k>, from
build_ambient_manifest.py) with `--init-checkpoint` (weights + the run's
normalizer, fresh optimizer) and trains on the SAME term-aligned pooled store
gated to the target only (`--ambient-tmin` 0 for the target, 100 for the
others, noise-first): every batch is target data, in the layout and scaling
the policy was trained with. One fine-tune per (family, target, co-train seed,
lr); the fine-tune uses the co-train seed.

Eval/val/checkpoints as the recipe (target-only padded eval 32 x 1500, family
val store, keep-last 3), evaluated every epoch. Report from
rescore_selected.py like the co-train runs.

  python scripts/build_finetune_manifest.py --lrs 1e-4 1e-5 --epochs 10 \\
      --out slurm_jobs/finetune_manifest.json

From ambient runs (CHANGES.md item 74): --src-sigmas picks the source runs
<Task>_sigma<S>_seed<k> (default: 0 only) and puts the source sigma in the
output name (<Task>_sigma<S>_ft_<init>_lr<lr>_seed<k>); --size picks the
target store scarce<Hand>_K<size> the sources were trained on. E21:

  python scripts/build_finetune_manifest.py --families InHand-Rotation \\
      --src-sigmas 0 15 100 --seeds 0 1 2 --lrs 1e-5 --size 50kr \\
      --src-root outputs/diffusion/ambient_ta_r --root outputs/diffusion/ambient_ta_r_ft \\
      --out slurm_jobs/finetune_r_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from padded_grid import HANDS, store_path, target_ambient_tmin  # noqa: E402

SHORT = {"Grasp": "grasp", "InHand-Rotation": "rotation"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["Grasp", "InHand-Rotation"])
    ap.add_argument("--hands", nargs="+", default=HANDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--lrs", type=float, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--init", default="policy_best_val.pt", help="checkpoint file in each co-train run dir")
    ap.add_argument("--src-sigmas", type=int, nargs="+", default=None,
                    help="source runs' sigma (default 0, old output names); given -> sigma in the output name")
    ap.add_argument("--size", default="50k", help="target store scarce<Hand>_K<size> (50kr: random draw)")
    ap.add_argument("--src-root", default="outputs/diffusion/ambient_ta")
    ap.add_argument("--root", default="outputs/diffusion/ambient_ta_ft")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    init = args.init.removeprefix("policy_").removesuffix(".pt")
    tasks = []
    for fam in args.families:
        for hand in args.hands:
            task = f"{fam}-{hand}"
            for seed in args.seeds:
                for sigma in args.src_sigmas or [0]:
                    for lr in args.lrs:
                        tag = f"_sigma{sigma}" if args.src_sigmas else ""
                        tasks.append([{
                            "dataset": str(store_path(fam, f"scarce{hand}_K{args.size}")),
                            "init-checkpoint": f"{args.src_root}/{task}_sigma{sigma}_seed{seed}/{args.init}",
                            "output-dir": f"{args.root}/{task}{tag}_ft_{init}_lr{lr:g}_seed{seed}",
                            "num-epochs": args.epochs,
                            "batch-size": 256,
                            "lr": lr,
                            "obs-horizon": 2,
                            "action-horizon": 8,
                            "num-workers": 0,
                            "seed": seed,
                            "ambient-tmin": target_ambient_tmin(hand, 100),
                            "ambient-sampler": "noise-first",
                            "norm-mode": "frozen",
                            "norm-artifact": f"configs/norm_{SHORT[fam]}_minmax.json",
                            "x0-clamp": 1.0,
                            "val-dataset": f"data/mjlab_hand_demos/val/{SHORT[fam]}_val_20k.zarr",
                            "val-windows": 2048,
                            "keep-last": 3,
                            "save-every-epochs": args.epochs,
                            "latest-every-epochs": 1,
                            "eval-spec": json.dumps([{"task": task, "pad": True}]),
                            "eval-every-epochs": 1,
                            "eval-num-envs": 32,
                            "eval-num-steps": 1500,
                        }])
    args.out.write_text(json.dumps(tasks, indent=1) + "\n")
    print(f"[INFO] wrote {len(tasks)} runs -> {args.out}")


if __name__ == "__main__":
    main()
