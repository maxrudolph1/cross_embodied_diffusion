#!/usr/bin/env python3
"""E21: target-only fine-tuning from ambient-gated starting points (CHANGES.md item 76).

For each target hand and each source sigma (default 0, 15, 100): the starting policy (the E19 run's best_val
checkpoint) and the same policy after fine-tuning on the target (`--ft-which` checkpoint of
`<Task>_sigma<S>_ft_best_val_lr<lr>_seed<k>`). One panel per hand plus a mean-over-hands panel; dots are
training seeds, the line joins the seed means.

  --what target   score on the run's own hand, from final_eval.jsonl
  --what others   forgetting: mean over the four OTHER hands, from cross_eval.jsonl (eval_cross_embodiment.py)

Rows: 100 envs x 1500 steps at `--eval-seed` (4321 = held-out test seed, never used for selection). Runs without
the needed rows are skipped. Writes <out>/<prefix>_<what>_<metric>.png and <prefix>_<what>_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HANDS = ["Allegro", "LEAP", "Shadow", "Sharpa", "Wuji"]
LABEL = {"avg_successes_before_drop": "successes before drop", "success_rate_any": "episodes reaching >= 1 target",
         "per_target_success_rate": "per-target success rate", "avg_survival_time_s": "survival time (s)",
         "avg_rot_dist": "final rotation distance (rad, lower is better)", "drop_rate": "drop rate (lower is better)"}
# reference categorical slots 1-2, fixed order (validator needs node, absent on Vista)
SERIES = [("src", "starting policy (co-trained at σ, best_val)", "#2a78d6", "o"),
          ("ft", "after target-only fine-tune", "#eb6834", "s")]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SCRATCH_OUT = "$SCRATCH/cross_embodied_diffusion/outputs/diffusion"
FAMILY = "InHand-Rotation"


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def value(run: Path, which: str, own: str, what: str, seed: int, metric: str) -> float | None:
    if what == "target":
        rs = [r for r in rows(run / "final_eval.jsonl") if r["which"] == which and r["task"] == own
              and r["eval_seed"] == seed and r["envs"] == 100 and r["steps"] == 1500]
        return float(rs[-1]["metrics"][metric]) if rs else None
    per = {}
    for r in rows(run / "cross_eval.jsonl"):
        if r["which"] == which and r["task"] != own and r["eval_seed"] == seed and r["envs"] == 100:
            per[r["task"]] = float(r["metrics"][metric])
    return st.mean(per.values()) if len(per) == len(HANDS) - 1 else None


def collect(args) -> list[dict]:
    recs = []
    for hand in HANDS:
        own = f"{FAMILY}-{hand}"
        for s in args.sigmas:
            for k in args.seeds:
                src = args.runs / f"{own}_sigma{s}_seed{k}"
                ft = args.ft_runs / f"{own}_sigma{s}_ft_best_val_lr{args.lr}_seed{k}"
                for kind, run, which in (("src", src, "best_val"), ("ft", ft, args.ft_which)):
                    v = value(run, which, own, args.what, args.eval_seed, args.metric)
                    if v is not None:
                        recs.append({"target": hand, "sigma": s, "seed": k, "kind": kind, "which": which,
                                     "value": v, "run": run.name})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path(os.path.expandvars(f"{SCRATCH_OUT}/ambient_ta_r")))
    ap.add_argument("--ft-runs", type=Path, default=Path(os.path.expandvars(f"{SCRATCH_OUT}/ambient_ta_r_ft")))
    ap.add_argument("--sigmas", type=int, nargs="+", default=[0, 15, 100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", default="1e-05")
    ap.add_argument("--ft-which", default="last0")
    ap.add_argument("--what", choices=["target", "others"], default="target")
    ap.add_argument("--metric", default="avg_successes_before_drop", choices=list(LABEL))
    ap.add_argument("--eval-seed", type=int, default=4321)
    ap.add_argument("--out", type=Path, default=Path("outputs/plots"))
    ap.add_argument("--prefix", default="finetune_sources_rotation")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    recs = collect(args)
    if not recs:
        raise SystemExit("no scored runs found")
    stem = f"{args.prefix}_{args.what}"
    with open(args.out / f"{stem}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "sigma", "seed", "kind", "which", "value", "run"])
        w.writeheader()
        w.writerows(recs)

    panels = HANDS + ["mean over hands"]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.9), sharey=True, facecolor=SURFACE)
    xs = np.arange(len(args.sigmas))
    rng = np.random.default_rng(0)
    for ax, panel in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for si, (kind, name, color, marker) in enumerate(SERIES):
            off = (si - 0.5) * 0.18
            means = []
            for xi, s in enumerate(args.sigmas):
                if panel == "mean over hands":
                    # per seed: mean over the hands that have this (sigma, seed, kind)
                    vals = []
                    for k in args.seeds:
                        hv = [r["value"] for r in recs if r["kind"] == kind and r["sigma"] == s and r["seed"] == k]
                        if len(hv) == len(HANDS):
                            vals.append(st.mean(hv))
                else:
                    vals = [r["value"] for r in recs if r["kind"] == kind and r["sigma"] == s and r["target"] == panel]
                means.append(st.mean(vals) if vals else np.nan)
                if vals:
                    ax.scatter(xi + off + rng.uniform(-0.04, 0.04, len(vals)), vals, s=14, color=color, alpha=0.45,
                               linewidths=0, zorder=2)
            ax.plot(xs + off, means, color=color, linewidth=2, marker=marker, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label=name)
        ax.set_title(panel, fontsize=10.5, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"σ {s}" for s in args.sigmas], fontsize=9, color=INK2)
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.tick_params(colors=INK2, length=0, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    axes[0].set_ylabel(LABEL[args.metric] + ("\n(mean over the 4 other hands)" if args.what == "others" else ""),
                       color=INK2, fontsize=9)
    axes[0].set_ylim(bottom=0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"{lab[0]}", f"{lab[1]} ({args.ft_which})"], loc="upper left", bbox_to_anchor=(0.005, 0.89),
               ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    where = "on the target hand" if args.what == "target" else "on the four other hands (forgetting)"
    fig.suptitle(f"{FAMILY}: fine-tuning from co-trained (σ 0), ambient (σ 15) and target-only (σ 100) starting "
                 f"policies, {where}", x=0.01, ha="left", color=INK, fontsize=11.5)
    n = len({(r['target'], r['sigma'], r['seed']) for r in recs if r['kind'] == 'ft'})
    fig.text(0.01, 0.015, f"100 episodes per run at held-out eval seed {args.eval_seed}; dots = training seeds "
             f"({n} fine-tunes scored), line = seed mean. Target data: 50k random draw (_K50kr). Fine-tune: lr "
             f"{float(args.lr):g}, 10 epochs, target-only gating.", fontsize=7.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    out = args.out / f"{stem}_{args.metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] -> {out} ({len(recs)} values)")


if __name__ == "__main__":
    main()
