#!/usr/bin/env python3
"""Per-target box-and-whisker comparison of training regimes, one figure per eval metric
(CHANGES.md item 72).

For each target hand of a family, four boxes, each over individual runs (dots = runs):
  1. sigma = 0        full co-training                      (--runs)
  2. 0 < sigma < 100  every intermediate ambient gate pooled  (--runs)
  3. sigma = 100      target-only                           (--runs)
  4. fine-tuned       co-trained, then fine-tuned on the target (--ft-runs; every lr and seed)
plus an "all targets" cluster pooling the five hands.

Defaults: the random-draw 50k sweep `ambient_ta_r` (E18/E19) and the fine-tunes `ambient_ta_ft` (E15).
The fine-tunes were trained on the OLD first-episodes 50k subsets (`_K50k`), not the random draws, so
box 4 is not on exactly the same target data as boxes 1-3; the figure says so.

Score: the --which checkpoint (default best_val) from `<run>/final_eval.jsonl`, 100 envs x 1500 steps,
eval seed 1234 (rescore_selected.py). Runs without that row are skipped, so re-running after more runs
are scored adds them.

Writes <out>/<prefix>_<metric>.png per metric and <prefix>_summary.csv (every value).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HANDS = ["Allegro", "LEAP", "Shadow", "Sharpa", "Wuji"]
# (key, title, y label, better direction)
METRICS = {
    "InHand-Rotation": [
        ("avg_successes_before_drop", "Successes before drop", "rotations completed before the object drops", "higher"),
        ("success_rate_any", "Episodes reaching at least one target", "success rate", "higher"),
        ("per_target_success_rate", "Per-target success rate", "targets reached / targets offered", "higher"),
        ("avg_survival_time_s", "Survival time before drop", "seconds", "higher"),
        ("avg_rot_dist", "Rotation distance to target at episode end", "radians", "lower"),
        ("drop_rate", "Drop rate", "fraction of episodes ending in a drop", "lower"),
    ],
    "Grasp": [("success_rate", "Success rate", "success rate", "higher")],
}
GROUPS = [("s0", "σ = 0 (co-training)"), ("mid", "0 < σ < 100 (ambient gating)"),
          ("s100", "σ = 100 (target-only)"), ("ft", "co-trained + fine-tuned")]
# dataviz reference categorical slots 1-4, fixed order (validator needs node, absent on Vista;
# groups are also told apart by their fixed position in each cluster and the legend)
COLOR = dict(zip([g for g, _ in GROUPS], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SCRATCH_OUT = "$SCRATCH/cross_embodied_diffusion/outputs/diffusion"
SIGMA_RE = re.compile(r"(Grasp|InHand-Rotation)-(\w+?)_sigma(\d+)_seed(\d+)$")
FT_RE = re.compile(r"(Grasp|InHand-Rotation)-(\w+?)_ft_\w+?_lr([0-9.e-]+)_seed(\d+)$")


def final_metrics(run: Path, which: str) -> dict | None:
    fe = run / "final_eval.jsonl"
    if not fe.exists():
        return None
    rows = [json.loads(x) for x in fe.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["which"] == which and r["envs"] == 100 and r["steps"] == 1500]
    return rows[-1]["metrics"] if rows else None


def collect(runs: Path, ft_runs: Path, family: str, which: str) -> list[dict]:
    recs = []
    for run in sorted(runs.glob(f"{family}-*_sigma*_seed*")):
        m = SIGMA_RE.search(run.name)
        met = final_metrics(run, which) if m and m.group(1) == family else None
        if met is None:
            continue
        s = int(m.group(3))
        group = "s0" if s == 0 else "s100" if s == 100 else "mid"
        recs.append({"target": m.group(2), "group": group, "sigma": s, "seed": int(m.group(4)), "lr": "",
                     "run": run.name, "metrics": met})
    for run in sorted(ft_runs.glob(f"{family}-*_ft_*")):
        m = FT_RE.search(run.name)
        met = final_metrics(run, which) if m and m.group(1) == family else None
        if met is None:
            continue
        recs.append({"target": m.group(2), "group": "ft", "sigma": "", "seed": int(m.group(4)), "lr": m.group(3),
                     "run": run.name, "metrics": met})
    return recs


def figure(recs: list[dict], family: str, metric: tuple, which: str, out: Path) -> None:
    key, title, ylabel, better = metric
    clusters = HANDS + ["all targets"]
    fig, ax = plt.subplots(figsize=(13, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    width, step = 0.17, 0.2
    rng = np.random.default_rng(0)
    centers = np.arange(len(clusters), dtype=float)
    centers[-1] += 0.35  # gap before the pooled cluster
    for ci, cl in enumerate(clusters):
        for gi, (g, _) in enumerate(GROUPS):
            vals = [float(r["metrics"][key]) for r in recs
                    if r["group"] == g and (cl == "all targets" or r["target"] == cl) and key in r["metrics"]]
            if not vals:
                continue
            x = centers[ci] + (gi - 1.5) * step
            if len(vals) >= 2:
                bp = ax.boxplot([vals], positions=[x], widths=width, patch_artist=True, showfliers=False,
                                whis=(0, 100), zorder=2,
                                medianprops={"color": INK, "linewidth": 1.6},
                                whiskerprops={"color": COLOR[g], "linewidth": 1.2},
                                capprops={"color": COLOR[g], "linewidth": 1.2},
                                boxprops={"linewidth": 0})
                for patch in bp["boxes"]:
                    patch.set_facecolor(COLOR[g])
                    patch.set_alpha(0.35)
            jit = rng.uniform(-width * 0.32, width * 0.32, len(vals)) if len(vals) > 1 else np.zeros(1)
            size = 34 if len(vals) == 1 else 16 if cl != "all targets" else 9
            ax.scatter(x + jit, vals, s=size, color=COLOR[g], edgecolors=SURFACE,
                       linewidths=0.8, zorder=3)
    ax.set_xticks(centers)
    ax.set_xticklabels(clusters, fontsize=10, color=INK)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_ylabel(f"{ylabel}  ({better} is better)", color=INK2, fontsize=9)
    ax.set_ylim(bottom=0)
    if key in ("success_rate_any", "per_target_success_rate", "drop_rate", "success_rate"):
        ax.set_ylim(0, 1.05)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[g], alpha=0.6, linewidth=0) for g, _ in GROUPS]
    n = {g: sum(r["group"] == g for r in recs) for g, _ in GROUPS}
    labels = [f"{name} (n={n[g]})" for g, name in GROUPS]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.9), ncol=4, frameon=False,
               fontsize=8.5, labelcolor=INK2)
    fig.suptitle(f"{family}: {title}, per target hand  (checkpoint {which}; 100 episodes, eval seed 1234)",
                 x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.text(0.01, 0.015,
             "Boxes: median, quartiles, whiskers = min/max; dots = individual runs (n = runs across all targets; a lone "
             "dot is a single run).\nσ runs: target 50k random draw + four hands at 1M. Fine-tuned: co-trained (σ=0) "
             "then fine-tuned on the target, lr 1e-4 and 1e-5, 2 seeds, on the older first-episodes 50k subset (E15).",
             fontsize=7.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="InHand-Rotation", choices=list(METRICS))
    ap.add_argument("--runs", type=Path, default=Path(os.path.expandvars(f"{SCRATCH_OUT}/ambient_ta_r")))
    ap.add_argument("--ft-runs", type=Path, default=Path(os.path.expandvars(f"{SCRATCH_OUT}/ambient_ta_ft")))
    ap.add_argument("--which", default="best_val")
    ap.add_argument("--out", type=Path, default=Path("outputs/plots"))
    ap.add_argument("--prefix", default=None, help="default: <family short>_conditions_box")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    short = "rotation" if args.family == "InHand-Rotation" else "grasp"
    prefix = args.prefix or f"{short}_conditions_box"

    recs = collect(args.runs, args.ft_runs, args.family, args.which)
    if not recs:
        raise SystemExit("no scored runs found")
    keys = [m[0] for m in METRICS[args.family]]
    with open(args.out / f"{prefix}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "group", "sigma", "seed", "lr", "run"] + keys)
        w.writeheader()
        for r in recs:
            w.writerow({**{k: r[k] for k in ("target", "group", "sigma", "seed", "lr", "run")},
                        **{k: r["metrics"].get(k) for k in keys}})
    counts = defaultdict(int)
    for r in recs:
        counts[r["group"]] += 1
    for metric in METRICS[args.family]:
        out = args.out / f"{prefix}_{metric[0]}.png"
        figure(recs, args.family, metric, args.which, out)
        print(f"[INFO] -> {out}")
    print(f"[INFO] runs per group: {dict(counts)}")


if __name__ == "__main__":
    main()
