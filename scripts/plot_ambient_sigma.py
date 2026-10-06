#!/usr/bin/env python3
"""Ambient sigma sweep: target-hand score vs sigma, one panel per target hand plus
the mean over targets (CHANGES.md item 71).

Runs: build_ambient_manifest.py output, <runs>/<Family>-<Hand>_sigma<s>_seed<k>/
(default: the random-draw 50k sweep, outputs/diffusion/ambient_ta_r, CHANGES item 70).
--ref overlays the old first-episodes-subset runs (outputs/diffusion/ambient_ta,
sigma 0/100 only) as separate points, to check that the subset draw does not
change the co-train vs target-only comparison.

Score per run, checkpoint best_val:
  - <run>/final_eval.jsonl row which=best_val, 100 envs x 1500 steps (rescore_selected.py,
    eval seed 1234) when present;
  - else PROVISIONAL: the in-training eval (32 envs) at the epoch with the lowest val loss
    so far. Works on unfinished runs (no selection.json needed). In-training evals use the
    train seed (0); with the random-draw subsets 1-2 of 32 eval starts are training starts
    (docs/COLLECTIONS.md), with the old subsets 30-32 of 32 -- so provisional points
    of --ref runs are not comparable and are not drawn.

Writes <out>/ambient_sigma_<family>.png and ambient_sigma_<family>_summary.csv (every
value: the table view); copy the final ones into docs/plots/ (docs/experiment_log_book.md E18). For rotation also ambient_sigma_rotation_metrics.png: one panel per
eval metric (successes before drop, success rate on >= 1 target, per-target success rate,
survival time, drop rate, final rotation distance), one line per target hand + the mean over
targets, sweep runs only; every metric is in the summary CSV.
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
FAMILIES = {"Grasp": ("grasp", "success_rate", "success rate"),
            "InHand-Rotation": ("rotation", "avg_successes_before_drop", "successes before drop")}
# dataviz reference categorical palette, slots 1-2 (validated all-pairs; as plot_cotrain_vs_target.py)
SWEEP, REF = "#2a78d6", "#eb6834"
# Per-hand lines: reference categorical slots 1-5 in fixed order. The validator could not be run
# (no node on Vista), so each hand also gets its own marker; 5 series -> legend, no end labels.
HAND_COLOR = dict(zip(HANDS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
HAND_MARKER = dict(zip(HANDS, ["o", "s", "^", "D", "v"]))
# (key, panel title, better direction) -- rotation eval metrics (eval/rotation.py report())
ROT_METRICS = [
    ("avg_successes_before_drop", "successes before drop (rotations completed)", "higher"),
    ("success_rate_any", "episodes reaching >= 1 target (success rate)", "higher"),
    ("per_target_success_rate", "per-target success rate", "higher"),
    ("avg_survival_time_s", "survival time before drop (s)", "higher"),
    ("drop_rate", "drop rate (episodes ending in a drop)", "lower"),
    ("avg_rot_dist", "rotation distance to target at end (rad)", "lower"),
]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RUN_RE = re.compile(r"(Grasp|InHand-Rotation)-(\w+)_sigma(\d+)_seed(\d+)$")
SCRATCH_OUT = "$SCRATCH/cross_embodied_diffusion/outputs/diffusion"


def score(run: Path) -> tuple[dict, str] | None:
    """(metrics, source) for the run's best_val checkpoint; None if nothing to score yet."""
    fe = run / "final_eval.jsonl"
    if fe.exists():
        rows = [json.loads(x) for x in fe.read_text().splitlines() if x.strip()]
        for r in rows:
            if r["which"] == "best_val" and r["envs"] == 100 and r["steps"] == 1500:
                return r["metrics"], "rescore"
    em = run / "eval_metrics.jsonl"
    if not em.exists():
        return None
    rows = [json.loads(x) for x in em.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("val") and np.isfinite(r["val"]["score"])]
    if not rows:
        return None
    best = min(rows, key=lambda r: r["val"]["score"])
    return best["metrics"], "provisional"


def collect(root: Path, family: str, metric: str, provisional_ok: bool) -> list[dict]:
    recs = []
    for run in sorted(root.glob(f"{family}-*_sigma*_seed*")):
        m = RUN_RE.search(run.name)
        if not m or m.group(1) != family:
            continue
        s = score(run)
        if s is None or (s[1] == "provisional" and not provisional_ok):
            continue
        recs.append({"target": m.group(2), "sigma": int(m.group(3)), "seed": int(m.group(4)),
                     "value": float(s[0][metric]), "source": s[1], "metrics": s[0], "done": (run / "selection.json").exists(),
                     "run": run.name})
    return recs


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2, length=0, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="InHand-Rotation", choices=list(FAMILIES))
    ap.add_argument("--runs", type=Path, default=Path(os.path.expandvars(f"{SCRATCH_OUT}/ambient_ta_r")))
    ap.add_argument("--ref", type=Path, default=Path(os.path.expandvars(f"{SCRATCH_OUT}/ambient_ta")),
                    help="old first-episodes-subset runs to overlay (rescored points only); '' to skip")
    ap.add_argument("--label", default="random 50k draw", help="legend name of the sweep")
    ap.add_argument("--ref-label", default="first-100-episodes 50k (old)", help="legend name of --ref")
    ap.add_argument("--out", type=Path, default=Path("outputs/plots"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    short, metric, ylabel = FAMILIES[args.family]

    sweep = collect(args.runs, args.family, metric, provisional_ok=True)
    ref = collect(args.ref, args.family, metric, provisional_ok=False) if str(args.ref) not in ("", ".") else []
    if not sweep:
        raise SystemExit(f"no {args.family} runs with evals under {args.runs}")

    with open(args.out / f"ambient_sigma_{short}_summary.csv", "w", newline="") as f:
        mkeys = sorted({k for r in sweep + ref for k, v in r["metrics"].items() if isinstance(v, (int, float))})
        w = csv.DictWriter(f, fieldnames=["set", "target", "sigma", "seed", "value", "source", "done", "run"] + mkeys)
        w.writeheader()
        for name, recs in ((args.label, sweep), (args.ref_label, ref)):
            for r in recs:
                row = {k: v for k, v in r.items() if k != "metrics"}
                row.update({k: r["metrics"].get(k) for k in mkeys})
                w.writerow({"set": name, **row})

    sigmas = sorted({r["sigma"] for r in sweep})
    n_prov = sum(r["source"] == "provisional" for r in sweep)
    n_done = sum(r["done"] for r in sweep)
    groups = HANDS + ["mean over targets"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), facecolor=SURFACE, sharex=True)
    ymax = max(r["value"] for r in sweep + ref) * 1.15 + 0.05
    for ax, g in zip(axes.flat, groups):
        style(ax)
        if g in HANDS:
            pts = {r["sigma"]: r["value"] for r in sweep if r["target"] == g}
            prov = {r["sigma"] for r in sweep if r["target"] == g and r["source"] == "provisional"}
            refs = [r for r in ref if r["target"] == g]
        else:  # mean over targets, only at sigmas every target has
            by = defaultdict(list)
            for r in sweep:
                by[r["sigma"]].append(r["value"])
            pts = {s: float(np.mean(v)) for s, v in by.items() if len(v) == len(HANDS)}
            prov = {r["sigma"] for r in sweep if r["source"] == "provisional"}
            rb = defaultdict(list)
            for r in ref:
                rb[(r["sigma"], r["seed"])].append(r["value"])
            refs = [{"sigma": s, "seed": k, "value": float(np.mean(v))}
                    for (s, k), v in rb.items() if len(v) == len(HANDS)]
        xs = sorted(pts)
        if xs:
            ax.plot(xs, [pts[x] for x in xs], "-", color=SWEEP, linewidth=2, zorder=3, label=args.label)
            solid = [x for x in xs if x not in prov]
            hollow = [x for x in xs if x in prov]
            ax.scatter(solid, [pts[x] for x in solid], s=36, color=SWEEP, edgecolors=SURFACE, linewidths=2, zorder=4)
            ax.scatter(hollow, [pts[x] for x in hollow], s=36, facecolors=SURFACE, edgecolors=SWEEP,
                       linewidths=1.6, zorder=4)
        if refs:
            ax.scatter([r["sigma"] + (-2.0 if r["seed"] == 0 else 2.0) for r in refs], [r["value"] for r in refs],
                       s=30, marker="D", color=REF, edgecolors=SURFACE, linewidths=1.5, zorder=5,
                       label=f"{args.ref_label}, seeds 0/1")
        ax.set_title(g, color=INK, fontsize=10, loc="left")
        ax.set_ylim(0, ymax)
        ax.set_xticks(sigmas if len(sigmas) <= 11 else sigmas[::2])
    fig.supxlabel("σ: the other four hands train only at diffusion steps t ≥ σ  (0 = full co-training, "
                  "100 = target-only); target 50k, others 1M", color=INK2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    h, lab = axes.flat[-1].get_legend_handles_labels()
    if not h:
        h, lab = axes.flat[0].get_legend_handles_labels()
    if n_prov:
        h.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=SWEEP))
        lab.append("provisional (in-training eval, 32 envs)")
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=len(h), frameon=False,
               fontsize=8.5, labelcolor=INK2)
    status = (f"{n_done}/{len(sweep)} runs finished, {n_prov} points provisional" if n_prov
              else "re-scored: 100 episodes, eval seed 1234")
    fig.suptitle(f"{args.family}: target-hand score vs ambient σ, checkpoint = best val loss  [{status}]",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = args.out / f"ambient_sigma_{short}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] {len(sweep)} sweep points ({n_prov} provisional, {n_done} runs done), {len(ref)} ref points -> {out}")
    if args.family == "InHand-Rotation":
        metrics_figure(sweep, args.family, args.out / f"ambient_sigma_{short}_metrics.png", status)


def metrics_figure(sweep: list[dict], family: str, out: Path, status: str) -> None:
    """One panel per rotation metric: value vs sigma, one line per target hand + mean."""
    sigmas = sorted({r["sigma"] for r in sweep})
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8), facecolor=SURFACE, sharex=True)
    for ax, (key, title, better) in zip(axes.flat, ROT_METRICS):
        style(ax)
        for h in HANDS:
            pts = {r["sigma"]: float(r["metrics"][key]) for r in sweep if r["target"] == h and key in r["metrics"]}
            xs = sorted(pts)
            if not xs:
                continue
            ys = [pts[x] for x in xs]
            ax.plot(xs, ys, "-", color=HAND_COLOR[h], linewidth=1.6, zorder=3)
            ax.scatter(xs, ys, s=26, marker=HAND_MARKER[h], color=HAND_COLOR[h], edgecolors=SURFACE,
                       linewidths=1.2, zorder=4, label=h)
        by = defaultdict(list)
        for r in sweep:
            if key in r["metrics"]:
                by[r["sigma"]].append(float(r["metrics"][key]))
        mx = [x for x in sigmas if len(by[x]) == len(HANDS)]
        if mx:
            ax.plot(mx, [np.mean(by[x]) for x in mx], "--", color=INK, linewidth=2, zorder=5, label="mean over hands")
        ax.set_title(f"{title}  ({better} is better)", color=INK, fontsize=9.5, loc="left")
        ax.set_xticks(sigmas)
        if key in ("success_rate_any", "per_target_success_rate", "drop_rate"):
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
    fig.supxlabel("σ: the other four hands train only at diffusion steps t ≥ σ  (0 = full co-training, "
                  "100 = target-only); target 50k (random draw), others 1M", color=INK2, fontsize=9)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=len(h), frameon=False,
               fontsize=8.5, labelcolor=INK2)
    fig.suptitle(f"{family}: every eval metric vs ambient σ per target hand, checkpoint = best val loss, "
                 f"1 seed  [{status}]", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] -> {out}")


if __name__ == "__main__":
    main()
