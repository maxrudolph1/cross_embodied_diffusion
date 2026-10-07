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


def score(run: Path, task: str, eval_seed: int = 1234) -> tuple[dict, str] | None:
    """(metrics, source) for the run's best_val checkpoint on its own task at eval_seed; None if nothing
    to score yet. Rows for other hands (--other-hands) or other seeds are ignored (CHANGES.md item 74)."""
    fe = run / "final_eval.jsonl"
    if fe.exists():
        rows = [json.loads(x) for x in fe.read_text().splitlines() if x.strip()]
        for r in rows:
            if (r["which"] == "best_val" and r["envs"] == 100 and r["steps"] == 1500
                    and r.get("task", task) == task and r.get("eval_seed", 1234) == eval_seed):
                return r["metrics"], "rescore"
    em = run / "eval_metrics.jsonl"
    if eval_seed != 1234 or not em.exists():
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
        s = score(run, f"{family}-{m.group(2)}")
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

    n_prov = sum(r["source"] == "provisional" for r in sweep)
    n_done = sum(r["done"] for r in sweep)
    groups = HANDS + ["mean over targets"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), facecolor=SURFACE, sharex=True)
    ymax = max(r["value"] for r in sweep + ref) * 1.12 + 0.05
    for ax, g in zip(axes.flat, groups):
        style(ax)
        if g in HANDS:
            means, seeds = seed_means(sweep, g, None)
            refs = [r for r in ref if r["target"] == g]
        else:
            means, seeds = target_means(sweep, None)
            rb = defaultdict(list)
            for r in ref:
                rb[(r["sigma"], r["seed"])].append(r["value"])
            refs = [{"sigma": s_, "seed": k, "value": float(np.mean(v))}
                    for (s_, k), v in rb.items() if len(v) == len(HANDS)]
        xs = sorted(means)
        if xs:
            sx = [x for x in xs for _ in seeds[x]]
            sy = [v for x in xs for v in seeds[x]]
            ax.scatter(sx, sy, s=12, color=SWEEP, alpha=0.35, linewidths=0, zorder=2,
                       label="single seed" if g == groups[-1] else None)
            ax.plot(xs, [means[x] for x in xs], "-", color=SWEEP, linewidth=2, zorder=3,
                    label=f"{args.label}, mean of seeds")
            ax.scatter(xs, [means[x] for x in xs], s=30, color=SWEEP, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        if refs:
            ax.scatter([r["sigma"] for r in refs], [r["value"] for r in refs], s=34, marker="D",
                       facecolors="none", edgecolors=REF, linewidths=1.5, zorder=5,
                       label=f"{args.ref_label}, seeds 0/1")
        ax.set_title(g, color=INK, fontsize=10, loc="left")
        ax.set_ylim(0, ymax)
        sigma_axis(ax)
    fig.supxlabel("σ (square-root spacing): the other four hands train only at diffusion steps t ≥ σ  "
                  "(0 = full co-training, 100 = target-only); target 50k, others 1M", color=INK2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    h, lab = axes.flat[-1].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=len(h), frameon=False,
               fontsize=8.5, labelcolor=INK2)
    status = (f"{n_done}/{len(sweep)} runs finished, {n_prov} points provisional" if n_prov
              else f"re-scored: 100 episodes, eval seed 1234\n{seed_note(sweep)}")
    fig.suptitle(f"{args.family}: target-hand score vs ambient σ, checkpoint = best val loss  [{status}]",
                 x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    out = args.out / f"ambient_sigma_{short}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] {len(sweep)} sweep points ({n_prov} provisional, {n_done} runs done), {len(ref)} ref points -> {out}")
    if args.family == "InHand-Rotation":
        metrics_figure(sweep, args.family, args.out / f"ambient_sigma_{short}_metrics.png", status)


def _val(r: dict, key: str | None) -> float | None:
    if key is None:
        return r["value"]
    v = r["metrics"].get(key)
    return None if v is None else float(v)


def seed_means(recs: list[dict], target: str, key: str | None) -> tuple[dict, dict]:
    """Per sigma for one target: (mean over seeds, list of seed values)."""
    seeds = defaultdict(list)
    for r in recs:
        v = _val(r, key)
        if r["target"] == target and v is not None:
            seeds[r["sigma"]].append(v)
    return {s_: float(np.mean(v)) for s_, v in seeds.items()}, dict(seeds)


def target_means(recs: list[dict], key: str | None) -> tuple[dict, dict]:
    """Per sigma: mean over targets of each target's seed mean (only sigmas every target has), and per seed
    the mean over targets (only seeds every target has at that sigma)."""
    per = {h: seed_means(recs, h, key)[0] for h in HANDS}
    common = set.intersection(*(set(m) for m in per.values())) if per else set()
    means = {s_: float(np.mean([per[h][s_] for h in HANDS])) for s_ in common}
    by = defaultdict(list)
    for r in recs:
        v = _val(r, key)
        if v is not None:
            by[(r["sigma"], r["seed"])].append(v)
    seeds = defaultdict(list)
    for (s_, k), v in by.items():
        if s_ in common and len(v) == len(HANDS):
            seeds[s_].append(float(np.mean(v)))
    return means, dict(seeds)


SIGMA_TICKS = [0, 1, 2, 3, 6, 10, 20, 30, 50, 70, 100]


def sigma_axis(ax) -> None:
    """Square-root sigma axis, so the dense low-sigma grid (1-20) is readable next to 30-100."""
    ax.set_xscale("function", functions=(lambda x: np.sign(x) * np.sqrt(np.abs(x)),
                                         lambda x: np.sign(x) * np.square(x)))
    ax.set_xlim(-0.5, 108)
    ax.set_xticks(SIGMA_TICKS)
    ax.set_xticklabels([str(t) for t in SIGMA_TICKS], fontsize=7.5)


def seed_note(recs: list[dict]) -> str:
    n = defaultdict(set)
    for r in recs:
        n[r["sigma"]].add(r["seed"])
    multi = sorted(s_ for s_, k in n.items() if len(k) > 1)
    single = sorted(s_ for s_, k in n.items() if len(k) == 1)
    if not multi:
        return "1 seed"
    k = max(len(v) for v in n.values())
    return f"{k} seeds at σ {', '.join(map(str, multi))}; 1 seed at σ {', '.join(map(str, single))}" if single \
        else f"{k} seeds"


def metrics_figure(sweep: list[dict], family: str, out: Path, status: str) -> None:
    """One panel per rotation metric: value vs sigma (mean over seeds), one line per target hand + mean."""
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8), facecolor=SURFACE, sharex=True)
    for ax, (key, title, better) in zip(axes.flat, ROT_METRICS):
        style(ax)
        for h in HANDS:
            means, _ = seed_means(sweep, h, key)
            xs = sorted(means)
            if not xs:
                continue
            ys = [means[x] for x in xs]
            ax.plot(xs, ys, "-", color=HAND_COLOR[h], linewidth=1.5, zorder=3)
            ax.scatter(xs, ys, s=22, marker=HAND_MARKER[h], color=HAND_COLOR[h], edgecolors=SURFACE,
                       linewidths=1.0, zorder=4, label=h)
        means, _ = target_means(sweep, key)
        mx = sorted(means)
        if mx:
            ax.plot(mx, [means[x] for x in mx], "--", color=INK, linewidth=2, zorder=5, label="mean over hands")
        ax.set_title(f"{title}  ({better} is better)", color=INK, fontsize=9.5, loc="left")
        sigma_axis(ax)
        if key in ("success_rate_any", "per_target_success_rate", "drop_rate"):
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
    fig.supxlabel("σ (square-root spacing): the other four hands train only at diffusion steps t ≥ σ  "
                  "(0 = full co-training, 100 = target-only); target 50k (random draw), others 1M",
                  color=INK2, fontsize=9)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=len(h), frameon=False,
               fontsize=8.5, labelcolor=INK2)
    fig.suptitle(f"{family}: every eval metric vs ambient σ per target hand (mean of seeds), checkpoint = "
                 f"best val loss  [{status}]", x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] -> {out}")


if __name__ == "__main__":
    main()
