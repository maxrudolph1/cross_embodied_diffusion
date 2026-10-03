#!/usr/bin/env python3
"""Co-training (sigma 0) vs target-only (sigma 100) per target hand, for each
checkpoint rule (CHANGES.md item 66). Runs: build_ambient_manifest.py output,
outputs/diffusion/ambient_ta/<Task>_sigma{0,100}_seed<k>.

Score per run and checkpoint rule:
  - from <run>/final_eval.jsonl (rescore_selected.py: 100 fresh episodes,
    1500 steps, eval seed 1234) when present -- which = best_rollout /
    best_val / last0;
  - else PROVISIONAL from the in-training evals (32 envs): best eval = max
    row (biased upward by the selection), best val = the row at
    selection.json's best_val epoch (validation runs only at eval epochs), last
    = the final epoch's row.
Metric: grasp success_rate, rotation avg_successes_before_drop.

Writes outputs/plots/cotrain_vs_target_{grasp,rotation}.png (3 panels: best
eval, best val, last; bars = mean of seeds, dots = seeds),
cotrain_vs_target_overview.png (both families, one checkpoint rule --
--overview-ckpt, default last0 -- per target plus the mean over targets) and
cotrain_vs_target_summary.csv (every value: the table view).
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
RULES = [("best_rollout", "best in-training eval"), ("best_val", "best val loss"), ("last0", "last epoch")]
ARMS = [("0", "co-training (σ=0)"), ("100", "target-only (σ=100)")]
# dataviz reference categorical palette, slots 1-2 (validated all-pairs)
SERIES = {"0": "#2a78d6", "100": "#eb6834"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RUN_RE = re.compile(r"(Grasp|InHand-Rotation)-(\w+)_sigma(\d+)_seed(\d+)$")


def scores(run: Path, metric: str) -> tuple[dict, bool]:
    fe = run / "final_eval.jsonl"
    if fe.exists():
        rows = [json.loads(x) for x in fe.read_text().splitlines() if x.strip()]
        out = {r["which"]: r["metrics"][metric] for r in rows if r["steps"] == 1500 and r["envs"] == 100}
        if all(w in out for w, _ in RULES):
            return out, False
    rows = [json.loads(x) for x in (run / "eval_metrics.jsonl").read_text().splitlines() if x.strip()]
    by_epoch = {r["epoch"]: r["metrics"][metric] for r in rows}
    sel = json.loads((run / "selection.json").read_text())
    bv = sel.get("best_val") or {}
    return {
        "best_rollout": max(by_epoch.values()),
        "best_val": by_epoch.get(bv.get("epoch"), np.nan),
        "last0": by_epoch[max(by_epoch)],
    }, True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path,
                    default=Path(os.path.expandvars("$SCRATCH/cross_embodied_diffusion/outputs/diffusion/ambient_ta")))
    ap.add_argument("--out", type=Path, default=Path("outputs/plots"))
    ap.add_argument("--overview-ckpt", default="last0", choices=[r for r, _ in RULES])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recs = []
    for run in sorted(args.runs.glob("*_sigma*_seed*")):
        m = RUN_RE.search(run.name)
        if not m or not (run / "selection.json").exists():
            continue
        fam, hand, sigma, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        vals, prov = scores(run, FAMILIES[fam][1])
        for rule, _ in RULES:
            recs.append({"family": fam, "target": hand, "sigma": sigma, "seed": seed, "checkpoint": rule,
                         "value": vals[rule], "source": "in-training (provisional)" if prov else "rescore"})
    with open(args.out / "cotrain_vs_target_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)

    for fam, (short, _metric, label) in FAMILIES.items():
        fr = [r for r in recs if r["family"] == fam]
        if not fr:
            continue
        prov = any(r["source"].startswith("in-training") for r in fr)
        by = defaultdict(list)
        for r in fr:
            by[(r["checkpoint"], r["target"], r["sigma"])].append(r["value"])
        ymax = max(v for r in fr for v in [r["value"]] if np.isfinite(v)) * 1.15
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE, sharey=True)
        x = np.arange(len(HANDS))
        width = 0.36
        for ax, (rule, title) in zip(axes, RULES):
            ax.set_facecolor(SURFACE)
            for j, (sig, name) in enumerate(ARMS):
                off = (j - 0.5) * (width + 0.04)
                means = [np.nanmean(by[(rule, h, sig)]) if by[(rule, h, sig)] else np.nan for h in HANDS]
                ax.bar(x + off, means, width, color=SERIES[sig], label=name, zorder=2,
                       edgecolor=SURFACE, linewidth=2)
                for i, h in enumerate(HANDS):
                    vs = by[(rule, h, sig)]
                    ax.scatter([x[i] + off] * len(vs), vs, s=14, color=INK, alpha=0.55, zorder=3, linewidths=0)
                # mean co-train minus target-only, as a muted label over each pair
            diffs = [np.nanmean(by[(rule, h, "0")]) - np.nanmean(by[(rule, h, "100")]) for h in HANDS]
            for i, d in enumerate(diffs):
                top = max(np.nanmax(by[(rule, HANDS[i], s)] or [0]) for s in ("0", "100"))
                ax.text(x[i], top + ymax * 0.03, f"{d:+.2f}", ha="center", fontsize=8, color=INK2)
            ax.set_title(f"{title}   (mean diff {np.nanmean(diffs):+.2f})", color=INK, fontsize=10, loc="left")
            ax.set_xticks(x)
            ax.set_xticklabels(HANDS, fontsize=9, color=INK2)
            ax.tick_params(colors=INK2, length=0)
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.set_ylim(0, ymax * 1.08)
        axes[0].set_ylabel(label, color=INK2, fontsize=9)
        axes[0].set_xlabel("target hand (50k demos; co-training adds the other four at 1M)", color=INK2, fontsize=8)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
        src = "PROVISIONAL: in-training evals, 32 envs" if prov else "fresh-seed re-score, 100 episodes"
        fig.suptitle(f"{fam}: co-training vs target-only by checkpoint rule (bars = mean of seeds, dots = seeds)  [{src}]",
                     x=0.01, ha="left", color=INK, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(args.out / f"cotrain_vs_target_{short}.png", dpi=150)
        plt.close(fig)
        print(f"[INFO] {fam}: {len(fr) // len(RULES)} runs, provisional={prov} -> cotrain_vs_target_{short}.png")

    overview(recs, args.out, args.overview_ckpt)


def overview(recs: list[dict], out: Path, rule: str) -> None:
    """Both families side by side: per target and the mean over targets."""
    fams = [f for f in FAMILIES if any(r["family"] == f for r in recs)]
    fig, axes = plt.subplots(1, len(fams), figsize=(6.6 * len(fams), 4.2), facecolor=SURFACE, squeeze=False)
    prov = any(r["source"].startswith("in-training") for r in recs)
    for ax, fam in zip(axes[0], fams):
        fr = [r for r in recs if r["family"] == fam and r["checkpoint"] == rule]
        groups = HANDS + ["mean"]
        x = np.arange(len(groups), dtype=float)
        x[-1] += 0.4  # visual gap before the mean
        width = 0.36
        for j, (sig, name) in enumerate(ARMS):
            off = (j - 0.5) * (width + 0.04)
            per = {h: [r["value"] for r in fr if r["target"] == h and r["sigma"] == sig] for h in HANDS}
            seed_means = defaultdict(list)  # mean over targets, per seed
            for r in fr:
                if r["sigma"] == sig:
                    seed_means[r["seed"]].append(r["value"])
            per["mean"] = [float(np.mean(v)) for v in seed_means.values()]
            means = [np.mean(per[g]) for g in groups]
            ax.bar(x + off, means, width, color=SERIES[sig], label=name, zorder=2, edgecolor=SURFACE, linewidth=2)
            for i, g in enumerate(groups):
                ax.scatter([x[i] + off] * len(per[g]), per[g], s=14, color=INK, alpha=0.55, zorder=3, linewidths=0)
        for i, g in enumerate(groups):
            a = [r["value"] for r in fr if r["sigma"] == "0" and (g == "mean" or r["target"] == g)]
            b = [r["value"] for r in fr if r["sigma"] == "100" and (g == "mean" or r["target"] == g)]
            top = max(a + b)
            ax.text(x[i], top * 1.03 + 0.02, f"{np.mean(a) - np.mean(b):+.2f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9, color=INK2)
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK2, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_ylabel(FAMILIES[fam][2], color=INK2, fontsize=9)
        ax.set_title(fam, color=INK, fontsize=10, loc="left", pad=10)
        ax.set_ylim(0, max(r["value"] for r in fr) * 1.2 + 0.05)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    label = dict(RULES)[rule]
    src = "PROVISIONAL: in-training evals" if prov else "fresh-seed re-score, 100 episodes"
    fig.suptitle(f"Co-training (target 50k + four hands at 1M) vs target-only 50k; checkpoint = {label}  [{src}]",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "cotrain_vs_target_overview.png", dpi=150)
    plt.close(fig)
    print("[INFO] -> cotrain_vs_target_overview.png")


if __name__ == "__main__":
    main()
