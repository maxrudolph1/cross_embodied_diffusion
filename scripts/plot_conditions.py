#!/usr/bin/env python3
"""Training conditions x policy-selection methods, and cross-embodiment
transfer of the generalist policies (CHANGES.md item 68).

Conditions (per family, per target hand, 2 seeds):
  target-only   outputs/diffusion/ambient_ta/<Task>_sigma100_seed<k>
  co-trained    outputs/diffusion/ambient_ta/<Task>_sigma0_seed<k>
  FT lr 1e-4    outputs/diffusion/ambient_ta_ft/<Task>_ft_best_val_lr0.0001_seed<k>
  FT lr 1e-5    outputs/diffusion/ambient_ta_ft/<Task>_ft_best_val_lr1e-05_seed<k>
Selection methods: best in-training eval (best_rollout), best val loss
(best_val), last epoch (last0). Scores from <run>/final_eval.jsonl
(rescore_selected.py: 100 fresh episodes, 1500 steps, seed 1234); runs not yet
re-scored are left out (counted in the titles).

Writes to --out (default outputs/plots):
  conditions_<family>.png         3 panels (selection method): per target + mean,
                                   bars = conditions, dots = seeds
  conditions_overview.png         mean over targets: x = selection method, bars =
                                   conditions, one panel per family
  cross_embodiment_<family>.png   heatmaps (last0): row = the policy's target
                                   hand, column = hand it is evaluated on; one
                                   panel per generalist condition
  conditions_summary.csv, cross_embodiment_summary.csv (the table views)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HANDS = ["Allegro", "LEAP", "Shadow", "Sharpa", "Wuji"]
FAMILIES = {"Grasp": ("grasp", "success_rate", "success rate"),
            "InHand-Rotation": ("rotation", "avg_successes_before_drop", "successes before drop")}
RULES = [("best_rollout", "best in-training eval"), ("best_val", "best val loss"), ("last0", "last epoch")]
CONDS = [  # (key, label, run-dir template relative to the outputs/diffusion root)
    ("target_only", "target-only 50k", "ambient_ta/{task}_sigma100_seed{seed}"),
    ("cotrain", "co-trained", "ambient_ta/{task}_sigma0_seed{seed}"),
    ("ft_1e-4", "co-trained + FT lr 1e-4", "ambient_ta_ft/{task}_ft_best_val_lr0.0001_seed{seed}"),
    ("ft_1e-5", "co-trained + FT lr 1e-5", "ambient_ta_ft/{task}_ft_best_val_lr1e-05_seed{seed}"),
]
GENERALISTS = ["cotrain", "ft_1e-4", "ft_1e-5"]
SEEDS = [0, 1]
# dataviz reference categorical palette, slots 1-4 in fixed order
COLOR = {"target_only": "#eb6834", "cotrain": "#2a78d6", "ft_1e-4": "#1baf7a", "ft_1e-5": "#eda100"}
ORDER = ["target_only", "cotrain", "ft_1e-4", "ft_1e-5"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def rows_of(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def collect(root: Path) -> tuple[list[dict], list[dict]]:
    target_recs, cross_recs = [], []
    for fam, (_s, metric, _l) in FAMILIES.items():
        for hand in HANDS:
            task = f"{fam}-{hand}"
            for key, _label, tmpl in CONDS:
                for seed in SEEDS:
                    run = root / tmpl.format(task=task, seed=seed)
                    fe = [r for r in rows_of(run / "final_eval.jsonl")
                          if r["task"] == task and r["envs"] == 100 and r["steps"] == 1500]
                    for r in fe:
                        target_recs.append({"family": fam, "target": hand, "condition": key, "seed": seed,
                                            "checkpoint": r["which"], "value": r["metrics"][metric]})
                    if key in GENERALISTS:
                        for r in rows_of(run / "cross_eval.jsonl") + [x for x in fe if x["which"] == "last0"]:
                            if r["which"] != "last0" or r["envs"] != 100 or r["steps"] != 1500:
                                continue
                            cross_recs.append({"family": fam, "target": hand, "condition": key, "seed": seed,
                                               "eval_on": r["task"].rsplit("-", 1)[1], "value": r["metrics"][metric]})
    return target_recs, cross_recs


def bars(ax, groups, series, values, width=0.19) -> None:
    """values[(group, series)] -> list of seed values."""
    x = np.arange(len(groups), dtype=float)
    if groups and groups[-1] == "mean":
        x[-1] += 0.5
    n = len(series)
    for j, s in enumerate(series):
        off = (j - (n - 1) / 2) * (width + 0.02)
        means = [np.mean(values[(g, s)]) if values.get((g, s)) else np.nan for g in groups]
        ax.bar(x + off, means, width, color=COLOR[s], zorder=2, edgecolor=SURFACE, linewidth=1.5,
               label=dict((k, lab) for k, lab, _ in CONDS)[s])
        for i, g in enumerate(groups):
            vs = values.get((g, s), [])
            ax.scatter([x[i] + off] * len(vs), vs, s=9, color=INK, alpha=0.5, zorder=3, linewidths=0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9, color=INK2)


def plot_conditions(recs, out: Path) -> None:
    for fam, (short, _m, label) in FAMILIES.items():
        fr = [r for r in recs if r["family"] == fam]
        if not fr:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), facecolor=SURFACE, sharey=True)
        for ax, (rule, title) in zip(axes, RULES):
            vals = {}
            for r in fr:
                if r["checkpoint"] == rule:
                    vals.setdefault((r["target"], r["condition"]), []).append(r["value"])
            for c in ORDER:  # mean over targets, per seed
                per_seed = {}
                for r in fr:
                    if r["checkpoint"] == rule and r["condition"] == c:
                        per_seed.setdefault(r["seed"], []).append(r["value"])
                vals[("mean", c)] = [np.mean(v) for v in per_seed.values() if len(v) == len(HANDS)]
            bars(ax, HANDS + ["mean"], ORDER, vals)
            style(ax)
            ax.set_title(title, color=INK, fontsize=10, loc="left")
        axes[0].set_ylabel(label, color=INK2, fontsize=9)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper right", ncol=4, frameon=False, fontsize=9, labelcolor=INK2)
        n = len({(r["condition"], r["target"], r["seed"]) for r in fr})
        fig.suptitle(f"{fam}: training condition x policy selection, on the target hand "
                     f"(100 fresh episodes; bars = mean of 2 seeds, dots = seeds; {n}/40 runs re-scored)",
                     x=0.01, ha="left", color=INK, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(out / f"conditions_{short}.png", dpi=150)
        plt.close(fig)

    fams = [f for f in FAMILIES if any(r["family"] == f for r in recs)]
    fig, axes = plt.subplots(1, len(fams), figsize=(7 * len(fams), 4.2), facecolor=SURFACE, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        vals = {}
        for rule, title in RULES:
            for c in ORDER:
                per_seed = {}
                for r in recs:
                    if r["family"] == fam and r["checkpoint"] == rule and r["condition"] == c:
                        per_seed.setdefault(r["seed"], []).append(r["value"])
                vals[(title, c)] = [np.mean(v) for v in per_seed.values() if len(v) == len(HANDS)]
        bars(ax, [t for _, t in RULES], ORDER, vals)
        style(ax)
        ax.set_title(f"{fam} (mean over 5 target hands)", color=INK, fontsize=10, loc="left")
        ax.set_ylabel(FAMILIES[fam][2], color=INK2, fontsize=9)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=4, frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Policy selection method x training condition (100 fresh episodes; dots = seeds)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "conditions_overview.png", dpi=150)
    plt.close(fig)


def plot_cross(recs, out: Path) -> None:
    for fam, (short, _m, label) in FAMILIES.items():
        fr = [r for r in recs if r["family"] == fam]
        if not fr:
            continue
        mats, vmax = {}, max(r["value"] for r in fr)
        for c in GENERALISTS:
            m = np.full((len(HANDS), len(HANDS)), np.nan)
            for i, t in enumerate(HANDS):
                for j, e in enumerate(HANDS):
                    v = [r["value"] for r in fr if r["condition"] == c and r["target"] == t and r["eval_on"] == e]
                    if v:
                        m[i, j] = np.mean(v)
            mats[c] = m
        fig, axes = plt.subplots(1, len(GENERALISTS), figsize=(5.2 * len(GENERALISTS), 4.6), facecolor=SURFACE)
        for ax, c in zip(axes, GENERALISTS):
            m = mats[c]
            cmap = matplotlib.colormaps["Blues"].copy()
            cmap.set_bad(GRID)  # not evaluated yet: grey, distinct from a 0 score
            im = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=0, vmax=vmax)
            for i in range(len(HANDS)):
                for j in range(len(HANDS)):
                    if np.isfinite(m[i, j]):
                        ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                                color="#ffffff" if m[i, j] > 0.6 * vmax else INK,
                                fontweight="bold" if i == j else "normal")
            ax.set_xticks(range(len(HANDS)))
            ax.set_xticklabels(HANDS, fontsize=8, color=INK2)
            ax.set_yticks(range(len(HANDS)))
            ax.set_yticklabels(HANDS, fontsize=8, color=INK2)
            ax.set_xlabel("evaluated on", color=INK2, fontsize=9)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(dict((k, lab) for k, lab, _ in CONDS)[c], color=INK, fontsize=10, loc="left")
        axes[0].set_ylabel("policy's target hand (50k demos)", color=INK2, fontsize=9)
        cb = fig.colorbar(im, ax=list(axes), shrink=0.8, pad=0.02)
        cb.set_label(label, color=INK2, fontsize=9)
        cb.outline.set_visible(False)
        fig.suptitle(f"{fam}: generalist policies on every hand (last epoch, 100 fresh episodes, "
                     f"mean of 2 seeds; diagonal = own target)", x=0.01, ha="left", color=INK, fontsize=11)
        fig.savefig(out / f"cross_embodiment_{short}.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path(os.path.expandvars("$SCRATCH/cross_embodied_diffusion/outputs/diffusion")))
    ap.add_argument("--out", type=Path, default=Path("outputs/plots"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trecs, crecs = collect(args.root)
    for name, recs in (("conditions_summary.csv", trecs), ("cross_embodiment_summary.csv", crecs)):
        if recs:
            with open(args.out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(recs[0]))
                w.writeheader()
                w.writerows(recs)
    plot_conditions(trecs, args.out)
    plot_cross(crecs, args.out)
    print(f"[INFO] {len(trecs)} target scores, {len(crecs)} cross-embodiment scores -> {args.out}")


if __name__ == "__main__":
    main()
