#!/usr/bin/env python3
"""Plots for the ambient-diffusion InHand-Rotation sweep (5 target hands x
16 sigmas x 4 seeds; `build_ambient_rotation_manifest.py --kind sweep`).

Score = successes before drop on the run's target hand. Source per run:
`posthoc_eval.json` (fresh-seed reporting eval, eval_checkpoints.py) when
present, else the in-training evals (`eval_metrics.jsonl`: best = max row,
latest = last row, best_val = the row at best_val.json's epoch) -- figures
built from in-training numbers say "PROVISIONAL" in the title.

Writes to --out (default outputs/plots):
  ambient_rot_by_hand.png    one panel per target hand: score vs sigma for the
                             three checkpoint rules (mean of seeds, seeds as dots)
  ambient_rot_hands.png      all hands on one axis (one checkpoint rule), raw
                             and normalized to each hand's sigma=100 score
  ambient_rot_control.png    (if control evals exist) a gated non-target hand's
                             score vs sigma, per target hand
  ambient_rot_summary.csv    every (hand, sigma, checkpoint, seed) value -- the
                             table view of the figures
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
SIGMAS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 25, 100]
CKPTS = ["best_eval", "best_val", "latest"]
CKPT_LABEL = {
    "best_eval": "best in-training eval",
    "best_val": "best target val loss",
    "latest": "last epoch",
}
# Reference categorical palette (dataviz skill, references/palette.md, light
# mode), slots in fixed order; slots 1-3 are validated all-pairs, 1-5 adjacent.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RUN_RE = re.compile(r"InHand-Rotation-(\w+)_sigma(\d+)_seed(\d+)$")
METRIC = "avg_successes_before_drop"


def xpos(sigma: int) -> float:
    """Even spacing for 0..25, then a visible break before sigma=100."""
    return SIGMAS.index(sigma) + (1.0 if sigma == 100 else 0.0)


def in_training(run: Path) -> dict[str, float]:
    rows = [json.loads(x) for x in (run / "eval_metrics.jsonl").read_text().splitlines() if x.strip()]
    if not rows:
        return {}
    vals = {r["epoch"]: r["metrics"].get(METRIC, np.nan) for r in rows}
    out = {"best_eval": max(vals.values()), "latest": vals[max(vals)]}
    bv = run / "best_val.json"
    if bv.exists():
        out["best_val"] = vals.get(json.loads(bv.read_text())["epoch"], np.nan)
    return out


def load(root: Path) -> tuple[list[dict], bool]:
    recs, provisional = [], False
    for run in sorted(root.glob("InHand-Rotation-*_sigma*_seed*")):
        m = RUN_RE.search(run.name)
        if not m or not (run / "eval_metrics.jsonl").exists():
            continue
        hand, sigma, seed = m.group(1), int(m.group(2)), int(m.group(3))
        target = f"InHand-Rotation-{hand}"
        ph = run / "posthoc_eval.json"
        post = json.loads(ph.read_text()) if ph.exists() else {}
        for ck in CKPTS:
            key = f"policy_{ck}.pt@{target}"
            if key in post:
                val, src = post[key]["metrics"][METRIC], "posthoc"
            else:
                val, src = in_training(run).get(ck, np.nan), "in-training"
                provisional = True
            recs.append({"hand": hand, "sigma": sigma, "seed": seed, "ckpt": ck, "eval_on": hand,
                         "role": "target", "value": val, "source": src})
            for k, v in post.items():  # control hands, post-hoc only
                ck_k, emb = k.split("@")
                if ck_k == f"policy_{ck}.pt" and emb != target:
                    recs.append({"hand": hand, "sigma": sigma, "seed": seed, "ckpt": ck,
                                 "eval_on": emb.split("-")[-1], "role": "control",
                                 "value": v["metrics"][METRIC], "source": "posthoc"})
    return recs, provisional


def style(ax, ylabel: bool) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks([xpos(s) for s in SIGMAS])
    ax.set_xticklabels([str(s) for s in SIGMAS], fontsize=7)
    # break marker between 25 and 100
    xb = (xpos(25) + xpos(100)) / 2
    ax.text(xb, 0, "//", transform=ax.get_xaxis_transform(), ha="center", va="center",
            color=INK2, fontsize=9, clip_on=False)
    ax.set_xlabel("sigma (other hands trained only at t >= sigma)", color=INK2, fontsize=8)
    if ylabel:
        ax.set_ylabel("successes before drop", color=INK2, fontsize=9)


def agg(recs, **sel):
    by = defaultdict(list)
    for r in recs:
        if all(r[k] == v for k, v in sel.items()) and np.isfinite(r["value"]):
            by[r["sigma"]].append(r["value"])
    xs = [s for s in SIGMAS if by.get(s)]
    return xs, [float(np.mean(by[s])) for s in xs], by


def line(ax, xs, means, by, color, label=None, seeds=True, z=3):
    if seeds:
        for s in xs:
            ax.scatter([xpos(s)] * len(by[s]), by[s], s=10, color=color, alpha=0.35, linewidths=0, zorder=z - 1)
    px = [xpos(s) for s in xs]
    # split the line at the axis break so 25 -> 100 isn't drawn as a trend
    for seg in ([i for i, s in enumerate(xs) if s != 100], [i for i, s in enumerate(xs) if s == 100]):
        if seg:
            ax.plot([px[i] for i in seg], [means[i] for i in seg], color=color, linewidth=2, zorder=z,
                    marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=None)
    if label is not None:
        ax.plot([], [], color=color, linewidth=2, marker="o", markersize=4.5, label=label)


def direct_labels(ax, items: list[tuple[float, float, str]], min_gap_frac: float = 0.06) -> None:
    """Label line ends at (x, y), nudging labels apart vertically so none collide."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda t: t[1])
    ys = [y for _, y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (x, y, text), yl in zip(items, ys):
        ax.annotate(text, (x, y), xytext=(x + 0.35, yl), textcoords="data", va="center",
                    fontsize=8, color=INK2, arrowprops=None)


def title_suffix(provisional: bool) -> str:
    return "  [PROVISIONAL: in-training evals]" if provisional else "  [fresh-seed post-hoc evals]"


def fig_by_hand(recs, provisional, out: Path) -> None:
    hands = [h for h in HANDS if any(r["hand"] == h for r in recs)]
    fig, axes = plt.subplots(1, len(hands), figsize=(3.6 * len(hands), 3.6), facecolor=SURFACE, squeeze=False)
    for ax, hand in zip(axes[0], hands):
        for i, ck in enumerate(CKPTS):
            xs, means, by = agg(recs, hand=hand, ckpt=ck, role="target")
            line(ax, xs, means, by, SERIES[i], label=CKPT_LABEL[ck])
        style(ax, ylabel=ax is axes[0][0])
        ax.set_title(f"target: {hand}", color=INK, fontsize=10, loc="left")
        ax.set_ylim(bottom=0)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=3, frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Ambient diffusion, InHand-Rotation: target-hand score vs sigma (mean of 4 seeds; dots = seeds)"
                 + title_suffix(provisional), x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "ambient_rot_by_hand.png", dpi=150)
    plt.close(fig)


def fig_hands(recs, provisional, out: Path, ckpt: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    for ax, norm in zip(axes, (False, True)):
        ends = []
        for i, hand in enumerate(HANDS):
            xs, means, by = agg(recs, hand=hand, ckpt=ckpt, role="target")
            if not xs:
                continue
            if norm:
                ref = dict(zip(xs, means)).get(100)
                if not ref:
                    continue
                means = [m / ref for m in means]
                by = {s: [v / ref for v in vs] for s, vs in by.items()}
            line(ax, xs, means, by, SERIES[i], seeds=False)
            # direct label (mandatory for >3 series) at the sigma=25 end, where
            # the lines separate; at sigma=100 the normalized ones all sit at 1
            last = max(j for j, s in enumerate(xs) if s != 100)
            ends.append((xpos(xs[last]), means[last], hand))
        ax.set_ylim(bottom=0)
        direct_labels(ax, ends)
        style(ax, ylabel=not norm)
        if norm:
            ax.set_ylabel("score / score at sigma=100", color=INK2, fontsize=9)
            ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.set_title("normalized to target-only (sigma=100)" if norm else "raw", color=INK, fontsize=10, loc="left")
        ax.set_xlim(right=xpos(100) + 1.6)
        ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=SERIES[i], linewidth=2, marker="o", markersize=4.5) for i in range(len(HANDS))]
    fig.legend(handles, HANDS, loc="upper right", ncol=5, frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f"All target hands, checkpoint = {CKPT_LABEL[ckpt]} (mean of 4 seeds)" + title_suffix(provisional),
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "ambient_rot_hands.png", dpi=150)
    plt.close(fig)


def fig_control(recs, out: Path, ckpt: str) -> bool:
    ctrl = [r for r in recs if r["role"] == "control" and r["ckpt"] == ckpt]
    if not ctrl:
        return False
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    for i, hand in enumerate(HANDS):
        xs, means, by = agg(ctrl, hand=hand)
        if not xs:
            continue
        emb = next(r["eval_on"] for r in ctrl if r["hand"] == hand)
        line(ax, xs, means, by, SERIES[i], seeds=False)
        ax.annotate(f"{hand} run, scored on {emb}", (xpos(xs[0]), means[0]), xytext=(6, 4),
                    textcoords="offset points", fontsize=7, color=INK2)
    style(ax, ylabel=True)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Control: a gated (non-target) hand's score vs sigma, checkpoint = {CKPT_LABEL[ckpt]}",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "ambient_rot_control.png", dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path,
                    default=Path(os.path.expandvars("$SCRATCH/cross_embodied_diffusion/outputs/diffusion/ambient_rot")))
    ap.add_argument("--out", type=Path, default=Path("outputs/plots"))
    ap.add_argument("--ckpt", choices=CKPTS, default="latest", help="checkpoint rule for the all-hands figure")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recs, provisional = load(args.runs)
    if not recs:
        raise SystemExit(f"no sweep runs with evals under {args.runs}")
    with open(args.out / "ambient_rot_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    fig_by_hand(recs, provisional, args.out)
    fig_hands(recs, provisional, args.out, args.ckpt)
    ctrl = fig_control(recs, args.out, args.ckpt)
    n = len({(r["hand"], r["sigma"], r["seed"]) for r in recs})
    print(f"[INFO] {n} runs, provisional={provisional}, control={ctrl} -> {args.out}")


if __name__ == "__main__":
    main()
