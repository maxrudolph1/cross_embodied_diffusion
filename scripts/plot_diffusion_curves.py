#!/usr/bin/env python3
"""Plot diffusion-policy training/eval curves.

Diffusion runs do not write TensorBoard; they append one JSON row per
in-training eval to `outputs/diffusion/<run>/eval_metrics.jsonl` (`epoch`,
`train_loss`, `train_steps`, `eval_task`, `metrics{...}`). This script reads
those and writes per-run curves, cross-run summaries, and (`plot_scaling`)
headline-metric-vs-dataset-size scaling figures.

The two task families do not share eval metric names: grasp reports
`success_rate` and `avg_final_dist_to_first_goal_m`; rotation reports
`avg_successes_before_drop`, `drop_rate`, `avg_survival_time_s`,
`avg_rot_dist` and has no `success_rate` at all. See CHANGES.md item 6g.

X-axis defaults to gradient steps, not epochs (`--x epoch` overrides),
because runs at different data scales are deliberately sized for matched
compute (e.g. 1M x 200 epochs ~= 10M x 20 epochs -- plotting against epochs
would make matched-compute runs look 10x apart). See items 6b/6f.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SIZE_ORDER = ("10k", "50k", "100k", "400k", "1M", "10M")
SIZE_RE = re.compile(r"_(10k|50k|100k|400k|1M|10M|smoke|full)$")

HEADLINE = {
    "grasp": ("success_rate", "rollout success_rate", (-0.02, 1.02)),
    "rotation": ("avg_successes_before_drop", "avg successes before drop", None),
}
SECONDARY = {
    "grasp": ("avg_final_dist_to_first_goal_m", "avg final dist to goal (m)"),
    "rotation": ("drop_rate", "drop rate"),
}


def family_of(task: str | None) -> str:
    return "rotation" if task and "rotation" in task.lower() else "grasp"


def parse_run(run_dir: Path) -> tuple[str, str]:
    """Best-effort (task, size) for a diffusion output dir."""
    task = None
    size = None
    cfg_path = run_dir / "train_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            cfg = {}
        task = cfg.get("eval_task")
        if task is None and cfg.get("eval_specs"):
            task = cfg["eval_specs"][0].get("task")
        dataset = str(cfg.get("dataset", ""))
        m = re.search(r"expert(?:_(\w+))?\.zarr", dataset)
        if m and m.group(1):
            size = m.group(1)
    if size is None:
        m = SIZE_RE.search(run_dir.name)
        size = m.group(1) if m else "unknown"
    if task is None:
        task = run_dir.name
    return task, size


def load_rows(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        return []
    rows = []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a torn final line while training appends
        rows.append(row)
    return rows


def rows_for_task(rows: list[dict], task: str) -> list[dict]:
    """Rows scored against `task`, tolerating rows with no `eval_task` key
    (single-target runs predating multi-target eval)."""
    out = []
    for r in rows:
        et = r.get("eval_task")
        if et is not None and et != task:
            continue
        out.append(r)
    return out


def discover(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p.parent for p in root.glob("*/eval_metrics.jsonl"))


def _size_key(size: str) -> int:
    return SIZE_ORDER.index(size) if size in SIZE_ORDER else 99


def plot_run(
    task: str, size_to_rows: dict[str, list[dict]], out_dir: Path, x: str
) -> Path | None:
    if not size_to_rows:
        return None
    family = family_of(task)
    headline_key, headline_label, ylim = HEADLINE[family]
    secondary_key, secondary_label = SECONDARY[family]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for size, rows in sorted(size_to_rows.items(), key=lambda kv: _size_key(kv[0])):
        if not rows:
            continue
        xs = [r["train_steps"] if x == "steps" else r["epoch"] for r in rows]
        losses = [r.get("train_loss") for r in rows]
        headline = [r["metrics"].get(headline_key, np.nan) for r in rows]
        secondary = [r["metrics"].get(secondary_key, np.nan) for r in rows]
        style = "--" if size == "10M" else "-"
        axes[0].plot(xs, losses, style, label=size)
        axes[1].plot(xs, headline, style, label=size)
        axes[2].plot(xs, secondary, style, label=size)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("train loss")
    axes[1].set_ylabel(headline_label)
    if ylim is not None:
        axes[1].set_ylim(*ylim)
    axes[2].set_ylabel(secondary_label)
    for ax in axes:
        ax.set_xlabel(x)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(task)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dp_{task}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_family_summary(
    task_to_data: dict[str, dict[str, list[dict]]], family: str, out_dir: Path, x: str
) -> Path | None:
    tasks = [t for t in task_to_data if family_of(t) == family]
    if not tasks:
        return None
    headline_key, headline_label, ylim = HEADLINE[family]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, task in enumerate(sorted(tasks)):
        color = f"C{i % 10}"
        for size, rows in task_to_data[task].items():
            if not rows:
                continue
            xs = [r["train_steps"] if x == "steps" else r["epoch"] for r in rows]
            headline = [r["metrics"].get(headline_key, np.nan) for r in rows]
            style = "--" if size == "10M" else "-"
            label = task if size in ("1M", "full") else None
            ax.plot(xs, headline, style, color=color, alpha=0.85, label=label)
    ax.set_xlabel(x)
    ax.set_ylabel(headline_label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(f"{family} summary (solid=1M, dashed=10M)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dp_summary_{family}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_summary_md(task_to_data: dict[str, dict[str, list[dict]]], out_dir: Path) -> Path:
    lines = [
        "# Diffusion policy training summary",
        "",
        "| task | size | epochs | grad steps | last | best | tail10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for task in sorted(task_to_data):
        family = family_of(task)
        headline_key = HEADLINE[family][0]
        for size, rows in sorted(task_to_data[task].items(), key=lambda kv: _size_key(kv[0])):
            if not rows:
                continue
            epochs = rows[-1]["epoch"]
            steps = rows[-1]["train_steps"]
            vals = [r["metrics"].get(headline_key, np.nan) for r in rows]
            last = vals[-1]
            best = float(np.nanmax(vals))
            # `last` alone swings enough at small data scales to be
            # unstable between consecutive evals; tail10 is a steadier read.
            tail10 = float(np.nanmean(vals[-10:]))
            lines.append(
                f"| {task} | {size} | {epochs} | {steps} | {last:.3f} | {best:.3f} | {tail10:.3f} |"
            )
    out_path = out_dir / "dp_summary.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def load_expert_reference(task: str) -> float | None:
    path = Path("outputs/expert_eval") / f"{task}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    metrics = data.get("metrics", data)
    key = HEADLINE[family_of(task)][0]
    return metrics.get(key)


def plot_scaling(
    task_to_data: dict[str, dict[str, list[dict]]], family: str, out_dir: Path
) -> Path | None:
    """dp_scaling_<family>.png: headline metric vs dataset size on a log-ish
    (ordinal) x-axis, one line per hand. Circles are the mean of the last 10
    evals, triangles the best-of-run, dotted horizontal lines the RL expert
    reference loaded from outputs/expert_eval/<Task>.json."""
    tasks = [t for t in task_to_data if family_of(t) == family]
    if not tasks:
        return None
    headline_key, headline_label, ylim = HEADLINE[family]
    size_to_x = {s: i for i, s in enumerate(SIZE_ORDER)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, task in enumerate(sorted(tasks)):
        color = f"C{i % 10}"
        points = []
        for size, rows in task_to_data[task].items():
            if size not in size_to_x or not rows:
                continue
            vals = [r["metrics"].get(headline_key, np.nan) for r in rows]
            points.append(
                (size_to_x[size], float(np.nanmean(vals[-10:])), float(np.nanmax(vals)))
            )
        if not points:
            continue
        points.sort(key=lambda p: p[0])
        xs, means, bests = zip(*points, strict=True)
        ax.plot(xs, means, "o-", color=color, label=task)
        ax.plot(xs, bests, "^", color=color, alpha=0.6)
        ref = load_expert_reference(task)
        if ref is not None:
            ax.axhline(ref, color=color, linestyle=":", alpha=0.5)

    ax.set_xticks(range(len(SIZE_ORDER)))
    ax.set_xticklabels(SIZE_ORDER)
    ax.set_xlabel("dataset size")
    ax.set_ylabel(headline_label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(f"{family}: {headline_label} vs dataset size")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dp_scaling_{family}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diffusion-root", type=Path, default=Path("outputs/diffusion"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    ap.add_argument("--x", choices=["steps", "epoch"], default="steps")
    args = ap.parse_args()

    run_dirs = discover(args.diffusion_root.resolve())
    task_to_data: dict[str, dict[str, list[dict]]] = {}
    for run_dir in run_dirs:
        task, size = parse_run(run_dir)
        rows = rows_for_task(load_rows(run_dir / "eval_metrics.jsonl"), task)
        if not rows:
            continue
        task_to_data.setdefault(task, {})[size] = rows

    saved: list[Path] = []
    for task, size_to_rows in task_to_data.items():
        p = plot_run(task, size_to_rows, args.output_dir.resolve(), args.x)
        if p is not None:
            saved.append(p)
            print(f"Saved {p}")

    for family in ("grasp", "rotation"):
        p = plot_family_summary(task_to_data, family, args.output_dir.resolve(), args.x)
        if p is not None:
            saved.append(p)
            print(f"Saved {p}")
        p = plot_scaling(task_to_data, family, args.output_dir.resolve())
        if p is not None:
            saved.append(p)
            print(f"Saved {p}")

    if task_to_data:
        md_path = write_summary_md(task_to_data, args.output_dir.resolve())
        print(f"Saved {md_path}")

    if not saved:
        raise SystemExit(f"No diffusion eval_metrics.jsonl found under {args.diffusion_root}")
    print(f"Generated {len(saved)} plot(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
