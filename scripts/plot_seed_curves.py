#!/usr/bin/env python3
"""Seed-aggregating training-curve plotter.

`plot_training_curves.py` emits one PNG per run (e.g. 30 disconnected
figures for a 3-seed x 10-task sweep) and hides seed structure. This script
groups runs by experiment, interpolates each seed onto a shared step grid,
and plots mean +/- 1 std with faint per-seed traces, plus
`summary_grasp.png` / `summary_rotation.png` overviews across tasks.

Safe to run mid-training: seeds are truncated to their shortest common step
range. See CHANGES.md item 11 (new) and item 7 (run-dir regex fix -- a
longer re-run named `<timestamp>_seed<N>_<variant>`, e.g. `_seed42_40k`, is
grouped as its own experiment rather than being silently invisible or
merged into the original).
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Anchored at end of string; captures an optional trailing variant suffix.
SEED_RE = re.compile(r"_seed(\d+)(?:_([A-Za-z0-9]+))?$")

HEADLINE_TAG = {
    "grasp": "Metrics/grasp_metrics/object_height_beyond_table",
    "rotation": "Metrics/rotation/episode_success",
}


def family_of(exp_name: str) -> str:
    return "rotation" if "rotation" in exp_name.lower() else "grasp"


def discover(log_root: Path) -> dict[str, dict[int, Path]]:
    """{experiment_key: {seed: run_dir}}. `experiment_key` folds in the
    variant suffix (if any) so a longer re-run plots as its own experiment."""
    found: dict[str, dict[int, Path]] = defaultdict(dict)
    if not log_root.exists():
        return found
    for exp_dir in sorted(log_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = SEED_RE.search(run_dir.name)
            if not m:
                continue
            seed = int(m.group(1))
            variant = m.group(2)
            key = f"{exp_dir.name}_{variant}" if variant else exp_dir.name
            prev = found[key].get(seed)
            if prev is None or run_dir.name > prev.name:
                found[key][seed] = run_dir
    return found


def load_series(run: Path, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    events = sorted(run.glob("events.out.tfevents.*"))
    if not events:
        return None
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    series = acc.Scalars(tag)
    if not series:
        return None
    return np.array([s.step for s in series]), np.array([s.value for s in series])


def aggregate(seed_runs: dict[int, Path], tag: str, n_grid: int = 200):
    """Interpolate each seed onto a shared step grid, truncated to the
    shortest common range. Returns (grid, mean, std, per_seed) or None."""
    per_seed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    starts, ends = [], []
    for seed, run in seed_runs.items():
        s = load_series(run, tag)
        if s is None:
            continue
        steps, values = s
        per_seed[seed] = (steps, values)
        starts.append(steps.min())
        ends.append(steps.max())
    if not per_seed:
        return None
    lo, hi = max(starts), min(ends)
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, n_grid)
    interped = {seed: np.interp(grid, s, v) for seed, (s, v) in per_seed.items()}
    stacked = np.stack(list(interped.values()), axis=0)
    return grid, stacked.mean(axis=0), stacked.std(axis=0), interped


def plot_experiment(key: str, seed_runs: dict[int, Path], tag: str, out_dir: Path) -> Path | None:
    agg = aggregate(seed_runs, tag)
    if agg is None:
        return None
    grid, mean, std, per_seed = agg

    fig, ax = plt.subplots(figsize=(7, 4))
    for seed, values in per_seed.items():
        ax.plot(grid, values, alpha=0.25, linewidth=1.0, label=f"seed {seed}")
    ax.plot(grid, mean, color="C0", linewidth=2.0, label="mean")
    ax.fill_between(grid, mean - std, mean + std, color="C0", alpha=0.2)
    ax.set_xlabel("iteration")
    ax.set_ylabel(tag.split("/")[-1])
    ax.set_title(f"{key} ({len(per_seed)} seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key}_seeds.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_summary(experiments: dict[str, dict], family: str, out_dir: Path) -> Path | None:
    keys = [k for k, v in experiments.items() if v["family"] == family]
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, key in enumerate(sorted(keys)):
        grid, mean, std, _ = experiments[key]["agg"]
        color = f"C{i % 10}"
        ax.plot(grid, mean, color=color, label=key, linewidth=1.5)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("iteration")
    ax.set_ylabel(HEADLINE_TAG[family].split("/")[-1])
    ax.set_title(f"{family} summary")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{family}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    args = ap.parse_args()

    runs = discover(args.log_root.resolve())
    experiments: dict[str, dict] = {}
    saved: list[Path] = []
    for key, seed_runs in sorted(runs.items()):
        family = family_of(key)
        tag = HEADLINE_TAG[family]
        path = plot_experiment(key, seed_runs, tag, args.output_dir.resolve())
        if path is not None:
            saved.append(path)
            print(f"Saved {path}")
        agg = aggregate(seed_runs, tag)
        if agg is not None:
            experiments[key] = {"family": family, "agg": agg}

    for family in ("grasp", "rotation"):
        path = plot_summary(experiments, family, args.output_dir.resolve())
        if path is not None:
            saved.append(path)
            print(f"Saved {path}")

    if not saved:
        raise SystemExit(f"No seed-grouped runs with usable TensorBoard data under {args.log_root}")
    print(f"Generated {len(saved)} plot(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
