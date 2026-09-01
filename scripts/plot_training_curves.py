#!/usr/bin/env python3
"""Plot training curves from TensorBoard event files under logs/rsl_rl/."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Metrics/grasp_metrics/object_height_beyond_table",
    "Metrics/rotation/episode_success",
    "Loss/value",
    "Loss/surrogate",
    "Loss/learning_rate",
    "Perf/total_fps",
]


def load_scalars(log_dir: Path, tags: list[str]) -> dict[str, tuple[list[int], list[float]]]:
    events = sorted(log_dir.glob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"No TensorBoard events found in {log_dir}")

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    acc.Reload()

    available = set(acc.Tags().get("scalars", []))
    out: dict[str, tuple[list[int], list[float]]] = {}
    for tag in tags:
        if tag not in available:
            continue
        series = acc.Scalars(tag)
        out[tag] = ([s.step for s in series], [s.value for s in series])
    return out


def plot_run(log_dir: Path, output_dir: Path, tags: list[str]) -> Path | None:
    scalars = load_scalars(log_dir, tags)
    if not scalars:
        return None

    n = len(scalars)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (tag, (steps, values)) in zip(axes, scalars.items(), strict=True):
        ax.plot(steps, values, linewidth=1.5)
        ax.set_ylabel(tag.split("/")[-1])
        ax.grid(True, alpha=0.3)
        ax.set_title(tag)

    axes[-1].set_xlabel("iteration")
    fig.suptitle(log_dir.name, fontsize=12)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{log_dir.parent.name}_{log_dir.name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def discover_runs(log_root: Path) -> list[Path]:
    runs: list[Path] = []
    for exp_dir in sorted(log_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if run_dir.is_dir() and list(run_dir.glob("events.out.tfevents.*")):
                runs.append(run_dir)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs/rsl_rl"),
        help="Root directory containing experiment logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plots"),
        help="Directory for saved plot PNGs",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=DEFAULT_TAGS,
        help="TensorBoard scalar tags to plot",
    )
    args = parser.parse_args()

    runs = discover_runs(args.log_root.resolve())
    if not runs:
        raise SystemExit(f"No completed runs with TensorBoard logs under {args.log_root}")

    saved: list[Path] = []
    for run_dir in runs:
        try:
            path = plot_run(run_dir, args.output_dir.resolve(), args.tags)
        except FileNotFoundError:
            continue
        if path is not None:
            saved.append(path)
            print(f"Saved {path}")

    if not saved:
        raise SystemExit("No plots generated (missing scalar tags in event files).")
    print(f"Generated {len(saved)} plot(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
