#!/usr/bin/env python3
"""Confusion-matrix heatmaps of the mixed-embodiment grid.

Rows = Allegro data size, columns = LEAP data size. Left panel: absolute
headline metric (sequential palette). Right panel: delta vs the
single-embodiment baseline at the same Allegro size (diverging palette,
neutral at zero). The diagonal (same size on both sides) is drawn as an
inert surface -- `build_mixed.sbatch` skips a == b, so it was never run.

Two correctness fixes baked in (CHANGES.md item 22):

- **Cells are gated on the run reaching its last *scheduled* eval.** Eval
  fires at the midpoint and the end, so a still-running job already has
  rows on disk; taking "the last row present" would plot a midpoint value
  styled the same as a completed one.
- **`last_scheduled = 2 * (total // 2)`**, not `epoch == num_epochs`:
  `EVAL_EVERY = EPOCHS / 2` is integer division, so for an odd
  `num_epochs` the last eval fires one epoch before the end, and requiring
  an exact match at `num_epochs` silently blanks those runs.

`--seeds` (item 29): pass one seed to plot it alone, or several to average
per cell; the filename/subtitle record which, and the per-cell seed count
is printed so a 1-seed and a 2-seed cell are never drawn identically.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SIZES = ["10k", "50k", "100k", "400k", "1M"]
FAMILIES = {
    "grasp": ("Grasp-Allegro", "Grasp-LEAP"),
    "rotation": ("InHand-Rotation-Allegro", "InHand-Rotation-LEAP"),
}
HEADLINE = {"grasp": "success_rate", "rotation": "avg_successes_before_drop"}

RUN_RE = re.compile(r"^(?P<family>.+)_A(?P<size_a>\w+)_L(?P<size_b>\w+)(?:_s(?P<seed>\d+))?$")


def load_rows(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        return []
    out = []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def final_value(rows: list[dict], eval_task: str | None, headline_key: str) -> float | None:
    """Value at the last *scheduled* eval, not merely the last row present."""
    if not rows:
        return None
    total = max(r["epoch"] for r in rows)
    last_scheduled = 2 * (total // 2)
    final = [
        r
        for r in rows
        if r.get("epoch") == last_scheduled
        and (eval_task is None or r.get("eval_task") == eval_task)
    ]
    if not final:
        return None
    return final[-1]["metrics"].get(headline_key)


def discover_mixed_runs(root: Path, family_key: str) -> dict[tuple[str, str, int], Path]:
    """{(size_a, size_b, seed): run_dir} for a family's mixed runs."""
    out: dict[tuple[str, str, int], Path] = {}
    if not root.exists():
        return out
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m or family_key not in run_dir.name.lower():
            continue
        seed = int(m.group("seed")) if m.group("seed") else 0
        out[(m.group("size_a"), m.group("size_b"), seed)] = run_dir
    return out


def discover_solo_runs(root: Path, task: str) -> dict[str, Path]:
    """{size: run_dir} for a task's single-embodiment diffusion runs."""
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for run_dir in root.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith(f"{task}_"):
            out[run_dir.name[len(task) + 1 :]] = run_dir
    return out


def build_matrix(
    family: str, task_a: str, mixed_root: Path, solo_root: Path, seeds: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    headline_key = HEADLINE[family]
    mixed = discover_mixed_runs(mixed_root, family.lower())
    solo_a = discover_solo_runs(solo_root, task_a)

    n = len(SIZES)
    abs_mat = np.full((n, n), np.nan)
    delta_mat = np.full((n, n), np.nan)
    count_mat = np.zeros((n, n), dtype=int)

    for i, size_a in enumerate(SIZES):
        baseline = None
        if size_a in solo_a:
            baseline = final_value(
                load_rows(solo_a[size_a] / "eval_metrics.jsonl"), None, headline_key
            )
        for j, size_b in enumerate(SIZES):
            if size_a == size_b:
                continue  # diagonal never run
            vals = []
            for seed in seeds:
                run_dir = mixed.get((size_a, size_b, seed))
                if run_dir is None:
                    continue
                v = final_value(load_rows(run_dir / "eval_metrics.jsonl"), task_a, headline_key)
                if v is not None:
                    vals.append(v)
            if not vals:
                continue
            abs_mat[i, j] = float(np.mean(vals))
            count_mat[i, j] = len(vals)
            if baseline is not None:
                delta_mat[i, j] = abs_mat[i, j] - baseline

    return abs_mat, delta_mat, count_mat


def plot_family(
    family: str,
    abs_mat: np.ndarray,
    delta_mat: np.ndarray,
    count_mat: np.ndarray,
    seeds: list[int],
    out_path: Path,
) -> None:
    n = len(SIZES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    vmax = np.nanmax(abs_mat) if np.isfinite(abs_mat).any() else 1.0
    vmin = np.nanmin(abs_mat) if np.isfinite(abs_mat).any() else 0.0
    im0 = axes[0].imshow(abs_mat, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{family}: absolute {HEADLINE[family]}")
    fig.colorbar(im0, ax=axes[0])

    lim = np.nanmax(np.abs(delta_mat)) if np.isfinite(delta_mat).any() else 1.0
    im1 = axes[1].imshow(delta_mat, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[1].set_title(f"{family}: delta vs solo baseline")
    fig.colorbar(im1, ax=axes[1])

    for ax, mat in ((axes[0], abs_mat), (axes[1], delta_mat)):
        ax.set_xticks(range(n))
        ax.set_xticklabels(SIZES)
        ax.set_yticks(range(n))
        ax.set_yticklabels(SIZES)
        ax.set_xlabel("LEAP data size")
        ax.set_ylabel("Allegro data size")
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="lightgray"))
                    continue
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)

    seed_str = "+".join(str(s) for s in seeds)
    fig.suptitle(f"{family} mixed-embodiment grid (seeds={seed_str})")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] {family} per-cell seed counts:")
    for i, size_a in enumerate(SIZES):
        for j, size_b in enumerate(SIZES):
            if i != j and count_mat[i, j] > 0:
                print(f"  A{size_a}_L{size_b}: {count_mat[i, j]} seed(s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mixed-root", type=Path, default=Path("outputs/mixed"))
    ap.add_argument("--diffusion-root", type=Path, default=Path("outputs/diffusion"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    saved = 0
    for family, (task_a, _task_b) in FAMILIES.items():
        abs_mat, delta_mat, count_mat = build_matrix(
            family, task_a, args.mixed_root.resolve(), args.diffusion_root.resolve(), args.seeds
        )
        if not np.isfinite(abs_mat).any():
            print(f"[WARN] No mixed runs found for {family}, skipping")
            continue
        seed_tag = "".join(str(s) for s in args.seeds)
        display_family = "in-hand_rotation" if family == "rotation" else "grasp"
        out_path = args.output_dir.resolve() / f"mixed_matrix_{display_family}_seed{seed_tag}.png"
        plot_family(family, abs_mat, delta_mat, count_mat, args.seeds, out_path)
        print(f"Saved {out_path}")
        saved += 1

    if saved == 0:
        raise SystemExit(f"No mixed-embodiment runs found under {args.mixed_root}")


if __name__ == "__main__":
    main()
