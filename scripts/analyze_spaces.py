#!/usr/bin/env python3
"""Distributional analysis of observation/action spaces across embodiments.

Per-dim and per-term statistics, action saturation (fraction of stored
action entries with |a| >= 1 -- actions are stored pre-clip: `collect_demos`
records the raw policy output while `RslRlVecEnvWrapper` clamps inside
`env.step`, so a BC policy sees the same clip at rollout), and intrinsic
dimensionality via PCA on standardised features.

Works in float64 and drops near-constant dims (std < CONST_STD_EPS) rather
than standardising them: Grasp-Shadow has an observation dim with std
9.4e-08, so standardising in float32 amplifies float32 quantisation noise
(eps ~1.2e-7 relative) to O(1), producing a correlated noise block that
dominates the SVD (measured: reported pc_95 = 1 for a 189-d space before
this fix). See CHANGES.md item 19.

Writes outputs/analysis/spaces_summary.md, spaces_stats.json, and
space_action_ranges.png / space_pca.png / space_term_matrix.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mjlab_hand.diffusion.dataset import TrajectoryStore

CONST_STD_EPS = 1e-6

TASKS = [
    "Grasp-Allegro",
    "Grasp-LEAP",
    "Grasp-Shadow",
    "Grasp-Sharpa",
    "Grasp-Wuji",
    "InHand-Rotation-Allegro",
    "InHand-Rotation-LEAP",
    "InHand-Rotation-Shadow",
    "InHand-Rotation-Sharpa",
    "InHand-Rotation-Wuji",
]


def pca_95(x: np.ndarray) -> tuple[int, int]:
    """(components for 95% variance, number of dims dropped as constant).

    Works in float64; drops dims with std < CONST_STD_EPS instead of
    standardising them, which would otherwise amplify float32 quantisation
    noise on near-constant dims into a spurious dominant SVD direction.
    """
    x = x.astype(np.float64)
    std = x.std(axis=0)
    keep = std >= CONST_STD_EPS
    n_const = int((~keep).sum())
    if not keep.any():
        return 0, n_const
    xs = (x[:, keep] - x[:, keep].mean(axis=0)) / std[keep]
    s = np.linalg.svd(xs, full_matrices=False, compute_uv=False)
    var = s**2
    total = var.sum()
    if total <= 0:
        return 0, n_const
    frac = np.cumsum(var) / total
    n95 = int(np.searchsorted(frac, 0.95) + 1)
    return n95, n_const


def analyze_task(task: str, path: Path, schema: dict | None) -> dict:
    store = TrajectoryStore(path, mode="r")
    obs = np.asarray(store.data["obs"][:], dtype=np.float32)
    action = np.asarray(store.data["action"][:], dtype=np.float32)

    obs_pc95, obs_const = pca_95(obs)
    act_pc95, act_const = pca_95(action)

    act_abs = np.abs(action)
    saturation = float((act_abs >= 1.0).mean())
    percentiles = {str(p): float(np.percentile(act_abs, p)) for p in (50, 90, 99)}

    result: dict = {
        "task": task,
        "obs_dim": int(obs.shape[1]),
        "obs_const": obs_const,
        "obs_pc95": obs_pc95,
        "action_dim": int(action.shape[1]),
        "action_const": act_const,
        "action_pc95": act_pc95,
        "action_saturation": saturation,
        "action_abs_percentiles": percentiles,
        "action_abs_max": float(act_abs.max()),
    }
    if schema is not None:
        result["obs_terms"] = schema.get("obs_terms")
        result["obs_term_dims"] = schema.get("obs_term_dims")
    return result


def plot_action_ranges(results: dict[str, dict], out_path: Path) -> None:
    tasks = sorted(results)
    saturations = [results[t]["action_saturation"] for t in tasks]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(tasks)), saturations, color="C0")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("fraction of |action| >= 1 (stored pre-clip)")
    ax.set_title("Action saturation by task")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca(results: dict[str, dict], out_path: Path) -> None:
    tasks = sorted(results)
    x = np.arange(len(tasks))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(x - 0.2, [results[t]["obs_dim"] for t in tasks], width=0.4, label="nominal dim", color="C1")
    axes[0].bar(x + 0.2, [results[t]["obs_pc95"] for t in tasks], width=0.4, label="pc_95", color="C0")
    axes[0].set_title("Observation intrinsic dimensionality")
    axes[1].bar(x - 0.2, [results[t]["action_dim"] for t in tasks], width=0.4, label="nominal dim", color="C1")
    axes[1].bar(x + 0.2, [results[t]["action_pc95"] for t in tasks], width=0.4, label="pc_95", color="C0")
    axes[1].set_title("Action intrinsic dimensionality")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_term_matrix(results: dict[str, dict], out_path: Path) -> None:
    """Presence/dim of each observation term across tasks."""
    tasks = sorted(t for t in results if results[t].get("obs_terms"))
    if not tasks:
        print("[WARN] No schema info available; skipping space_term_matrix.png")
        return
    all_terms = sorted({t for r in results.values() for t in (r.get("obs_terms") or [])})
    mat = np.zeros((len(tasks), len(all_terms)))
    for i, task in enumerate(tasks):
        names = results[task]["obs_terms"]
        dims = results[task]["obs_term_dims"]
        for name, dim in zip(names, dims, strict=True):
            mat[i, all_terms.index(name)] = dim

    fig, ax = plt.subplots(figsize=(max(8, len(all_terms) * 0.6), max(4, len(tasks) * 0.5)))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(all_terms)))
    ax.set_xticklabels(all_terms, rotation=90)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    fig.colorbar(im, ax=ax, label="term dim (0 = absent)")
    ax.set_title("Observation term layout by task")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_md(results: dict[str, dict], out_path: Path) -> None:
    lines = [
        "# Observation / action space summary",
        "",
        "| Task | obs dim | const | obs pc_95 | act dim | act pc_95 | act saturation |",
        "|---|---|---|---|---|---|---|",
    ]
    for task in sorted(results):
        r = results[task]
        lines.append(
            f"| {task} | {r['obs_dim']} | {r['obs_const']} | {r['obs_pc95']} | "
            f"{r['action_dim']} | {r['action_pc95']} | {r['action_saturation']:.1%} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos-dir", type=Path, default=Path("data/demos"))
    ap.add_argument(
        "--pattern",
        default="{task}_expert.zarr",
        help="Dataset filename pattern relative to --demos-dir, with {task} substituted.",
    )
    ap.add_argument("--schemas", type=Path, default=Path("outputs/analysis/schemas.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    ap.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    args = ap.parse_args()

    schemas = json.loads(args.schemas.read_text()) if args.schemas.exists() else {}

    results: dict[str, dict] = {}
    for task in TASKS:
        path = args.demos_dir / args.pattern.format(task=task)
        if not path.exists():
            print(f"[WARN] Missing dataset for {task}: {path}")
            continue
        print(f"[INFO] Analyzing {task} ({path})")
        results[task] = analyze_task(task, path, schemas.get(task))

    if not results:
        raise SystemExit(f"No datasets found under {args.demos_dir} matching {args.pattern}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "spaces_stats.json").write_text(json.dumps(results, indent=2))
    write_summary_md(results, args.output_dir / "spaces_summary.md")

    plot_action_ranges(results, args.plots_dir / "space_action_ranges.png")
    plot_pca(results, args.plots_dir / "space_pca.png")
    plot_term_matrix(results, args.plots_dir / "space_term_matrix.png")

    print(f"[INFO] Wrote {args.output_dir / 'spaces_stats.json'}")
    print(f"[INFO] Wrote {args.output_dir / 'spaces_summary.md'}")


if __name__ == "__main__":
    main()
