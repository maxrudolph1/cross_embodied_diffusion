#!/usr/bin/env python3
"""Elementwise comparison of Allegro vs LEAP -- the only observation/action
space pair that matches in both dimension and term layout (see
`analyze_spaces.py` / ANALYSIS.md), so it's the one pair where a
per-dimension comparison means "same physical quantity, dimension j in
both".

Metric: 1-D Wasserstein-1 per dimension, divided by the pooled std of that
dimension (no scipy needed -- quantile functions), aggregated per
observation term. Reports both raw and mean-centred distance: a large *raw*
W1 on a low-variance feature is a systematic offset, not a distribution-shape
difference (e.g. rotation `object_pos` scores ~4.4 raw on a feature with std
~0.034, because the two palms hold the object ~9cm apart; centred it drops
to ~0.31 -- reading raw distances alone would flag the most trivially
correctable feature as the biggest obstacle). See CHANGES.md item 20.

`wasserstein1()` is reused by `measure_ambient_threshold.py`.

Writes outputs/analysis/matched_pairs.json, matched_pairs_summary.md, and
outputs/plots/space_matched_pairs.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mjlab_hand.diffusion.dataset import TrajectoryStore

FAMILIES = {
    "grasp": ("Grasp-Allegro", "Grasp-LEAP"),
    "rotation": ("InHand-Rotation-Allegro", "InHand-Rotation-LEAP"),
}


def wasserstein1(a: np.ndarray, b: np.ndarray, n_quantiles: int = 1000) -> float:
    """1-D Wasserstein-1 distance via quantile functions (no scipy)."""
    qs = np.linspace(0.0, 1.0, n_quantiles)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def per_dim_w1_over_std(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-dimension W1 divided by the pooled std of that dimension."""
    d = a.shape[1]
    out = np.zeros(d)
    for j in range(d):
        pooled_std = max(float(np.concatenate([a[:, j], b[:, j]]).std()), 1e-8)
        out[j] = wasserstein1(a[:, j], b[:, j]) / pooled_std
    return out


def term_slices(names: list[str], dims: list[int]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    start = 0
    for name, d in zip(names, dims, strict=True):
        out[name] = (start, start + d)
        start += d
    return out


def compare(task_a: str, task_b: str, demos_dir: Path, pattern: str, schema_a: dict | None) -> dict:
    store_a = TrajectoryStore(demos_dir / pattern.format(task=task_a), mode="r")
    store_b = TrajectoryStore(demos_dir / pattern.format(task=task_b), mode="r")
    obs_a = np.asarray(store_a.data["obs"][:], dtype=np.float64)
    obs_b = np.asarray(store_b.data["obs"][:], dtype=np.float64)
    act_a = np.asarray(store_a.data["action"][:], dtype=np.float64)
    act_b = np.asarray(store_b.data["action"][:], dtype=np.float64)

    if obs_a.shape[1] != obs_b.shape[1] or act_a.shape[1] != act_b.shape[1]:
        raise ValueError(
            f"{task_a} and {task_b} have mismatched dims "
            f"(obs {obs_a.shape[1]} vs {obs_b.shape[1]}, act {act_a.shape[1]} vs {act_b.shape[1]})"
        )

    obs_raw = per_dim_w1_over_std(obs_a, obs_b)
    act_raw = per_dim_w1_over_std(act_a, act_b)
    obs_centred = per_dim_w1_over_std(obs_a - obs_a.mean(axis=0), obs_b - obs_b.mean(axis=0))
    act_centred = per_dim_w1_over_std(act_a - act_a.mean(axis=0), act_b - act_b.mean(axis=0))

    result: dict = {
        "obs_raw_mean": float(obs_raw.mean()),
        "obs_centred_mean": float(obs_centred.mean()),
        "action_raw_mean": float(act_raw.mean()),
        "action_centred_mean": float(act_centred.mean()),
        "obs_raw_per_dim": obs_raw.tolist(),
        "obs_centred_per_dim": obs_centred.tolist(),
        "action_raw_per_dim": act_raw.tolist(),
        "action_centred_per_dim": act_centred.tolist(),
    }

    if schema_a is not None and schema_a.get("obs_terms"):
        slices = term_slices(schema_a["obs_terms"], schema_a["obs_term_dims"])
        result["obs_by_term"] = {
            name: {"raw": float(obs_raw[s:e].mean()), "centred": float(obs_centred[s:e].mean())}
            for name, (s, e) in slices.items()
        }

    return result


def plot_summary(results: dict[str, dict], out_path: Path) -> None:
    labels, raw_vals, centred_vals = [], [], []
    for fam in sorted(results):
        for kind in ("obs", "action"):
            labels.append(f"{fam} {kind}")
            raw_vals.append(results[fam][f"{kind}_raw_mean"])
            centred_vals.append(results[fam][f"{kind}_centred_mean"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, raw_vals, width=0.4, label="raw", color="C1")
    ax.bar(x + 0.2, centred_vals, width=0.4, label="mean-centred", color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mean per-dim W1 / pooled std")
    ax.set_title("Allegro vs LEAP: raw vs mean-centred distance")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos-dir", type=Path, default=Path("data/demos"))
    ap.add_argument("--pattern", default="{task}_expert.zarr")
    ap.add_argument("--schemas", type=Path, default=Path("outputs/analysis/schemas.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    ap.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    args = ap.parse_args()

    schemas = json.loads(args.schemas.read_text()) if args.schemas.exists() else {}

    results: dict[str, dict] = {}
    for family, (task_a, task_b) in FAMILIES.items():
        path_a = args.demos_dir / args.pattern.format(task=task_a)
        path_b = args.demos_dir / args.pattern.format(task=task_b)
        if not path_a.exists() or not path_b.exists():
            print(f"[WARN] Missing dataset(s) for {family}: {path_a}, {path_b}")
            continue
        print(f"[INFO] Comparing {task_a} vs {task_b}")
        results[family] = compare(
            task_a, task_b, args.demos_dir, args.pattern, schemas.get(task_a)
        )

    if not results:
        raise SystemExit(
            "No matched-pair datasets found (need both Allegro and LEAP for a family)"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "matched_pairs.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Allegro vs LEAP matched-pair comparison",
        "",
        "| family | obs raw | obs centred | act raw | act centred |",
        "|---|---|---|---|---|",
    ]
    for family, r in results.items():
        lines.append(
            f"| {family} | {r['obs_raw_mean']:.3f} | {r['obs_centred_mean']:.3f} | "
            f"{r['action_raw_mean']:.3f} | {r['action_centred_mean']:.3f} |"
        )
        if "obs_by_term" in r:
            lines += ["", f"### {family}, by term", "", "| term | raw | centred |", "|---|---|---|"]
            for term, vals in sorted(r["obs_by_term"].items(), key=lambda kv: -kv[1]["raw"]):
                lines.append(f"| {term} | {vals['raw']:.3f} | {vals['centred']:.3f} |")
            lines.append("")
    (args.output_dir / "matched_pairs_summary.md").write_text("\n".join(lines) + "\n")

    plot_summary(results, args.plots_dir / "space_matched_pairs.png")
    print(f"[INFO] Wrote {args.output_dir / 'matched_pairs.json'}")
    print(f"[INFO] Wrote {args.output_dir / 'matched_pairs_summary.md'}")


if __name__ == "__main__":
    main()
