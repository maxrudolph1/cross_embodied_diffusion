#!/usr/bin/env python3
"""Diagnostic: does noise actually hide embodiment identity in the action?

Noises both embodiments' actions on the policy's own cosine schedule and
measures W1 per dimension over pooled std (reusing `wasserstein1()` from
`compare_matched_embodiments.py`), as a function of diffusion timestep t.
Answers whether the Ambient-Diffusion premise -- that the sample becomes
unidentifiable at high noise -- actually holds here, before spending any
training compute on the sigma* sweep. See CHANGES.md item 28 / ANALYSIS.md
"the premise check, before the sweep lands".

Only the action is noised: the observation is never noised in this
pipeline (it enters `global_cond` at full fidelity at every reverse step),
so this measures an upper bound on how hard the offset is to handle, not
whether the network actually handles it -- the condition really required is
that p(a_sigma | o) converge between embodiments, which this does not test.

Writes outputs/analysis/ambient_threshold.json and
outputs/plots/ambient_threshold.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from compare_matched_embodiments import wasserstein1

from mjlab_hand.diffusion.dataset import TrajectoryStore
from mjlab_hand.diffusion.policy import cosine_beta_schedule

FAMILIES = {
    "grasp": ("Grasp-Allegro", "Grasp-LEAP"),
    "rotation": ("InHand-Rotation-Allegro", "InHand-Rotation-LEAP"),
}


def per_dim_w1_over_std(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a.shape[1]
    out = np.zeros(d)
    for j in range(d):
        pooled_std = max(float(np.concatenate([a[:, j], b[:, j]]).std()), 1e-8)
        out[j] = wasserstein1(a[:, j], b[:, j]) / pooled_std
    return out


def noise_at(action: np.ndarray, alpha_bar_t: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(action.shape).astype(np.float64)
    return np.sqrt(alpha_bar_t) * action + np.sqrt(max(1.0 - alpha_bar_t, 0.0)) * noise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos-dir", type=Path, default=Path("data/demos"))
    ap.add_argument("--pattern", default="{task}_expert.zarr")
    ap.add_argument("--num-train-timesteps", type=int, default=100)
    ap.add_argument("--timesteps", type=int, nargs="+", default=[0, 25, 50, 75, 90, 99])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    ap.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    args = ap.parse_args()

    betas = cosine_beta_schedule(args.num_train_timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).numpy()

    results: dict[str, dict] = {}
    for family, (task_a, task_b) in FAMILIES.items():
        path_a = args.demos_dir / args.pattern.format(task=task_a)
        path_b = args.demos_dir / args.pattern.format(task=task_b)
        if not path_a.exists() or not path_b.exists():
            print(f"[WARN] Missing dataset(s) for {family}: {path_a}, {path_b}")
            continue
        act_a = np.asarray(TrajectoryStore(path_a, mode="r").data["action"][:], dtype=np.float64)
        act_b = np.asarray(TrajectoryStore(path_b, mode="r").data["action"][:], dtype=np.float64)
        if act_a.shape[1] != act_b.shape[1]:
            print(f"[WARN] Mismatched action dims for {family}, skipping")
            continue

        raw_curve, centred_curve = [], []
        for t in args.timesteps:
            ab = float(alphas_cumprod[t])
            na = noise_at(act_a, ab, args.seed)
            nb = noise_at(act_b, ab, args.seed + 1)
            raw_curve.append(float(per_dim_w1_over_std(na, nb).mean()))
            centred_curve.append(
                float(per_dim_w1_over_std(na - na.mean(axis=0), nb - nb.mean(axis=0)).mean())
            )
        results[family] = {"timesteps": args.timesteps, "raw": raw_curve, "mean_centred": centred_curve}
        print(
            f"[INFO] {family}: raw={[round(v, 3) for v in raw_curve]} "
            f"centred={[round(v, 3) for v in centred_curve]}"
        )

    if not results:
        raise SystemExit("No matched-pair datasets found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ambient_threshold.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (family, r) in enumerate(results.items()):
        ax.plot(r["timesteps"], r["raw"], "o-", color=f"C{i * 2}", label=f"{family} raw")
        ax.plot(r["timesteps"], r["mean_centred"], "o--", color=f"C{i * 2 + 1}", label=f"{family} centred")
    ax.set_xlabel("diffusion timestep t")
    ax.set_ylabel("mean per-dim W1 / pooled std")
    ax.set_title("Allegro vs LEAP action distance vs noise level")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.plots_dir / "ambient_threshold.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"[INFO] Wrote {args.output_dir / 'ambient_threshold.json'}")
    print(f"[INFO] Wrote {out_png}")


if __name__ == "__main__":
    main()
