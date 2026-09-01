#!/usr/bin/env python3
"""Ambient-diffusion sigma* sweep: performance vs gating threshold.

sigma* (ambient t_min, as a percentage of the schedule) on x, headline
performance on y, one line per target data budget N, one panel per (family,
hand). Routes each cell to whichever run actually holds it:

- sigma*=0 (naive mixing, no gating)   -> outputs/mixed_noc/
- interior sigma* (the sweep proper)   -> outputs/ambient/
- sigma*=100 (target-only, no source)  -> outputs/diffusion/
- N=400k, sigma*=0 special case        -> outputs/ambient/..._amb0

Carries the same strict final-epoch guard as `plot_mixed_matrix.py`, and
prefers `eval_metrics_100.jsonl` (from `reeval_endpoints.sbatch`) over
`eval_metrics.jsonl` when present, so 32-rollout and 100-rollout numbers
are never mixed inside one curve. `--require-100` blanks any point not
scored on >= 100 rollouts.

N is an ordered quantity, so it is drawn as one blue hue, light -> dark,
with direct labels rather than a categorical palette.

Bug fixed here (CHANGES.md item 36): once `train.py` gained multi-target
eval (`eval_specs`), even a solo (single-target) run started writing an
`eval_task` key into every row -- `TrainConfig.eval_task` is internally
promoted to a one-element spec list. The original `_final()` identified a
solo run by requiring that key to be *absent*, which held for early runs
but silently blanked every later solo `sigma*=100` cell trained after that
change. Fixed by having `_final` take a `solo_task` argument and accepting
rows tagged with either no `eval_task` or the run's own task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FAMILIES = {
    "grasp": ("Grasp-Allegro", "Grasp-LEAP"),
    "rotation": ("InHand-Rotation-Allegro", "InHand-Rotation-LEAP"),
}
HEADLINE = {"grasp": "success_rate", "rotation": "avg_successes_before_drop"}
# Dataset/run-dir path component -- matches the literal paths in ANALYSIS.md
# (e.g. data/mixed_noc/Grasp_A400k_L400k.zarr), distinct from the lowercase
# `family` key used elsewhere for metric lookups.
DISPLAY_FAMILY = {"grasp": "Grasp", "rotation": "InHand-Rotation"}
N_BUDGETS = ["10k", "50k", "100k", "400k"]
SIGMAS = [0, 50, 75, 90, 100]


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


def _final(
    rows: list[dict], eval_task: str | None, solo_task: str, headline_key: str
) -> tuple[float, float | None] | None:
    """(value, n_rollouts) at the last scheduled eval for `eval_task` (or,
    on the solo sigma*=100 branch where `eval_task=None`, for any row
    tagged with no eval_task or with the run's own `solo_task`)."""
    if not rows:
        return None
    total = max(r["epoch"] for r in rows)
    last_scheduled = 2 * (total // 2)

    def _matches(r: dict) -> bool:
        if eval_task is not None:
            # Mixed/ambient run: the row must name the target being scored.
            return r.get("eval_task") == eval_task
        if "eval_task" in r and r["eval_task"] not in (None, solo_task):
            # Solo run whose rows happen to be tagged: accept only its own task.
            return False
        return True

    final = [r for r in rows if r.get("epoch") == last_scheduled and _matches(r)]
    if not final:
        return None
    row = final[-1]
    return row["metrics"].get(headline_key), row["metrics"].get("completed_episodes")


def _best_jsonl(run_dir: Path) -> Path:
    hi_res = run_dir / "eval_metrics_100.jsonl"
    return hi_res if hi_res.exists() else run_dir / "eval_metrics.jsonl"


def point(
    family: str, n_budget: str, sigma: int, hand: str, seed: int, require_100: bool
) -> tuple[float | None, float | None]:
    """(value, n_rollouts) for one (N, sigma*, hand, seed) cell."""
    task = FAMILIES[family][0] if hand == "Allegro" else FAMILIES[family][1]
    headline_key = HEADLINE[family]
    seed_suffix = f"_s{seed}" if seed else ""
    dfam = DISPLAY_FAMILY[family]

    # The foreign-data partner (LEAP) is fixed at 400k throughout the sweep --
    # only the target (Allegro) budget N varies. N=400k is the one cell where
    # target and partner coincide, which build_mixed.sbatch structurally
    # cannot build (it skips a == b), hence the dedicated `_amb0` dataset
    # from build_ambient.sbatch.
    if sigma == 100:
        run_dir = Path("outputs/diffusion") / f"{task}_{n_budget}{seed_suffix}"
        eval_task = None
    elif sigma == 0 and n_budget == "400k":
        run_dir = Path("outputs/ambient") / f"{dfam}_A400k_L400k_amb0{seed_suffix}"
        eval_task = task
    elif sigma == 0:
        run_dir = Path("outputs/mixed_noc") / f"{dfam}_A{n_budget}_L400k{seed_suffix}"
        eval_task = task
    else:
        run_dir = Path("outputs/ambient") / f"{dfam}_A{n_budget}_L400k_sigma{sigma}{seed_suffix}"
        eval_task = task

    result = _final(load_rows(_best_jsonl(run_dir)), eval_task, task, headline_key)
    if result is None:
        return None, None
    value, n_rollouts = result
    if require_100 and (n_rollouts is None or n_rollouts < 100):
        return None, n_rollouts
    return value, n_rollouts


def plot_panel(family: str, hand: str, seed: int, require_100: bool, out_dir: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(6, 5))
    blues = plt.cm.Blues(np.linspace(0.35, 0.95, len(N_BUDGETS)))
    any_data = False
    for n_budget, color in zip(N_BUDGETS, blues, strict=True):
        xs, ys = [], []
        for sigma in SIGMAS:
            v, _ = point(family, n_budget, sigma, hand, seed, require_100)
            if v is not None:
                xs.append(sigma)
                ys.append(v)
        if xs:
            any_data = True
            ax.plot(xs, ys, "o-", color=color, label=f"N={n_budget}")
    if not any_data:
        plt.close(fig)
        return None
    ax.set_xlabel("sigma* (ambient gating threshold, % of schedule)")
    ax.set_ylabel(HEADLINE[family])
    ax.set_title(f"{family} / {hand} (seed {seed})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ambient_sweep_{family}_{hand.lower()}_seed{seed}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--require-100", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    args = ap.parse_args()

    saved = []
    for family in FAMILIES:
        for hand in ("Allegro", "LEAP"):
            for seed in args.seeds:
                p = plot_panel(family, hand, seed, args.require_100, args.output_dir.resolve())
                if p is not None:
                    saved.append(p)
                    print(f"Saved {p}")

    if not saved:
        raise SystemExit("No ambient sweep runs found under outputs/{mixed_noc,ambient,diffusion}")
    print(f"Generated {len(saved)} plot(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
