#!/usr/bin/env python3
"""Select the best-performing seed/checkpoint per task+embodiment combo.

Scans `logs/rsl_rl/<experiment>/<timestamp>_seed<N>[_<variant>]` run
directories, reads each seed's headline TensorBoard metric, and picks either
the last checkpoint (`--pick last`, default) or the checkpoint nearest the
smoothed peak of the headline curve (`--pick best`). Writes a manifest
(default `outputs/experts.json`) consumed by demo collection.

See CHANGES.md items 6c ("--variant" filter) and 8 (best-checkpoint pick).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HAND_DISPLAY = {
    "allegro": "Allegro",
    "leap": "LEAP",
    "shadow": "Shadow",
    "sharpa": "Sharpa",
    "wuji": "Wuji",
}
HEADLINE_TAG = {
    "grasp": "Metrics/grasp_metrics/object_height_beyond_table",
    "rotation": "Metrics/rotation/episode_success",
}
# Longer re-runs are named `<timestamp>_seed<N>_<variant>` (e.g. `_seed42_40k`).
# Anchored at end of string, unlike the old `_seed(\d+)$` which silently
# dropped every variant run from selection/plotting -- CHANGES.md item 7.
SEED_RE = re.compile(r"_seed(\d+)(?:_([A-Za-z0-9]+))?$")
CKPT_RE = re.compile(r"^model_(\d+)\.pt$")


def parse_experiment_dir(name: str) -> tuple[str, str] | None:
    """Return (hand, family) parsed from an experiment directory name, e.g.
    'allegro_grasp' -> ('allegro', 'grasp'), 'sharpa_in_hand_rotation' ->
    ('sharpa', 'rotation'). None if no known hand is found in the name."""
    lname = name.lower()
    hand = next((h for h in HAND_DISPLAY if h in lname), None)
    if hand is None:
        return None
    family = "rotation" if "rotation" in lname else "grasp"
    return hand, family


def task_id(hand: str, family: str) -> str:
    display = HAND_DISPLAY[hand]
    return f"InHand-Rotation-{display}" if family == "rotation" else f"Grasp-{display}"


def checkpoint_iters(run: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in run.glob("model_*.pt"):
        m = CKPT_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def load_headline(run: Path, tag: str) -> tuple[list[int], list[float]] | None:
    events = sorted(run.glob("events.out.tfevents.*"))
    if not events:
        return None
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    series = acc.Scalars(tag)
    return [s.step for s in series], [s.value for s in series]


def best_checkpoint(run: Path, tag: str, window: int = 25) -> tuple[Path, float, int] | None:
    """Smooth the headline curve with a `window`-point moving average, find
    its argmax, and snap to the newest checkpoint at or before that step.

    Not cosmetic: some tasks peak well before their final iteration and decay
    afterward, so `--pick last` can select measurably worse weights.
    """
    series = load_headline(run, tag)
    ckpts = checkpoint_iters(run)
    if series is None or not ckpts:
        return None
    steps, values = series
    if not steps:
        return None
    values_arr = np.asarray(values, dtype=np.float64)
    w = max(1, min(window, len(values_arr)))
    kernel = np.ones(w) / w
    smoothed = np.convolve(values_arr, kernel, mode="same")
    peak_i = int(np.argmax(smoothed))
    peak_step = steps[peak_i]
    candidates = [it for it in ckpts if it <= peak_step]
    iteration = max(candidates) if candidates else min(ckpts)
    return ckpts[iteration], float(values_arr[peak_i]), iteration


def last_checkpoint(run: Path, tag: str) -> tuple[Path, float, int] | None:
    ckpts = checkpoint_iters(run)
    if not ckpts:
        return None
    iteration = max(ckpts)
    series = load_headline(run, tag)
    headline = float("nan")
    if series is not None and series[0]:
        steps, values = series
        candidates = [(s, v) for s, v in zip(steps, values, strict=True) if s <= iteration]
        headline = candidates[-1][1] if candidates else values[-1]
    return ckpts[iteration], headline, iteration


def discover_runs(log_root: Path, variant_filter: str) -> dict[str, dict[int, Path]]:
    """{task_id: {seed: run_dir}}, filtered by --variant, keeping the newest
    run directory per (task, seed) if more than one matches."""
    found: dict[str, dict[int, Path]] = {}
    if not log_root.exists():
        return found
    for exp_dir in sorted(log_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        parsed = parse_experiment_dir(exp_dir.name)
        if parsed is None:
            continue
        hand, family = parsed
        tid = task_id(hand, family)
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = SEED_RE.search(run_dir.name)
            if not m:
                continue
            seed = int(m.group(1))
            variant = m.group(2)
            if variant_filter == "none" and variant is not None:
                continue
            if variant_filter not in ("any", "none") and variant != variant_filter:
                continue
            found.setdefault(tid, {})
            prev = found[tid].get(seed)
            if prev is None or run_dir.name > prev.name:
                found[tid][seed] = run_dir
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    ap.add_argument("--out", type=Path, default=Path("outputs/experts.json"))
    ap.add_argument("--pick", choices=["last", "best"], default="last")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--only", choices=["all", "grasp", "rotation"], default="all")
    ap.add_argument(
        "--variant",
        default="any",
        help="'any', 'none' for plain <timestamp>_seed<N> dirs only, "
        "or an explicit suffix such as '40k'.",
    )
    args = ap.parse_args()

    runs = discover_runs(args.log_root.resolve(), args.variant)
    manifest: dict[str, dict] = {}
    for tid, seeds in sorted(runs.items()):
        family = "rotation" if "Rotation" in tid else "grasp"
        if args.only != "all" and args.only != family:
            continue
        tag = HEADLINE_TAG[family]
        scored = []
        for seed, run_dir in seeds.items():
            result = (
                best_checkpoint(run_dir, tag, window=args.window)
                if args.pick == "best"
                else last_checkpoint(run_dir, tag)
            )
            if result is None:
                print(f"[WARN] {tid} seed{seed}: no usable checkpoint/metric in {run_dir}")
                continue
            ckpt, headline, iteration = result
            scored.append((seed, ckpt, headline, iteration))
        if not scored:
            print(f"[WARN] {tid}: no seeds with usable results, skipping")
            continue
        scored.sort(key=lambda x: x[2], reverse=True)
        best_seed, best_ckpt, best_headline, best_iter = scored[0]
        others = [round(h, 4) for _, _, h, _ in scored[1:]]
        manifest[tid] = {
            "task": tid,
            "checkpoint": str(best_ckpt),
            "seed": best_seed,
            "checkpoint_iter": best_iter,
            "headline": round(best_headline, 4),
            "headline_tag": tag,
            "other_seeds": others,
            "pick_mode": args.pick,
            "variant": args.variant,
        }
        print(
            f"{tid}: seed{best_seed} iter{best_iter} headline={best_headline:.4f} "
            f"(others={others})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"[INFO] Wrote {args.out} ({len(manifest)} tasks)")


if __name__ == "__main__":
    main()
