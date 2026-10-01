#!/usr/bin/env python3
"""Assemble the held-out validation stores (--val-dataset) from fresh expert
rollouts (slurm_jobs/vista_collect_val.sbatch). CHANGES.md item 63.

Per hand: take the raw val collection's success episodes, drop every episode
whose first obs row equals the first obs row of any episode in that hand's 1M
training store (the 50k/10k subsets are prefixes of it), and keep whole
episodes up to --steps (default 20k). Bundle found 254 of 400 episode starts
colliding between independent collections, so the filter matters. Then pool
the five hands into a term-aligned store:
data/mjlab_hand_demos/val/{grasp,rotation}_val_20k.zarr.

    python scripts/build_val_split.py [--seed 1000] [--steps 20000]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_padded_dataset import build  # noqa: E402

from mjlab_hand.diffusion.dataset import TrajectoryStore  # noqa: E402
from mjlab_hand.diffusion.padding import HANDS  # noqa: E402

DEMOS = Path("data/mjlab_hand_demos")
SHORT = {"Grasp": "grasp", "InHand-Rotation": "rotation"}


def starts(store: TrajectoryStore) -> set[bytes]:
    obs = store.data["obs"]
    ends = np.asarray(store.data["episode_ends"][:], dtype=np.int64)
    first = np.concatenate([[0], ends[:-1]])
    return {np.asarray(obs[int(s)], dtype=np.float32).tobytes() for s in first}


def filtered(task: str, seed: int, steps: int, out: Path) -> Path:
    raw = TrajectoryStore(DEMOS / "val" / "raw" / f"{task}_val_seed{seed}.zarr", mode="r")
    train_starts = starts(TrajectoryStore(DEMOS / f"{task}_expert_1M.zarr", mode="r"))
    obs = np.asarray(raw.data["obs"][:], np.float32)
    act = np.asarray(raw.data["action"][:], np.float32)
    rew = np.asarray(raw.data["reward"][:], np.float32)
    eps = raw.episode_slices(success_only=True)
    keep, dropped, total = [], 0, 0
    for s, e, _ in eps:
        if obs[s].tobytes() in train_starts:
            dropped += 1
            continue
        if total >= steps:
            break
        keep.append((s, e))
        total += e - s
    if out.exists():
        shutil.rmtree(out)
    dst = TrajectoryStore(out, mode="w")
    dst.initialize(obs_dim=obs.shape[1], action_dim=act.shape[1], task=task,
                   checkpoint=str(raw.root.attrs.get("checkpoint", "")),
                   extra_meta={"val_from": str(raw.path), "dropped_start_collisions": dropped})
    for s, e in keep:
        dst.append_episode(obs[s:e], act[s:e], rew[s:e], success=True)
    print(f"[INFO] {task}: {len(eps)} success episodes, {dropped} dropped (start collides with 1M), "
          f"kept {len(keep)} episodes / {total} steps")
    if total < steps:
        print(f"[WARN] {task}: only {total} < {steps} val steps; collect more episodes")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=20_000)
    args = ap.parse_args()
    k = f"{args.steps // 1000}k"
    for fam, short in SHORT.items():
        per_hand = [filtered(f"{fam}-{h}", args.seed, args.steps, DEMOS / "val" / f"{fam}-{h}_val_{k}.zarr")
                    for h in HANDS]
        build(per_hand, DEMOS / "val" / f"{short}_val_{k}.zarr", overwrite=True)


if __name__ == "__main__":
    main()
