#!/usr/bin/env python3
"""Build a smaller demo dataset by copying whole episodes from a larger one.

Episodes are copied whole -- never truncated mid-episode, because
`DiffusionDataset` builds obs/action windows within episode boundaries and a
partial trailing episode would generate windows running off the end. The
greedy selector stops at whichever episode boundary lands closest to the
target step count.

See CHANGES.md item 6e.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from mjlab_hand.diffusion.dataset import TrajectoryStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-steps", type=int, required=True)
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output} already exists (pass --overwrite)")
        shutil.rmtree(args.output)

    src = TrajectoryStore(args.source, mode="r")
    episodes = src.episode_slices(success_only=args.success_only)
    if not episodes:
        raise SystemExit(f"No episodes found in {args.source}")

    # Greedy: add whole episodes, stopping at whichever episode boundary
    # (before or after the crossing episode) lands closer to target_steps.
    selected: list[tuple[int, int, bool]] = []
    total = 0
    for start, end, succ in episodes:
        if total >= args.target_steps:
            break
        length = end - start
        if selected and total + length > args.target_steps:
            if abs(total - args.target_steps) <= abs(total + length - args.target_steps):
                break
        selected.append((start, end, succ))
        total += length

    if not selected:
        raise SystemExit("No episodes selected -- source dataset too small or empty")

    obs = np.asarray(src.data["obs"][:], dtype=np.float32)
    action = np.asarray(src.data["action"][:], dtype=np.float32)
    reward = np.asarray(src.data["reward"][:], dtype=np.float32)

    summary = src.summary()
    dst = TrajectoryStore(args.output, mode="w")
    dst.initialize(
        obs_dim=summary["obs_dim"],
        action_dim=summary["action_dim"],
        task=summary["task"],
        checkpoint=summary["checkpoint"],
        extra_meta={
            "subsampled_from": str(args.source),
            "target_steps": args.target_steps,
            "source_n_steps": summary["n_steps"],
            "source_n_episodes": summary["n_episodes"],
        },
    )
    for start, end, succ in selected:
        dst.append_episode(obs[start:end], action[start:end], reward[start:end], success=succ)

    out_summary = dst.summary()
    print(json.dumps(out_summary, indent=2))
    frac = out_summary["n_steps"] / max(args.target_steps, 1)
    print(
        f"[INFO] {out_summary['n_steps']} steps / {out_summary['n_episodes']} episodes "
        f"({frac:.1%} of target {args.target_steps})"
    )


if __name__ == "__main__":
    main()
