#!/usr/bin/env python3
"""Concatenate two single-embodiment demo datasets into a mixed,
one-hot-conditioned dataset.

obs_mixed = [obs (D), onehot (K)], actions unchanged. The label goes on the
observation, not the action -- it conditions the policy, it is not
something to predict -- so it reaches the model through the same
`global_cond` path as the rest of the observation, on every obs_horizon
frame.

Refuses mismatched observation/action spaces rather than zero-padding: only
Allegro and LEAP share both dimensionality and term layout, and padding two
different layouts would silently place unrelated physical quantities in the
same column.

Episodes are copied whole and `episode_ends` recomputed against a running
offset. Provenance (`mixed`, `onehot_dim`, `onehot_order`, `base_obs_dim`,
per-source paths/steps) goes into the store's `extra` attrs -- consumed by
`TrajectoryStore.source_step_bounds()` for ambient-diffusion timestep
gating.

Assembles the whole mixture in memory and bulk-writes each zarr array once,
rather than calling `append_episode` per episode (which resizes 4 zarr
arrays per call -- for ~2000-episode sources that is thousands of resizes
and, over 40 mixtures, hours of wall time). See CHANGES.md item 16.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from mjlab_hand.diffusion.dataset import TrajectoryStore


def load_source(path: Path, success_only: bool) -> dict:
    store = TrajectoryStore(path, mode="r")
    episodes = store.episode_slices(success_only=success_only)
    return {
        "path": path,
        "task": str(store.root.attrs.get("task", "")),
        "episodes": episodes,
        "obs": np.asarray(store.data["obs"][:], dtype=np.float32),
        "action": np.asarray(store.data["action"][:], dtype=np.float32),
        "reward": np.asarray(store.data["reward"][:], dtype=np.float32),
    }


def gather(
    src: dict, onehot: np.ndarray, onehot_dim: int, mixed_obs_dim: int, action_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    obs_parts, act_parts, rew_parts, succ_parts = [], [], [], []
    for start, end, succ in src["episodes"]:
        length = end - start
        oh = np.broadcast_to(onehot, (length, onehot_dim))
        obs_parts.append(np.concatenate([src["obs"][start:end], oh], axis=1))
        act_parts.append(src["action"][start:end])
        rew_parts.append(src["reward"][start:end])
        succ_parts.append(np.full((length,), 1 if succ else 0, dtype=np.uint8))
    obs = (
        np.concatenate(obs_parts, axis=0)
        if obs_parts
        else np.zeros((0, mixed_obs_dim), dtype=np.float32)
    )
    action = (
        np.concatenate(act_parts, axis=0)
        if act_parts
        else np.zeros((0, action_dim), dtype=np.float32)
    )
    reward = np.concatenate(rew_parts, axis=0) if rew_parts else np.zeros((0,), dtype=np.float32)
    success = np.concatenate(succ_parts, axis=0) if succ_parts else np.zeros((0,), dtype=np.uint8)
    ep_lengths = [e - s for s, e, _ in src["episodes"]]
    return obs, action, reward, success, ep_lengths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-a", type=Path, required=True)
    ap.add_argument("--source-b", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--success-only", action="store_true", default=True)
    ap.add_argument("--no-success-only", dest="success_only", action="store_false")
    ap.add_argument(
        "--onehot",
        action="store_true",
        default=True,
        help="Append a one-hot embodiment label to obs (default). Use --no-onehot for an "
        "unconditioned mixture (e.g. the ambient-diffusion sweep's naive-mixing baseline).",
    )
    ap.add_argument("--no-onehot", dest="onehot", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output} already exists (pass --overwrite)")
        shutil.rmtree(args.output)

    a = load_source(args.source_a, args.success_only)
    b = load_source(args.source_b, args.success_only)

    if a["obs"].shape[1] != b["obs"].shape[1] or a["action"].shape[1] != b["action"].shape[1]:
        raise ValueError(
            f"Mismatched spaces: {args.source_a} obs={a['obs'].shape[1]} "
            f"act={a['action'].shape[1]} vs {args.source_b} obs={b['obs'].shape[1]} "
            f"act={b['action'].shape[1]}. Refusing to zero-pad -- only embodiments with "
            "matching dimensionality and term layout can be mixed."
        )

    base_obs_dim = a["obs"].shape[1]
    action_dim = a["action"].shape[1]
    onehot_dim = 2 if args.onehot else 0
    mixed_obs_dim = base_obs_dim + onehot_dim

    onehot_a = np.array([1.0, 0.0], dtype=np.float32)[:onehot_dim]
    onehot_b = np.array([0.0, 1.0], dtype=np.float32)[:onehot_dim]
    obs_a, act_a, rew_a, succ_a, lens_a = gather(a, onehot_a, onehot_dim, mixed_obs_dim, action_dim)
    obs_b, act_b, rew_b, succ_b, lens_b = gather(b, onehot_b, onehot_dim, mixed_obs_dim, action_dim)

    obs = np.concatenate([obs_a, obs_b], axis=0)
    action = np.concatenate([act_a, act_b], axis=0)
    reward = np.concatenate([rew_a, rew_b], axis=0)
    success = np.concatenate([succ_a, succ_b], axis=0)

    ends: list[int] = []
    offset = 0
    for length in lens_a + lens_b:
        offset += length
        ends.append(offset)
    ep_ends = np.asarray(ends, dtype=np.int64)

    extra_meta = {
        "mixed": True,
        "onehot_dim": onehot_dim,
        "onehot_order": [a["task"], b["task"]],
        "base_obs_dim": base_obs_dim,
        "sources": [
            {"path": str(args.source_a), "task": a["task"], "n_steps": int(obs_a.shape[0])},
            {"path": str(args.source_b), "task": b["task"], "n_steps": int(obs_b.shape[0])},
        ],
    }

    dst = TrajectoryStore(args.output, mode="w")
    dst.initialize(
        obs_dim=mixed_obs_dim,
        action_dim=action_dim,
        task=f"{a['task']}+{b['task']}",
        checkpoint=f"{args.source_a}+{args.source_b}",
        extra_meta=extra_meta,
    )
    for name, arr in (("obs", obs), ("action", action), ("reward", reward), ("success", success)):
        ds = dst.data[name]
        ds.resize((arr.shape[0], *arr.shape[1:]))
        ds[:] = arr
    ends_ds = dst.data["episode_ends"]
    ends_ds.resize((ep_ends.shape[0],))
    ends_ds[:] = ep_ends

    summary = dst.summary()
    print(json.dumps(summary, indent=2))
    print(f"[INFO] onehot order: {extra_meta['onehot_order']}")


if __name__ == "__main__":
    main()
