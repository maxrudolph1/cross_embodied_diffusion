#!/usr/bin/env python3
"""Pool N single-embodiment demo datasets into one zero-padded,
per-source-normalized dataset for cross-embodiment BC.

For each source dataset (one embodiment, one task family):

  1. Fit a `GaussianNormalizer` (mean 0, var 1) independently from that
     source's own raw obs and raw action arrays. Static: computed once here
     from this source alone, never from the pooled mixture and never
     refit later -- the same fitted mean/std is what training and eval both
     use, forever, for that embodiment. See `agent_logbook/` for why this
     matters: a normalizer fit on the pooled/padded data would let one
     embodiment's scale dominate another's, and a per-batch-refit normalizer
     wouldn't be reproducible between training and eval.
  2. Normalize that source's own obs/action with its own stats.
  3. Zero-pad the normalized obs out to `max_obs_dim` and the normalized
     action out to `max_action_dim` (the max over all sources being pooled),
     real dims front-packed at [0:real_dim], padding always at the tail.

Sources are then concatenated (episodes copied whole, `episode_ends`
recomputed), matching `build_mixed_dataset.py`'s approach but generalized to
N sources of *mismatched* dimensionality -- the thing that script explicitly
refuses to do, because there padding would silently conflate unrelated
physical quantities in a 2-source onehot-conditioned mixture. Here it's the
opposite tradeoff on purpose: a single shared padded action space is what
lets one diffusion policy drive an arbitrary embodiment, and the per-source
static normalization + `extra.sources[i]` real-dim/mean/std provenance is
what lets `DiffusionDataset` build a correct per-row loss mask
(`TrajectoryStore.source_real_dims`) and lets eval-time code
(`CrossEmbodimentActionChunkPolicy`) normalize/pad a live embodiment's raw
obs and un-normalize/un-pad its predicted action using the exact same
per-embodiment stats used at training time.

`extra.padded = True` distinguishes this scheme from the existing 2-source
`extra.mixed` onehot scheme; `DiffusionDataset`/`train_diffusion.py` key off
this flag to know a dataset needs identity normalizers + a loss mask instead
of a single fitted LinearNormalizer.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from mjlab_hand.diffusion.dataset import TrajectoryStore
from mjlab_hand.diffusion.normalizer import GaussianNormalizer


def load_source(path: Path, success_only: bool) -> dict:
    store = TrajectoryStore(path, mode="r")
    episodes = store.episode_slices(success_only=success_only)
    return {
        "path": path,
        "embodiment": path.stem.split("_expert")[0],
        "task": str(store.root.attrs.get("task", "")),
        "episodes": episodes,
        "obs": np.asarray(store.data["obs"][:], dtype=np.float32),
        "action": np.asarray(store.data["action"][:], dtype=np.float32),
        "reward": np.asarray(store.data["reward"][:], dtype=np.float32),
    }


def gather_padded(
    src: dict,
    obs_norm: GaussianNormalizer,
    act_norm: GaussianNormalizer,
    max_obs_dim: int,
    max_action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    real_obs_dim = src["obs"].shape[1]
    real_action_dim = src["action"].shape[1]

    obs_parts, act_parts, rew_parts, succ_parts = [], [], [], []
    for start, end, succ in src["episodes"]:
        length = end - start
        raw_obs = src["obs"][start:end]
        raw_act = src["action"][start:end]
        norm_obs = (raw_obs - obs_norm.mean.numpy()) / obs_norm.std.numpy()
        norm_act = (raw_act - act_norm.mean.numpy()) / act_norm.std.numpy()
        padded_obs = np.zeros((length, max_obs_dim), dtype=np.float32)
        padded_obs[:, :real_obs_dim] = norm_obs
        padded_act = np.zeros((length, max_action_dim), dtype=np.float32)
        padded_act[:, :real_action_dim] = norm_act
        obs_parts.append(padded_obs)
        act_parts.append(padded_act)
        rew_parts.append(src["reward"][start:end])
        succ_parts.append(np.full((length,), 1 if succ else 0, dtype=np.uint8))

    obs = np.concatenate(obs_parts, axis=0) if obs_parts else np.zeros((0, max_obs_dim), "f4")
    action = (
        np.concatenate(act_parts, axis=0) if act_parts else np.zeros((0, max_action_dim), "f4")
    )
    reward = np.concatenate(rew_parts, axis=0) if rew_parts else np.zeros((0,), "f4")
    success = np.concatenate(succ_parts, axis=0) if succ_parts else np.zeros((0,), "u1")
    ep_lengths = [e - s for s, e, _ in src["episodes"]]
    return obs, action, reward, success, ep_lengths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, nargs="+", required=True, help="N source .zarr dirs")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--task-family", type=str, required=True, help="e.g. Grasp, InHand-Rotation")
    ap.add_argument("--success-only", action="store_true", default=True)
    ap.add_argument("--no-success-only", dest="success_only", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output} already exists (pass --overwrite)")
        shutil.rmtree(args.output)

    sources = [load_source(p, args.success_only) for p in args.sources]
    for s in sources:
        print(
            f"[INFO] source {s['embodiment']}: obs_dim={s['obs'].shape[1]} "
            f"action_dim={s['action'].shape[1]} n_steps={s['obs'].shape[0]}"
        )

    max_obs_dim = max(s["obs"].shape[1] for s in sources)
    max_action_dim = max(s["action"].shape[1] for s in sources)
    print(f"[INFO] padded obs_dim={max_obs_dim} action_dim={max_action_dim}")

    all_obs, all_act, all_rew, all_succ = [], [], [], []
    ep_lengths_all: list[int] = []
    source_meta = []
    n_steps_running = 0
    for s in sources:
        obs_norm = GaussianNormalizer.fit(s["obs"])
        act_norm = GaussianNormalizer.fit(s["action"])
        obs, action, reward, success, ep_lengths = gather_padded(
            s, obs_norm, act_norm, max_obs_dim, max_action_dim
        )
        all_obs.append(obs)
        all_act.append(action)
        all_rew.append(reward)
        all_succ.append(success)
        ep_lengths_all.extend(ep_lengths)
        source_meta.append(
            {
                "path": str(s["path"]),
                "embodiment": s["embodiment"],
                "task": s["task"],
                "n_steps": int(obs.shape[0]),
                "n_episodes": len(ep_lengths),
                "obs_dim": int(s["obs"].shape[1]),
                "action_dim": int(s["action"].shape[1]),
                "obs_mean": obs_norm.mean.tolist(),
                "obs_std": obs_norm.std.tolist(),
                "action_mean": act_norm.mean.tolist(),
                "action_std": act_norm.std.tolist(),
            }
        )
        n_steps_running += obs.shape[0]

    obs = np.concatenate(all_obs, axis=0)
    action = np.concatenate(all_act, axis=0)
    reward = np.concatenate(all_rew, axis=0)
    success = np.concatenate(all_succ, axis=0)

    ends: list[int] = []
    offset = 0
    for length in ep_lengths_all:
        offset += length
        ends.append(offset)
    ep_ends = np.asarray(ends, dtype=np.int64)

    extra_meta = {
        "padded": True,
        "max_obs_dim": max_obs_dim,
        "max_action_dim": max_action_dim,
        "task_family": args.task_family,
        "sources": source_meta,
    }

    dst = TrajectoryStore(args.output, mode="w")
    dst.initialize(
        obs_dim=max_obs_dim,
        action_dim=max_action_dim,
        task=args.task_family,
        checkpoint="+".join(str(p) for p in args.sources),
        extra_meta=extra_meta,
    )
    for name, arr in (("obs", obs), ("action", action), ("reward", reward), ("success", success)):
        ds = dst.data[name]
        ds.resize((arr.shape[0], *arr.shape[1:]))
        ds[:] = arr
    ends_ds = dst.data["episode_ends"]
    ends_ds.resize((ep_ends.shape[0],))
    ends_ds[:] = ep_ends

    print(json.dumps(dst.summary(), indent=2))
    print(f"[INFO] embodiments: {[s['embodiment'] for s in sources]}")


if __name__ == "__main__":
    main()
