"""Pool N single-embodiment demo datasets into one zero-padded,
per-source-normalized dataset for cross-embodiment BC.

For each source dataset (one embodiment, one task family):

  1. Fit a `GaussianNormalizer` (mean 0, var 1) independently from that
     source's own raw obs and raw action arrays -- all rows, including
     failed episodes. Static: computed once from this source alone, never
     from the pooled mixture and never refit later -- the same fitted
     mean/std is what training and eval both use, forever, for that
     embodiment. A normalizer fit on the pooled/padded data would let one
     embodiment's scale dominate another's, and a per-batch-refit normalizer
     wouldn't be reproducible between training and eval.
  2. Normalize that source's own obs/action with its own stats.
  3. Zero-pad the normalized obs out to `max_obs_dim` and the normalized
     action out to `max_action_dim` (the max over all sources being pooled),
     real dims front-packed at [0:real_dim], padding always at the tail.

Sources are then concatenated (episodes copied whole, `episode_ends`
recomputed). The per-source real-dim/mean/std provenance in
`extra.sources[i]` is what lets `DiffusionDataset` build a per-row loss mask
(`TrajectoryStore.source_real_dims`) and lets eval-time code
(`CrossEmbodimentActionChunkPolicy`) normalize/pad a live embodiment's raw
obs and un-normalize/un-pad its predicted action with the exact same stats.
`extra.padded = True` is what `train.py` keys off to use identity
normalizers + a loss mask instead of a single fitted LinearNormalizer.

Two consumers, one implementation (CHANGES.md item 56):
  - `scripts/build_padded_dataset.py` writes the pool to a zarr on disk;
  - `train-diffusion --dataset A.zarr B.zarr ...` builds it in memory at
    load time (`pooled_store`), so no padded zarr has to exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from mjlab_hand.diffusion.dataset import TrajectoryStore
from mjlab_hand.diffusion.normalizer import GaussianNormalizer, LinearNormalizer


def embodiment_name(path: Path) -> str:
    """`Grasp-Allegro_expert_50k.zarr` -> `Grasp-Allegro`. Eval matches
    `eval_specs[i].embodiment` against this name in `source_stats.json`."""
    return Path(path).stem.split("_expert")[0]


def infer_task_family(names: list[str]) -> str:
    """`Grasp-Allegro`, `Grasp-LEAP` -> `Grasp`; mixed families -> joined."""
    families = sorted({n.rsplit("-", 1)[0] for n in names})
    return "+".join(families)


def load_source(path: Path, success_only: bool) -> dict:
    store = TrajectoryStore(path, mode="r")
    return {
        "path": path,
        "embodiment": embodiment_name(path),
        "task": str(store.root.attrs.get("task", "")),
        "episodes": store.episode_slices(success_only=success_only),
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


def fit_source_normalizer(data: np.ndarray, kind: str) -> GaussianNormalizer:
    """Per-source static normalizer, as an affine (x - mean) / std map.

    "gaussian": mean 0 / var 1 (the original scheme). "minmax": [min, max]
    -> [-1, 1], exactly LinearNormalizer's map (mean = midpoint, std =
    half-range, same zero-range guard), so a single-source pooled run sees
    the same numbers as a plain run -- and eval, which only applies
    (x - mean) / std, needs no change. CHANGES.md item 58.
    """
    if kind == "gaussian":
        return GaussianNormalizer.fit(data)
    if kind != "minmax":
        raise ValueError(f"unknown source normalization {kind!r}")
    lin = LinearNormalizer.fit(data)
    center = (lin.high + lin.low) / 2
    half = (lin.high - lin.low) / 2
    return GaussianNormalizer(mean=center, std=half)


def pool_padded(
    paths: list[Path],
    *,
    task_family: str | None = None,
    success_only: bool = True,
    source_norm: str = "gaussian",
    verbose: bool = True,
) -> dict[str, Any]:
    """Pool `paths` (in order) into padded arrays + the `extra` metadata.

    Returns {"obs", "action", "reward", "success", "episode_ends", "extra",
    "checkpoint"}; `extra` is exactly what `build_padded_dataset.py` stores
    in the zarr's `extra` attr.
    """
    paths = [Path(p) for p in paths]
    names = [embodiment_name(p) for p in paths]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate embodiment names in pooled sources: {names}")
    if task_family is None:
        task_family = infer_task_family(names)

    # Load one source at a time and keep only its padded result, so peak RAM
    # is ~ one raw source + the pooled arrays, not every raw source at once.
    dims = []
    for p in paths:
        store = TrajectoryStore(p, mode="r")
        dims.append((int(store.data["obs"].shape[1]), int(store.data["action"].shape[1])))
    max_obs_dim = max(d[0] for d in dims)
    max_action_dim = max(d[1] for d in dims)
    if verbose:
        print(f"[INFO] padded obs_dim={max_obs_dim} action_dim={max_action_dim}")

    all_obs, all_act, all_rew, all_succ = [], [], [], []
    ep_lengths_all: list[int] = []
    source_meta = []
    for p in paths:
        s = load_source(p, success_only)
        if verbose:
            print(
                f"[INFO] source {s['embodiment']}: obs_dim={s['obs'].shape[1]} "
                f"action_dim={s['action'].shape[1]} n_steps={s['obs'].shape[0]}"
            )
        obs_norm = fit_source_normalizer(s["obs"], source_norm)
        act_norm = fit_source_normalizer(s["action"], source_norm)
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
        del s

    return {
        "obs": np.concatenate(all_obs, axis=0),
        "action": np.concatenate(all_act, axis=0),
        "reward": np.concatenate(all_rew, axis=0),
        "success": np.concatenate(all_succ, axis=0),
        "episode_ends": np.cumsum(np.asarray(ep_lengths_all, dtype=np.int64)),
        "extra": {
            "padded": True,
            "max_obs_dim": max_obs_dim,
            "max_action_dim": max_action_dim,
            "task_family": task_family,
            "source_norm": source_norm,
            "sources": source_meta,
        },
        "checkpoint": "+".join(str(p) for p in paths),
    }


class InMemoryTrajectoryStore(TrajectoryStore):
    """Read-only `TrajectoryStore` over numpy arrays instead of a zarr.

    Exposes the same `data[name]` / `root.attrs` surface the readers use
    (`episode_slices`, `source_real_dims`, `summary`, `DiffusionDataset`).
    `DiffusionDataset` does `np.asarray(store.data["obs"][:], float32)`,
    which on a float32 ndarray is a view, so the train and val datasets
    share one copy of the pooled arrays (a zarr-backed store decodes a
    separate copy for each).
    """

    def __init__(self, label: str, arrays: dict[str, np.ndarray], attrs: dict[str, Any]):
        self.path = Path(label)
        self.root = SimpleNamespace(attrs=dict(attrs))
        self.data = dict(arrays)


def pooled_store(
    paths: list[Path],
    *,
    task_family: str | None = None,
    success_only: bool = True,
    source_norm: str = "gaussian",
) -> InMemoryTrajectoryStore:
    """`pool_padded` wrapped as a store, with the same attrs
    `build_padded_dataset.py` writes. `success_only` here only decides which
    episodes are pooled (as in the builder); `DiffusionDataset` filters again."""
    pool = pool_padded(
        paths, task_family=task_family, success_only=success_only, source_norm=source_norm
    )
    extra = pool["extra"]
    attrs = {
        "obs_dim": extra["max_obs_dim"],
        "action_dim": extra["max_action_dim"],
        "task": extra["task_family"],
        "checkpoint": pool["checkpoint"],
        "extra": json.dumps(extra),
    }
    arrays = {k: pool[k] for k in ("obs", "action", "reward", "success", "episode_ends")}
    label = "pooled:" + "+".join(embodiment_name(p) for p in paths)
    return InMemoryTrajectoryStore(label, arrays, attrs)
