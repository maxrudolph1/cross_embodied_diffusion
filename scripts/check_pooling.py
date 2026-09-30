#!/usr/bin/env python3
"""Check that in-memory pooling (`train-diffusion --dataset A B ...`) is
identical to the padded zarrs `build_padded_dataset.py` wrote (CHANGES.md
item 56).

For each padded zarr: re-pool its recorded sources on the fly (their
recorded absolute paths are mapped onto data/mjlab_hand_demos), then compare

  1. obs / action / reward / success / episode_ends -- bitwise, with the
     padding/concatenation run on the normalizer stats recorded in the zarr;
  2. the `extra` metadata (dims, task_family, per-source names, counts and
     the normalizer mean/std that eval reads from source_stats.json);
     refitting the stats on this machine may differ from the recorded ones
     by float32 rounding (torch's CPU reduction order differs across CPUs,
     e.g. x86 vs Vista's aarch64), so they -- and the arrays pooled with
     them -- are compared to a tolerance, reported as max abs diff;
  3. what training sees: `DiffusionDataset` episodes, per-episode action
     masks, per-window source ids, the train/val split, `balanced` sampling
     weights and a sample of windows, for train and val splits.

Usage: python scripts/check_pooling.py [padded.zarr ...]
       (default: every data/mjlab_hand_demos/padded/*.zarr)
Exit status 1 if anything differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from mjlab_hand.diffusion.dataset import DiffusionDataset, TrajectoryStore
from mjlab_hand.diffusion.normalizer import GaussianNormalizer
from mjlab_hand.diffusion.pooling import gather_padded, load_source, pooled_store

DATA = Path("data/mjlab_hand_demos")
CHUNK = 1 << 18
STAT_TOL = 1e-5  # abs, on normalizer mean/std
DATA_TOL = 1e-4  # abs, on normalized values pooled with refit stats


def local_source(recorded: str) -> Path:
    """Recorded source path -> this machine's copy (path after the
    `mjlab_hand_demos/` component, resolved under data/mjlab_hand_demos)."""
    parts = Path(recorded).parts
    return DATA.joinpath(*parts[parts.index("mjlab_hand_demos") + 1 :])


def compare_arrays(name: str, disk, mem: np.ndarray, errs: list[str], tol: float = 0.0) -> float:
    """Returns max abs diff; records an error if it exceeds `tol` (0 = bitwise)."""
    if tuple(disk.shape) != mem.shape:
        errs.append(f"{name}: shape {tuple(disk.shape)} on disk vs {mem.shape} pooled")
        return float("inf")
    worst = 0.0
    for start in range(0, mem.shape[0], CHUNK):
        a = np.asarray(disk[start : start + CHUNK])
        b = mem[start : start + CHUNK]
        if a.dtype != b.dtype:
            errs.append(f"{name}: dtype {a.dtype} on disk vs {b.dtype} pooled")
            return float("inf")
        if not np.array_equal(a, b):
            worst = max(worst, float(np.max(np.abs(a.astype(np.float64) - b))))
    if worst > tol:
        errs.append(f"{name}: max abs diff {worst:.3g} (tolerance {tol:g})")
    return worst


def pool_with_recorded_stats(sources: list[Path], extra: dict) -> dict[str, np.ndarray]:
    """The pooling path with normalizer *fitting* swapped for the stats stored
    in the zarr: isolates the padding/concatenation logic, which must then
    reproduce the zarr bitwise on any machine."""
    parts: dict[str, list] = {"obs": [], "action": [], "reward": [], "success": [], "len": []}
    for path, meta in zip(sources, extra["sources"], strict=True):
        s = load_source(path, success_only=True)
        obs_norm = GaussianNormalizer(
            mean=torch.tensor(meta["obs_mean"]), std=torch.tensor(meta["obs_std"])
        )
        act_norm = GaussianNormalizer(
            mean=torch.tensor(meta["action_mean"]), std=torch.tensor(meta["action_std"])
        )
        obs, action, reward, success, lens = gather_padded(
            s, obs_norm, act_norm, extra["max_obs_dim"], extra["max_action_dim"]
        )
        for k, v in (("obs", obs), ("action", action), ("reward", reward), ("success", success)):
            parts[k].append(v)
        parts["len"].extend(lens)
    out = {k: np.concatenate(parts[k]) for k in ("obs", "action", "reward", "success")}
    out["episode_ends"] = np.cumsum(np.asarray(parts["len"], dtype=np.int64))
    return out


def compare_extra(disk: dict, mem: dict, errs: list[str]) -> None:
    for k in ("padded", "max_obs_dim", "max_action_dim", "task_family"):
        if disk.get(k) != mem.get(k):
            errs.append(f"extra.{k}: {disk.get(k)!r} on disk vs {mem.get(k)!r} pooled")
    if len(disk["sources"]) != len(mem["sources"]):
        errs.append("extra.sources: different number of sources")
        return
    for i, (d, m) in enumerate(zip(disk["sources"], mem["sources"], strict=True)):
        for k in ("embodiment", "task", "n_steps", "n_episodes", "obs_dim", "action_dim"):
            if d[k] != m[k]:
                errs.append(f"sources[{i}].{k}: {d[k]!r} on disk vs {m[k]!r} pooled")
        for k in ("obs_mean", "obs_std", "action_mean", "action_std"):
            diff = float(np.max(np.abs(np.asarray(d[k]) - np.asarray(m[k]))))
            if diff > STAT_TOL:
                errs.append(f"sources[{i}].{k}: max abs diff {diff:.3g} (tolerance {STAT_TOL:g})")


def compare_datasets(disk_store, mem_store, errs: list[str]) -> None:
    for split in ("train", "val"):
        kw = dict(obs_horizon=2, action_horizon=8, split=split, val_fraction=0.1, val_seed=0)
        a = DiffusionDataset(disk_store, **kw)
        b = DiffusionDataset(mem_store, **kw)
        tag = f"DiffusionDataset[{split}]"
        if a.episodes != b.episodes:
            errs.append(f"{tag}: episode lists differ")
            continue
        if not np.array_equal(a.action_mask_per_episode, b.action_mask_per_episode):
            errs.append(f"{tag}: action masks differ")
        if not np.array_equal(a.source_id_per_window, b.source_id_per_window):
            errs.append(f"{tag}: per-window source ids differ")
        if not np.array_equal(a.source_sample_weights("balanced"), b.source_sample_weights("balanced")):
            errs.append(f"{tag}: balanced sampling weights differ")
        rng = np.random.default_rng(0)
        for idx in rng.integers(0, len(a), size=2000):
            x, y = a[int(idx)], b[int(idx)]
            same = x.keys() == y.keys() and all(
                np.array_equal(x[k], y[k])
                if k == "action_mask"
                else np.allclose(x[k], y[k], rtol=0, atol=DATA_TOL)
                for k in x
            )
            if not same:
                errs.append(f"{tag}: window {idx} differs")
                break
        print(f"    {split}: {len(a.episodes)} episodes, {len(a)} windows match")


def check(padded: Path) -> list[str]:
    disk = TrajectoryStore(padded, mode="r")
    extra = json.loads(disk.root.attrs["extra"])
    sources = [local_source(s["path"]) for s in extra["sources"]]
    print(f"[CHECK] {padded}  <-  {[str(s) for s in sources]}")
    errs: list[str] = []
    exact = pool_with_recorded_stats(sources, extra)
    for name in ("obs", "action", "reward", "success", "episode_ends"):
        compare_arrays(f"{name} (recorded stats)", disk.data[name], exact[name], errs)
    del exact
    print(f"    recorded stats -> bitwise: {'yes' if not errs else 'NO'}")

    mem = pooled_store(sources, task_family=extra["task_family"])
    for name in ("reward", "success", "episode_ends"):
        compare_arrays(name, disk.data[name], mem.data[name], errs)
    for name in ("obs", "action"):
        d = compare_arrays(name, disk.data[name], mem.data[name], errs, tol=DATA_TOL)
        print(f"    refit stats -> {name} max abs diff {d:.2g}")
    compare_extra(extra, json.loads(mem.root.attrs["extra"]), errs)
    for k in ("obs_dim", "action_dim", "task"):
        if disk.root.attrs[k] != mem.root.attrs[k]:
            errs.append(f"attrs.{k}: {disk.root.attrs[k]!r} vs {mem.root.attrs[k]!r}")
    if not errs:
        compare_datasets(disk, mem, errs)
    print(f"    {'OK' if not errs else 'MISMATCH'} ({mem.n_steps} steps)")
    for e in errs:
        print(f"    - {e}")
    return errs


def main() -> None:
    targets = [Path(p) for p in sys.argv[1:]] or sorted((DATA / "padded").glob("*.zarr"))
    failed = [t for t in targets if check(t)]
    print(
        f"\n{len(targets) - len(failed)}/{len(targets)} padded datasets reproduced "
        "(bitwise with recorded stats, within tolerance when refit)"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
