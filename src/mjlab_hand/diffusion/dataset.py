"""On-disk trajectory store (zarr) and horizon sampling dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class TrajectoryStore:
    """Append-only trajectory writer / reader backed by zarr."""

    def __init__(self, path: str | Path, mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(str(self.path), mode=mode)
        if "data" not in self.root:
            self.root.create_group("data")
        self.data = self.root["data"]

    @property
    def n_steps(self) -> int:
        if "obs" not in self.data:
            return 0
        return int(self.data["obs"].shape[0])

    @property
    def n_episodes(self) -> int:
        if "episode_ends" not in self.data:
            return 0
        return int(self.data["episode_ends"].shape[0])

    def initialize(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        task: str,
        checkpoint: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        if "obs" in self.data:
            return
        self.data.create_array(
            "obs",
            shape=(0, obs_dim),
            chunks=(4096, obs_dim),
            dtype="f4",
        )
        self.data.create_array(
            "action",
            shape=(0, action_dim),
            chunks=(4096, action_dim),
            dtype="f4",
        )
        self.data.create_array("reward", shape=(0,), chunks=(4096,), dtype="f4")
        self.data.create_array("success", shape=(0,), chunks=(4096,), dtype="u1")
        self.data.create_array("episode_ends", shape=(0,), chunks=(1024,), dtype="i8")
        self.root.attrs["obs_dim"] = int(obs_dim)
        self.root.attrs["action_dim"] = int(action_dim)
        self.root.attrs["task"] = task
        self.root.attrs["checkpoint"] = checkpoint
        if extra_meta:
            self.root.attrs["extra"] = json.dumps(extra_meta)

    def append_episode(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray | None = None,
        *,
        success: bool = False,
    ) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        assert obs.ndim == 2 and action.ndim == 2
        assert obs.shape[0] == action.shape[0]
        t = obs.shape[0]
        if reward is None:
            reward = np.zeros((t,), dtype=np.float32)
        else:
            reward = np.asarray(reward, dtype=np.float32).reshape(-1)
            assert reward.shape[0] == t

        for name, arr in (
            ("obs", obs),
            ("action", action),
            ("reward", reward),
            ("success", np.full((t,), 1 if success else 0, dtype=np.uint8)),
        ):
            ds = self.data[name]
            start = int(ds.shape[0])
            new_shape = list(ds.shape)
            new_shape[0] = start + t
            ds.resize(tuple(new_shape))
            ds[start : start + t] = arr

        ends = self.data["episode_ends"]
        start = int(ends.shape[0])
        ends.resize((start + 1,))
        ends[start] = self.n_steps

    def episode_slices(self, *, success_only: bool = False) -> list[tuple[int, int, bool]]:
        ends = np.asarray(self.data["episode_ends"][:], dtype=np.int64)
        starts = np.concatenate([[0], ends[:-1]]) if len(ends) else np.array([], dtype=np.int64)
        out: list[tuple[int, int, bool]] = []
        for s, e in zip(starts, ends, strict=True):
            succ = bool(self.data["success"][e - 1]) if e > s else False
            if success_only and not succ:
                continue
            out.append((int(s), int(e), succ))
        return out

    def summary(self) -> dict[str, Any]:
        episodes = self.episode_slices()
        n_succ = sum(1 for *_, s in episodes if s)
        return {
            "path": str(self.path),
            "n_steps": self.n_steps,
            "n_episodes": len(episodes),
            "n_success": n_succ,
            "obs_dim": int(self.root.attrs.get("obs_dim", -1)),
            "action_dim": int(self.root.attrs.get("action_dim", -1)),
            "task": str(self.root.attrs.get("task", "")),
            "checkpoint": str(self.root.attrs.get("checkpoint", "")),
        }


class DiffusionDataset(Dataset):
    """Sample (obs_horizon, action_horizon) windows from trajectories."""

    def __init__(
        self,
        store: TrajectoryStore,
        *,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        success_only: bool = True,
        pad_before: bool = True,
    ):
        self.store = store
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pad_before = pad_before
        self.episodes = store.episode_slices(success_only=success_only)
        if not self.episodes:
            self.episodes = store.episode_slices(success_only=False)
        if not self.episodes:
            raise RuntimeError(f"No episodes found in {store.path}")

        self.indices: list[tuple[int, int]] = []
        for epi_i, (start, end, _) in enumerate(self.episodes):
            length = end - start
            for t in range(length):
                self.indices.append((epi_i, t))

        self.obs = np.asarray(store.data["obs"][:], dtype=np.float32)
        self.action = np.asarray(store.data["action"][:], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        epi_i, t_local = self.indices[idx]
        start, end, _ = self.episodes[epi_i]
        obs_idx = []
        for k in range(self.obs_horizon):
            j = t_local - (self.obs_horizon - 1 - k)
            if j < 0:
                j = 0
            obs_idx.append(start + min(j, end - start - 1))
        act_idx = []
        for k in range(self.action_horizon):
            j = t_local + k
            if j >= end - start:
                j = end - start - 1
            act_idx.append(start + j)

        return {
            "obs": torch.from_numpy(self.obs[obs_idx]),
            "action": torch.from_numpy(self.action[act_idx]),
        }
