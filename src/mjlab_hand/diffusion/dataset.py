"""On-disk trajectory store (zarr) and horizon sampling dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, Sampler


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

    def source_step_bounds(self) -> list[tuple[int, int, str]]:
        """Per-source (start, end, task) step ranges of a multi-source store,
        from the cumulative `extra.sources[i].n_steps`; [] for a single-source
        store. Raises if the counts don't sum to n_steps, rather than
        silently mis-attributing rows to the wrong source."""
        extra = json.loads(self.root.attrs.get("extra", "{}"))
        sources = extra.get("sources")
        if not sources:
            return []
        bounds: list[tuple[int, int, str]] = []
        start = 0
        for src in sources:
            n = int(src["n_steps"])
            bounds.append((start, start + n, str(src.get("task", ""))))
            start += n
        if start != self.n_steps:
            raise RuntimeError(
                f"{self.path}: source step counts sum to {start}, but n_steps={self.n_steps}"
            )
        return bounds

    def source_action_dims(self) -> list[int]:
        """Real (pre-padding) action width per source; [] if not recorded."""
        extra = json.loads(self.root.attrs.get("extra", "{}"))
        return [int(s["action_dim"]) for s in extra.get("sources", []) if "action_dim" in s]

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
    """Sample (obs_horizon, action_horizon) windows from trajectories.

    `__getitem__(i)` returns obs/action windows (+ `t_min` for an ambient
    store: the window's source may only be trained at t >= t_min; the step
    is then drawn in `compute_loss`, data-first). `__getitem__((i, t))` --
    what `AmbientNoiseFirstBatchSampler` yields -- returns the window with the
    step already chosen as `t` (noise-first). With `mask_pad_loss`, both add
    `action_valid` (True on the row's real action channels). CHANGES.md item 63.
    """

    def __init__(
        self,
        store: TrajectoryStore,
        *,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        success_only: bool = True,
        pad_before: bool = True,
        ambient_tmin: list[int] | None = None,
        mask_pad_loss: bool = False,
    ):
        self.store = store
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pad_before = pad_before
        self.mask_pad_loss = mask_pad_loss
        self.episodes = store.episode_slices(success_only=success_only)
        if not self.episodes:
            self.episodes = store.episode_slices(success_only=False)
        if not self.episodes:
            raise RuntimeError(f"No episodes found in {store.path}")

        self.indices: list[tuple[int, int]] = []
        for epi_i, (start, end, _) in enumerate(self.episodes):
            for t in range(end - start):
                self.indices.append((epi_i, t))
        lengths = np.asarray([e - s for s, e, _ in self.episodes], dtype=np.int64)
        self._win_epi = np.repeat(np.arange(len(self.episodes), dtype=np.int64), lengths)

        self.obs = np.asarray(store.data["obs"][:], dtype=np.float32)
        self.action = np.asarray(store.data["action"][:], dtype=np.float32)

        # Source of each episode (episodes never straddle a source boundary).
        bounds = store.source_step_bounds()
        self.episode_source = np.zeros(len(self.episodes), dtype=np.int64)
        for i, (start, _end, _succ) in enumerate(self.episodes):
            for j, (b0, b1, _task) in enumerate(bounds):
                if b0 <= start < b1:
                    self.episode_source[i] = j
                    break
            else:
                if bounds:
                    raise RuntimeError(f"episode at step {start} not within any source range")

        full_act = self.action.shape[1]
        src_act = store.source_action_dims()
        self.episode_act_dim = np.asarray(
            [src_act[s] if src_act else full_act for s in self.episode_source], dtype=np.int64
        )
        self.window_act_dim = self.episode_act_dim[self._win_epi]

        self.ambient_tmin = ambient_tmin
        if ambient_tmin is not None:
            if len(ambient_tmin) != max(len(bounds), 1):
                raise ValueError(
                    f"ambient_tmin has {len(ambient_tmin)} entries but the store has "
                    f"{len(bounds)} sources (in its stored order)"
                )
            self.episode_tmin = np.asarray(
                [ambient_tmin[s] for s in self.episode_source], dtype=np.int64
            )
        else:
            self.episode_tmin = np.zeros(len(self.episodes), dtype=np.int64)
        self.window_tmin = self.episode_tmin[self._win_epi]

    def __len__(self) -> int:
        return len(self.indices)

    def _window(self, epi_i: int, t_local: int) -> tuple[np.ndarray, np.ndarray]:
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
        return self.obs[obs_idx], self.action[act_idx]

    def __getitem__(self, item) -> dict[str, torch.Tensor]:
        t_fixed = None
        if isinstance(item, (tuple, list)):
            idx, t_fixed = int(item[0]), int(item[1])
        else:
            idx = int(item)
        epi_i, t_local = self.indices[idx]
        obs_window, action_window = self._window(epi_i, t_local)
        out = {
            "obs": torch.from_numpy(obs_window),
            "action": torch.from_numpy(action_window),
        }
        if t_fixed is not None:
            out["t"] = torch.tensor(t_fixed, dtype=torch.long)
        elif self.ambient_tmin is not None:
            out["t_min"] = torch.tensor(int(self.window_tmin[idx]), dtype=torch.long)
        if self.mask_pad_loss:
            valid = torch.zeros(self.action.shape[1], dtype=torch.bool)
            valid[: int(self.window_act_dim[idx])] = True
            out["action_valid"] = valid
        return out


class AmbientNoiseFirstBatchSampler(Sampler):
    """Noise-first ambient batches: for each row draw the diffusion step t
    uniformly from [0, T) FIRST, then a window uniformly among those whose
    t_min <= t. Yields lists of (index, t) for DiffusionDataset.__getitem__.

    Drawing the window first and then t >= t_min (data-first) makes the
    training mass at a low t proportional to the share of windows admitted
    there, starving the low-noise steps by (N_target + N_other) / N_target
    (41x on 2-hand, 79-97x on padded stores per MIGRATION section 7).
    Noise-first keeps every t equally likely. Uses torch's global RNG
    (seeded by torch.manual_seed), no explicit generator.
    """

    def __init__(self, dataset: DiffusionDataset, batch_size: int, num_train_timesteps: int):
        self.batch_size = batch_size
        self.T = num_train_timesteps
        tmin = torch.from_numpy(dataset.window_tmin)
        self.sorted_tmin, self.order = torch.sort(tmin, stable=True)
        self.n_batches = len(dataset) // batch_size
        if int(self.sorted_tmin[0]) != 0:
            raise ValueError("noise-first sampling needs a source with t_min = 0 (the target)")

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            t = torch.randint(0, self.T, (self.batch_size,))
            count = torch.searchsorted(self.sorted_tmin, t, right=True)
            pos = (torch.rand(self.batch_size) * count).long().clamp_max(count - 1)
            idx = self.order[pos]
            yield list(zip(idx.tolist(), t.tolist(), strict=True))
