"""Held-out validation by full DDIM denoising (`--val-dataset`).

`DenoisedValidator` samples a fixed set of windows from a separate validation
store (scripts/build_val_split.py: fresh expert rollouts, episodes that also
start a training episode removed), runs the policy's sampler on their obs and
scores the predicted action chunk against the demo action, per hand on that
hand's own action channels, plus pooled over all scored windows.

Units: raw stored actions, i.e. pre-clip policy outputs (roughly +-15), not
radians. Which hands are scored follows the run's eval spec (a target-only
spec gives target-only validation); with no spec, every hand in the store.
Sampling noise comes from a forked RNG seeded per call, so validation is
deterministic and does not perturb the training RNG stream.

Re-implementation of Bundle's validate.py from MIGRATION.md (CHANGES.md item 63).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mjlab_hand.diffusion.dataset import DiffusionDataset, TrajectoryStore
from mjlab_hand.diffusion.padding import build_plan


class DenoisedValidator:
    def __init__(
        self,
        val_path: str | Path,
        *,
        obs_horizon: int,
        action_horizon: int,
        tasks: list[str] | None = None,
        padded: bool = True,
        n_windows: int = 2048,
        seed: int = 0,
        batch_size: int = 512,
    ):
        store = TrajectoryStore(val_path, mode="r")
        extra = json.loads(store.root.attrs.get("extra", "{}"))
        if extra.get("pad_scheme") != "term_aligned":
            raise ValueError(f"{val_path}: val store must be a term-aligned padded store")
        plan = build_plan(extra["family"])
        src_tasks = [s["task"] for s in extra["sources"]]
        tasks = tasks or src_tasks
        missing = [t for t in tasks if t not in src_tasks]
        if missing:
            raise ValueError(f"{val_path} has no val data for {missing} (has {src_tasks})")
        if not padded and len(tasks) != 1:
            raise ValueError("a single-hand (unpadded) run validates exactly one task")

        ds = DiffusionDataset(store, obs_horizon=obs_horizon, action_horizon=action_horizon)
        rng = np.random.default_rng(seed)
        per = n_windows // len(tasks)
        self.batches: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
        for task in tasks:
            src = src_tasks.index(task)
            cand = np.flatnonzero(ds.episode_source[ds._win_epi] == src)
            pick = np.sort(rng.choice(cand, size=min(per, len(cand)), replace=False))
            items = [ds[int(i)] for i in pick]
            obs = torch.stack([it["obs"] for it in items])
            act = torch.stack([it["action"] for it in items])
            na = plan.native[task]["action_dim"]
            if not padded:  # single-hand policy: back to the native layout
                obs = obs[..., torch.from_numpy(plan.obs_index(task))]
                act = act[..., :na]
            self.batches[task] = (obs, act, na)
        self.padded = padded
        self.plan = plan
        self.seed = seed
        self.batch_size = batch_size
        self.tasks = tasks

    @torch.no_grad()
    def evaluate(self, policy, device: torch.device) -> dict:
        devices = [device] if device.type == "cuda" else []
        per_hand, all_err = {}, []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            for task, (obs, act, na) in self.batches.items():
                errs = []
                for i in range(0, len(obs), self.batch_size):
                    o = obs[i : i + self.batch_size].to(device)
                    a = act[i : i + self.batch_size].to(device)
                    valid = None
                    if self.padded and policy.cfg.mask_pad_loss:
                        valid = torch.zeros(len(o), a.shape[-1], dtype=torch.bool, device=device)
                        valid[:, :na] = True
                    pred = policy.predict_action(o, action_valid=valid)
                    errs.append(((pred[..., :na] - a[..., :na]) ** 2).mean(dim=(1, 2)).cpu())
                e = torch.cat(errs)
                per_hand[task] = float(e.mean())
                all_err.append(e)
        pooled = float(torch.cat(all_err).mean())
        return {"pooled": pooled, "per_hand": per_hand, "score": pooled}
