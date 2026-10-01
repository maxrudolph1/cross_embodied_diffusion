"""Frozen per-family normalizer: one LinearNormalizer per task family in the
term-aligned padded layout, fit once and stored as a JSON artifact, so every
multi-hand run of a family uses identical scaling (--norm-mode frozen).

Per-run fits made the same column scale differently across data mixtures
(Bundle saw up to 2.42x on one Allegro column, MIGRATION section 7). Each
column's range is fit over only the rows of hands that have that column, and
widened to contain 0 so padding zeros stay inside [-1, 1] (without that,
padding mapped to -198,322 on 17 grasp columns in Bundle).

Artifact JSON: {"family", "stat", "plan", "obs_low", "obs_high", "act_low",
"act_high", "sources", "digest"}. "digest" = sha256 over the four arrays
(float32), as `normalizer.norm_digest`.

Deviation from Bundle (CHANGES.md item 63): Bundle fit its artifacts on the
five 10M single-hand stores per family, which are not on Vista; ours are fit
on the 1M stores (scripts/build_family_normalizer.py), so the numbers differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mjlab_hand.diffusion.normalizer import LinearNormalizer, norm_digest
from mjlab_hand.diffusion.padding import build_plan, family_of


def build_artifact(family: str, sources: list[Path], stat: str = "minmax") -> dict:
    from mjlab_hand.diffusion.dataset import TrajectoryStore

    plan = build_plan(family)
    obs_parts, act_parts, groups_o, groups_a, meta, start = [], [], [], [], [], 0
    for p in sources:
        st = TrajectoryStore(p, mode="r")
        task = str(st.root.attrs["task"])
        if family_of(task) != family:
            raise ValueError(f"{p} is {task}, not in family {family}")
        o = plan.pad_obs(task, np.asarray(st.data["obs"][:], np.float32))
        a = plan.pad_action(task, np.asarray(st.data["action"][:], np.float32))
        n = len(o)
        obs_parts.append(o)
        act_parts.append(a)
        groups_o.append((start, start + n, plan.obs_valid(task)))
        groups_a.append((start, start + n, plan.action_valid(task)))
        meta.append({"path": str(p), "task": task, "n_steps": n})
        start += n
    obs, act = np.concatenate(obs_parts), np.concatenate(act_parts)
    if stat == "minmax":
        on = LinearNormalizer.fit_masked(obs, groups_o, include_zero=True)
        an = LinearNormalizer.fit_masked(act, groups_a, include_zero=True)
    elif stat == "zscore":
        on = LinearNormalizer.fit_standardized(obs, groups_o)
        an = LinearNormalizer.fit_standardized(act, groups_a)
    else:
        raise ValueError(f"unknown stat {stat!r}")
    return {
        "family": family,
        "stat": stat,
        "plan": plan.to_json(),
        "obs_low": on.low.tolist(),
        "obs_high": on.high.tolist(),
        "act_low": an.low.tolist(),
        "act_high": an.high.tolist(),
        "sources": meta,
        "digest": norm_digest(on, an),
    }


def normalizers(artifact: dict) -> tuple[LinearNormalizer, LinearNormalizer]:
    f = lambda k: torch.tensor(artifact[k], dtype=torch.float32)  # noqa: E731
    return (
        LinearNormalizer(low=f("obs_low"), high=f("obs_high")),
        LinearNormalizer(low=f("act_low"), high=f("act_high")),
    )


def load_artifact(path: str | Path) -> dict:
    """Load and verify (digest recomputed from the stored arrays)."""
    art = json.loads(Path(path).read_text())
    on, an = normalizers(art)
    got = norm_digest(on, an)
    if got != art["digest"]:
        raise ValueError(f"{path}: digest mismatch (stored {art['digest']}, arrays give {got})")
    return art


def select(artifact: dict, family: str, obs_dim: int, action_dim: int) -> tuple[LinearNormalizer, LinearNormalizer]:
    """The artifact's normalizers, after checking they fit this store."""
    if artifact["family"] != family:
        raise ValueError(f"norm artifact is for {artifact['family']}, store is {family}")
    plan = build_plan(family)
    if artifact["plan"] != plan.to_json():
        raise ValueError("norm artifact layout differs from the current padding plan")
    if (len(artifact["obs_low"]), len(artifact["act_low"])) != (obs_dim, action_dim):
        raise ValueError(
            f"norm artifact widths {len(artifact['obs_low'])}/{len(artifact['act_low'])} "
            f"!= store {obs_dim}/{action_dim}"
        )
    return normalizers(artifact)
