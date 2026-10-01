#!/usr/bin/env python3
"""Pool single-hand demo stores of one task family into a term-aligned padded
store (CHANGES.md item 63).

Each source's raw obs are scattered into the family layout of
`mjlab_hand.diffusion.padding` (every obs term in its own block, as wide as
the family's widest version of it) and its raw actions are front-packed;
padding is 0. Data is stored raw: the policy normalizes (--norm-mode).
Sources are concatenated in HANDS order (Allegro, LEAP, Shadow, Sharpa, Wuji)
whatever order they are given in, episodes copied whole (success-only by
default). `--ambient-tmin` vectors must follow that stored order
(`padded_grid.target_ambient_tmin`).

Writes extra = {"pad_scheme": "term_aligned", "family", "plan",
"sources": [{path, task, n_steps, n_episodes, obs_dim, action_dim}]}.

    python scripts/build_padded_dataset.py --sources A.zarr B.zarr ... --output OUT.zarr
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from mjlab_hand.diffusion.dataset import TrajectoryStore
from mjlab_hand.diffusion.padding import HANDS, build_plan, family_of


def hand_of(task: str) -> str:
    return task.rsplit("-", 1)[1]


def build(sources: list[Path], output: Path, success_only: bool = True, overwrite: bool = False) -> dict:
    if output.exists():
        if not overwrite:
            raise SystemExit(f"{output} already exists (pass --overwrite)")
        shutil.rmtree(output)
    stores = [(Path(p), TrajectoryStore(p, mode="r")) for p in sources]
    tasks = [str(s.root.attrs["task"]) for _, s in stores]
    if len(set(tasks)) != len(tasks):
        raise SystemExit(f"duplicate tasks among sources: {tasks}")
    family = {family_of(t) for t in tasks}
    if len(family) != 1:
        raise SystemExit(f"sources span several families: {sorted(family)}")
    family = family.pop()
    plan = build_plan(family)
    order = sorted(range(len(stores)), key=lambda i: HANDS.index(hand_of(tasks[i])))

    obs_parts, act_parts, rew_parts, succ_parts, lens, meta = [], [], [], [], [], []
    for i in order:
        path, store = stores[i]
        task = tasks[i]
        obs = np.asarray(store.data["obs"][:], dtype=np.float32)
        act = np.asarray(store.data["action"][:], dtype=np.float32)
        rew = np.asarray(store.data["reward"][:], dtype=np.float32)
        eps = store.episode_slices(success_only=success_only)
        rows = np.concatenate([np.arange(s, e) for s, e, _ in eps])
        obs_parts.append(plan.pad_obs(task, obs[rows]))
        act_parts.append(plan.pad_action(task, act[rows]))
        rew_parts.append(rew[rows])
        succ_parts.append(np.concatenate([np.full(e - s, 1 if ok else 0, np.uint8) for s, e, ok in eps]))
        lens += [e - s for s, e, _ in eps]
        meta.append({
            "path": str(path),
            "task": task,
            "n_steps": int(len(rows)),
            "n_episodes": len(eps),
            "obs_dim": plan.native[task]["obs_dim"],
            "action_dim": plan.native[task]["action_dim"],
        })
        print(f"[INFO] {task}: {len(eps)} episodes, {len(rows)} steps")

    extra = {"pad_scheme": "term_aligned", "family": family, "plan": plan.to_json(), "sources": meta}
    dst = TrajectoryStore(output, mode="w")
    dst.initialize(
        obs_dim=plan.obs_dim,
        action_dim=plan.action_dim,
        task=family,
        checkpoint="+".join(m["path"] for m in meta),
        extra_meta=extra,
    )
    arrays = {
        "obs": np.concatenate(obs_parts),
        "action": np.concatenate(act_parts),
        "reward": np.concatenate(rew_parts),
        "success": np.concatenate(succ_parts),
        "episode_ends": np.cumsum(np.asarray(lens, dtype=np.int64)),
    }
    for name, arr in arrays.items():
        ds = dst.data[name]
        ds.resize((arr.shape[0], *arr.shape[1:]))
        ds[:] = arr
    summary = dst.summary()
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, nargs="+", required=True, help="single-hand .zarr stores")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--no-success-only", dest="success_only", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    build(args.sources, args.output, success_only=args.success_only, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
