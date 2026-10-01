#!/usr/bin/env python3
"""Check a term-aligned padded store against its source stores (CHANGES.md item 63).

For --rows random rows of every source block: the native obs scattered through
`plan.obs_index(task)` equals the source row bit for bit, every other obs
column is exactly 0, the action prefix equals the source action and the tail
is 0. Also checks per-source step counts and the stored plan against the
current schemas.

    python scripts/check_padded_dataset.py STORE.zarr [--rows 1000]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from mjlab_hand.diffusion.dataset import TrajectoryStore
from mjlab_hand.diffusion.padding import build_plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("--rows", type=int, default=1000)
    args = ap.parse_args()

    st = TrajectoryStore(args.store, mode="r")
    extra = json.loads(st.root.attrs["extra"])
    errs = []
    if extra.get("pad_scheme") != "term_aligned":
        sys.exit(f"{args.store}: not a term-aligned store (pad_scheme={extra.get('pad_scheme')})")
    plan = build_plan(extra["family"])
    if plan.to_json() != extra["plan"]:
        errs.append("stored plan differs from the plan built from the current schemas")
    obs, act = st.data["obs"], st.data["action"]
    rng = np.random.default_rng(0)
    start = 0
    for src in extra["sources"]:
        task, n = src["task"], src["n_steps"]
        s = TrajectoryStore(src["path"], mode="r")
        rows = np.concatenate([np.arange(a, b) for a, b, _ in s.episode_slices(success_only=True)])
        if len(rows) != n:
            errs.append(f"{task}: source has {len(rows)} success-episode steps, store says {n}")
        pick = np.sort(rng.choice(n, size=min(args.rows, n), replace=False))
        p_obs, p_act = obs.get_orthogonal_selection((start + pick, slice(None))), act.get_orthogonal_selection((start + pick, slice(None)))
        s_obs = np.asarray(s.data["obs"].get_orthogonal_selection((rows[pick], slice(None))))
        s_act = np.asarray(s.data["action"].get_orthogonal_selection((rows[pick], slice(None))))
        idx, valid = plan.obs_index(task), plan.obs_valid(task)
        na = plan.native[task]["action_dim"]
        checks = {
            "obs scattered == source": np.array_equal(p_obs[:, idx], s_obs),
            "obs padding == 0": not np.any(p_obs[:, ~valid]),
            "action prefix == source": np.array_equal(p_act[:, :na], s_act),
            "action tail == 0": not np.any(p_act[:, na:]),
        }
        bad = [k for k, ok in checks.items() if not ok]
        errs += [f"{task}: {b}" for b in bad]
        print(f"  {task:26s} rows {start:>9,}-{start + n - 1:>9,}  {len(pick)} sampled  {'OK' if not bad else 'FAIL ' + ', '.join(bad)}")
        start += n
    if start != st.n_steps:
        errs.append(f"source n_steps sum {start} != store n_steps {st.n_steps}")
    print("OK" if not errs else "\n".join(["FAILED:"] + errs))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
