#!/usr/bin/env python3
"""Check that live env observations land in the same padded columns as the
training data (CHANGES.md item 63). Needs a GPU node.

For each hand of a term-aligned store: reset --envs eval envs, scatter the
native obs through `padding.build_plan(family).obs_index(task)` exactly as
`DiffusionActionChunkPolicy(pad_task=...)` does, and compare per-column
statistics with the store's episode-start rows of that hand:
  - padded (unused) columns must be exactly 0 in both;
  - for used columns, z = |mean_env - mean_store| / (std_store + 1e-3).
A misordered or shifted term shows up as large z on whole term blocks; reset
states are random, so expect z of order 1 at most for a correct layout.

    python scripts/check_env_alignment.py --store data/mjlab_hand_demos/padded_ta/<store>.zarr
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mjlab_hand.diffusion.dataset import TrajectoryStore
from mjlab_hand.diffusion.padding import build_plan
from mjlab_hand.eval.base import _policy_input


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--z-max", type=float, default=4.0)
    args = ap.parse_args()

    import mjlab_hand  # noqa: F401
    from mjlab_hand.eval.config import EvalConfig
    from mjlab_hand.eval.env_setup import setup_eval_env

    st = TrajectoryStore(args.store, mode="r")
    extra = json.loads(st.root.attrs["extra"])
    plan = build_plan(extra["family"])
    obs_all = st.data["obs"]
    device = torch.device("cuda:0")
    ok = True
    start = 0
    ends = np.asarray(st.data["episode_ends"][:])
    ep_starts = np.concatenate([[0], ends[:-1]])
    for src in extra["sources"]:
        task, n = src["task"], src["n_steps"]
        rows = ep_starts[(ep_starts >= start) & (ep_starts < start + n)]
        store_obs = np.asarray(obs_all.get_orthogonal_selection((rows, slice(None))), np.float32)
        env, _ = setup_eval_env(task, EvalConfig(num_envs=args.envs, seed=0, device=str(device)), device)
        obs, _ = env.reset()
        raw = _policy_input(obs, device).float().reshape(args.envs, -1).cpu().numpy()
        env.close()
        if raw.shape[1] != plan.native[task]["obs_dim"]:
            print(f"FAIL {task}: env obs width {raw.shape[1]} != schema {plan.native[task]['obs_dim']}")
            ok = False
            start += n
            continue
        env_pad = plan.pad_obs(task, raw)
        valid = plan.obs_valid(task)
        pad_zero = not env_pad[:, ~valid].any() and not store_obs[:, ~valid].any()
        z = np.abs(env_pad.mean(0) - store_obs.mean(0)) / (store_obs.std(0) + 1e-3)
        worst = []
        for term, off, w, nat in zip(plan.terms, plan.term_offsets, plan.term_widths, plan.native[task]["term_dims"], strict=True):
            if nat:
                worst.append((float(z[off : off + nat].max()), term))
        zmax, zterm = max(worst)
        good = pad_zero and zmax <= args.z_max
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} {task:26s} padding zero both: {pad_zero}  "
              f"max z {zmax:.2f} ({zterm}); {len(rows)} store episode starts vs {args.envs} env resets")
        start += n
    print("ALL OK" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
