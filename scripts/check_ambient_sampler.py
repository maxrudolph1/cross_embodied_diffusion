#!/usr/bin/env python3
"""Check ambient sampling on a store (CHANGES.md item 63).

Draws batches with AmbientNoiseFirstBatchSampler and reports, per diffusion
step t: total samples (should be flat, ~ n/T each) and the share coming from
each source (only sources with t_min <= t may appear). Also the data-first
mass per t for comparison: drawing the window first makes the mass at t
proportional to the windows admitted there, i.e. the low-noise steps get
N_admitted(t) / N_total of the flat rate.

    python scripts/check_ambient_sampler.py --dataset STORE --tmin 0 20 20 20 20 [--batches 400]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from mjlab_hand.diffusion.dataset import (
    AmbientNoiseFirstBatchSampler,
    DiffusionDataset,
    TrajectoryStore,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tmin", type=int, nargs="+", required=True)
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--T", type=int, default=100)
    args = ap.parse_args()

    torch.manual_seed(0)
    store = TrajectoryStore(args.dataset, mode="r")
    ds = DiffusionDataset(store, ambient_tmin=args.tmin)
    sampler = AmbientNoiseFirstBatchSampler(ds, 256, args.T)
    src_of_win = ds.episode_source[ds._win_epi]
    n_src = int(src_of_win.max()) + 1
    counts = np.zeros((args.T, n_src), dtype=np.int64)
    viol = 0
    for b, batch in enumerate(sampler):
        if b >= args.batches:
            break
        for idx, t in batch:
            s = src_of_win[idx]
            counts[t, s] += 1
            viol += int(args.tmin[s] > t)
    tot = counts.sum(1)
    n = tot.sum()
    names = [task.rsplit("-", 1)[-1] for _, _, task in store.source_step_bounds()] or ["single"]
    print(f"{n:,} samples; gating violations (source with t_min > t): {viol}")
    print(f"noise-first mass per t: min {tot.min()} max {tot.max()} (flat = {n / args.T:.0f}); "
          f"max/min {tot.max() / max(tot.min(), 1):.2f}")
    win_tmin = ds.window_tmin
    df = np.array([(win_tmin <= t).sum() for t in range(args.T)], dtype=float)
    df_mass = df / df.sum()  # data-first: t ~ U[t_min, T) per window -> approx admitted share
    print("data-first relative mass at t (vs flat 1.0): "
          + " ".join(f"t{t}:{df_mass[t] * args.T:.2f}" for t in sorted({0, *args.tmin, args.T - 1})))
    for t in sorted({0, *[x for x in args.tmin if x < args.T], args.T - 1}):
        share = counts[t] / max(tot[t], 1)
        print(f"  t={t:3d} n={tot[t]:6d} " + " ".join(f"{names[i]}:{share[i]:.3f}" for i in range(n_src)))
    if viol or tot.max() / max(tot.min(), 1) > 1.5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
