#!/usr/bin/env python3
"""Manifests for ambient diffusion on mixed-embodiment InHand-Rotation data
(CHANGES.md item 57). One run per manifest task, for
`slurm_jobs/vista_train_manifest.sbatch` with PACK=4 (4 runs per node).

Every pooled run lists the per-hand zarrs directly (pooled in memory, item
56): the target hand's 50k subset first, then the other four hands' full 1M
sets. `--ambient-tmin 0 s s s s` admits the target at every diffusion
timestep and the other hands only at t >= s (T = 100 timesteps, t in
[0, 99]): s = 0 is plain co-training on the pool, s = 100 is target-only.

--kind sweep (320 runs): 5 target hands x 16 sigmas x 4 seeds. The 4 seeds
of one (hand, sigma) are consecutive, so with PACK=4 each node runs exactly
one config and every node does the same amount of work.

--kind diagnostic (8 runs, target = Allegro, seed 0): is the padded /
pooled path what zeroed the 2026-09-28 scarce sweep? Node 0 = specialists
(50k and 1M, each plain and through --pool-sources), node 1 = the 5-hand
pool (plain co-training, and ambient s = 0, 10, 100).

Checkpoint selection (see JOURNAL 2026-09-29): in-training eval is on the
target hand only (so policy_best_eval.pt is picked by target success), the
val loss is on the target hand's held-out trajectories only
(--val-embodiment, so policy_best_val.pt is picked by target val loss), and
policy_latest.pt is the last epoch. No per-epoch snapshots. All three are
re-scored afterwards with a fresh-seed eval for reporting.

Step budget: ~700k gradient steps everywhere (50 epochs of the ~4M pool,
4000 of 50k, 200 of 1M), as in the scarce/specialist manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILY = "InHand-Rotation"
HANDS = ["Allegro", "LEAP", "Shadow", "Sharpa", "Wuji"]
SIGMAS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 25, 100]
SEEDS = [0, 1, 2, 3]
DATA = "data/mjlab_hand_demos"
COMMON = {
    "obs-horizon": 2,
    "action-horizon": 8,
    "batch-size": 256,
    "num-workers": 8,
    "val-fraction": 0.1,
    "val-seed": 0,
    "val-every-epochs": 1,
    "val-max-batches": 20,
    "eval-num-envs": 32,
    "eval-num-steps": 800,
    "wandb-project": "mjlab",
}


def task(hand: str) -> str:
    return f"{FAMILY}-{hand}"


def cadence(epochs: int) -> dict:
    return {
        "num-epochs": epochs,
        "eval-every-epochs": max(1, epochs // 10),
        "latest-every-epochs": max(1, epochs // 10),
        "save-every-epochs": epochs + 1,  # never: keep only latest/best_val/best_eval
    }


def pool_sources(target: str) -> list[str]:
    return [f"{DATA}/subsets_50k/{task(target)}_expert_50k.zarr"] + [
        f"{DATA}/{task(h)}_expert_1M.zarr" for h in HANDS if h != target
    ]


def target_eval(target: str, padded: bool) -> dict:
    if not padded:
        return {"eval-task": task(target)}
    return {"eval-spec": json.dumps([{"task": task(target), "embodiment": task(target)}])}


def pooled_run(target: str, sigma: int | None, seed: int, out: str, name: str, tags: list) -> dict:
    run = {
        **COMMON,
        **cadence(50),
        "dataset": pool_sources(target),
        "output-dir": out,
        **target_eval(target, padded=True),
        "val-embodiment": task(target),
        # Per-source min/max normalization: the gaussian default zeroed every
        # pooled run (CHANGES.md item 58); minmax matches the plain path.
        "source-norm": "minmax",
        "seed": seed,
        "wandb-run-name": name,
        "wandb-tags": tags,
    }
    if sigma is not None:
        run["ambient-tmin"] = [0] + [sigma] * (len(HANDS) - 1)
    return run


def sweep() -> list[list[dict]]:
    tasks = []
    for hand in HANDS:
        for sigma in SIGMAS:
            for seed in SEEDS:
                name = f"{task(hand)}-ambient-s{sigma}-seed{seed}"
                out = f"outputs/diffusion/ambient_rot/{task(hand)}_sigma{sigma}_seed{seed}"
                tags = ["ambient-rot", FAMILY, f"target-{hand}", f"sigma{sigma}"]
                tasks.append([pooled_run(hand, sigma, seed, out, name, tags)])
    return tasks


def diagnostic() -> list[list[dict]]:
    hand, seed, root = "Allegro", 0, "outputs/diffusion/diag_rot"
    tags = ["diag-rot", FAMILY, f"target-{hand}"]
    runs = []
    for scale, epochs, dataset in [
        ("50k", 4000, f"{DATA}/subsets_50k/{task(hand)}_expert_50k.zarr"),
        ("1M", 200, f"{DATA}/{task(hand)}_expert_1M.zarr"),
    ]:
        for padded in (False, True):
            label = f"spec{scale}_{'padded' if padded else 'plain'}"
            run = {
                **COMMON,
                **cadence(epochs),
                "dataset": dataset,
                "output-dir": f"{root}/{task(hand)}_{label}_seed{seed}",
                **target_eval(hand, padded),
                "seed": seed,
                "wandb-run-name": f"diag-{task(hand)}-{label}",
                "wandb-tags": [*tags, label],
            }
            if padded:
                run["pool-sources"] = True
            runs.append(run)
    for label, sigma in [("pool_cotrain", None), ("ambient_s0", 0), ("ambient_s10", 10), ("ambient_s100", 100)]:
        out = f"{root}/{task(hand)}_{label}_seed{seed}"
        runs.append(pooled_run(hand, sigma, seed, out, f"diag-{task(hand)}-{label}", [*tags, label]))
    return [[r] for r in runs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["sweep", "diagnostic"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tasks = sweep() if args.kind == "sweep" else diagnostic()
    args.out.write_text(json.dumps(tasks, indent=1) + "\n")
    print(f"[INFO] wrote {len(tasks)} tasks ({sum(len(t) for t in tasks)} runs) -> {args.out}")


if __name__ == "__main__":
    main()
