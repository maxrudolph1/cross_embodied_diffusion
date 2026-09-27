#!/usr/bin/env python3
"""Build the run manifest for the specialist + scarce-co-training sweep.

80 total `train-diffusion` runs, packed 2-per-Slurm-array-task (the 2 seeds
of the same config share a task, run sequentially) = 40 array tasks:

- 40 **specialist** runs: 10 (task, embodiment) combos x {50k, 1M} x 2 seeds.
  One dataset, one embodiment, ordinary "linear" normalizer -- the existing
  training path, unchanged.
- 40 **scarce co-training** runs: 2 task families x 5 "scarce" embodiment
  choices x {uniform, balanced} sampling mode x 2 seeds. Each dataset pools
  all 5 embodiments of a family with one embodiment (the "scarce" one)
  subsampled to 50k and the other four at their full ~1M, via
  `build_padded_dataset.py` (see `build_scarce_pools.sh`). "balanced"
  sampling (--source-sample-mode) upweights the scarce source so it gets
  equal expected representation per epoch instead of ~1/80th.

Epoch counts follow this project's fixed ~780k-total-gradient-step
convention (batch size 256): epochs = round(780000 * 256 / n_steps),
snapped to the same round numbers already used elsewhere in this repo
(200 for ~1M, 4000 for ~50k, 50 for the ~4.05M scarce pools -- c.f. 40 for
the ~5M full AllHands pool in slurm_jobs/train_cross_embodiment.sbatch).

val_seed is fixed (0) across both seeds of a config -- so seed 0 and seed 1
of the same dataset validate on the *same* held-out trajectories, making
their val losses directly comparable (only --seed, the training seed,
differs between the two runs packed in one task).
"""
from __future__ import annotations

import json
from pathlib import Path

HANDS = ["Allegro", "LEAP", "Shadow", "Sharpa", "Wuji"]
FAMILIES = ["Grasp", "InHand-Rotation"]
SEEDS = [0, 1]
DATASTOR = "/datastor2/mrudolph/mjlab_hand_demos"
COMMON = {
    "obs-horizon": 2,
    "action-horizon": 8,
    "batch-size": 256,
    "num-workers": 8,
    "val-fraction": 0.1,
    "val-seed": 0,
    "val-every-epochs": 1,
    "val-max-batches": 20,
    "eval-num-envs": 16,
    "eval-num-steps": 800,
    "wandb-project": "mjlab",
}


def cadence(epochs: int) -> dict:
    save_every = max(1, epochs // 10)
    eval_every = max(1, epochs // 10)
    return {
        "num-epochs": epochs,
        "save-every-epochs": save_every,
        "latest-every-epochs": save_every,
        "eval-every-epochs": eval_every,
    }


def specialist_runs(families: list[str] = FAMILIES) -> list[dict]:
    runs = []
    for family in families:
        for hand in HANDS:
            task = f"{family}-{hand}"
            for scale, epochs, dataset in [
                (
                    "50k",
                    4000,
                    f"{DATASTOR}/subsets_50k/{task}_expert_50k.zarr",
                ),
                ("1M", 200, f"{DATASTOR}/{task}_expert_1M.zarr"),
            ]:
                for seed in SEEDS:
                    out = f"outputs/diffusion/specialist/{task}_{scale}_seed{seed}"
                    args = {
                        **COMMON,
                        **cadence(epochs),
                        "dataset": dataset,
                        "output-dir": out,
                        "eval-task": task,
                        "seed": seed,
                        "wandb-run-name": f"{task}-{scale}-seed{seed}",
                        "wandb-tags": ["specialist", scale, family, hand],
                    }
                    runs.append({"kind": "specialist", "seed": seed, "config_key": (task, scale), "args": args})
    return runs


def scarce_runs(families: list[str] = FAMILIES) -> list[dict]:
    runs = []
    for family in families:
        eval_spec = json.dumps(
            [{"task": f"{family}-{h}", "embodiment": f"{family}-{h}"} for h in HANDS]
        )
        for scarce_hand in HANDS:
            dataset = f"{DATASTOR}/padded/{family}-Scarce-{scarce_hand}.zarr"
            for mode in ["uniform", "balanced"]:
                for seed in SEEDS:
                    out = f"outputs/diffusion/scarce/{family}-Scarce-{scarce_hand}_{mode}_seed{seed}"
                    args = {
                        **COMMON,
                        **cadence(50),
                        "dataset": dataset,
                        "output-dir": out,
                        "eval-spec": eval_spec,
                        "source-sample-mode": mode,
                        "seed": seed,
                        "wandb-run-name": f"{family}-Scarce-{scarce_hand}-{mode}-seed{seed}",
                        "wandb-tags": ["scarce-co-training", family, f"scarce-{scarce_hand}", mode],
                    }
                    runs.append(
                        {
                            "kind": "scarce",
                            "seed": seed,
                            "config_key": (family, scarce_hand, mode),
                            "args": args,
                        }
                    )
    return runs


def pack(runs: list[dict]) -> list[list[dict]]:
    by_config: dict[tuple, list[dict]] = {}
    for r in runs:
        by_config.setdefault(r["config_key"], []).append(r)
    jobs = []
    for key, group in by_config.items():
        assert len(group) == len(SEEDS), f"{key}: expected {len(SEEDS)} seeds, got {len(group)}"
        jobs.append([g["args"] for g in sorted(group, key=lambda g: g["seed"])])
    return jobs


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=FAMILIES, default=None, help="restrict to one task family")
    ap.add_argument("--kind", choices=["specialist", "scarce"], default=None)
    ap.add_argument("--out", default="slurm_jobs/scarce_specialist_manifest.json")
    a = ap.parse_args()
    families = [a.family] if a.family else FAMILIES

    specialist_jobs = pack(specialist_runs(families)) if a.kind != "scarce" else []
    scarce_jobs = pack(scarce_runs(families)) if a.kind != "specialist" else []
    all_jobs = specialist_jobs + scarce_jobs
    print(f"[INFO] {len(specialist_jobs)} specialist job-tasks, {len(scarce_jobs)} scarce job-tasks, "
          f"{len(all_jobs)} total tasks, {sum(len(j) for j in all_jobs)} total runs")
    out_path = Path(a.out)
    out_path.write_text(json.dumps(all_jobs, indent=2))
    print(f"[INFO] wrote {out_path}")


if __name__ == "__main__":
    main()
