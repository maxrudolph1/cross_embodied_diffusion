#!/usr/bin/env python3
"""Chain a pre-train manifest and its fine-tune manifest into ONE job (CHANGES.md item 79).

For every pre-train run, the fine-tune whose `init-checkpoint` lives in that run's output dir is chained after
it, with the reporting evals, as one manifest task (a `{"chain": [...]}` item, run by run_manifest_task.py):

  1. pre-train
  2. in parallel: the fine-tune, and the pre-train's evals (last0 on its own task at --seed-test, then on the
     family's other hands at --seed-test, eval_cross_embodiment.py --skip-target)
  3. in parallel: the fine-tune scored on its own task at --seed-test and --seed-select (best_rollout, best_val,
     last0), and on the other hands at --seed-test (best_val, last0; one process each)

All evals are 100 envs x 1500 steps (rescore_selected.py / eval_cross_embodiment.py, resumable). One task per
node (PACK=1).

Without --ft (item 80): each run is chained with its own evals only: train, then --which (default best_val)
on its own task at --seed-select and --seed-test in parallel. E23 (grasp sweep).

E22:

  python scripts/build_chain_manifest.py --pre slurm_jobs/loo_pretrain_manifest.json \\
      --ft slurm_jobs/loo_finetune_manifest.json --out slurm_jobs/loo_chain_manifest.json
  python scripts/build_chain_manifest.py --pre slurm_jobs/grasp_sweep_manifest.json \\
      --out slurm_jobs/grasp_sweep_chain_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rescore(run: str, which: list[str], seed: int) -> dict:
    return {"cmd": ["python3", "scripts/rescore_selected.py", "--run", run, "--which", *which,
                    "--envs", "100", "--steps", "1500", "--eval-seed", str(seed)]}


def cross(run: str, which: list[str], seed: int) -> dict:
    return {"cmd": ["python3", "scripts/eval_cross_embodiment.py", "--run", run, "--which", *which,
                    "--envs", "100", "--steps", "1500", "--eval-seed", str(seed), "--skip-target"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", type=Path, required=True)
    ap.add_argument("--ft", type=Path, default=None, help="fine-tune manifest; omit to chain evals only")
    ap.add_argument("--which", nargs="+", default=["best_val"], help="checkpoints scored when --ft is omitted")
    ap.add_argument("--seed-test", type=int, default=4321)
    ap.add_argument("--seed-select", type=int, default=1234)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pre = [r for task in json.loads(args.pre.read_text()) for r in task]
    if args.ft is None:
        tasks = [[{"chain": [p, {"parallel": [rescore(p["output-dir"], args.which, args.seed_select),
                                              rescore(p["output-dir"], args.which, args.seed_test)]}]}]
                 for p in pre]
        args.out.write_text(json.dumps(tasks, indent=1) + "\n")
        print(f"[INFO] wrote {len(tasks)} train+eval chains -> {args.out}")
        return
    ft = {str(Path(r["init-checkpoint"]).parent): r for task in json.loads(args.ft.read_text()) for r in task}
    tasks = []
    for p in pre:
        f = ft.pop(p["output-dir"], None)
        if f is None:
            raise SystemExit(f"no fine-tune starts from {p['output-dir']}")
        P, F, T, S = p["output-dir"], f["output-dir"], args.seed_test, args.seed_select
        tasks.append([{"chain": [
            p,
            {"parallel": [f, {"chain": [rescore(P, ["last0"], T), cross(P, ["last0"], T)]}]},
            {"parallel": [rescore(F, ["best_rollout", "best_val", "last0"], T),
                          rescore(F, ["best_rollout", "best_val", "last0"], S),
                          cross(F, ["best_val"], T), cross(F, ["last0"], T)]},
        ]}])
    if ft:
        raise SystemExit(f"fine-tunes without a pre-train in --pre: {sorted(ft)}")
    args.out.write_text(json.dumps(tasks, indent=1) + "\n")
    print(f"[INFO] wrote {len(tasks)} chained tasks -> {args.out}")


if __name__ == "__main__":
    main()
