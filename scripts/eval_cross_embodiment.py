#!/usr/bin/env python3
"""Score generalist (term-aligned multi-hand) policies on every hand of their
family -> <run>/cross_eval.jsonl (CHANGES.md item 68).

For each run (a dir with selection.json and a padded eval spec) and each
--which checkpoint (resolved like rescore_selected.py: best_rollout, best_val,
best, latest, lastK), evaluate on all five hands of the run's family with
`pad=True`, `--envs` envs x `--steps` steps, env seed `--eval-seed`. Same
protocol as rescore_selected.py, so the run's own target in final_eval.jsonl
and its cross_eval.jsonl row for that hand are interchangeable (skipped here
with --skip-target if final_eval.jsonl already has it). Resumable: rows
already present are skipped.

  python scripts/eval_cross_embodiment.py --run RUN [RUN ...] --which last0 \\
      --envs 100 --steps 1500 --eval-seed 1234 [--shard i/n] [--skip-target]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rescore_selected import resolve, specs_of  # noqa: E402

from mjlab_hand.diffusion.padding import family_of, family_tasks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, nargs="+", required=True)
    ap.add_argument("--which", nargs="+", default=["last0"])
    ap.add_argument("--envs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--skip-target", action="store_true",
                    help="skip the run's own target if final_eval.jsonl already scored it identically")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from mjlab_hand.diffusion.evaluate import evaluate_diffusion_policy

    i, n = (int(x) for x in args.shard.split("/"))
    runs = sorted(r for r in args.run if (r / "selection.json").exists())[i::n]
    for run in runs:
        target = specs_of(run)[0]["task"]
        out = run / "cross_eval.jsonl"
        done = set()
        for f in (out, run / "final_eval.jsonl") if args.skip_target else (out,):
            if f.exists():
                for line in f.read_text().splitlines():
                    r = json.loads(line)
                    if f.name == "final_eval.jsonl" and r["task"] != target:
                        continue
                    done.add((r["which"], r["task"], r["envs"], r["steps"], r["eval_seed"]))
        for which in args.which:
            ckpt = resolve(run, which)
            if ckpt is None:
                print(f"[SKIP] {run.name}: no checkpoint for {which}")
                continue
            for task in family_tasks(family_of(target)):
                key = (which, task, args.envs, args.steps, args.eval_seed)
                if key in done:
                    continue
                m = evaluate_diffusion_policy(task=task, policy_path=ckpt, num_envs=args.envs,
                                              num_steps=args.steps, device=args.device,
                                              seed=args.eval_seed, pad=True)
                row = {"which": which, "checkpoint": ckpt.name, "target": target, "task": task,
                       "envs": args.envs, "steps": args.steps, "eval_seed": args.eval_seed,
                       "metrics": {k: v for k, v in m.items() if k != "per_episode"}}
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[EVAL] {run.name} {which} on {task}", flush=True)


if __name__ == "__main__":
    main()
