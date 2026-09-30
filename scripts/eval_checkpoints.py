#!/usr/bin/env python3
"""Re-score each run's selected checkpoints with a fresh-seed eval, for
reporting (CHANGES.md item 57).

In-training evals pick `policy_best_eval.pt`, so their numbers are biased
upward by the selection; they are not reported. For every run dir given,
this evaluates

    policy_best_eval.pt   best in-training closed-loop eval (target hand)
    policy_best_val.pt    best val loss (target hand, --val-embodiment)
    policy_latest.pt      last epoch

on the run's target (its first eval spec / --eval-task), in a fresh env
seeded with `--seed-offset + <run's training seed>` -- the same env seed for
all three checkpoints of a run, so they face the same initial conditions --
with one episode per env (the evaluators score each env's first episode).
`--also EMB` (repeatable) scores extra embodiments too (e.g. a control hand; pooled
runs only). Results go to `<run>/posthoc_eval.json`, keyed
"<checkpoint>@<embodiment>"; entries already there are skipped, so the script
can be re-run after an interruption.

Usage:
  python scripts/eval_checkpoints.py outputs/diffusion/ambient_rot/*_seed*
  python scripts/eval_checkpoints.py --shard 2/8 RUN_DIR...   # every 8th run from #2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHECKPOINTS = ["policy_best_eval.pt", "policy_best_val.pt", "policy_latest.pt"]


def targets(cfg: dict, also: list[str]) -> list[tuple[str, str | None]]:
    """(task, embodiment) pairs to evaluate; embodiment None = unpooled policy."""
    if cfg.get("eval_specs"):
        spec = cfg["eval_specs"][0]
        first = (spec["task"], spec.get("embodiment"))
    elif cfg.get("eval_task"):
        first = (cfg["eval_task"], None)
    else:
        raise ValueError("run has neither eval_specs nor eval_task")
    out = [first]
    for emb in also:
        if first[1] is None:
            raise ValueError("--also needs a pooled (padded) run")
        if emb != first[1]:
            out.append((emb, emb))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--num-envs", type=int, default=100)
    ap.add_argument("--num-steps", type=int, default=800)
    ap.add_argument("--seed-offset", type=int, default=1000)
    ap.add_argument(
        "--also", action="append", default=[], help="extra embodiment to score (repeatable)"
    )
    ap.add_argument("--shard", default="0/1", help="i/n: take every n-th run starting at i")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from mjlab_hand.diffusion.evaluate import evaluate_diffusion_policy

    i, n = (int(x) for x in args.shard.split("/"))
    runs = sorted(r for r in args.runs if (r / "train_config.json").exists())[i::n]
    for run in runs:
        cfg = json.loads((run / "train_config.json").read_text())
        out_path = run / "posthoc_eval.json"
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
        seed = args.seed_offset + int(cfg.get("seed", 0))
        for task, emb in targets(cfg, args.also):
            for ckpt in CHECKPOINTS:
                key = f"{ckpt}@{emb or task}"
                if key in results or not (run / ckpt).exists():
                    continue
                metrics = evaluate_diffusion_policy(
                    task=task,
                    policy_path=run / ckpt,
                    num_envs=args.num_envs,
                    num_steps=args.num_steps,
                    device=args.device,
                    seed=seed,
                    embodiment=emb,
                )
                results[key] = {
                    "task": task,
                    "embodiment": emb,
                    "env_seed": seed,
                    "num_envs": args.num_envs,
                    "num_steps": args.num_steps,
                    "metrics": metrics,
                }
                out_path.write_text(json.dumps(results, indent=2))
                print(f"[EVAL] {run.name} {key}: {json.dumps(metrics)}", flush=True)


if __name__ == "__main__":
    main()
