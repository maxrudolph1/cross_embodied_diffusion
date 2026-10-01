#!/usr/bin/env python3
"""Fresh-seed re-score of a run's selected checkpoints -> <run>/final_eval.jsonl
(CHANGES.md item 63; replaces scripts/eval_checkpoints.py).

In-training evals pick policy_best_rollout.pt, so their numbers overstate it
(+0.21 on rotation in Bundle). Report from this instead: each --which
checkpoint is scored on the run's own eval spec (task + pad flag, or
--eval-task), in a fresh env seeded --eval-seed, one episode per env.

  --which   best_rollout | best_val | best | last0 | last1 | last2 | latest
            (lastK resolved through selection.json)
Rows already in final_eval.jsonl (same which, env count, steps, seed) are
skipped, so it can be re-run after an interruption. Like Bundle's, it scores
only the run's eval spec (no 5-hand diagnostic).

  python scripts/rescore_selected.py --run RUN [RUN ...] \\
      --which best_rollout best_val last0 --envs 100 --steps 1500 --eval-seed 1234
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve(run: Path, which: str) -> Path | None:
    if which in ("best_rollout", "best_val", "best", "latest"):
        p = run / f"policy_{which}.pt"
        return p if p.exists() else None
    if which.startswith("last"):
        sel = json.loads((run / "selection.json").read_text())
        k = int(which[len("last"):])
        for item in sel.get("last", []):
            if item["k"] == k:
                return run / item["path"]
    return None


def specs_of(run: Path) -> list[dict]:
    cfg = json.loads((run / "train_config.json").read_text())
    if cfg.get("eval_specs"):
        return cfg["eval_specs"]
    if cfg.get("eval_task"):
        return [{"task": cfg["eval_task"]}]
    raise ValueError(f"{run}: no eval spec or eval task")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, nargs="+", required=True)
    ap.add_argument("--which", nargs="+", default=["best_rollout"])
    ap.add_argument("--envs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th run starting at i")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from mjlab_hand.diffusion.evaluate import evaluate_diffusion_policy

    i, n = (int(x) for x in args.shard.split("/"))
    runs = sorted(r for r in args.run if (r / "selection.json").exists())[i::n]
    skipped = sorted(set(args.run) - set(r for r in args.run if (r / "selection.json").exists()))
    for r in skipped:
        print(f"[SKIP] {r}: no selection.json (unfinished)")
    for run in runs:
        out = run / "final_eval.jsonl"
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                row = json.loads(line)
                done.add((row["which"], row["task"], row["envs"], row["steps"], row["eval_seed"]))
        for spec in specs_of(run):
            for which in args.which:
                key = (which, spec["task"], args.envs, args.steps, args.eval_seed)
                if key in done:
                    continue
                ckpt = resolve(run, which)
                if ckpt is None:
                    print(f"[SKIP] {run.name}: no checkpoint for {which}")
                    continue
                m = evaluate_diffusion_policy(
                    task=spec["task"],
                    policy_path=ckpt,
                    num_envs=args.envs,
                    num_steps=args.steps,
                    device=args.device,
                    seed=args.eval_seed,
                    onehot=spec.get("onehot"),
                    pad=bool(spec.get("pad", False)),
                )
                row = {"which": which, "checkpoint": ckpt.name, "task": spec["task"],
                       "pad": bool(spec.get("pad", False)), "envs": args.envs, "steps": args.steps,
                       "eval_seed": args.eval_seed, "metrics": m}
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[EVAL] {run.name} {which} {spec['task']}: "
                      + json.dumps({k: v for k, v in m.items() if k != "per_episode"}), flush=True)


if __name__ == "__main__":
    main()
