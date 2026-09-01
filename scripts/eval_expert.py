#!/usr/bin/env python3
"""Evaluate an RL expert checkpoint, working around a broken `eval-policy`.

`eval-policy` (`mjlab_hand.eval.base.run_eval`) calls
`policy(_policy_input(obs, device))`, which flattens the observation
TensorDict before the policy sees it. An rsl_rl policy indexes obs by group
name and dies with::

    IndexError: too many indices for tensor of dimension 2
    (rsl_rl/models/mlp_model.py: obs_list = [obs[g] for g in self.obs_groups])

`collect_demos` already sidesteps this by calling `policy(obs)` on the raw
observation. This script duplicates `run_eval`'s loop and does the same, so
RL-expert reference numbers (e.g. for `plot_diffusion_curves.py`'s scaling
plots) can be produced without touching the hot `run_eval` path.

Deliberately not fixed in `run_eval` itself: that function is on the hot
path of concurrently-running training jobs and destabilising it for a
reporting convenience is not worth it. Proper fix, for when nothing is
mid-flight: drop the pre-flattening in `run_eval` and let each policy
handle its own input -- `DiffusionActionChunkPolicy.__call__` already calls
`_policy_input` internally, so only `eval.py`/`evaluate.py` callers of
`run_eval` would be affected.

See CHANGES.md item 6i.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--num-steps", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    import mjlab_hand  # noqa: F401
    from mjlab_hand.eval.base import select_evaluator
    from mjlab_hand.eval.config import EvalConfig
    from mjlab_hand.eval.env_setup import (
        default_eval_steps,
        load_policy,
        make_runner,
        setup_eval_env,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    eval_cfg = EvalConfig(num_envs=args.num_envs, seed=args.seed, device=str(device))
    env, rl_cfg = setup_eval_env(args.task, eval_cfg, device)

    EvaluatorCls = select_evaluator(env)
    num_steps = args.num_steps or default_eval_steps(EvaluatorCls.name)

    class _Args:
        grasp_success_tolerance = None
        grasp_success_steps = None
        rotation_success_tolerance = None
        rotation_success_steps = None

    evaluator = EvaluatorCls(env, _Args(), device)

    runner = make_runner(args.task, env, rl_cfg, device)
    policy = load_policy(runner, args.checkpoint, device)

    # Duplicated from mjlab_hand.eval.base.run_eval, passing raw `obs` to
    # `policy` instead of the flattened `_policy_input(obs, device)`.
    obs, _ = env.reset()
    evaluator.bootstrap(torch.arange(env.num_envs, device=device))
    evaluator.snapshot()

    t0 = time.time()
    for _ in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, extras = env.step(actions)
        evaluator.on_step(obs, rewards, dones, extras)
        if dones.any():
            evaluator.on_reset(dones.bool())
        evaluator.snapshot()

    metrics = evaluator.finalize()
    metrics["eval_time_s"] = time.time() - t0

    evaluator.report(metrics)
    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"task": args.task, "checkpoint": str(args.checkpoint), "metrics": metrics},
                indent=2,
            )
        )

    env.close()


if __name__ == "__main__":
    main()
