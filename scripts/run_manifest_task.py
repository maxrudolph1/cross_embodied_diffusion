#!/usr/bin/env python3
"""Run one or more array tasks' runs from a manifest built by
build_scarce_specialist_manifest.py: manifest[task_id] is a list of
train-diffusion arg-dicts (the 2 seeds packed into that task).

By default the runs execute sequentially. With --parallel, every run of every
given task is started at once and shares the GPU -- for clusters that bill
whole nodes (TACC Vista: 4 concurrent 50k runs on one GH200 give ~1.8x the
throughput of 1). Each parallel run gets its own WARP_CACHE_PATH subdir, since
concurrent processes sharing one Warp kernel cache race (CCD-kernel
FileNotFoundError)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def build_cmd(run_args: dict) -> list[str]:
    cmd = ["train-diffusion"]
    for k, v in run_args.items():
        cmd.append(f"--{k}")
        if isinstance(v, list):
            cmd.extend(str(x) for x in v)
        elif not isinstance(v, bool):
            cmd.append(str(v))
        # bare bool flags aren't used by this manifest; nothing to append
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=str)
    ap.add_argument("task_ids", type=int, nargs="+")
    ap.add_argument("--parallel", action="store_true", help="run all runs concurrently")
    ap.add_argument(
        "--compile-mode",
        default=None,
        help="add --compile-mode to every run that doesn't set it (CHANGES.md item 53)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print commands, don't run them")
    args = ap.parse_args()

    manifest = json.loads(open(args.manifest).read())
    runs = []
    for task_id in args.task_ids:
        if not (0 <= task_id < len(manifest)):
            raise SystemExit(f"task_id {task_id} out of range for manifest of {len(manifest)} tasks")
        runs.extend((task_id, run_args) for run_args in manifest[task_id])
    if args.compile_mode is not None:
        runs = [(t, {"compile-mode": args.compile_mode, **r}) for t, r in runs]

    procs = []
    for i, (task_id, run_args) in enumerate(runs):
        cmd = build_cmd(run_args)
        print(f"[INFO] task={task_id} run {i + 1}/{len(runs)}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        if not args.parallel:
            subprocess.run(cmd, check=True)
            continue
        env = dict(os.environ)
        if "WARP_CACHE_PATH" in env:
            env["WARP_CACHE_PATH"] = os.path.join(env["WARP_CACHE_PATH"], f"run{i}")
            os.makedirs(env["WARP_CACHE_PATH"], exist_ok=True)
        procs.append((cmd, subprocess.Popen(cmd, env=env)))

    failed = [cmd for cmd, p in procs if p.wait() != 0]
    for cmd in failed:
        print(f"[ERROR] failed: {' '.join(cmd)}", file=sys.stderr, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
