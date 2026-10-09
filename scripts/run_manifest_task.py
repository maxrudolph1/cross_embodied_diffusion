#!/usr/bin/env python3
"""Run one or more array tasks' runs from a manifest built by
build_scarce_specialist_manifest.py: manifest[task_id] is a list of
train-diffusion arg-dicts (the 2 seeds packed into that task).

By default the runs execute sequentially. With --parallel, every run of every
given task is started at once and shares the GPU -- for clusters that bill
whole nodes (TACC Vista: 4 concurrent 50k runs on one GH200 give ~1.8x the
throughput of 1). Each parallel run gets its own WARP_CACHE_PATH subdir, since
concurrent processes sharing one Warp kernel cache race (CCD-kernel
FileNotFoundError).

Chains (CHANGES.md item 79): an item of a task may also be {"chain": [stage, ...]}, whose stages run in order
(each must succeed before the next starts). A stage is a train-diffusion arg dict, {"cmd": [argv ...]} (any
command, e.g. scripts/rescore_selected.py), {"parallel": [stage, ...]} (run together, wait for all) or a nested
{"chain": [...]}. A chain counts as one item: with --parallel it runs alongside the task's other items. Every
process gets its own WARP_CACHE_PATH subdir. E22 uses one chain per node: pre-train -> fine-tune (+ pre-train
evals alongside) -> fine-tune evals."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading


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


_counter = iter(range(10**9))
_lock = threading.Lock()


def _env() -> dict:
    env = dict(os.environ)
    if "WARP_CACHE_PATH" in env:
        with _lock:
            i = next(_counter)
        env["WARP_CACHE_PATH"] = os.path.join(env["WARP_CACHE_PATH"], f"run{i}")
        os.makedirs(env["WARP_CACHE_PATH"], exist_ok=True)
    return env


def run_stage(stage: dict, dry_run: bool, depth: int = 0) -> None:
    """Run one chain stage; raise CalledProcessError / RuntimeError on failure."""
    pad = "  " * depth
    if "chain" in stage:
        for st in stage["chain"]:
            run_stage(st, dry_run, depth + 1)
        return
    if "parallel" in stage:
        errors = []

        def worker(st: dict) -> None:
            try:
                run_stage(st, dry_run, depth + 1)
            except Exception as e:  # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(st,)) for st in stage["parallel"]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise RuntimeError(f"{len(errors)} parallel stage(s) failed: {errors}")
        return
    cmd = [str(x) for x in stage["cmd"]] if "cmd" in stage else build_cmd(stage)
    print(f"[INFO] {pad}stage: {' '.join(cmd)}", flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=_env())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=str)
    ap.add_argument("task_ids", type=int, nargs="+")
    ap.add_argument("--parallel", action="store_true", help="run all runs concurrently")
    ap.add_argument("--dry-run", action="store_true", help="print commands, don't run them")
    args = ap.parse_args()

    manifest = json.loads(open(args.manifest).read())
    runs = []
    for task_id in args.task_ids:
        if not (0 <= task_id < len(manifest)):
            raise SystemExit(f"task_id {task_id} out of range for manifest of {len(manifest)} tasks")
        runs.extend((task_id, run_args) for run_args in manifest[task_id])

    chains = [(t, r) for t, r in runs if "chain" in r]
    runs = [(t, r) for t, r in runs if "chain" not in r]
    chain_errors = []

    def chain_worker(task_id: int, item: dict) -> None:
        try:
            run_stage(item, args.dry_run)
            print(f"[INFO] task={task_id} chain done", flush=True)
        except Exception as e:  # noqa: BLE001 - reported below
            print(f"[ERROR] task={task_id} chain failed: {e}", file=sys.stderr, flush=True)
            chain_errors.append(task_id)

    chain_threads = []
    for task_id, item in chains:
        print(f"[INFO] task={task_id} chain of {len(item['chain'])} stages", flush=True)
        th = threading.Thread(target=chain_worker, args=(task_id, item))
        th.start()
        if not args.parallel:
            th.join()
        chain_threads.append(th)

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
    for th in chain_threads:
        th.join()
    for cmd in failed:
        print(f"[ERROR] failed: {' '.join(cmd)}", file=sys.stderr, flush=True)
    if failed or chain_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
