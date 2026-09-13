#!/usr/bin/env python3
"""Run one array task's runs (sequentially) from a manifest built by
build_scarce_specialist_manifest.py: manifest[task_id] is a list of
train-diffusion arg-dicts (the 2 seeds packed into that task)."""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("task_id", type=int)
    ap.add_argument("--dry-run", action="store_true", help="print commands, don't run them")
    args = ap.parse_args()

    manifest = json.loads(open(args.manifest).read())
    if not (0 <= args.task_id < len(manifest)):
        raise SystemExit(f"task_id {args.task_id} out of range for manifest of {len(manifest)} tasks")
    job = manifest[args.task_id]

    for i, run_args in enumerate(job):
        cmd = build_cmd(run_args)
        print(f"[INFO] task={args.task_id} run {i + 1}/{len(job)}: {' '.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
