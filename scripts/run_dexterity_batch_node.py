#!/usr/bin/env python3
"""Map a Vista array/node slot to independent local dexterity workflows.

This is called by vista_dexterity_experts.sbatch from an immutable source snapshot.
It uses no scheduler commands itself. --dry-run only prints mapping and commands.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def plan(manifest: dict, array_index: int, nodes: int, node_id: int, pack: int) -> list[dict]:
    if min(array_index, node_id) < 0 or min(nodes, pack) < 1 or node_id >= nodes:
        raise ValueError("Invalid array/node/PACK mapping")
    first = (array_index * nodes + node_id) * pack
    jobs = []
    for index in range(first, min(first + pack, len(manifest["tasks"]))):
        task = manifest["tasks"][index]
        output = f"outputs/{manifest['output_prefix']}/{task['task']}_seed{task['seed']}"
        command = [".venv/bin/python", "-u", "scripts/run_dexterity_local.py",
                   "--task", task["task"], "--output-dir", output,
                   "--stages", *map(str, manifest["stages"]),
                   "--num-envs", str(task["num_envs"]), "--seed", str(task["seed"]),
                   "--eval-seed", str(manifest["eval_seed"]),
                   "--eval-envs", str(manifest["eval_envs"]),
                   "--save-interval", str(manifest["save_interval"]),
                   "--keep-last", str(manifest["keep_last"]), "--device", "cuda:0"]
        jobs.append({"task_index": index, "task": task["task"], "output": output,
                     "argv": command, "command": shlex.join(command)})
        if manifest.get("resume_root"):
            jobs[-1]["resume_search_root"] = (
                f"{manifest['resume_root']}/{task['task']}_seed{task['seed']}"
            )
    return jobs


def resume_checkpoint(root: Path) -> Path:
    """Find the newest intact torch.save ZIP after a bootstrap dependency.

    A time limit can interrupt writing the latest checkpoint. Check ZIP records
    and CRCs without unpickling, then fall back to an older retained checkpoint.
    """
    candidates = []
    for checkpoint in root.glob("stage_*/model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", checkpoint.name)
        if match and checkpoint.is_file() and checkpoint.stat().st_size:
            candidates.append((int(match[1]), checkpoint))
    for _, checkpoint in sorted(candidates, reverse=True):
        try:
            with zipfile.ZipFile(checkpoint) as archive:
                if archive.namelist() and archive.testzip() is None:
                    return checkpoint
        except (zipfile.BadZipFile, EOFError, OSError, RuntimeError):
            pass
        print(f"[WARN] Skipping incomplete/corrupt bootstrap checkpoint: {checkpoint}",
              file=sys.stderr, flush=True)
    raise FileNotFoundError(f"No intact bootstrap checkpoint under {root}; refusing a fresh restart")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--array-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--nodes", type=int, default=int(os.environ.get("SLURM_NNODES", 1)))
    parser.add_argument("--node-id", type=int, default=int(os.environ.get("SLURM_NODEID", 0)))
    parser.add_argument("--pack", type=int, default=int(os.environ.get("PACK", 1)))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    jobs = plan(manifest, args.array_index, args.nodes, args.node_id, args.pack)
    payload = {"host": socket.gethostname(), "array_index": args.array_index,
               "node_id": args.node_id, "nodes": args.nodes, "pack": args.pack,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "source_revision": manifest["source_revision"],
               "started_utc": datetime.now(timezone.utc).isoformat(), "jobs": jobs}
    print(json.dumps(payload, indent=2), flush=True)
    if args.dry_run or not jobs:
        return
    cache_root = Path(os.environ["DEXTERITY_NODE_TMP"])
    batch = Path("outputs") / manifest["output_prefix"] / "batch"
    batch.mkdir(parents=True, exist_ok=True)
    metadata = batch / f"array_{args.array_index}_node_{args.node_id}.json"
    metadata.write_text(json.dumps(payload, indent=2) + "\n")
    processes = []
    try:
        for job in jobs:
            if job.get("resume_search_root"):
                checkpoint = resume_checkpoint(Path(job["resume_search_root"]))
                job["argv"] += ["--resume", str(checkpoint)]
                job["command"] = shlex.join(job["argv"])
                job["resume_checkpoint"] = str(checkpoint)
            root = cache_root / f"task_{job['task_index']}"
            environment = dict(os.environ)
            environment["DEXTERITY_SOURCE_REVISION"] = manifest["source_revision"]
            for name, leaf in (("TMPDIR", "tmp"), ("DEXTERITY_CACHE_ROOT", "cache"),
                               ("XDG_CACHE_HOME", "xdg"), ("MPLCONFIGDIR", "mpl")):
                target = root / leaf
                target.mkdir(parents=True, exist_ok=True)
                environment[name] = str(target)
            environment.update(MUJOCO_GL="egl", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            log = batch / f"task_{job['task_index']}.log"
            with log.open("w") as stream:
                process = subprocess.Popen(job["argv"], stdout=stream, stderr=subprocess.STDOUT,
                                           env=environment)
            processes.append((process, job))
            job["pid"] = process.pid
            job["log"] = str(log)
        metadata.write_text(json.dumps(payload, indent=2) + "\n")
        codes = []
        for process, job in processes:
            job["exit_code"] = process.wait()
            codes.append(job["exit_code"])
            metadata.write_text(json.dumps(payload, indent=2) + "\n")
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        payload["status"] = "complete" if not any(codes) else "failed"
        metadata.write_text(json.dumps(payload, indent=2) + "\n")
        if any(codes):
            raise SystemExit(1)
    except BaseException:
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
        raise


if __name__ == "__main__":
    main()
