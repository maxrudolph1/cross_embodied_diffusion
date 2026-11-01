#!/usr/bin/env python3
"""Use an existing GPU allocation for staged expert rounds (CHANGES 85).

Run every manifest task locally; no scheduler commands are issued. Every local
stage evaluates and renders before the next hand starts. State and subprocess
logs survive exit. Callers must avoid duplicate remote work before launching.
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from run_dexterity_batch_node import resume_checkpoint


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wait-root", type=Path, required=True)
    parser.add_argument("--deadline", required=True, help="ISO timestamp with UTC offset")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    manifest = read(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "serial.json"
    if state_path.exists():
        raise SystemExit("Use a fresh output directory; refusing to replay an existing schedule")
    state = {"status": "waiting", "pid": os.getpid(), "jobs": [],
             "deadline": args.deadline}

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    # The already active LEAP workflow owns this GPU until completion.
    save()
    while read(args.wait_root / "workflow.json").get("status") == "running":
        if time.time() >= deadline - 1800:
            state["status"] = "allocation_budget_exhausted"
            save()
            return
        time.sleep(30)
    initial = read(args.wait_root / "workflow.json")
    if initial.get("status") != "complete":
        state["status"] = "initial_workflow_failed"
        save()
        return
    previous = {}
    skipped = {i for i, task in enumerate(manifest["tasks"])
               if task["task"] == initial["task"]}
    for stage in (1000, 3000, 10000):
        for index, task in enumerate(manifest["tasks"]):
            if index in skipped:
                continue
            # Reserve time for evaluation and clean shutdown. Checkpoints also
            # survive an unexpected allocation expiry during a long stage.
            if time.time() >= deadline - 1800:
                state["status"] = "allocation_budget_exhausted"
                save()
                return
            output = args.output_dir / task["task"] / f"round_{stage}"
            command = [".venv/bin/python", "-u", "scripts/run_dexterity_local.py",
                       "--task", task["task"], "--output-dir", str(output),
                       "--stages", str(stage), "--num-envs", str(task["num_envs"]),
                       "--seed", str(task["seed"]), "--eval-seed", str(manifest["eval_seed"]),
                       "--eval-envs", str(manifest["eval_envs"])]
            if index in previous:
                command += ["--resume", str(resume_checkpoint(previous[index]))]
            env = dict(os.environ)
            cache = Path(f"/tmp/dex_serial_{os.environ.get('SLURM_JOB_ID', os.getpid())}/{index}")
            for key, leaf in (("DEXTERITY_CACHE_ROOT", "cache"), ("TMPDIR", "tmp"),
                              ("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", "xdg")):
                directory = cache / leaf
                directory.mkdir(parents=True, exist_ok=True)
                env[key] = str(directory)
            env.update(MUJOCO_GL="egl", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            row = {"task": task["task"], "stage": stage, "argv": command,
                   "status": "running", "output": str(output)}
            state["status"] = "running"
            state["jobs"].append(row)
            save()
            with (args.output_dir / f"{task['task']}_{stage}.log").open("w") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                row["pid"] = child.pid
                save()
                row["exit_code"] = child.wait()
            row["status"] = "complete" if row["exit_code"] == 0 else "failed"
            if row["exit_code"]:
                skipped.add(index)
            else:
                previous[index] = output
            save()
    state["status"] = "finished"
    save()


if __name__ == "__main__":
    main()
