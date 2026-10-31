#!/usr/bin/env python3
"""Sequential local train/evaluate stages on one allocated GPU; never submits jobs.

Example (stage counts are cumulative):
  .venv/bin/python scripts/run_dexterity_local.py --task Baoding-Allegro \
    --output-dir outputs/dexterity/baoding_staged --stages 1000 3000 10000

Each stage writes a checkpoint directory, an evaluation/video directory, and two
combined stdout/stderr logs. workflow.json tracks progress atomically. Completion
means the workflow ran successfully, not that the resulting policy is an expert.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from train_dexterity_expert import positive_int, repo_path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    hands = ("Allegro", "LEAP", "Shadow", "Sharpa", "Wuji")
    parser.add_argument("--task", required=True,
                        choices=[f"{family}-{hand}" for family in ("Baoding", "Pen-Spinning") for hand in hands])
    parser.add_argument("--output-dir", required=True, type=repo_path)
    parser.add_argument("--stages", type=positive_int, nargs="+", default=[1000, 3000, 10000])
    parser.add_argument("--num-envs", type=positive_int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--eval-envs", type=positive_int, default=100)
    parser.add_argument("--save-interval", type=positive_int, default=250)
    parser.add_argument("--keep-last", type=positive_int, default=3)
    parser.add_argument("--video-steps", type=positive_int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if any(right <= left for left, right in zip(args.stages, args.stages[1:])):
        parser.error("--stages must be strictly increasing cumulative iteration counts")
    if not re.fullmatch(r"cuda:\d+", args.device):
        parser.error("--device must name one allocated local GPU, e.g. cuda:0")
    if args.eval_seed == 0 or args.eval_seed == args.seed:
        parser.error("--eval-seed must be nonzero and different from the training seed")
    if not args.output_dir.parts or args.output_dir.parts[0] not in {"outputs", "logs"}:
        parser.error("--output-dir must be under outputs/ or logs/")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("--output-dir must be a fresh empty directory")
    return args


def write_workflow(path: Path, workflow: dict) -> None:
    workflow["updated_utc"] = now()
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as file:
        json.dump(workflow, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    paths = sorted(Path("src/mjlab_hand").rglob("*.py"))
    paths.extend(Path("scripts") / name for name in (
        "run_dexterity_local.py", "train_dexterity_expert.py", "eval_dexterity_expert.py",
        "dexterity_runtime.py",
    ))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def execute(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    print(f"[{now()}] {shlex.join(command)}\nLog: {log_path}", flush=True)
    with log_path.open("w") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                       env=environment, check=True)


def main() -> None:
    args = parse_args()
    python = ".venv/bin/python"
    if not Path("src/mjlab_hand").is_dir() or not Path(python).is_file():
        raise RuntimeError("Run from the repository root with its .venv installed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = args.output_dir / "workflow.json"
    snapshot = source_hashes()
    for source in snapshot:
        destination = args.output_dir / "source_snapshot" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    workflow = {
        "task": args.task, "status": "running", "started_utc": now(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "source_sha256": snapshot, "stages": [],
        "note": "A completed stage is a policy checkpoint, not a verified expert.",
    }
    for count in args.stages:
        workflow["stages"].append({"cumulative_iterations": count, "status": "pending"})
    write_workflow(workflow_path, workflow)
    previous_count = 0
    checkpoint = None
    active_stage = None
    try:
        for stage in workflow["stages"]:
            active_stage = stage
            count = stage["cumulative_iterations"]
            training_dir = args.output_dir / f"stage_{count}"
            evaluation_dir = args.output_dir / f"eval_{count}"
            train_log = args.output_dir / f"stage_{count}.train.log"
            eval_log = args.output_dir / f"stage_{count}.eval.log"
            if source_hashes() != snapshot:
                raise RuntimeError("Source changed during the workflow; stopped before mixing task implementations")
            train_command = [python, "scripts/train_dexterity_expert.py", "--task", args.task,
                             "--output-dir", str(training_dir), "--num-envs", str(args.num_envs),
                             "--iterations", str(count - previous_count), "--seed", str(args.seed),
                             "--save-interval", str(args.save_interval), "--keep-last", str(args.keep_last),
                             "--device", args.device]
            if checkpoint is not None:
                train_command += ["--resume", str(checkpoint)]
            stage.update({"status": "training", "started_utc": now(),
                          "train_argv": train_command, "train_command": shlex.join(train_command),
                          "train_log": str(train_log), "eval_log": str(eval_log)})
            write_workflow(workflow_path, workflow)
            environment = dict(os.environ)
            environment["MUJOCO_GL"] = "egl"
            environment["OMP_NUM_THREADS"] = "1"
            execute(train_command, train_log, environment)
            if source_hashes() != snapshot:
                raise RuntimeError("Source changed during training; evaluation stopped to preserve task identity")
            training = json.loads((training_dir / "training.json").read_text())
            if training.get("status") != "complete" or training.get("final_iteration") != count - 1:
                raise RuntimeError(f"Stage {count} did not finish at expected iteration {count - 1}: {training}")
            checkpoint = training_dir / f"model_{training['final_iteration']}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Final checkpoint missing: {checkpoint}")
            eval_command = [python, "scripts/eval_dexterity_expert.py", "--task", args.task,
                            "--checkpoint", str(checkpoint), "--output-dir", str(evaluation_dir),
                            "--num-envs", str(args.eval_envs), "--seed", str(args.eval_seed),
                            "--device", args.device, "--video", "--video-steps", str(args.video_steps)]
            stage.update({"status": "evaluating", "checkpoint": str(checkpoint),
                          "eval_argv": eval_command, "eval_command": shlex.join(eval_command)})
            write_workflow(workflow_path, workflow)
            environment["WARP_CACHE_PATH"] = str(args.output_dir / f"eval_{count}_warp_cache")
            execute(eval_command, eval_log, environment)
            metrics_path = evaluation_dir / f"policy_seed{args.eval_seed}.json"
            video_path = evaluation_dir / f"policy_seed{args.eval_seed}_env0.mp4"
            metrics = json.loads(metrics_path.read_text())
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise RuntimeError(f"Evaluation did not produce its video: {video_path}")
            stage.update({"status": "complete", "finished_utc": now(),
                          "metrics": str(metrics_path), "video": str(video_path),
                          "summary": metrics["summary"]})
            previous_count = count
            write_workflow(workflow_path, workflow)
        workflow["status"] = "complete"
    except BaseException as exc:
        workflow["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        workflow["error"] = f"{type(exc).__name__}: {exc}"
        if active_stage is not None:
            active_stage["failed_phase"] = active_stage["status"]
            active_stage["status"] = workflow["status"]
        raise
    finally:
        workflow["finished_utc"] = now()
        write_workflow(workflow_path, workflow)
    print(f"Local staged workflow complete: {workflow_path}", flush=True)


if __name__ == "__main__":
    main()
