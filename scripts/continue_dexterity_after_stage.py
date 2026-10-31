#!/usr/bin/env python3
"""Wait for an existing local stage, evaluate it, then resume a fresh workflow.

This never signals workers or submits jobs. The old driver may stop at its source
guard after completing the stage; the completed checkpoint remains usable.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from train_dexterity_expert import positive_int, repo_path


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(path, record):
    record["updated_utc"] = now()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=repo_path, required=True)
    parser.add_argument("--output-dir", type=repo_path, required=True)
    parser.add_argument("--stages", type=positive_int, nargs="+", default=[3000, 10000])
    parser.add_argument("--completed-stage", type=positive_int, default=1000)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--poll-seconds", type=positive_int, default=10)
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    args = parser.parse_args(argv)
    if not (args.previous_root / "workflow.json").is_file():
        parser.error("--previous-root must contain workflow.json")
    if args.output_dir == args.previous_root:
        parser.error("Continuation needs a different output directory")
    if not args.output_dir.parts or args.output_dir.parts[0] not in {"outputs", "logs"}:
        parser.error("--output-dir must be under outputs/ or logs/")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty for the new staged driver")
    if args.stages[0] <= args.completed_stage or any(b <= a for a, b in zip(args.stages, args.stages[1:])):
        parser.error("--stages must increase strictly beyond --completed-stage")
    return args


def completed_training(previous_root, completed_stage):
    training_dir = previous_root / f"stage_{completed_stage}"
    training = read_json(training_dir / "training.json")
    if training.get("status") in {"failed", "interrupted"}:
        raise RuntimeError(f"Previous training did not complete: {training.get('error', training['status'])}")
    if training.get("status") != "complete":
        return None
    if training.get("final_iteration") != completed_stage - 1:
        raise RuntimeError(f"Expected iteration {completed_stage - 1}, found {training.get('final_iteration')}")
    checkpoint = training_dir / f"model_{completed_stage - 1}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return training, checkpoint


def evaluation_matches(directory, checkpoint, task, seed, num_envs):
    metrics_path = directory / f"policy_seed{seed}.json"
    video = directory / f"policy_seed{seed}_env0.mp4"
    metrics = read_json(metrics_path)
    return (metrics.get("task") == task and metrics.get("checkpoint") == str(checkpoint)
            and metrics.get("seed") == seed and metrics.get("num_envs") == num_envs
            and metrics.get("summary", {}).get("episodes") == num_envs
            and metrics.get("summary", {}).get("censored_rate") == 0
            and video.is_file() and video.stat().st_size > 0)


def execute(command, environment, log):
    print(f"[{now()}] {shlex.join(command)}", flush=True)
    log.write(shlex.join(command) + "\n")
    log.flush()
    subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    args = parse_args()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir.with_name(args.output_dir.name + ".continuation.json")
    log_path = args.output_dir.with_name(args.output_dir.name + ".continuation.log")
    record = {"status": "waiting_for_training", "started_utc": now(), "pid": os.getpid(),
              "previous_root": str(args.previous_root), "output_dir": str(args.output_dir),
              "completed_stage": args.completed_stage, "stages": args.stages, "log": str(log_path)}
    save(metadata_path, record)
    try:
        ready = None
        while ready is None:
            ready = completed_training(args.previous_root, args.completed_stage)
            if ready is None:
                time.sleep(args.poll_seconds)
        training, checkpoint = ready
        record.update(status="waiting_for_previous_driver", checkpoint=str(checkpoint), task=training["task"])
        save(metadata_path, record)
        while True:
            previous = read_json(args.previous_root / "workflow.json")
            if previous.get("status") in {"failed", "interrupted", "complete"}:
                break
            time.sleep(args.poll_seconds)
        later_started = [stage for stage in previous.get("stages", [])
                         if stage["cumulative_iterations"] > args.completed_stage
                         and stage.get("status") != "pending"]
        if later_started:
            raise RuntimeError("Previous workflow advanced beyond the requested checkpoint; avoid duplicate continuation")
        record.update(previous_workflow_status=previous["status"],
                      previous_workflow_error=previous.get("error"), supersedes_previous_workflow=True)
        old_args = previous["arguments"]
        task = training["task"]
        seed = int(old_args.get("seed", 0))
        eval_seed = int(old_args.get("eval_seed", 1234))
        eval_envs = int(old_args.get("eval_envs", 100))
        environment = dict(os.environ, MUJOCO_GL="egl", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        if args.cache_root:
            environment["DEXTERITY_CACHE_ROOT"] = str(args.cache_root)
            environment["WARP_CACHE_PATH"] = str(args.cache_root / "warp")
        else:
            environment["WARP_CACHE_PATH"] = str(checkpoint.parent / "warp_cache")
        evaluation_dir = args.previous_root / f"eval_{args.completed_stage}"
        with log_path.open("w") as log:
            record["status"] = "evaluating_previous_checkpoint"
            save(metadata_path, record)
            if not evaluation_matches(evaluation_dir, checkpoint, task, eval_seed, eval_envs):
                command = [".venv/bin/python", "-u", "scripts/eval_dexterity_expert.py", "--task", task,
                           "--checkpoint", str(checkpoint), "--output-dir", str(evaluation_dir),
                           "--num-envs", str(eval_envs), "--seed", str(eval_seed), "--device", args.device,
                           "--video", "--video-steps", str(old_args.get("video_steps", 1500))]
                record["eval_argv"] = command
                save(metadata_path, record)
                execute(command, environment, log)
            if not evaluation_matches(evaluation_dir, checkpoint, task, eval_seed, eval_envs):
                raise RuntimeError("Previous-checkpoint evaluation is missing or does not match the requested protocol")
            record.update(status="continuing", evaluation=str(evaluation_dir))
            command = [".venv/bin/python", "-u", "scripts/run_dexterity_local.py", "--task", task,
                       "--output-dir", str(args.output_dir), "--resume", str(checkpoint),
                       "--stages", *map(str, args.stages), "--num-envs", str(old_args.get("num_envs", 1024)),
                       "--seed", str(seed), "--eval-seed", str(eval_seed), "--eval-envs", str(eval_envs),
                       "--device", args.device, "--save-interval", str(old_args.get("save_interval", 250)),
                       "--keep-last", str(old_args.get("keep_last", 3)),
                       "--video-steps", str(old_args.get("video_steps", 1500))]
            record["continue_argv"] = command
            save(metadata_path, record)
            execute(command, environment, log)
        record["status"] = "complete"
    except BaseException as exc:
        record["failed_phase"] = record["status"]
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_utc"] = now()
        save(metadata_path, record)


if __name__ == "__main__":
    main()
