#!/usr/bin/env python3
"""Train one dexterity expert on one already allocated local GPU (no Slurm).

Example pilot:
  .venv/bin/python scripts/train_dexterity_expert.py --task Baoding-Allegro \
    --output-dir outputs/dexterity/baoding_allegro_pilot --num-envs 1024 \
    --iterations 100 --save-interval 50

Use a fresh output directory for every invocation, including a resumed run.
--resume loads actor, critic, normalizers and optimizer; --iterations specifies
additional iterations. Only --keep-last numbered checkpoints are retained.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dexterity_runtime import check_checkpoint_task, task_signature


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError("use a repository-relative path without '..'")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    hands = ("Allegro", "LEAP", "Shadow", "Sharpa", "Wuji")
    parser.add_argument("--task", required=True,
                        choices=[f"{task}-{hand}" for task in ("Baoding", "Pen-Spinning") for hand in hands])
    parser.add_argument("--output-dir", required=True, type=repo_path)
    parser.add_argument("--num-envs", type=positive_int, default=1024)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-interval", type=positive_int, default=50)
    parser.add_argument("--keep-last", type=positive_int, default=3)
    parser.add_argument("--resume", type=repo_path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"cuda:\d+", args.device):
        parser.error("--device must name one local GPU, e.g. cuda:0")
    if not args.output_dir.parts or args.output_dir.parts[0] not in {"outputs", "logs"}:
        parser.error("--output-dir must be under outputs/ or logs/")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty; use a new directory even with --resume")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"Resume checkpoint does not exist: {args.resume}")
    return args


def prune_checkpoints(directory: Path, keep_last: int) -> None:
    """Only remove numbered RL checkpoints in this invocation's fresh directory."""
    numbered = []
    for path in directory.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            numbered.append((int(match[1]), path))
    for _, path in sorted(numbered)[:-keep_last]:
        path.unlink()


def warp_cache_path(output_dir: Path, environment=None) -> Path:
    """A launcher supplies a task-private cache shared by sequential stages."""
    environment = os.environ if environment is None else environment
    root = environment.get("DEXTERITY_CACHE_ROOT")
    return Path(root) / "warp" if root else output_dir / "warp_cache"


def write_progress(path: Path, progress: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(progress, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not Path("src/mjlab_hand").is_dir():
        raise RuntimeError("Run this script from the repository root")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", args.device.split(":")[1])
    os.environ["WARP_CACHE_PATH"] = str(warp_cache_path(args.output_dir))

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    import mjlab_hand  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on this node; no CPU fallback or job submission")
    torch.cuda.set_device(args.device)
    configure_torch_backends()
    torch.manual_seed(args.seed)
    env_cfg = load_env_cfg(args.task)
    agent_cfg = load_rl_cfg(args.task)
    signature = task_signature(args.task, env_cfg)
    if args.resume:
        check_checkpoint_task(args.resume, args.task, signature)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    agent_cfg.save_interval = args.save_interval
    agent_cfg.logger = "tensorboard"
    agent_cfg.resume = args.resume is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = args.output_dir / "params"
    params.mkdir(exist_ok=True)
    # Save before runner construction mutates the argument dictionary.
    dump_yaml(params / "env.yaml", asdict(env_cfg))
    dump_yaml(params / "agent.yaml", asdict(agent_cfg))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    source_revision = (os.environ.get("DEXTERITY_SOURCE_REVISION")
                       or os.environ.get("SOURCE_REVISION")
                       or (revision.stdout.strip() if revision.returncode == 0 else None))
    provenance = {
        "task": args.task,
        "task_signature": signature,
        "command": shlex.join([sys.executable, *sys.argv]),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "git_revision": source_revision,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device_name": torch.cuda.get_device_name(args.device),
        "status": "running",
    }
    metadata_path = args.output_dir / "training.json"
    metadata_path.write_text(json.dumps(provenance, indent=2) + "\n")
    env = None
    try:
        raw = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
        env = raw
        env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), str(args.output_dir), device=args.device)
        runner.add_git_repo_to_log(__file__)
        if args.resume:
            runner.load(str(args.resume), map_location=args.device)
            # rsl-rl saves the last completed iteration index and starts its next
            # learn() at current_learning_iteration. Advance past that saved step.
            runner.current_learning_iteration += 1
        original_save = runner.save

        def save_with_retention(path: str, infos=None):
            original_save(path, infos)
            prune_checkpoints(args.output_dir, args.keep_last)

        runner.save = save_with_retention
        # Record actual update completion using the monotonic clock, independently
        # of upstream log buffering/ETA. This performs no tensor-to-host transfers.
        original_update = runner.alg.update
        start_iteration = runner.current_learning_iteration
        completed_updates = 0
        learn_start = time.monotonic()
        last_update = learn_start

        def update_with_progress():
            nonlocal completed_updates, last_update
            result = original_update()
            now = time.monotonic()
            completed_updates += 1
            duration = now - last_update
            last_update = now
            if completed_updates == 1 or completed_updates % 10 == 0 or completed_updates == args.iterations:
                progress = {
                    "task": args.task, "pid": os.getpid(), "status": "training",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_iteration": start_iteration + completed_updates - 1,
                    "completed_updates_this_run": completed_updates,
                    "requested_updates_this_run": args.iterations,
                    "elapsed_learning_s": now - learn_start,
                    "last_iteration_s": duration,
                    "mean_iteration_s": (now - learn_start) / completed_updates,
                    "transitions_this_run": completed_updates * args.num_envs * agent_cfg.num_steps_per_env,
                }
                write_progress(args.output_dir / "progress.json", progress)
                print(f"[PROGRESS] {args.task} iteration={progress['completed_iteration']} "
                      f"updates={completed_updates}/{args.iterations} "
                      f"last={duration:.3f}s mean={progress['mean_iteration_s']:.3f}s", flush=True)
            return result

        runner.alg.update = update_with_progress
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        provenance["status"] = "complete"
        provenance["final_iteration"] = runner.current_learning_iteration
    except BaseException as exc:
        provenance["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(provenance, indent=2) + "\n")
        progress_path = args.output_dir / "progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text())
            progress.update(status=provenance["status"], updated_utc=provenance["finished_utc"])
            write_progress(progress_path, progress)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
