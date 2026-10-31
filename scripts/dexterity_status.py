#!/usr/bin/env python3
"""Read live dexterity workflow/checkpoint/TensorBoard status without using a GPU.

Example: .venv/bin/python scripts/dexterity_status.py --roots outputs/dexterity/*staged*
Training success is labeled separately from held-out episode success.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def latest_checkpoint(root: Path) -> dict:
    files = []
    for path in root.glob("stage_*/model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            files.append((int(match[1]), path))
    if not files:
        return {}
    iteration, path = max(files)
    return {"path": str(path), "iteration": iteration,
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def status(root: Path) -> dict:
    workflow = read_json(root / "workflow.json")
    stages = workflow.get("stages", [])
    active = next((stage for stage in stages if stage.get("status") in {"training", "evaluating"}), None)
    if active is None:
        active = next((stage for stage in reversed(stages) if stage.get("status") != "pending"), {})
    count = active.get("cumulative_iterations")
    training_dir = root / f"stage_{count}" if count is not None else root
    progress = read_json(training_dir / "progress.json")
    row = {"task": workflow.get("task", "unknown"), "root": str(root),
           "workflow_status": workflow.get("status", "unknown"),
           "stage_status": active.get("status"), "stage_target": count,
           "progress": progress, "checkpoint": latest_checkpoint(root)}
    command = "baoding" if row["task"].startswith("Baoding-") else "pen_spin"
    if any(training_dir.glob("events.out.tfevents.*")):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        events = EventAccumulator(str(training_dir), size_guidance={"scalars": 1}).Reload()
        metrics = {}
        for key, tag in (("success", f"Metrics/{command}/episode_success"),
                         ("best_revolutions", f"Metrics/{command}/best_revolutions"),
                         ("mean_return", "Train/mean_reward")):
            if tag in events.Tags()["scalars"]:
                event = events.Scalars(tag)[-1]
                metrics[key] = event.value
                metrics["iteration"] = max(metrics.get("iteration", 0), event.step)
                metrics["updated_utc"] = datetime.fromtimestamp(event.wall_time, timezone.utc).isoformat()
        row["training_metrics"] = metrics
    completed = [stage for stage in stages if stage.get("summary")]
    if completed:
        last = completed[-1]
        row["held_out"] = {"stage": last["cumulative_iterations"], **last["summary"]}
    else:
        # A continuation waiter can evaluate a completed checkpoint after the
        # original driver's source guard stops; its report still belongs here.
        reports = []
        for path in root.glob("eval_*/policy_seed*.json"):
            match = re.fullmatch(r"eval_(\d+)", path.parent.name)
            report = read_json(path)
            if match and report.get("summary"):
                reports.append((int(match[1]), report["summary"]))
        if reports:
            count, summary = max(reports, key=lambda value: value[0])
            row["held_out"] = {"stage": count, **summary}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    roots = set()
    for root in args.roots:
        if (root / "workflow.json").is_file():
            roots.add(root)
        elif root.is_dir():
            roots.update(path.parent for pattern in ("*/workflow.json", "*/*/workflow.json")
                         for path in root.glob(pattern))
    rows = [status(root) for root in sorted(roots)]
    if args.json:
        print(json.dumps(rows, indent=2, allow_nan=False))
        return
    print("TASK                     WORKFLOW / STAGE          ITER/TARGET  SEC/ITER  TRAIN-SUCCESS  EVAL-SUCCESS  CHECKPOINT")
    for row in rows:
        progress, train, ckpt = row["progress"], row.get("training_metrics", {}), row["checkpoint"]
        iteration = progress.get("completed_iteration", train.get("iteration", "?"))
        speed = progress.get("last_iteration_s")
        train_success = train.get("success")
        eval_success = row.get("held_out", {}).get("success_rate")
        def fmt(value):
            return "—" if value is None else f"{value:.3f}"
        print(f"{row['task']:24s} {row['workflow_status']}/{row['stage_status']!s:18s} "
              f"{iteration!s:>5s}/{row['stage_target']!s:<6s} {fmt(speed):>8s} "
              f"{fmt(train_success):>14s} {fmt(eval_success):>13s} {ckpt.get('path', '—')}")
    if not rows:
        print("No workflow.json files found at the supplied roots or their first two directory levels.")


if __name__ == "__main__":
    main()
