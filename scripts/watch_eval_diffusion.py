#!/usr/bin/env python3
"""Periodically evaluate a diffusion policy while training is in progress.

Watches the training log for completed epochs and runs env eval on
``policy_latest.pt`` every ``--every-epochs`` epochs (and once at start).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

EPOCH_RE = re.compile(r"\[epoch\s+(\d+)/(\d+)\]\s+loss=([0-9.eE+-]+)\s+steps=(\d+)")


def latest_epoch_from_log(log_path: Path) -> tuple[int, float, int] | None:
    if not log_path.exists():
        return None
    last = None
    for line in log_path.read_text().splitlines():
        m = EPOCH_RE.search(line)
        if m:
            last = (int(m.group(1)), float(m.group(3)), int(m.group(4)))
    return last


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def plot_eval_curve(jsonl_path: Path, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    rows = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    success = [r["metrics"].get("success_rate", 0.0) for r in rows]
    dist = [r["metrics"].get("avg_final_dist_to_first_goal_m", float("nan")) for r in rows]
    losses = [r.get("train_loss") for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, success, marker="o", linewidth=1.5)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Env eval success rate")

    ax2 = axes[1]
    ax2.plot(epochs, dist, marker="o", color="C1", linewidth=1.5, label="final dist (m)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("avg final dist to goal (m)")
    ax2.grid(True, alpha=0.3)
    if any(l is not None for l in losses):
        ax3 = ax2.twinx()
        ax3.plot(epochs, losses, marker="x", color="C2", alpha=0.7, label="train loss")
        ax3.set_ylabel("train loss")
        ax3.set_yscale("log")
    ax2.set_title("Env eval distance (+ train loss)")
    fig.suptitle(jsonl_path.parent.name)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[INFO] Wrote {out_png}")


def run_one_eval(
    *,
    task: str,
    policy_path: Path,
    device: str,
    num_envs: int,
    num_steps: int,
    seed: int,
) -> dict:
    from mjlab_hand.diffusion.evaluate import evaluate_diffusion_policy

    return evaluate_diffusion_policy(
        task=task,
        policy_path=policy_path,
        num_envs=num_envs,
        num_steps=num_steps,
        device=device,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--train-log", type=Path, default=None)
    parser.add_argument("--every-epochs", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stop-when-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit after training log shows final epoch completed",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    train_log = (args.train_log or Path("logs/train_diffusion_allegro_full.log")).resolve()
    metrics_path = run_dir / "eval_metrics.jsonl"
    plot_path = run_dir / "plots" / "eval_curves.png"
    policy_path = run_dir / "policy_latest.pt"

    evaluated_epochs: set[int] = set()
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if line.strip():
                evaluated_epochs.add(int(json.loads(line)["epoch"]))

    print(f"[INFO] Watching {train_log}")
    print(f"[INFO] Eval every {args.every_epochs} epochs -> {metrics_path}")

    while True:
        info = latest_epoch_from_log(train_log)
        if info is None:
            time.sleep(args.poll_seconds)
            continue
        epoch, train_loss, steps = info
        targets = [e for e in range(args.every_epochs, epoch + 1, args.every_epochs)]
        # Also evaluate the very first available epoch once.
        if epoch >= 1 and 1 not in evaluated_epochs and args.every_epochs > 1:
            # Prefer first multiple; if none yet, eval current once as warm-start.
            if not targets and epoch not in evaluated_epochs:
                targets = [epoch]

        for target in targets:
            if target in evaluated_epochs:
                continue
            if not policy_path.exists():
                print(f"[WARN] Missing {policy_path}, skip eval @ epoch {target}")
                continue
            # Only eval when training has reached this epoch.
            if epoch < target:
                continue
            print(
                f"[INFO] {datetime.now(tz=timezone.utc).isoformat()} "
                f"Evaluating epoch>={target} (log epoch={epoch}, loss={train_loss:.6f})"
            )
            try:
                metrics = run_one_eval(
                    task=args.task,
                    policy_path=policy_path,
                    device=args.device,
                    num_envs=args.num_envs,
                    num_steps=args.num_steps,
                    seed=args.seed + target,
                )
            except Exception as exc:
                print(f"[ERROR] Eval failed at epoch {target}: {exc}")
                # Don't mark evaluated so we retry next poll.
                continue

            row = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "epoch": target,
                "log_epoch": epoch,
                "train_loss": train_loss,
                "train_steps": steps,
                "policy": str(policy_path),
                "metrics": metrics,
            }
            append_jsonl(metrics_path, row)
            evaluated_epochs.add(target)
            print(
                f"[INFO] epoch={target} success_rate={metrics.get('success_rate', float('nan')):.3f} "
                f"final_dist={metrics.get('avg_final_dist_to_first_goal_m', float('nan')):.4f}"
            )
            try:
                plot_eval_curve(metrics_path, plot_path)
            except Exception as exc:
                print(f"[WARN] plot failed: {exc}")

        # Stop if training finished.
        if args.stop_when_done and info is not None:
            # Detect final epoch line like [epoch 0150/150]
            text = train_log.read_text() if train_log.exists() else ""
            if "Training done" in text:
                # Final eval on last epoch if not covered.
                if epoch not in evaluated_epochs and policy_path.exists():
                    print(f"[INFO] Final eval at epoch {epoch}")
                    metrics = run_one_eval(
                        task=args.task,
                        policy_path=policy_path,
                        device=args.device,
                        num_envs=args.num_envs,
                        num_steps=args.num_steps,
                        seed=args.seed + epoch,
                    )
                    append_jsonl(
                        metrics_path,
                        {
                            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                            "epoch": epoch,
                            "log_epoch": epoch,
                            "train_loss": train_loss,
                            "train_steps": steps,
                            "policy": str(policy_path),
                            "metrics": metrics,
                        },
                    )
                    plot_eval_curve(metrics_path, plot_path)
                print("[INFO] Training done; watcher exiting.")
                break

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
