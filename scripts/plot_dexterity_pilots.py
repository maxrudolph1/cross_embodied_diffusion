"""Plot local PPO pilot curves and held-out evaluations (not expert claims)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    rows = []
    for run in args.runs:
        metadata = json.loads((run / "training.json").read_text())
        task = metadata["task"]
        command = "baoding" if task.startswith("Baoding-") else "pen_spin"
        events = EventAccumulator(str(run)).Reload()
        for axis, tag in zip(axes, ("Train/mean_reward", f"Metrics/{command}/best_revolutions")):
            if tag not in events.Tags()["scalars"]:
                raise ValueError(f"Missing {tag} in {run}; available: {events.Tags()['scalars']}")
            values = events.Scalars(tag)
            axis.plot([x.step for x in values], [x.value for x in values], label=task)
        report = json.loads((run / "eval" / "policy_seed1234.json").read_text())
        summary = report["summary"]
        rows.append({
            "task": task, "checkpoint": report["checkpoint"], "seed": report["seed"],
            "episodes": summary["episodes"], "success_rate": summary["success_rate"],
            "mean_best_revolutions": summary["mean_final_command_metrics"]["best_revolutions"],
            "mean_duration_s": summary["mean_duration_s"],
            "failure_rate": summary["failure_rate"], "censored_rate": summary["censored_rate"],
        })
    for axis, title in zip(axes, ("Training episode return (task-specific scale)", "Training best uninterrupted revolutions")):
        axis.set(title=title, xlabel="PPO iteration")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Allegro local pilots — training diagnostics, not expert performance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    with args.output.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
