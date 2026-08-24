# Agent logbook

Persistent memory for Cursor agents working in this repo. Read it at session start; update it when you finish meaningful work.

## Files

| File | Purpose |
|------|---------|
| [`JOURNAL.md`](JOURNAL.md) | Chronological narrative of agent sessions and decisions |
| [`RUNS.md`](RUNS.md) | Registry of training runs (RL, diffusion, etc.) |
| [`COLLECTIONS.md`](COLLECTIONS.md) | Registry of data collection / demo datasets |

## When to update

- Starting or finishing a training run → `RUNS.md` (+ short journal note)
- Collecting demos / datasets → `COLLECTIONS.md` (+ short journal note)
- Non-trivial code, scripts, Slurm jobs, evals, or debugging → `JOURNAL.md`
- Prefer facts: paths, commands, configs, metrics, status. Skip fluff.

## Artifact locations (gitignored)

- RL checkpoints / TB logs: `logs/rsl_rl/<experiment>/<run>/`
- Console / process logs: `logs/*.log`
- Demo datasets: `data/demos/`
- Diffusion outputs: `outputs/diffusion/`
- Plots / videos: `outputs/plots/`, `outputs/videos/`
- Slurm arrays: `slurm_jobs/`

Keep the logbook itself under version control; do not commit large artifacts under `logs/`, `data/`, or `outputs/`.
