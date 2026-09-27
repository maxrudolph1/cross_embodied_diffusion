# Agent logbook

Persistent, shared memory for anyone working in this repo — coding agents
(Claude Code, Cursor, ...) and people. Read it at session start; update it
when you finish meaningful work. The full protocol is in
[`../AGENTS.md`](../AGENTS.md) ("Logbook protocol").

## Files

| File | Purpose |
|------|---------|
| [`JOURNAL.md`](JOURNAL.md) | Dated narrative of sessions, decisions, failures and handoffs, newest first |
| [`RUNS.md`](RUNS.md) | Registry of training / eval / Slurm runs (RL, diffusion) |
| [`COLLECTIONS.md`](COLLECTIONS.md) | Registry of demo datasets: provenance, sizes, derived sets, HF mirror |
| [`../CHANGES.md`](../CHANGES.md) | Numbered source edits (cited from code comments as "CHANGES.md item N") |
| [`../ANALYSIS.md`](../ANALYSIS.md) | Analyses and post-mortems that are neither runs nor code changes |

`CHANGES.md` and `ANALYSIS.md` stay at the repo root because code comments
and sbatch files reference them by that path.

## When to update

- Starting or finishing a training run → `RUNS.md` (+ journal note)
- Collecting, deriving or moving datasets → `COLLECTIONS.md` (+ journal note)
- Any source, script or sbatch change → next numbered item in `CHANGES.md`
- An analysis conclusion → `ANALYSIS.md` (and `AGENTS.md` standing findings)
- Every non-trivial session → `JOURNAL.md`
- Prefer facts: paths, commands, configs, metrics, status. Skip fluff.

## Artifact locations (gitignored, repo-relative)

Bulk storage is symlinked into these paths per machine — see
[`../README.md`](../README.md) "Data layout".

- Main demo datasets: `data/mjlab_hand_demos/` (`*_expert_1M.zarr`, `subsets_10k/`,
  `subsets_50k/`, `padded/`); older ad-hoc sets in `data/demos/`, `data/mixed_noc/`
- RL checkpoints / TB logs: `logs/rsl_rl/<experiment>/<run>/`
- Console / process logs: `logs/*.log`
- Diffusion outputs: `outputs/diffusion/`, `outputs/ambient/`
- Plots / analysis JSON (tracked): `outputs/plots/*.png`, `outputs/analysis/*.json`
- Videos: `outputs/videos/`
- Slurm scripts (tracked) and per-array logs/manifests (untracked): `slurm_jobs/`

Keep the logbook itself under version control; do not commit large artifacts under `logs/`,
`data/`, or `outputs/`.
