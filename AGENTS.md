# AGENTS.md

Instructions for any coding agent (Claude Code, Cursor, Codex, ...) and for
people working in this repo, covering what `README.md` does not. This is the
single source: `CLAUDE.md` imports it and `.cursor/rules/agent-logbook.mdc`
points to it, so edit here, not there.

## Documentation map

| File | What it is | Who writes it |
|---|---|---|
| `AGENTS.md` (this file) | Summary: what the repo is, env setup, data paths, protocol, invariants, standing findings | Update when a new invariant or standing finding is established |
| `README.md` | Human-facing: install, upstream RL usage, "Data layout" for new machines | Humans / agents on user-visible changes |
| `agent_logbook/JOURNAL.md` | Dated narrative per session, newest first: requests, decisions, failures, "pick up here" handoffs | Every session doing non-trivial work |
| `agent_logbook/RUNS.md` | Registry of training / eval / Slurm runs | Whenever a run starts, finishes or fails |
| `agent_logbook/COLLECTIONS.md` | Registry of datasets: source checkpoint, sizes, how derived sets were built, where they're mirrored | Whenever a dataset is collected, derived or moved |
| `CHANGES.md` | Numbered source edits, exact enough to reproduce or revert. Code comments cite "CHANGES.md item N" | Every source / script / sbatch change |
| `ANALYSIS.md` | Studies that are neither runs nor code changes (probes, post-mortems); raw numbers in `outputs/analysis/*.json`, figures in `outputs/plots/*.png` | When an analysis produces a conclusion |

Per-user agent memory (e.g. Claude Code's `~/.claude/projects/.../memory/`)
is private to one person's machine. Anything another collaborator needs must
go in the files above, not only there.

## Logbook protocol

1. **Before non-trivial work**, read `agent_logbook/JOURNAL.md` (at least the
   recent entries), `RUNS.md` and `COLLECTIONS.md`, so you reuse prior runs,
   datasets and lessons instead of redoing them.
2. **After meaningful work, in the same turn**, update:
   - training / eval / Slurm jobs → `RUNS.md`
   - dataset collection or derivation → `COLLECTIONS.md`
   - any source, script or sbatch change → a new numbered item in
     `CHANGES.md` (next number; never renumber, code cites these)
   - an analysis conclusion → `ANALYSIS.md`, and the standing findings
     below if it changes them
   - always → a dated `JOURNAL.md` entry, prepended (newest first)
3. Record facts: date, task, command / config highlights, artifact paths,
   metrics, status (`running` / `done` / `failed` / `aborted`). Do not invent
   results; say what was verified and what was not.
4. Do not commit gitignored artifacts (`logs/`, `data/`, most of `outputs/`
   — see `.gitignore` and `CHANGES.md` item 23 for exactly what is tracked).

## What this repo actually is right now

The upstream `mjlab_hand` package is an RL benchmark (grasp / in-hand
rotation, five hand embodiments: Allegro, LEAP, Shadow, Sharpa, Wuji). The
**active work is cross-embodiment diffusion-policy behavior cloning on top
of that benchmark**, not the RL benchmark itself:

1. Train RL experts per (task, embodiment) with `mjlab_hand`'s own
   training entry point (see `README.md`).
2. Collect demonstrations from an expert checkpoint:
   `collect-demos` / `src/mjlab_hand/diffusion/collect.py`.
3. Train an action-chunk diffusion policy on the demos:
   `train-diffusion` / `src/mjlab_hand/diffusion/train.py`.
4. Evaluate / render rollouts: `eval-diffusion`,
   `src/mjlab_hand/diffusion/evaluate.py`.
5. Cross-embodiment experiments (mixed-embodiment training, padded
   all-hands pooling, scarce co-training, ambient diffusion,
   state-equivalence probes) build on steps 2-4 — see
   `agent_logbook/COLLECTIONS.md` and `ANALYSIS.md`.

## Data paths are repo-relative; bulk storage is symlinked in

The repo is used on more than one server and by more than one person. Code,
sbatch files and manifests must refer to data only by repo-relative paths
(`data/mjlab_hand_demos/...`, `logs/rsl_rl/...`, `outputs/...`) and run from
the repo root — never a machine-specific absolute path like `/datastor2/...`.
Each machine symlinks those repo paths to wherever its storage is (on the
original cluster: `data/mjlab_hand_demos -> /datastor2/mrudolph/mjlab_hand_demos`,
`data/hf_staging -> /datastor2/mrudolph/hf_staging/mjlab-hand-demos`). The
demos + source RL experts are mirrored to the private HF dataset repo
`maxrudolph/mjlab-hand-demos` via `scripts/hf_sync.py`; see README "Data
layout".

## Required environment variables (every GPU job)

```bash
# rlcompute (H200) nodes ship a stale /usr/local/cuda/compat libcuda that
# ldconfig prefers over the real driver -> CUDA error 803 on every call.
export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Concurrent array tasks sharing one Warp kernel cache race and fail with a
# CCD-kernel FileNotFoundError. Give each task its own.
export WARP_CACHE_PATH="$JOBDIR/warp_cache/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

export MUJOCO_GL=egl   # headless rendering

# wandb's service subprocess makes a tempfile.TemporaryDirectory() under
# $TMPDIR (default /tmp) on every wandb.init(). Node-local /tmp is small and
# shared across whatever else lands on that node; with >~10 concurrent
# wandb-logging array tasks on one node it fills up and every wandb.init()
# after that dies with ENOSPC (seen 2026-09-12: 36/40 tasks failed in under
# a minute on slurm-node-005 this way). Any job with wandb logging AND real
# array concurrency needs this too, not just WARP_CACHE_PATH.
# Keep the path short: torch DataLoader workers put an AF_UNIX socket under
# it, and paths over ~107 chars fail with "AF_UNIX path too long" (job 94864).
export TMPDIR="$JOBDIR/tmp/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
```

`srun` fails with "More processors requested than permitted" when invoked
from inside an existing interactive allocation — use `sbatch`.

### TACC Vista (GH200, aarch64)

The block above is for the original shared-node cluster. On Vista:

- Use `slurm_jobs/vista_train_manifest.sbatch` (partition `gh`, 48h max,
  20 running / 40 submitted jobs per user, whole-node allocation — no
  `--gres`/`--mem`). It packs several runs per node; see CHANGES.md item 51.
  Pass the project with `sbatch -A <project>`.
- The cap that matters is jobs, not nodes: 96 running nodes/user (64/job).
  Submit multi-node jobs (`-N 8`..`16`, CHANGES.md item 54) to go past 20
  nodes. Per-node throughput is flat from 4 to 16 packed runs (GPU-bound), so
  runs/node only sets wall time. See ANALYSIS.md "Vista job shape".
- `$WORK` is a 1 TB quota shared with every TACC system, and it was ~97% full
  on 2026-09-28. A checkpoint is 265 MB and each run keeps >=3, so check
  `lfs quota -u $USER /work` before launching and keep benchmark/scratch
  output on `$SCRATCH`. Going over quota makes `torch.save` fail
  mid-training ("unexpected pos").
- `sbatch` is disabled on compute nodes (including idev sessions), and ssh to
  the login nodes needs interactive 2FA, so an agent running inside idev can
  smoke-test on the GPU but cannot submit — hand the `sbatch` line to the user.
- `export CC=gcc` for anything using `torch.compile` (TACC sets `CC=nvc`,
  which Triton can't use). The Vista sbatch compiles by default
  (`COMPILE_MODE=reduce-overhead`, 2.3x per run; CHANGES.md item 53).
- No `LD_LIBRARY_PATH` workaround needed; torch comes from the cu128 index on
  aarch64 (CHANGES.md item 50).
- `$HOME` is 23 GB; `$WORK` (1 TB, shared with other projects) holds the
  bulk data, venv and outputs via symlinks (`data/mjlab_hand_demos`, `logs`,
  `outputs`, `.venv`). `$SCRATCH` is purged — don't keep results there.
- `vista_train_manifest.sbatch` stages the demos it needs to `$SCRATCH` (1-node) or node
  `/tmp` (multi-node) and writes run outputs to `$SCRATCH/cross_embodied_diffusion/outputs`
  (CHANGES.md item 55). Copy runs worth keeping to Stockyard (the repo `outputs` target) with
  `scripts/promote_outputs.sh <run or sweep dir, relative to the scratch outputs root>`
  before the purge gets them.

## Invariants worth knowing before touching this code

- **The diffusion sampler is DDIM (eta=0), not single-step ancestral
  updates.** `inference_timesteps` is a strided subsequence of the training
  schedule; a single-step DDPM update is only valid for a `t -> t-1`
  transition. See `CHANGES.md` item 1 and `scripts/check_sampler.py` for a
  standalone regression check.
- **`collect_demos` filters rotation success via
  `RotationCommand.metrics["episode_success"]`, not the `pose` command
  term** — grasp tasks have a `pose` term, rotation tasks have `rotation`
  instead, and treating their absence as "always successful" silently
  mislabels every rotation episode. See item 9.
- **Actions are stored pre-clip.** `collect_demos` records the raw policy
  output; `RslRlVecEnvWrapper` clips to `clip_actions=1.0` inside
  `env.step`. Not a bug — a BC policy is clipped identically at rollout —
  but `LinearNormalizer` fits min/max, so account for the long tail
  (measured range ~±15) before assuming the normalized range is well used.
- **Only Allegro and LEAP share observation/action dimensionality *and*
  term layout.** Every other embodiment pair needs padding, per-embodiment
  encoders, or a shared schema before it can be mixed. `build_mixed_dataset.py`
  refuses mismatched spaces rather than zero-padding; the N-hand padded
  scheme (`build_padded_dataset.py`, item 42) is the one that pads.
- **Padded datasets are normalized once, upstream, per source.** Training
  on them uses identity normalizers and a per-row `action_mask`; eval needs
  the run's `source_stats.json` and an `embodiment=`. Ambient gating +
  padded is refused on purpose (item 42).
  A padded zarr is optional: `train-diffusion --dataset A.zarr B.zarr ...`
  pools the per-hand zarrs in memory with the same code
  (`mjlab_hand.diffusion.pooling`, item 56). Stats refit on a different CPU
  can differ from a stored pool's by float32 rounding (~1e-7).
- **Checkpoint names changed on 2026-09-20 (item 45).** New runs write
  `policy_latest.pt`, `policy_best_val.pt` (only with `--val-fraction > 0`)
  and `policy_best_eval.pt`; older runs have `policy_best.pt`, selected by
  training loss, which is not a quality signal.
- **Eval rows are tagged with `eval_task`, always, including solo runs.**
  Once multi-target eval (`eval_specs`) shipped, a single-target
  `--eval-task` run is internally promoted to a one-element spec list and
  still writes an `eval_task` key. A plotter that identifies "solo" rows by
  the *absence* of that key will silently blank every run trained after
  that point — see item 36.
- **`src/mjlab_hand/env_cfg.py` is dead upstream leftover.** It imports
  `mjlab_hand.anymal_c`, which does not exist in this repo, so the module
  cannot be imported. Left in place; do not extend it.

## Standing findings (see `ANALYSIS.md` for the full analysis)

- The observation identifies the embodiment perfectly (linear probe AUC
  1.000) at every diffusion timestep, including t=99 — the observation is
  never noised, so masking/invariance schemes that rely on high-noise
  unidentifiability do not apply here.
- Ambient (per-source timestep-gated) diffusion is falsified for grasp
  (decisively negative, replicated across seeds) and null for rotation.
  The one robust result is the control: an embodiment trained only on the
  coarse end of the schedule is completely non-functional (0.000), not
  merely degraded.
- Adversarial invariance training does not work on the Allegro/LEAP pair:
  a fresh probe trained after the encoder freezes stays at ≥0.977 balanced
  accuracy regardless of reversal strength, while task information drops
  substantially. Always score invariance with a fresh probe on held-out
  data, never the adversary's own training accuracy.
- Sequential transfer (pretrain on one embodiment, fine-tune on another) is
  the one cross-embodiment-sharing idea that measurement has not yet ruled
  out.
