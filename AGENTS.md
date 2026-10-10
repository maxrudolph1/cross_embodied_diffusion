# AGENTS.md

Instructions for any coding agent (Claude Code, Cursor, Codex, ...) and for
people working in this repo, covering what `README.md` does not. This is the
single source: `CLAUDE.md` imports it and `.cursor/rules/agent-logbook.mdc`
points to it, so edit here, not there.

## Documentation map

Everything lives in [`docs/`](docs/README.md). **When the user says "update the log books", follow the
"How to update" paragraph at the top of each file below. Do it without asking.**

| File | What it is | When to write it |
|---|---|---|
| [`docs/agent_log_book.md`](docs/agent_log_book.md) | **Start here.** Current state, environment/install, and numbered, dated entries (newest first) for every session's implementation work, with a TOC | Every session that changes code, data, jobs or results |
| [`docs/experiment_log_book.md`](docs/experiment_log_book.md) | Numbered, dated experiments: question, data, policy inputs/outputs, protocol, results with embedded plots, takeaways, and links to runs and agent entries | Whenever an experiment produces results or a comparison |
| [`docs/RUNS.md`](docs/RUNS.md) | Every Slurm job (and result-producing idev run): exact submit command, how to regenerate its manifest, outputs, status | Whenever a job starts, finishes, fails or is cancelled |
| [`docs/COLLECTIONS.md`](docs/COLLECTIONS.md) | Datasets: what exists, how each was built, sizes, mirrors | Whenever a dataset is built, moved or deleted |
| [`docs/CHANGES.md`](docs/CHANGES.md) | Numbered source edits, exact enough to reproduce or revert. Code comments cite "CHANGES.md item N", which means this file | Every source / script / sbatch change (next number; never renumber) |
| [`docs/plots/`](docs/plots/) | Figures (+ summary CSVs) embedded in the experiment log book | With each experiment entry |
| [`docs/MIGRATION.md`](docs/MIGRATION.md) | Reference: the "Bundle" design this code re-implements (CHANGES item 63) | Read-only reference |
| [`docs/archive/`](docs/archive/) | Verbatim old `JOURNAL.md` and `ANALYSIS.md` (to 2026-10-04); the log books summarize them | Never edited |
| `README.md` | Human-facing: install, upstream RL usage, "Data layout" | On user-visible changes |

Per-user agent memory (e.g. Claude Code's `~/.claude/projects/.../memory/`) is private to one machine.
Anything another collaborator needs must go in the files above.

## Logbook protocol

1. **At session start**, read `docs/agent_log_book.md`: "Current state" plus the newest few entries. Before
   launching or analysing anything, also check `docs/RUNS.md`, `docs/COLLECTIONS.md` and the relevant
   `docs/experiment_log_book.md` entries, so you reuse prior runs, datasets and lessons.
2. **After meaningful work, in the same turn**, or whenever the user says "update the log books", update the
   agent log book (entry, TOC row, Current state), RUNS.md, COLLECTIONS.md, CHANGES.md and the experiment log
   book, as their headers describe. Copy figures into `docs/plots/`.
3. Record facts: date, command / config, artifact paths, metrics with their eval protocol, and status. Do not
   invent results. Say what was and was not verified.
4. Do not commit gitignored artifacts (`logs/`, `data/`, `outputs/`; see `.gitignore`). Figures go to
   `docs/plots/`, not `outputs/plots/`.

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
   state-equivalence probes, fine-tuning) build on steps 2-4 — see
   `docs/experiment_log_book.md`.

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

**Limits and environment**

- Partition `gh`: 1 GH200 per node, whole-node allocation (no `--gres`/`--mem`), 48 h max.
  Per user **and per partition** (count only `gh` jobs; idevs on `gg` do not count): **40 submitted jobs, 20
  running jobs, 96 running nodes, 64 nodes/job**. Every
  array element counts as one job. Pass the project with `sbatch -A ASC26008` (the user writes
  it upper-case; Slurm shows `asc26008`).
- `sbatch` (including `--test-only`) is disabled on compute nodes, idev included, and ssh to
  login nodes needs interactive 2FA. An agent inside idev can test on the node but cannot
  submit: hand the exact `sbatch` line to the user. idev sessions can land on CPU-only `gg`
  nodes (no `nvidia-smi`); check before planning GPU work there.
- No `torch.compile` since item 63 (it changes numerics; item 53's compile path is gone),
  so runs are ~2.3x slower per run than the 09-28..10-01 compiled Vista runs. No
  `LD_LIBRARY_PATH` workaround needed (item 50). `OMP_NUM_THREADS=1` makes `nproc` print 1;
  the node still has 72 cores.

**Storage**

- `$HOME` 23 GB; `$WORK`/Stockyard is a 1 TB quota shared with every TACC system (near full on
  2026-09-28; going over makes `torch.save` fail with "unexpected pos") and holds the repo,
  venv, bulk data and kept outputs via symlinks (`data/mjlab_hand_demos`, `outputs`, `.venv`).
- `vista_train_manifest.sbatch` stages the demos its runs use to `$SCRATCH` (1-node) or node
  `/tmp` (multi-node) and writes outputs to `$SCRATCH/cross_embodied_diffusion/outputs`
  (item 55). `$SCRATCH` is purged: copy runs worth keeping with
  `scripts/promote_outputs.sh <dir relative to the scratch outputs root>`.

**Running things**

- Training: `slurm_jobs/vista_train_manifest.sbatch` + a manifest (list of tasks, each a list
  of `train-diffusion` arg dicts). `PACK` = manifest tasks per node; `-N K` makes each node of
  a job take its own `PACK` tasks (node slot = array_index * K + node_id; item 54). `DRY_RUN=1`
  prints the commands. Per-node throughput, measured 2026-10-06 on the current (uncompiled) rotation recipe
  (A40): 1 run 33.4 steps/s; 2 packed 26.2 each (52.5/node); 4 packed 15.4 each (61.6/node). A
  ~784k-step run is ~6.8 h alone, ~8.7 h at PACK=2, ~14.7 h at PACK=4. (The older compiled-code
  numbers in ANALYSIS "Vista job shape" no longer apply.)
- Multi-hand data: one pre-built **term-aligned** store per mixture
  (`scripts/padded_grid.py --family F --config C --build` ->
  `data/mjlab_hand_demos/padded_ta/`), trained with the recipe in MIGRATION.md section 5.2:
  `--ambient-sampler noise-first --norm-mode frozen --norm-artifact configs/norm_<family>_minmax.json
  --x0-clamp 1.0 --val-dataset data/mjlab_hand_demos/val/<family>_val_20k.zarr --keep-last 3
  --num-workers 0`, eval spec `[{"task": T, "pad": true}]` at 1500 steps (item 63).
  Manifests: `scripts/build_ambient_rotation_manifest.py`.
- Reporting evals: `slurm_jobs/vista_eval_checkpoints.sbatch` runs
  `scripts/rescore_selected.py --which best_rollout best_val last0 --envs 100 --steps 1500
  --eval-seed 1234` -> `<run>/final_eval.jsonl`. Report `last0` (MIGRATION section 7).

### Getting Vista jobs scheduled fast (do this before every large submission)

The `gh` partition is usually ~100% allocated, so start time is set by queue position and
backfill, and it varies a lot with job shape. **Measure; do not assume** that bigger or smaller
jobs start sooner. On 2026-09-30 the user expected 8-node jobs to start slower than 16-node ones,
an agent then claimed "2-4 nodes start fastest" from a coarse bin, and the fine-grained numbers
said 2-node 2.6 h, 3-4-node 12.6 h, 5-8-node 5.8 h, 9-16-node 13.2 h median wait. Both guesses
were wrong in different ways.

1. Run `scripts/queue_wait_stats.sh [partition] [days] [time_limit]` (works on any node). It
   prints node states, submit->start waits of recently started jobs and ages of pending jobs
   per node count (fine bins: 1, 2, 3-4, 5-8, 9-16, ...), and the user's usage against the caps.
   Treat small bins (n < ~10) as weak evidence; started-job waits are biased low, pending ages
   biased high.
2. On a login node (or ask the user to), get the scheduler's own projection for each candidate
   shape with the exact time limit you intend: `sbatch --test-only -A ASC26008 -p gh -N <n>
   -t <limit> --wrap=true` (the script prints the loop). This is the strongest evidence.
3. Enumerate the shapes that fit the caps for the run count R (PACK runs per node, K nodes per
   job, J = R / (PACK*K) jobs): J + jobs already queued <= 40, min(J, 20) * K <= 96 to run at
   once, K <= 64. Remember to leave queue slots for the eval job(s) and any idev.
4. Pick by **expected completion**, not start: wait(K) + run time(PACK) (+ extra waves if
   J > 20 or J*K > 96). Packing more runs per node lengthens run time ~linearly but needs fewer
   nodes/jobs; a slightly later start with half the run time usually wins.
5. Use the tightest safe `--time` (measured run time + ~25-50%): shorter limits backfill into
   more gaps. A pending job's limit can be lowered with `scontrol update JobId=<id>
   TimeLimit=<t>` (raising it needs the user/admin). Record the choice and the evidence in
   `docs/RUNS.md`.

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
- **Multi-hand stores are term-aligned and raw (item 63, Bundle's design).** Each obs term
  (`schemas.json`, `padding.py`) has its own block as wide as the family's widest version,
  so a column means the same term for every hand (grasp 191/28, rotation 91/22); actions are
  front-packed; padding is 0. One shared normalizer in the policy (`--norm-mode`; experiments
  use the frozen per-family min/max artifact, which must contain 0). Joint order *within* a
  term still differs per hand (`outputs/analysis/spaces_reference.md`). The old tail-padded,
  per-hand-normalized `padded/*.zarr`, in-memory pooling and `--source-norm` (items 42, 56,
  58) are gone; train.py refuses an old tail-padded store.
- **`--ambient-tmin` follows the store's source order, which is HANDS order** (Allegro,
  LEAP, Shadow, Sharpa, Wuji) for term-aligned stores, not "target first". Use
  `padded_grid.target_ambient_tmin`; a wrong order silently gates the target.
- **Ambient runs must pass `--ambient-sampler noise-first`.** The code default
  `data-first` (kept for parity with Bundle) starves low-noise steps 50-100x on padded stores.
- **Checkpoints (item 63):** `policy_latest.pt`, `policy_best.pt` (train loss: not a quality
  signal), rolling `policy_last{0,1,2}_epoch{N}.pt` (last0 = newest), `policy_best_rollout.pt`,
  `policy_best_val.pt` (with `--val-dataset`); `selection.json` is written last and marks
  completion. Checkpoints from before item 63 do not load in this code (their config has
  `normalizer_type`): evaluate them from the `vista-ambient-rotation` branch.
- **Eval rows are tagged with `eval_task`, always, including solo runs.**
  Once multi-target eval (`eval_specs`) shipped, a single-target
  `--eval-task` run is internally promoted to a one-element spec list and
  still writes an `eval_task` key. A plotter that identifies "solo" rows by
  the *absence* of that key will silently blank every run trained after
  that point — see item 36.
- **`src/mjlab_hand/env_cfg.py` is dead upstream leftover.** It imports
  `mjlab_hand.anymal_c`, which does not exist in this repo, so the module
  cannot be imported. Left in place; do not extend it.

## Standing findings (details in `docs/experiment_log_book.md`)

- **Current best recipe for a scarce (50k) target** (E14-E16): co-train on the term-aligned 5-hand pool
  (target 50k + four 1M), then fine-tune 10 epochs on the target only. Grasp 0.91 / rotation 1.34 (last0)
  vs target-only 0.38 / 0.70. lr 1e-4 forgets the other hands completely; lr 1e-5 keeps grasp
  generality (0.82) but not rotation (0.25). Co-training alone: grasp +0.38..+0.44, rotation
  -0.04..-0.15 vs target-only.

- The observation identifies the embodiment perfectly (linear probe AUC
  1.000) at every diffusion timestep, including t=99 — the observation is
  never noised, so masking/invariance schemes that rely on high-noise
  unidentifiability do not apply here.
- Ambient (per-source timestep-gated) diffusion: the Aug "falsified for grasp" result used
  the data-first sampler and was retracted in Bundle (MIGRATION.md section 7). With
  noise-first Bundle reports, for a starved 50k rotation target, a peak at sigma 2-3 (0.655
  at 0 -> 1.137 at 2, 0.767 at 100; Sharpa best at 25), sign reversal at 1M, and grasp
  hurt monotonically. Reproduced here for rotation (E18/E19, 3 training seeds on one
  random-draw 50k target set): mean successes 0.42 at sigma 0, a plateau of 0.80-0.85 at sigma 9-20,
  0.63 at 100, consistent across seeds. Best sigma is hand-specific (Allegro/LEAP/Shadow 2-6, Sharpa
  >= 15, Wuji >= 9); sigma 15 is the only setting >= target-only for every hand. Grasp (E23, random-draw targets,
  1 seed; σ 0/100 3 seeds): gating hurts at every σ, 0.58 at σ 0 → 0.31 at 10 → ~0.08 from 30 on (seed 4321). Robust in both: an embodiment trained
  only on the coarse end of the schedule is completely non-functional (0.000; Vista
  2026-10-01: LEAP 1.8 at sigma 0 -> 0.00 at sigma 10).
- Adversarial invariance training does not work on the Allegro/LEAP pair:
  a fresh probe trained after the encoder freezes stays at ≥0.977 balanced
  accuracy regardless of reversal strength, while task information drops
  substantially. Always score invariance with a fresh probe on held-out
  data, never the adversary's own training accuracy.
- Sequential transfer works: co-train then fine-tune on the target (E15) is the best
  target policy found so far in both families (see the first bullet).
- Fine-tune from full co-training (σ 0), not from ambient-gated runs (E21, rotation, held-out seed 4321, 3 seeds):
  fine-tuned from σ 0 1.17 vs σ 15 0.90 vs σ 100 0.71 successes before drop, for every hand. Ambient σ 15 is the best
  policy *without* fine-tuning (0.86) but gains only +0.04 from it: its low-noise steps already saw only target
  data (the noise-first sampler gives the 50k target 1.3% of samples at σ 0, 100% at t < σ). The σ 0 fine-tune
  forgets the other hands (1.37 -> 0.26 at last0; its best_val, epochs 1-5, keeps 0.87 with 1.04 on the target).
- The target's 50k is not needed in pre-training when a fine-tune follows (E22, rotation, seed 4321): pre-training
  on the other four hands only, then fine-tuning, gives 1.16 vs 1.17 for co-training with the target. Zero-shot on
  the unseen hand is 0.00, and the fine-tuned policy then fails the other hands (0.00), unlike σ 0's best_val (0.87).
- **Never evaluate at env seed 0.** The 1M demo stores were collected at seed 0, so a seed-0 eval
  replays training start states, and the old `subsets_50k/` are the first ~100 episodes of the 1M
  stores (100% overlap). Report at seed 1234. Target-only 50k grasp policies memorize (0.92-0.95
  on training starts vs 0.32-0.39 on new ones); E17. New 50k subsets should be random draws
  (`subsample_dataset.py --random-seed`, CHANGES 70).
