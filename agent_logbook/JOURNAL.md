# Agent journal

Newest entries first. Link run/collection IDs from `RUNS.md` / `COLLECTIONS.md`.

---

## 2026-09-02 — First real Slurm-trained BC policies: dp-leap-rot-400k, dp-allegro-rot-400k

Jobs `80853` (LEAP) / `80854` (Allegro) -- the resubmit after fixing the
`SLURM_SUBMIT_DIR` bug (see the entry below) -- both ran to completion
overnight, 7-8h each, on separate GPU allocations (ran concurrently once
both got scheduled). Real, working BC rotation policies: LEAP reaches 2.31
goals before dropping on average, Allegro 1.62, both well above zero and in
the same ballpark as the historical ~27%-of-expert finding noted in
`ANALYSIS.md` for rotation BC. See `RUNS.md` for full numbers.

Backfill's displayed `StartTime` estimate (~1.8 days) was badly pessimistic
both times -- the first attempt (80250/80251, which failed on the
`SLURM_SUBMIT_DIR` bug) actually started ~3h after submission despite the
same estimate, and this successful pair started within a few hours too.
Don't take the scheduler's `StartTime` field as the real ETA on this
cluster; check back periodically instead of trusting it.

This also closes the loop on the ambient-sampling-bug fix from the
previous entry: these two runs used the *ordinary* (non-ambient) training
path, so they don't exercise `sample_ambient_batch` -- that fix is still
unverified by any long real run, only the short smoke test recorded there.

---

## 2026-09-01 (later) — Ambient diffusion sampling-order bug, found by the user

User: "For each training sample, you need to first sample a diffusion time
step, then sample a valid state action tuple so that you don't undersample
the early (ie low noise) tuples." The ambient mechanism reconstructed earlier
today (see below, and `CHANGES.md` items 24-26) sampled the tuple first (via
a standard shuffling `DataLoader`) and the timestep second, conditioned on
that tuple's `t_min`. Diagnosed why this is wrong: it makes the probability
of ever training below a source's `t_min` proportional to the target's share
of the *dataset*, not the schedule -- a ~40x suppression at the ambient
sweep's own N=10k/400k-source scale. Fixed by inverting the order:
`DiffusionDataset.sample_ambient_batch` now draws the timestep first, then
picks uniformly among currently-valid tuples via a sorted-index searchsorted.
`compute_loss` lost its `t_min`/reweighting path entirely -- once sampling is
correct, no downstream weighting is needed. See `CHANGES.md` for the full
writeup and the verification numbers (built a real 10k-target/400k-source
mixed dataset from data already collected today; old order put 2.5-3.4% of
the intended mass below t=50 where it should be ~50%, new order lands within
2% of ideal everywhere; a 3-epoch real training run with the fix completed
cleanly). Committed.

---

## 2026-09-01 — Reconstructed lost code from the logbook; GPU verification training

`CHANGES.md`/`ANALYSIS.md`/`COLLECTIONS.md` documented ~40 source edits and ~20
new scripts/Slurm jobs made on another box, but only the prose ever landed in
this repo (`git log` here stops right after the base diffusion pipeline,
commit `58946e8`; the commits `CHANGES.md` cites, e.g. `b3b458e`/`ad994ec`,
don't exist in this repo's history). Reconstructed the actual code from those
records: all of `src/mjlab_hand/diffusion/`'s documented fixes (DDIM sampler,
ambient-diffusion gating, checkpoint I/O, one-hot conditioning, multi-target
eval, rotation-success-filter fix), 19 new `scripts/`, 7 new Slurm job
scripts, `.gitignore`, `CLAUDE.md`. Verified what's verifiable on this box's
actual data (real regression test via `check_sampler.py` against the real
smoke checkpoint reproduced the documented before/after MSE numbers almost
exactly; real end-to-end tests of `build_mixed_dataset.py`,
`source_step_bounds`/`ambient_tmin`, `select_experts.py`, `plot_seed_curves.py`,
and the `train.py` checkpoint-cadence rewrite; unit tests of the
`plot_mixed_matrix.py`/`plot_ambient_sweep.py` correctness fixes against the
exact documented bug scenarios). Nothing was committed.

Then launched real GPU verification: **rl-leap-rot-gpu-verify** (see
`RUNS.md`), RL training on the local A40 (free, 0% util at start),
confirming the reconstructed `mjlab_hand` task registration / `train` CLI
path still works end-to-end (not something `CHANGES.md` touched, but a real
dependency of it). Healthy: reward and `episode_success` climbing from a
fresh policy within the first ~30 iterations, ~6.1s/iter steady state after
~1min of Warp/JIT warmup -- confirmed over a 6-minute sampling window (iter
17->64, reward 0.07->0.23).

**Aborted at iter ~64**, user pointed out `logs/rsl_rl/leap_inhand_rotation/2026-08-23_02-26-38_slurm`
and the Allegro equivalent already have `done` experts (`model_9999.pt`,
08-23/08-24) -- retraining was redundant. Killed both the wrapper script and
`train` process, freed the GPU, deleted the partial run dir / wandb run /
console log. The verification goal (confirm the pipeline trains) was already
met by the 64 iterations observed; no need for a trained checkpoint from
this run specifically. Lesson: check `RUNS.md` for an existing `done` expert
*before* launching a "verify it trains" run against a task that already has
one -- a shorter/synthetic smoke check would have answered the same question
without spending GPU time on a real task.

---

## 2026-08-24 — Agent logbook added

- Created `agent_logbook/` (`README.md`, `JOURNAL.md`, `RUNS.md`, `COLLECTIONS.md`) and always-on Cursor rule `.cursor/rules/agent-logbook.mdc`.
- Seeded registries from prior session artifacts (transcript [c5d65292-2612-48b0-8936-c7852d3f7ba7](c5d65292-2612-48b0-8936-c7852d3f7ba7)).

---

## 2026-08-22 → 2026-08-23 — Setup, RL training, demos, diffusion pipeline

Session: [c5d65292-2612-48b0-8936-c7852d3f7ba7](c5d65292-2612-48b0-8936-c7852d3f7ba7)

### Install & tooling

- Installed with `uv sync --group dev --default-index https://pypi.org/simple --system-certs` (Aliyun mirror TLS issues).
- Added helper scripts: `scripts/train_all.sh`, `scripts/plot_training_curves.py`, `scripts/record_trajectories.sh`, `scripts/render_checkpoint_video.py`, later `scripts/watch_eval_diffusion.py`.

### RL training

- Started **rl-allegro-grasp-abort**, user stopped; restarted as **rl-allegro-grasp** on local A40 (`logs/grasp_allegro_train.log`).
- Built Slurm array `slurm_jobs/array_20260823_022315` for 9 other hand/task combos (skip Allegro grasp).
- **rl-shadow-grasp** failed (Warp CCD kernel cache meta missing). Several grasp jobs look incomplete vs 10k iters; all in-hand rotation jobs except Sharpa reached `model_9999.pt`.
- Generated TB plots under `outputs/plots/` and rollout videos under `outputs/videos/`.

### Diffusion pipeline + data

- User asked for diffusion policies from expert checkpoints across embodiments; started with Allegro grasp.
- Smoke collect **demo-allegro-smoke** (100 eps) → smoke train **dp-allegro-smoke** (20 epochs, best loss ≈0.015, env eval 0% success).
- Full collect **demo-allegro-full** (2000 eps, ~1M steps, 1979 success) → **dp-allegro-full** (150 epochs). Watch-eval every 10 epochs still reporting 0% success through epoch 130; training loss ~0.0018.
- GitHub fork/push attempt was abandoned by user (“just forget it”); local commands were provided instead.

### Open issues / follow-ups

- Fix/retry **Grasp-Shadow** Slurm failure (Warp cache).
- Confirm whether incomplete grasp/Sharpa-rot Slurm jobs are still running or need resubmit.
- Allegro diffusion full train: finish remaining epochs; diagnose 0% rollout success despite low train loss (horizon, obs, action scaling, success filter, eval protocol).
- Collect demos + train diffusion for other hands/tasks once RL experts are solid.
