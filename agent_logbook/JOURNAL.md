# Agent journal

Newest entries first. Link run/collection IDs from `RUNS.md` / `COLLECTIONS.md`.

---

## 2026-09-11 — Real validation loss: held-out trajectories, denoised-action MSE

User pointed out the existing `"train_loss"` in `eval_metrics.jsonl` isn't a validation loss at
all -- no held-out split existed, and it's a noise-prediction MSE, not a measure of actual
action-reconstruction quality. Asked for both fixed: a val split from held-out *trajectories*
(not held-out individual states), and a validation loss computed over **denoised actions**, not
predicted noise.

- `DiffusionDataset` gained `split`/`val_fraction`/`val_seed`. Split is at the **episode** level,
  not the window level -- splitting individual (obs, action) windows would let near-identical
  neighboring states from the same trajectory leak across train/val, understating how much the
  val loss actually measures generalization. `_train_val_split` partitions episodes with a seeded
  `np.random.default_rng(val_seed).permutation`, **stratified per source** (`_episode_source_ids`,
  reusing `source_real_dims()`/`source_step_bounds()` with a single-group fallback for plain
  datasets) so a cross-embodiment or onehot-mixed dataset holds out trajectories from every
  embodiment, not just whichever source happens to shuffle to the front. `val_fraction=0.0`
  (default) is a pure no-op -- every existing caller/script that doesn't pass these kwargs trains
  on 100% of episodes exactly as before; confirmed via smoke test that a plain `DiffusionDataset()`
  call still gets all episodes.
- `DiffusionPolicy.action_reconstruction_loss`: runs the *actual* DDIM reverse-diffusion sampling
  (`predict_action`, same code path used at rollout time) on held-out obs windows, then MSE against
  the ground-truth action -- both in real (unnormalized) action units. Deliberately not a
  cheaper single-step x0 estimate from one random noisy timestep (which the training loss's
  machinery would have made trivial to bolt on) -- the user asked for denoised actions, and the
  thing a BC policy is actually judged on is what it outputs after full sampling, not a partial
  one-step reconstruction.
- `TrainConfig` gained `val_fraction`/`val_seed`/`val_every_epochs`/`val_max_batches` (default 20 --
  DDIM sampling is far pricier per batch than one training step, so validation is capped to a
  small slice of an epoch's compute rather than doubling it) and CLI flags in
  `cli/train_diffusion.py`. Val loss is written to a new `val_metrics.jsonl` (epoch,
  val_action_loss) and wandb (`val/action_loss`), printed alongside the train-loss line each
  epoch. Deliberately did **not** change `policy_best.pt` checkpoint selection (still lowest
  train loss) -- the ask was for a validation signal to look at, not a change to what "best"
  means; flagged to the user as an easy follow-up if wanted.
- Normalizer fitting (`LinearNormalizer.fit(dataset.obs/.action)` in the non-padded path) now
  fits from the train-split dataset only when `val_fraction > 0`, not the full store -- otherwise
  val trajectories' own statistics would leak into the normalization the model is trained under,
  which is itself a subtle form of train/val contamination independent of the loss/split fix.

**Verified via a synthetic smoke test** (not committed -- ad hoc script, small random
single-embodiment + fake 2-source mixed `TrajectoryStore`s): train/val episode sets are disjoint
and sum to the full episode count; `val_fraction=0` reproduces the old all-episodes behavior
exactly; `split="val"` with `val_fraction=0` raises rather than silently returning nothing;
the 2-source split gives both sources nonzero held-out episodes; a real 2-epoch CPU
`train_diffusion` run with `val_fraction=0.2` writes non-NaN `val_action_loss` to
`val_metrics.jsonl` every epoch. Not yet run against a real dataset/GPU job -- next real
cross-embodiment or scaling run should pass `--val-fraction` to pick this up.

---

## 2026-09-10 — New cross-embodiment BC scheme: pad-to-max + per-source static normalization + masked loss

User: one diffusion policy per task family (Grasp, InHand-Rotation) that takes obs from an
arbitrary embodiment and outputs its action -- via padding to the family's max obs/action dim,
careful static per-dataset normalization, and correct loss masking. Confirmed scope first
(per-task-family pooling, not one policy across both tasks) via a scoping question -- the
obs/action dims differ enough between task families (grasp max 189/28 vs rotation max 89/22)
that a single global pad would waste most of rotation's capacity on grasp-only padding.

This is a deliberate reversal of `build_mixed_dataset.py`'s stance ("refuses mismatched spaces
rather than zero-padding") for a good reason: that script mixes exactly 2 same-dim embodiments
with onehot conditioning, where padding really would conflate unrelated physical quantities.
Here the pool is all 5 embodiments of one task, which necessarily have different dims, and
zero-padding to the family max plus a real/pad mask is the only way to share one network.

**New pieces** (all additive, existing single-embodiment / 2-source-mixed training paths
unaffected -- verified via a full small-scale smoke test before running on real data):

- `GaussianNormalizer` (`normalizer.py`): mean-0/var-1, `.fit()` per source dataset (never on
  the pooled mixture), `.identity()` for a no-op pass-through.
- `scripts/build_padded_dataset.py`: takes N source `.zarr` dirs (one task family, all 5
  hands), fits a `GaussianNormalizer` per source from that source's own raw data, normalizes,
  zero-pads obs/action to the family's max width (real dims front-packed, pad at the tail),
  concatenates. Writes `extra.padded=True` + per-source `{embodiment, obs_dim, action_dim,
  obs_mean/std, action_mean/std}` -- this provenance is what makes the loss mask and eval-time
  normalization reproducible from stored data alone.
- `TrajectoryStore.source_real_dims()`: reads that per-source real-dim provenance (parallel to
  the existing `source_step_bounds()` used by the onehot-mixed/ambient scheme).
- `DiffusionDataset`: when a store has `source_real_dims()`, precomputes a per-episode
  `action_mask` (episodes never span a source boundary, so it's constant within an episode) and
  includes it in every `__getitem__` batch.
- `DiffusionPolicy.compute_loss(..., action_mask=...)`: masked MSE -- `(sq_err * mask).sum() /
  mask.sum()` -- excludes padded action dims from the loss rather than training the network to
  predict zero there (which would bias the shared denoiser for no reason). `None` (default)
  reproduces the exact old unmasked behavior.
- `DiffusionPolicyConfig.normalizer_type`: `"linear"` (default, old behavior) or `"gaussian"`.
  A padded dataset uses `"gaussian"` with **identity** normalizers inside the policy --
  normalization already happened once, statically, per-source, in `build_padded_dataset.py`;
  normalizing again at the pooled level would mix embodiments' scales and defeat the purpose.
  `train_diffusion.py` detects `extra.padded` and wires this automatically, plus writes
  `source_stats.json` next to the checkpoint (self-contained -- eval doesn't need the original
  training zarr on disk).
- `CrossEmbodimentActionChunkPolicy` path in `evaluate.py` (folded into the existing
  `DiffusionActionChunkPolicy` via an `embodiment_stats` arg, mutually exclusive with `onehot`):
  at eval/rollout time, normalizes a live env's raw (real-dim) obs with that one embodiment's
  static stats, zero-pads to the policy's width, predicts, un-normalizes and slices the
  prediction back to that embodiment's real action_dim before it's sent to `env.step`. Wired
  into `train_diffusion.py`'s existing in-loop `eval_specs` (`{"task": ..., "embodiment": ...}`
  entries) so periodic training-time eval scores each embodiment separately.
- Ambient-diffusion (`--ambient-tmin`) + padded datasets together is explicitly refused
  (`sample_ambient_batch` bypasses `__getitem__` and doesn't emit `action_mask` -- combining
  them would silently leak padded dims into the loss). Not needed for this work.

**Verified correctness on a small smoke dataset** (Grasp-Allegro + Grasp-Sharpa, 8k steps
each) before running for real: per-source real dims are genuinely mean~0/std~1 and the pad
tail is exactly 0 in both obs and action; `action_mask` sums to the right per-embodiment
real `action_dim` per episode; 2 epochs of real training ran end-to-end with the masked loss
decreasing; reloaded the saved checkpoint and confirmed `normalizer_type="gaussian"` with true
identity stats, and that the eval wrapper's normalize -> pad -> predict -> un-normalize ->
slice pipeline returns an action of exactly the source embodiment's real `action_dim`.

**Real datasets built** from the existing 1M-transition collections (`COLLECTIONS.md`):
`/datastor2/mrudolph/mjlab_hand_demos/padded/Grasp-AllHands.zarr` (5,001,608 steps, obs=189,
act=28) and `.../InHand-Rotation-AllHands.zarr` (4,897,978 steps, obs=89, act=22).

**Training launched**: `slurm_jobs/train_cross_embodiment.sbatch`, job `85823`, array `0-1`,
pinned to `slurm-node-011` (per the driver lessons above -- this is real multi-hour training,
not a one-off collection). 40 epochs each (matches this project's ~780k-gradient-step
convention at this data scale), periodic per-embodiment eval every 10 epochs via 5
`{"task", "embodiment"}` eval_specs, wandb project `mjlab`. Outputs:
`outputs/diffusion/{Grasp,InHand-Rotation}-AllHands_padded/`.

## 2026-09-10 (later) — 50k-scale pooled datasets + training

User: pick N trajectories per embodiment for a ~50k-sample pool, then train those too.
Deliberately did **not** subsample the already-pooled 5M padded dataset directly -- its
episodes are 5 concatenated ~1M-step blocks in source order, so `subsample_dataset.py`'s
front-to-back greedy selector would only ever have picked from the first source (Allegro),
and it also drops the `extra.padded`/`sources` metadata `source_real_dims()`/`action_mask`
depend on. Instead: subsampled each of the 10 individual 1M per-embodiment datasets to ~10k
steps first (`subsample_dataset.py`, unchanged), then re-ran `build_padded_dataset.py` over
those 5-per-family 10k subsets -- landed at 50,060 (Grasp) and 50,326 (Rotation) steps, all 5
hands genuinely represented, fresh per-source normalizer fit on the smaller data (not reused
from the 1M fit, per the "static per that dataset" spec). Verified `action_mask` sums match
each source's real action_dim on both. See `COLLECTIONS.md` for the dataset table.

Training launched: `slurm_jobs/train_cross_embodiment_50k.sbatch`, job `86050`, array `0-1`,
also pinned to node-011. 4000 epochs (this project's existing `EPOCHS_50K` convention, same
~780k total gradient steps as every other scale), eval every 1000 epochs, same 5-embodiment
eval_specs and wandb project as the 5M runs (tagged `50k` to distinguish). Outputs:
`outputs/diffusion/{Grasp,InHand-Rotation}-AllHands_50k_padded/`.

---

## 2026-09-09 (later still) — Pinned remaining grasp RL training to node-011 only

Surveyed more nodes for the stale-driver issue (see entries above). Confirmed bad, in addition
to node-004: **node-002, node-003, node-005, node-007, node-008 (also currently drained,
`Reason=Kill task failed` since 2026-09-05), node-009** -- via direct `nvidia-smi` on running
jobs and via the "CUDA version < 12.4" warning appearing in the **original 2026-08-23 array**
(`72897`) logs too, on 5 different nodes. So this has been true since the very first Slurm
array in this project; it just didn't show up as a problem because rotation tasks (lower
per-step contact cost) only pay a mild penalty (5.7-11s/iter) while grasp tasks pay heavily
(15-34s/iter). Only `slurm-node-011` (driver 580.173.02) confirmed clean.

Node survey for 001/006/010 never completed -- the probe jobs sat PENDING (priority) for
hours and got cancelled once the decision below made them moot. **Lesson: a job's
`#SBATCH --output` path must be on storage the compute node can actually reach.** Wrote the
probes' output to this session's `/tmp/claude-*/.../scratchpad/` dir, which isn't visible from
compute nodes -- the jobs ran and exited 0, but the result files were never created. Anything
meant to be read back after an sbatch job runs needs to live under the repo
(`/scratch/cluster/.../mjlab_hand/...`) or another genuinely shared path, never a
session-local scratch dir.

User decision: **only run the remaining RL expert training on node-011**, accepting that it
may queue rather than risk more bad-driver time. Cancelled `85163` tasks 0/1/2 (Allegro/LEAP/
Shadow grasp, all on bad nodes) and resubmitted as `slurm_jobs/train_rl_experts_node011.sbatch`
(job `85382`, array `0-2`, `#SBATCH --nodelist=slurm-node-011`), resuming from their current
checkpoints (5800/5200/4900) with the iteration-count bug fixed this time -- passed the
*remaining* budget (4200/4800/5100) as `--agent.max-iterations`, not 10000, since resume adds
that value on top of the loaded checkpoint's iteration rather than treating it as an absolute
target. `85163` task 3 (Grasp-Sharpa) had, by luck, already landed on node-011 directly and is
running fast (~4.5s/iter) -- left alone rather than restarted, but it still carries the
uncorrected overshoot bug (target 12500, not 10000); not fixed since killing a
healthy fast-running task felt worse than the overshoot itself. Node-011 is often near-full
(96/96 CPUs when `85382` was submitted, including this user's own long-running `flock` jobs
`84302-84304`, 48 CPUs) so `85382` may sit pending for a while -- accepted tradeoff.

---

## 2026-09-09 (later) — Root cause of the 7-14x slowdown: stale driver on slurm-node-004

User: "we need the actual algorithms to go much faster. there's something wrong here." Dug
into why the 4 remaining tasks from array `84348` were so much slower than Grasp-Wuji/
InHand-Rotation-Sharpa (tasks 4/5, which finished at 2.5s/iter and 5.5s/iter respectively).

All 4 slow tasks landed on **`slurm-node-004`**; the 2 fast ones ran on `slurm-node-011`.
Every slow task's `.err` log carries mujoco_warp's `check_toolkit_driver()` warning ("CUDA
version < 12.4 detected ... conditional graph nodes are not available"); neither fast task's
log does. Confirmed directly: `srun --jobid=<task's real JobId> --overlap --cpu-bind=none
nvidia-smi --query-gpu=driver_version --format=csv` inside the *running* job's own allocation
(the array-task alias like `84348_0` doesn't work as a `--jobid` for `srun --overlap`; need
the real numeric JobId from `scontrol show job 84348_0 | grep JobId=`, e.g. `84368`) --
node-004 reports driver **535.216.03** (CUDA 12.2 max). Ran the same `warp.init()` /
`wp.is_conditional_graph_supported()` check interactively on node-011: driver **580.173.02**,
CUDA 13.0, conditional graph capture **True**. Also ruled out the `LD_LIBRARY_PATH=/usr/lib64`
env workaround (from `CLAUDE.md`, meant for H200 rlcompute nodes) as the cause -- reproduced
it explicitly set on node-011 and conditional-graph support stayed `True`; neither node has a
`libcuda` under `/usr/lib64` at all, so that prepend is a no-op on both. This is a genuine
per-node driver mismatch on the cluster, not anything in this repo's code or env setup.

**Fix applied:** cancelled the 4 tasks stuck on node-004 (`scancel 84348_0..3`, all were in
`CG` afterward) and resubmitted as `slurm_jobs/train_rl_experts_resume.sbatch`, job `85163`,
array `0-3`, `#SBATCH --exclude=slurm-node-004`, `--agent.resume True` (defaults for
`--agent.load-run`/`--agent.load-checkpoint` pick the alphabetically-latest run dir / model
file, unambiguous since each hand has exactly one `*_slurm2` dir) so the 2400-4900 iterations
already trained are kept rather than restarted. Also dropped the accidental
`--agent.logger tensorboard` override from the original script (see the wandb entry above) --
these resumed runs use `train`'s actual default (`wandb`) and log live. Killed the background
tensorboard->wandb sync watcher (pid 965893) since it would otherwise have synced the
now-cancelled, truncated tensorboard logs for these 4 as if they were the final result.

**Open follow-up:** only checked node-004 (bad) and node-011 (good) directly -- driver
versions on node-001/002/003/005/006/007/009/010 are unknown. If a resumed task lands on
another stale-driver node, the same "CUDA version < 12.4" line will show up in its `.err`
within the first few log lines; check with the same `srun --jobid=<real JobId> --overlap
--cpu-bind=none nvidia-smi --query-gpu=driver_version --format=csv` trick and add that node to
`--exclude` too.

---

## 2026-09-09 — Don't trust `train`'s printed "Time elapsed"/"ETA"; use checkpoint mtimes

Gave the user a "close to done" ETA for the `84348` RL array (see previous entry) sourced
from the training script's own `.out` log lines (`Time elapsed: HH:MM:SS`, `ETA: HH:MM:SS`).
User pushed back -- these jobs had been running ~25h per `sacct`, but the script's own
"Time elapsed" showed under an hour. That field is **not cumulative since job start**; it
appears to reset periodically while `sacct`'s elapsed and the `Iteration time` (instantaneous
per-step) field stay trustworthy. Its "ETA" is derived from the same broken counter, so it was
off by 10-25x (printed ~2-10h remaining vs. real ~26-76h).

**Correct way to estimate remaining time for an in-progress RL run:** take two checkpoint
files' mtimes (`stat -c '%y'`) and their iteration numbers from the filenames, compute real
iter/hour from that, then `(max_iterations - current_iter) / rate`. Cross-check against the
log's own `Iteration time` line -- it should roughly match `elapsed / iters_in_that_window`
from the mtime method; if it does, trust the mtime-derived ETA over the script's printed one.

Also surfaced: **Grasp-Sharpa (task 3) is running ~2x slower per iteration** than the other
three still-running grasp tasks (36.2s/iter vs 18-21s/iter), with its log full of
`contact match overflow: please increase Option.contact_sensor_maxmatch` warnings -- plausibly
the cause of the slowdown, not yet confirmed. At current rate it's ~76h (~3.2 days) from
iteration 10000, vs. ~26-34h for Allegro/LEAP/Shadow. Not yet restarted with a higher
`contact_sensor_maxmatch` -- open question for the user.

---

## 2026-09-07 — Launched RL expert completion pass, all 10 combos

User asked for expert policies for all 5 embodiments x both tasks. Audited actual on-disk
state first (the `RUNS.md`/`COLLECTIONS.md` 2026-08-24 snapshot doesn't match this box -- see
the 2026-09-01 reconstruction entry below; verified real checkpoints and real `data/demos/`
contents directly). Found 4 of 10 combos already `done` at `model_9999.pt` (rotation:
Allegro/LEAP/Shadow/Wuji); the other 6 (all 5 grasp hands + rotation-Sharpa) had only partial
checkpoints (iters 3800-7500) or, for Grasp-Shadow, none at all (prior Warp-cache failure).

Asked the user two scoping questions rather than assuming: (1) retrain the 4 already-done
combos too, or leave them -- chose leave them; (2) resume the 6 partial runs from checkpoint
(`--agent.resume`/`--agent.load-run` are supported by `train`) or restart fresh -- chose
restart fresh for all 6.

Wrote `slurm_jobs/train_rl_experts.sbatch` (array `0-5`, one task per missing combo,
`run-name slurm2` so new runs land in fresh dirs rather than overwriting the partial ones) and
submitted it: Slurm job `84348`. See `RUNS.md` for the per-task table and script details.
Currently pending on priority (cluster already running the ambient sweep, job `82903`, plus
unrelated `flock` jobs `84302-84304`).

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
