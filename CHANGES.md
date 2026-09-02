# Code changes

Exact source edits made by agent sessions, so another agent can reproduce or revert them.
Newest first. Repo-relative paths. Artifacts and run registries live in
[`RUNS.md`](RUNS.md) / [`COLLECTIONS.md`](COLLECTIONS.md); narrative in [`JOURNAL.md`](JOURNAL.md).

---

## 2026-09-01 — Ambient diffusion: FIX — sampling order starved low-noise training

Found by the user, not observed independently: the reconstructed ambient mechanism
(items 24-26 below) sampled a training tuple first (uniformly across the whole
mixed dataset, via the standard shuffling `DataLoader`) and only then drew its
diffusion timestep from `[t_min_row, T)`. That order is wrong whenever the
admitted-everywhere (target) data is a small fraction of the mixed dataset --
which is exactly the ambient sweep's own design (e.g. N=10k target + 400k
source, target = 2.4% of rows). The probability that *any* training step lands
below a given `t` becomes `p_target * (t/T)`, not `t/T`: a factor of `p_target`
below the intended schedule. At N=10k this is a ~40x suppression of low-noise
training -- exactly the regime where, per `ANALYSIS.md`, embodiment identity is
realized and target-specific fine detail must be learned.

**Fix: sample the timestep first, then sample uniformly among the training
tuples valid at that timestep** (target tuples are always valid; source tuples
only once `t >= t_min_source`). Since `t` is now drawn independent of
everything else, its marginal is uniform by construction -- no reweighting is
needed downstream, which let the per-example `(T-t_min)/T` weight in
`compute_loss` be deleted entirely (it fixed *loss scale* for a row already
selected under the old, wrong order; it did nothing about selection
*frequency*, which is what was actually broken).

- `src/mjlab_hand/diffusion/policy.py`: `compute_loss(obs, action, timesteps=None)`
  replaces the old `t_min` parameter. If `timesteps` is given, use it directly
  (plain `F.mse_loss`, no weighting). If `None`, sample uniformly over
  `[0, T)` -- unchanged ungated behaviour.
- `src/mjlab_hand/diffusion/dataset.py`: `DiffusionDataset` no longer returns
  `t_min` from `__getitem__`. New `sample_ambient_batch(batch_size,
  num_train_timesteps, rng)`: draws timesteps first, then for each one uses
  `np.searchsorted` on a precomputed sort-by-`t_min` index (`_build_ambient_index`)
  to find how many windows are valid, and picks uniformly among them. O(log N)
  per query; ~2.5ms for a 256-batch against a 383k-row mixed dataset --
  negligible next to the GPU forward/backward pass.
- `src/mjlab_hand/diffusion/train.py`: when `cfg.ambient_tmin` is set, the
  epoch loop bypasses the `DataLoader` entirely (a fixed-row Dataset +
  shuffling sampler can't express "pick the tuple after the timestep") and
  calls `dataset.sample_ambient_batch` directly, `len(dataset)//batch_size`
  times per epoch to keep epoch semantics comparable to the non-ambient path.

**Verified concretely** (10k LEAP target + 400k Allegro source, unconditioned,
`t_min=[0, 50]`, 200k timestep draws): old (row-first) order put only 2.5-3.4%
of the intended mass below t=50 (ratio 0.025-0.034 vs the ideal 1.0) and ~2x
the intended mass at t>=50; new (timestep-first) order lands within 2% of
ideal at every checked t from 0 to 99. A 3-epoch real training run on this
mixed dataset with the fix completed cleanly (loss 0.084 -> 0.035 -> 0.028).

No ambient sweep has been run with either the old or the fixed sampler on this
box -- the `ANALYSIS.md` ambient-diffusion findings are from the other box's
runs and predate this fix; whether they used the same broken order is unknown.

---

## 2026-08-31 — logbook reconciliation (retroactive: commits `b3b458e`, `ad994ec`)

These entries were written on 2026-08-31 for work committed on 08-27 and 08-31 that
**shipped without any logbook update**. Both commits touched `scripts/` and produced
results under `outputs/`; neither appears in any logbook file before now. The numbers below
were re-derived from the run directories on 08-31, not copied from the commit messages —
see the verification note in [`JOURNAL.md`](JOURNAL.md) 2026-08-31.

### 36. `scripts/plot_ambient_sweep.py` — FIX: two eval-row schemas blanked every seed-1 endpoint

Committed in `b3b458e` (2026-08-27 22:20). **This fix changed a published conclusion**; see
item 39 and the correction in [`ANALYSIS.md`](ANALYSIS.md).

`train.py` gained `eval_specs` (multi-target eval) partway through the project — CHANGES
item 15. After that change, even the single `--eval-task` path writes an `eval_task` key
into every eval row. `_final()` identified a solo run by requiring that key to be **absent**,
which held for the seed-0 Allegro-only baselines but not for the seed-1 ones trained later.
Every seed-1 `sigma*=100` cell was silently blank.

Before:

```python
# Solo runs log a single untagged eval; mixed/ambient runs tag each target.
if eval_task is not None and r.get("eval_task") != eval_task:
    continue
if eval_task is None and "eval_task" in r:
    continue
```

After — `_final` takes a new `solo_task` argument, and `point()` passes `solo_task=task` on
the `sigma == 100` branch:

```python
if eval_task is not None:
    # Mixed/ambient run: the row must name the target being scored.
    if r.get("eval_task") != eval_task:
        continue
elif "eval_task" in r and r["eval_task"] not in (None, solo_task):
    # Solo run whose rows happen to be tagged: accept only its own task.
    continue
```

Verify: `point("InHand-Rotation", "400k", 100, "Allegro", 1, True)` returns `2.090`; before
the fix it returned `None`. Third occurrence of this same two-schema root cause in one day,
after the monitoring completion check and the baseline eval-row verification.

### 37. `scripts/probe_state_equivalence.py` — NEW

Committed in `b3b458e`. Checks the three preconditions for a usable Allegro/LEAP state
equivalence before any encoder is trained. Results: `outputs/analysis/state_equivalence.json`.

- **Q1 populated equivalence classes** — nearest-neighbour distance in the task subspace,
  cross-embodiment vs a within-embodiment floor.
- **Q2 joint invariance** — MLP embodiment separability from the task subspace alone. Must be
  checked on the *concatenation*: terms can be marginally at chance yet jointly identifying,
  which is exactly what rotation does.
- **Q3 action correspondence** — R² of Allegro-action → matched-LEAP-action, against a
  task-state-only baseline. The comparison, not the absolute R², is the answer.

Two methodology points worth preserving:

- The within-embodiment NN baseline loads **disjoint episode halves** (`load(..., half=0/1)`).
  Querying a set against itself returns the point itself or its adjacent near-identical frame,
  giving a median distance ~0.001 and a meaningless ratio.
- The task subspace is selected from measured per-term separability (`acc < 0.70` in
  `state_separability.json`), not by intuition about which terms "are" object state.

Run: `.venv/bin/python scripts/probe_state_equivalence.py` (defaults: `--size 400k
--max-episodes 120 --n-match 8000 --seed 0`). Needs `outputs/analysis/schemas.json` (from
`slurm_jobs/dump_schemas.sh`) and `outputs/analysis/state_separability.json` (item 30).

### 38. `scripts/train_state_equivalence.py` — NEW

Committed in `ad994ec` (2026-08-31 16:01). Trains an encoder with task-decode +
gradient-reversal-adversarial + cross-embodiment alignment losses, sweeping the reversal
strength lambda, and asks whether z can be embodiment-invariant *and* task-informative.
Trains no policy. Results: `outputs/analysis/state_equivalence_training.json`,
`outputs/plots/state_equivalence_tradeoff.png`.

**The methodology point that decides whether the numbers mean anything:** invariance is
scored by a **fresh probe trained after the encoder is frozen**, on held-out episodes — not
by the adversary's own head. The recorded run shows exactly why: the adversary's own training
accuracy falls to 0.59-0.76 while a fresh probe still separates the hands at 0.98-0.999
balanced accuracy. Reading the adversary's loss would have reported invariance that does not
exist.

Two further guards: shared (not per-hand) standardisation, or the encoder gets invariance for
free in a way a deployed policy could not; and `--disc-steps` to keep the discriminator near
optimal, since gradient reversal only supplies a useful signal when it is.

Run as recorded — **note the non-default lambdas**, which is what the committed JSON contains:

```bash
.venv/bin/python scripts/train_state_equivalence.py --lambdas 1 10 100
```

Defaults are `--lambdas 0.0 0.1 0.3 1.0 3.0 10.0 --size 400k --zdim 32 --epochs 400
--max-episodes 120 --seed 0`. Also needs `outputs/analysis/schemas.json`.

### 39. `scripts/make_plot_index.py` — classify the result figures

`GROUPS` had no entry for anything produced after 2026-08-26, so 16 of 50 figures — including
the ambient-sweep and separability **result** figures — landed in the "Unclassified" list with
no description and, more importantly, no record of which script regenerates them. Added
seven entries: `mixed_matrix_*`, `ambient_sweep*`, `ambient_threshold`, `state_separability`,
`state_equivalence_tradeoff`, `space_matched_pairs` (its own entry, since
`compare_matched_embodiments.py` writes it while the other `space_*.png` come from
`analyze_spaces.py`), and `space_*`.

`space_matched_pairs.png` must stay **above** the `space_*.png` glob: `GROUPS` is matched in
order and each file is claimed once.

Verify: `.venv/bin/python scripts/make_plot_index.py` prints `50 classified, 0 unclassified`.
`outputs/plots/INDEX.md` regenerated accordingly.

### 40. `CLAUDE.md` — NEW (repo root)

Written 2026-08-31 on request, for Claude Code sessions. No behaviour change; it is
documentation only. Summarises what `README.md` does not: that the active work is
cross-embodiment diffusion BC rather than the upstream mjlab_hand RL benchmark, the pipeline
order, the required env vars (`MUJOCO_GL`, the `/usr/lib64` `LD_LIBRARY_PATH` prepend,
per-job `WARP_CACHE_PATH`), the logbook protocol from `.cursor/rules/agent-logbook.mdc`, the
invariants recorded in this file, and the standing findings from [`ANALYSIS.md`](ANALYSIS.md).

It also records that `src/mjlab_hand/env_cfg.py` is dead upstream leftover — it imports
`mjlab_hand.anymal_c`, which does not exist in this repo, so the module cannot be imported.
Left in place, flagged do-not-extend.

---

## 2026-08-27 (evening) — separability analysis, plotting, and operational fixes

### 30. `scripts/measure_state_separability.py` — NEW

Probes whether the embodiment is identifiable from a single observation: linear and MLP
classifiers, held-out accuracy and AUC, plus per-term breakdown and a hypothetical
noise-vs-separability curve using the policy's own cosine schedule.

**Splits are by EPISODE, never by step.** Consecutive frames within an episode are
near-duplicates, so a random step-level split leaks test data into training and drives
accuracy toward 1.0 for *any* two datasets. Every reported number is on held-out episodes.

AUC is computed from the rank statistic rather than with sklearn — sklearn is not installed
in this venv and adding it while 60 jobs were running was not worth the risk. Same reason
the classifiers are hand-rolled torch/numpy.

### 31. `scripts/characterize_state_difference.py` — NEW

Decomposes *what* the difference is: marginal overlap per dimension, "constant-label" score
(between-hand mean gap / within-hand std), removability under per-hand
centring/standardisation/ZCA whitening, and how few dimensions suffice.

Two bugs found and fixed during development, both of which would have produced confident
wrong answers:

**Fix A — accuracy below chance.** First run reported 0.386 accuracy after centring.
Episode-level splits do not balance step counts, so a signal-free model that predicts one
class scores the majority fraction. Now reports **balanced accuracy and AUC** (both
imbalance-invariant) and applies class weights in the fit. `logreg()` deliberately does not
return plain accuracy.

**Fix B — test statistics leaked into the alignment.** `align()` originally computed
per-hand means/stds over all rows including test. For a question of the form "does centring
remove the difference?" that is precisely the leak that matters. It now takes train masks
and uses train-only statistics.

### 32. `scripts/plot_ambient_sweep.py` — NEW

sigma* on x, performance on y, one line per target budget, one panel per (family, hand).
Routes each cell to whichever run holds it: sigma*=0 from `outputs/mixed_noc`, interior from
`outputs/ambient`, sigma*=100 from `outputs/diffusion`, with the N=400k sigma*=0 special case
in `outputs/ambient/..._amb0`. Carries the same strict final-epoch guard as
`plot_mixed_matrix.py` and prefers `eval_metrics_100.jsonl` when present, so 32-rollout and
100-rollout numbers are never mixed inside one curve. `--require-100` blanks any point not
scored on >=100 rollouts.

N is an ordered quantity, so it is encoded sequentially (one blue hue, light->dark) with
direct labels rather than as a categorical palette.

### 33. `slurm_jobs/reeval_endpoints.sbatch` — NEW

Re-scores the sweep's endpoint runs at 100 rollouts from their saved `policy_latest.pt`.
Writes to **`eval_metrics_100.jsonl`**, never appending to the original
`eval_metrics.jsonl`, so no already-published number is mutated. Truncates its output file
on start so a requeue cannot double-append. Array `751473`, 14 tasks, ~1:50 each.

### 34. `slurm_jobs/train_diffusion.sbatch` — `NUM_WORKERS` parameterised

Was hardcoded `--num-workers 8`. Added `NUM_WORKERS` (default 8, back-compatible). Batch
composition is unaffected by worker count — shuffle order comes from the main-process
generator seeded by `torch.manual_seed(cfg.seed)` — so this is not a protocol change.

Introduced to work around apparent DataLoader deadlocks. **That diagnosis turned out to be
mostly wrong** (see item 35); the flag is still useful but is not the fix.

### 35. Operational: silent hangs, and Slurm's State field lying

Four training tasks reported `RUNNING` while doing no work. Symptoms: log file unwritten for
15-45 minutes, GPU at 0% utilisation, and `AveCPU` frozen. Examples:

| task | CPU used / elapsed | node |
|---|---|---|
| 751514_0 (grasp 10k s1) | 7 min / 36 min | rlcompute13 |
| 751460_1 (rot 50k s1) | 35 min / 4h06 = 14.3% | **rlcompute23** |
| 751460_2 (rot 100k s1) | 1s of CPU across 30 min | **rlcompute23** |
| 751608_1 (rot 50k s1, retry) | 16 min / 59 min = 27.7% | **rlcompute23** |

Lessons for whoever automates this next:

- **Do not trust `State=RUNNING`.** Detect hangs from **log-file mtime staleness**, and
  confirm with `AveCPU` vs `ElapsedRaw`.
- **Cumulative CPU-busy % hides a recent stall.** 751460_2 showed 47.9% busy and was
  described as "degraded but safe"; it had in fact already stopped dead. The reliable
  signal is *progress between two consecutive checks*, not a run-lifetime average.
- **`NUM_WORKERS=4` did not prevent a recurrence** on the same node, which is what shifted
  the diagnosis from DataLoader churn to the node itself. 3 of 4 stalls were on
  `sea112-rlcompute23`.
- Recovery procedure that worked: `scancel`, `mv <dir> <dir>.stalled-<timestamp>` (never
  delete, and never let a resubmit append into an existing `eval_metrics.jsonl`), resubmit
  that index alone with `--exclude` extended and `--time=48:00:00`.
- `scontrol update jobid=... TimeLimit=...` to raise a walltime is **permission-denied** for
  a normal user, so a job projecting past its limit must be restarted, not extended.
- Slurm node names must be **fully qualified** in `--exclude`: `rlcompute13` fails
  submission with `Invalid node name specified`; `sea112-rlcompute13` works.
- `squeue` prints a not-yet-expanded array as a single line (`751461_[0-3]`), so counting
  `squeue` lines undercounts submitted tasks. Use `sacct`.

---

## 2026-08-27 — ambient diffusion: per-source diffusion-timestep gating

### 23. Version control initialised

The project arrived as an unpacked zip with **no git history**. `git init` + a baseline
import commit; incremental commits from there. `.gitignore` extended: the original
ignored `slurm_jobs/` and `outputs/` wholesale, which excluded the job scripts (the
reproducibility record) and the figures (the results). Now:

```
slurm_jobs/*            outputs/*
!slurm_jobs/*.sbatch    !outputs/plots/    !outputs/analysis/
!slurm_jobs/*.sh        outputs/plots/*    outputs/analysis/*
                        !outputs/plots/*.png   !outputs/analysis/*.json
*.pt  *.zarr/  warp_cache/  data/
```

Checkpoints (265MB each), zarr stores, videos and the 6398 slurm log files stay out;
1066 files / 113MB tracked.

### 24. `src/mjlab_hand/diffusion/policy.py` — `compute_loss` gains `t_min` / `weight`

The mechanism for ambient diffusion. Previously the loss sampled timesteps uniformly and
reduced with `F.mse_loss`:

```python
timesteps = torch.randint(0, T, (b,), device=device, dtype=torch.long)
...
return F.mse_loss(pred, noise)
```

Now, with per-sample `t_min`, `t ~ Uniform[t_min, T)`:

```python
t_min = t_min.to(device=device, dtype=torch.long)
span  = (T - t_min).clamp(min=0)
u     = torch.rand(b, device=device)
timesteps = (t_min + (u * span).long()).clamp(max=T - 1)
...
per_sample = ((pred - noise) ** 2).mean(dim=tuple(range(1, noise.ndim)))
return (per_sample * weight).sum() / weight.sum().clamp_min(1e-8)
```

**Why the weight matters, and why it is `(T - t_min) / T`.** Restricting a sample to
`[t_min, T)` *concentrates* its probability mass there. Without compensation a large
`t_min` would make source data dominate the coarse steps more and more as the threshold
rises — the sweep would then move for a reason unrelated to the hypothesis. With
`w = (T - t_min)/T` the per-timestep gradient density is

    w * P(t) = (T - t_min)/T * 1/(T - t_min) = 1/T     for every t >= t_min

exactly the density of an ungated sample. So the intervention is precisely **"truncate the
schedule range below `t_min`, change nothing else"**, not "reweight the source". It also
gives the clean limits: `t_min=0` -> density 1/T everywhere (ungated), `t_min=T` -> the
sample contributes nothing.

Normalising by `weight.sum()` rather than `b` keeps the loss scale — and hence the
effective step size under the matched-gradient-step protocol — comparable across settings,
instead of shrinking it as source weight drops out.

**Back-compat is exact.** With `t_min=None, weight=None` the function is the original
code path verbatim. Verified: identical to 10 decimal places on a fixed seed, and the
weighted reduction with `w=1` matches `F.mse_loss` to 1.2e-07 worst case over several
shapes (float32 rounding only).

### 25. `src/mjlab_hand/diffusion/dataset.py` — source provenance, no rebuild needed

`TrajectoryStore.source_step_bounds()` -> `[(start, end, task)]` per source, recovered
from the cumulative `extra.sources[i].n_steps` that `build_mixed_dataset.py` already
writes. **No dataset rebuild was required** for the 6 existing mixed datasets. Raises if
the counts do not sum to `n_steps`, rather than silently mis-attributing samples.

`DiffusionDataset(..., ambient_tmin=[0, 50])` takes one `t_min` per source in dataset
order and resolves it per episode **from the episode's step offset**, not its index —
which keeps it correct after `success_only` drops episodes. `__getitem__` returns an extra
`"t_min"` key. Guards: refuses a non-mixed dataset, refuses an arity mismatch.

Verified on `data/mixed_noc/InHand-Rotation_A10k_L400k.zarr`: 10,038 windows at t_min=0
and 399,899 at t_min=50, exactly matching the stored per-source step counts.

### 26. `train.py` / `cli/train_diffusion.py` — `--ambient-tmin`

`TrainConfig.ambient_tmin: list[int] | None`, passed to the dataset and used to derive the
weight in the loop. Recorded in `train_config.json` so plotters can read sigma* off disk.
`--ambient-tmin 0 50` = admit source 0 everywhere, source 1 only at t >= 50.

### 27. `slurm_jobs/train_diffusion.sbatch` — `SEED` / `EVAL_ENVS` parameterised

Was hardcoded `--seed 0` and `--eval-num-envs 32`, so a second seed would have overwritten
the first run's directory and appended into its `eval_metrics.jsonl`. Now `SEED` env var
with a `_s{N}` output-dir suffix (matching the convention `train_mixed.sbatch` already
used) and `EVAL_ENVS` defaulting to 32 for back-compat. Also added
`--exclude=sea112-rlcompute13`.

Note the node name must be **fully qualified**: `--exclude=rlcompute13` fails submission
with `Invalid node name specified`. Nodes are `sea112-rlcompute{06,12,13,14,17,18,19,22,23}`
and `sea112-rtsrtx{01,02,12,14,15,16,17,18}`.

### 28. New scripts and job files

| File | Purpose |
|---|---|
| `scripts/measure_ambient_threshold.py` | Diagnostic: W1(Allegro, LEAP) actions vs diffusion timestep. Reuses `wasserstein1()` from `compare_matched_embodiments.py`. |
| `scripts/plot_ambient_sweep.py` | Result figure: performance vs sigma*, one line per target budget. Carries the same strict final-epoch guard as `plot_mixed_matrix.py`. |
| `slurm_jobs/build_ambient.sbatch` | The 2 `A400k_L400k` datasets `build_mixed.sbatch` structurally cannot make (it skips `a == b`). |
| `slurm_jobs/train_ambient.sbatch` | The sweep. `FAMILY` / `SEED` / `TARGET_SIZES` / `SIGMA_LIST`. |
| `slurm_jobs/reeval_endpoints.sbatch` | Re-scores endpoint runs at 100 rollouts, **non-destructively** into `eval_metrics_100.jsonl`. |

### 29. `scripts/plot_mixed_matrix.py` — `--seeds`

Had no seed support, so the 40 seed-1 unconditioned runs had never been plotted; every
published matrix was seed 0 only. `--seeds 1` plots that seed, `--seeds 0 1` plots the
per-cell mean, and the filename/subtitle record which. Reports the per-cell seed count to
stdout so a 1-seed and a 2-seed cell are not silently drawn identically. Also fixed a
subtitle that claimed one-hot conditioning on the *unconditioned* figures.

---

## 2026-08-26 (evening) — result-reporting correctness

### 22. `scripts/plot_mixed_matrix.py` — NEW, plus two correctness fixes

Confusion-matrix heatmaps of the mixed grid: rows = Allegro data size, columns = LEAP data
size. Top row absolute performance (sequential, one blue hue light->dark), bottom row delta vs
the single-embodiment baseline (diverging blue<->red, neutral gray at zero, symmetric about
0). Diagonal drawn as inert surface — a mixture of a size with itself was never run.

Palette taken from the dataviz reference instance. The bundled validator is Node and there is
no `node` on this cluster, so the checks were ported to Python: all matched-magnitude
diverging arm pairs clear dE >= 8 under both deuteranopia and protanopia (min 8.6), and the
midpoint sits at 1.12 contrast against the surface. The derived red arm needed this; the blue
sequential ramp is verbatim from the reference.

**Fix A — midpoint evals were being read as final.** Eval fires at the midpoint *and* the end,
so a still-running job already has rows on disk. Taking "the last row present" plotted a
midpoint value in the same style as a completed one, and produced a reported +0.41 that became
-0.03 once the run finished. Cells are now gated on the run actually reaching its last
scheduled eval.

**Fix B — odd-epoch runs never reach `num_epochs`.** The first version of Fix A required
`epoch == num_epochs`. But `EVAL_EVERY = EPOCHS / 2` is integer division, so for
`num_epochs = 3351` evals fire at 1675 and **3350**. Six finished runs were silently marked
incomplete and blanked from the figure. Correct test:

```python
last_scheduled = 2 * (total // 2)
final = [x for x in r if x.get("epoch") == last_scheduled and "eval_task" in x]
```

Both rules matter for any consumer of `eval_metrics.jsonl`, not just this script.

**Follow-up worth doing:** make `EVAL_EVERY` divide `EPOCHS`, or force an eval on the final
epoch, so "final" means the actual final weights. Currently an odd-epoch run is scored one
epoch early — 0.025% of a 780k-step budget, immaterial numerically but a trap for readers.

## 2026-08-26 — mixed-embodiment conditioning, checkpoint I/O fix, space analysis

### 13. `src/mjlab_hand/diffusion/train.py` — FIX: per-epoch checkpointing dominated runtime

`policy_latest.pt` was written **every epoch**. The checkpoint is 265 MB / 66.3M params and
`torch.save` to NFS measures ~0.75s — about the same as a 39-batch epoch at the 10k data
scale. Measured overhead: **10k 42%, 100k 3%, 1M ~0%** (same absolute cost, very different
epoch lengths). Over 20,000 epochs that is ~4.2h of I/O and ~5.3 TB written per run.

`TrainConfig` gained `latest_every_epochs: int = 1`. Save block became:

```python
is_last = epoch == cfg.num_epochs
if epoch % cfg.latest_every_epochs == 0 or is_last:
    policy.save(latest_path)
if mean_loss < best_loss:
    best_loss = mean_loss
    if epoch % cfg.latest_every_epochs == 0 or is_last:
        policy.save(cfg.output_dir / "policy_best.pt")
```

`policy_best` is guarded too — early on the loss improves nearly every epoch, so an
unguarded best-save doubles the I/O. A `policy.save(latest_path)` was added immediately
before each eval, because `evaluate_diffusion_policy` loads from disk and would otherwise
score stale weights. Numbered epoch checkpoints are unchanged.

CLI: `--latest-every-epochs`. Job scripts set it equal to `save_every`.

### 14. `src/mjlab_hand/diffusion/evaluate.py` — one-hot embodiment conditioning

For policies trained on mixed-embodiment data, the env emits the base observation but the
policy expects the training-time embodiment label appended. `DiffusionActionChunkPolicy`
gained an `onehot` argument:

```python
self.onehot = torch.tensor(onehot, dtype=torch.float32, device=device) if onehot is not None else None
...
if self.onehot is not None:
    obs_t = torch.cat([obs_t, self.onehot.expand(obs_t.shape[0], -1)], dim=1)
```

Appended in the same trailing position `build_mixed_dataset.py` uses. `onehot=` also added to
`evaluate_diffusion_policy` and `render_diffusion_rollout`, plus a warning when the policy's
`obs_dim` does not match the env's and no one-hot was supplied — the failure would otherwise
be a confusing shape error deep in the UNet.

### 15. `src/mjlab_hand/diffusion/train.py` — multi-target eval (`eval_specs`)

A mixed policy drives two bodies, so a single `eval_task` cannot express its evaluation.
Added `eval_specs: list[dict] | None`:

```python
[{"task": "Grasp-Allegro", "onehot": [1, 0]},
 {"task": "Grasp-LEAP",    "onehot": [0, 1]}]
```

At each eval point the loop iterates the specs, evaluating each env with its own one-hot, and
writes one JSONL row per spec with added `eval_task` and `onehot` fields so per-embodiment
performance is separable. `eval_task` still works and is internally promoted to a
single-element spec. The headline print falls back from `success_rate` to
`avg_successes_before_drop` so rotation runs log a real number.

CLI: `--eval-spec '<json>'`.

### 16. `scripts/build_mixed_dataset.py` — NEW

Concatenates two single-embodiment datasets and appends a one-hot embodiment label to every
observation: `obs_mixed = [obs (D), onehot (K)]`, actions unchanged. 115+2=117 for grasp,
69+2=71 for rotation.

The label goes on the **observation, not the action** — it conditions the policy, it is not
something to predict — so it reaches the model through the same `global_cond` path as the
rest of the observation, on all `obs_horizon` frames.

**Refuses mismatched spaces** rather than zero-padding. Only Allegro and LEAP share both
dimensionality and term layout; padding two different layouts would silently place unrelated
physical quantities in the same column.

Episodes are copied whole and `episode_ends` recomputed against a running offset.
Provenance (`mixed`, `onehot_dim`, `onehot_order`, `base_obs_dim`, per-source paths/steps)
goes into the store's `extra` attrs.

**Performance fix during development:** the first version called
`TrajectoryStore.append_episode` per episode, which resizes four zarr arrays each time.
For ~2000-episode sources that meant thousands of resizes and ~6h to build all 40 mixtures.
Rewritten to assemble in memory and bulk-write each array once (peak ~0.7 GB for the largest
mixture); combined with running the build as a Slurm array it took ~12 min.

### 17. `slurm_jobs/build_mixed.sbatch`, `slurm_jobs/train_mixed.sbatch` — NEW

`build_mixed.sbatch` — CPU-only array, 20 tasks per family, one mixture each.

`train_mixed.sbatch` — 20 jobs per family, every **ordered** distinct-size pair from
{10k, 50k, 100k, 400k, 1M}. Both orderings are run so hand identity is not confounded with
data budget (`A10k_L1M` and `A1M_L10k` are different experiments).

Epochs are **derived per run from the dataset**, not tabulated: mixed totals are irregular
(10k+400k = 410k), so the script reads `TrajectoryStore(...).n_steps` and computes
`epochs = ceil(784000 / (n_steps // 256))`. Result spans 3351 epochs for the smallest
mixture to 143 for the largest, all within 0.5% of the target gradient-step budget.

### 18. `slurm_jobs/train_diffusion.sbatch` — HANDS_LIST, new scales, eval mid+end

- `HANDS_LIST` env var restricts which embodiments an array covers. Necessary, not
  cosmetic: the script hardcoded five hands with `% 5` / `/ 5` index arithmetic, so a
  2-hand array would have silently mapped indices to the wrong task/size cells. Now
  derived from `${#HANDS[@]}`.
- Added `50k` (4000 epochs) and `400k` (500 epochs) scale entries.
- `EVAL_EVERY=$(( EPOCHS / 2 ))` — rollout eval exactly at the midpoint and the end, per
  user instruction, at every scale.
- `LATEST_EVERY="$SAVE_EVERY"` wires the checkpoint fix above.

### 19. `scripts/analyze_spaces.py` — NEW, plus a PCA numerical bug

Distributional analysis of observation and action spaces per embodiment: per-dim and
per-term statistics, action saturation, and intrinsic dimensionality via PCA on standardised
features. Writes `outputs/analysis/spaces_summary.md`, `spaces_stats.json`, and
`space_action_ranges.png` / `space_pca.png` / `space_term_matrix.png`.

**Bug found and fixed during development.** The first version guarded near-constant dims with
`sd[sd < 1e-8] = 1.0` and worked in float32. Grasp-Shadow has an observation dim with
std **9.4e-08** — above that threshold — so it was standardised by its own std, amplifying
float32 quantisation noise (eps ~1.2e-7 relative) to O(1). Several near-constant dims
quantise on the same rows, producing a correlated noise block that dominated the SVD and
reported **pc_95 = 1 for a 189-d space**. Fixed by working in float64 and *dropping* dims
with std < `CONST_STD_EPS = 1e-6` rather than standardising them, and reporting the dropped
count. Shadow then reports pc_95 = 43 with 27 constant dims, in line with the other hands.

### 20. `scripts/compare_matched_embodiments.py` — NEW

Elementwise comparison for the only dimension-matched pair (Allegro vs LEAP, both families).
Metric is 1-D Wasserstein-1 per dimension divided by the pooled std of that dimension (no
scipy needed — quantile functions), aggregated per observation term.

Reports **raw and mean-centred** distance. The split matters: rotation `object_pos` has the
largest raw shift of anything (4.417) but a std of only 0.034 — the palms hold the object
~9 cm apart, so it is a rigid translation, and centring drops it to 0.312. Raw distance alone
is actively misleading on low-variance features. Added the decomposition to the script rather
than leaving it as a one-off check.

### 21. `slurm_jobs/dump_schemas.sh` — NEW

Builds each task env once and dumps observation/action term names and dims to
`outputs/analysis/schemas.json`. The zarr stores hold flat vectors with no column names, so
without this the analysis cannot say which slice of a 115-d observation is "object position".
Probes several attribute names (`group_obs_term_dim` / `group_obs_term_dims`) since these
differ across mjlab versions.

## 2026-08-25 — diffusion sampler fix, per-epoch eval, rollout rendering

### 1. `src/mjlab_hand/diffusion/policy.py` — FIX: sampler was mathematically invalid

**Function:** `DiffusionPolicy.predict_action`

**Bug.** `inference_timesteps` is a strided subsequence
(`torch.linspace(0, num_train_timesteps - 1, num_inference_steps).long()` →
`[0, 6, 13, …, 99]`, stride ≈ 6.6), but the loop applied the **single-step** DDPM ancestral
update at each grid point. That update is only valid for a `t → t-1` transition, so across a
~6-step jump it removed roughly a sixth of the required noise. `compute_loss` was unaffected
(it uses `sqrt_alphas_cumprod` with `t` uniform over the full range), so training loss looked
healthy while every sampled action was noise-dominated.

**Before:**

```python
x = torch.randn(b, self.cfg.action_horizon, self.cfg.action_dim, device=device)
timesteps = self.inference_timesteps.tolist()
for i in reversed(range(len(timesteps))):
    t = torch.full((b,), int(timesteps[i]), device=device, dtype=torch.long)
    eps = self.noise_pred_net(x, t, nobs)
    alpha = self.alphas[t].view(-1, 1, 1)
    alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
    beta = self.betas[t].view(-1, 1, 1)
    x = (1.0 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps)
    if i > 0:
        x = x + torch.sqrt(beta) * torch.randn_like(x)
return self.action_normalizer.unnormalize(x)
```

**After** (DDIM, eta=0 — valid for arbitrary strides because it uses cumulative alphas at the
two *inference* timesteps):

```python
for i in reversed(range(len(timesteps))):
    t_cur = int(timesteps[i])
    t = torch.full((b,), t_cur, device=device, dtype=torch.long)
    eps = self.noise_pred_net(x, t, nobs)
    alpha_bar_t = self.alphas_cumprod[t_cur]
    if i > 0:
        alpha_bar_prev = self.alphas_cumprod[int(timesteps[i - 1])]
    else:
        alpha_bar_prev = torch.ones_like(alpha_bar_t)
    x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
    x0 = x0.clamp(-1.0, 1.0)
    x = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * eps
return self.action_normalizer.unnormalize(x)
```

Keeps 16 inference steps (no slowdown) and is deterministic, which makes eval reproducible.
The alternative fix — keeping the original update but iterating all 100 timesteps — is also
correct but 6x slower per rollout.

**Verify:** `python scripts/check_sampler.py --checkpoint <policy.pt> --dataset <zarr>`.
Expect DDIM MSE ≈ 0.002 vs old ≈ 0.53 (worse than the 0.09 predict-zeros baseline) on a
checkpoint trained only a few epochs.

### 2. `scripts/check_sampler.py` — NEW

Standalone A/B of the two samplers against recorded expert actions on real observation
windows, in normalized action space. No env needed. Reproduces the old loop inline as
`sample_old()` purely for comparison. Reports MSE and std against `noise` and `zeros`
baselines. Use as a regression check if rollout success ever collapses again.

### 3. `src/mjlab_hand/diffusion/evaluate.py` — env reuse for per-epoch eval

**Added** module-level `_ENV_CACHE: dict[tuple[str, int, str], tuple]` and
`close_cached_envs()`.

**Changed** `evaluate_diffusion_policy` signature: new keyword `reuse_env: bool = False`.
When true, the env is looked up in / stored into `_ENV_CACHE` keyed by
`(task, num_envs, device)` and **not** closed at the end:

```python
cache_key = (task, num_envs, str(device_t))
if reuse_env and cache_key in _ENV_CACHE:
    env, _rl_cfg = _ENV_CACHE[cache_key]
else:
    eval_cfg = EvalConfig(num_envs=num_envs, seed=seed, device=str(device_t))
    env, _rl_cfg = setup_eval_env(task, eval_cfg, device_t)
    if reuse_env:
        _ENV_CACHE[cache_key] = (env, _rl_cfg)
...
if not reuse_env:
    env.close()
```

Safe because `run_eval` (`src/mjlab_hand/eval/base.py`) calls `env.reset()` itself. The
evaluator is still rebuilt per call since it accumulates episode state.

Default is `False`, so the standalone `eval-diffusion` CLI keeps the old build-and-close
behaviour. **Behavioural note:** with reuse the `seed` argument only affects the first build,
so every epoch is scored on identical eval conditions — intentional, it removes env-sampling
variance from the success curve.

### 4. `src/mjlab_hand/diffusion/evaluate.py` — NEW `render_diffusion_rollout()`

Records an mp4 of a diffusion-policy rollout. Mirrors `scripts/render_checkpoint_video.py`
but drives `DiffusionActionChunkPolicy` instead of an RL runner.

```python
render_diffusion_rollout(
    task=..., policy_path=..., output_dir=...,
    num_steps=400, num_envs=1, device="cuda:0", seed=0, tag="epoch0020",
) -> Path | None
```

Builds `ManagerBasedRlEnv(..., render_mode="rgb_array")` → `VideoRecorder` →
`RslRlVecEnvWrapper`, rolls out, closes. Writes to `<output_dir>/<task>_<tag>/*.mp4`.
Requires `MUJOCO_GL=egl`.

Deliberately builds a **fresh** env per call (unlike the eval cache): `VideoRecorder` keeps an
internal step counter driving `step_trigger`, so reuse would need trigger bookkeeping to open
a new file each time.

### 5. `src/mjlab_hand/diffusion/train.py` — eval/render wiring

- `TrainConfig`: added `render_every_epochs: int = 0`, `render_num_steps: int = 400`,
  `render_num_envs: int = 1`.
- Eval call now passes `seed=cfg.seed` (was `cfg.seed + epoch`) and `reuse_env=True`.
- Added `policy.train()` after the eval block. Redundant — the epoch loop already calls
  `policy.train()` at its top — kept only so the block is self-contained.
- New render block after the eval block, gated on `cfg.render_every_epochs > 0`, wrapped in
  `try/except` so a rendering failure (EGL, GPU memory) can never kill a training run.
- `close_cached_envs()` called once after the epoch loop when `eval_task` is set.

### 6. `src/mjlab_hand/cli/train_diffusion.py` — new flags

Added and forwarded to `TrainConfig`:

| Flag | Default | Why |
|------|---------|-----|
| `--save-every-epochs` | 10 | `save_every_epochs` existed in `TrainConfig` but was unreachable from the CLI. A 20-epoch 10M run would otherwise emit only two numbered checkpoints. |
| `--render-every-epochs` | 0 (off) | mp4 rollout cadence. |
| `--render-num-steps` | 400 | |
| `--render-num-envs` | 1 | |

### 6b. `scripts/plot_diffusion_curves.py` — NEW

Diffusion runs do not write TensorBoard, so `plot_seed_curves.py` does not see them. They
append one JSON row per epoch to `outputs/diffusion/<Task>_<size>/eval_metrics.jsonl`
(`epoch`, `train_loss`, `train_steps`, `metrics{success_rate, avg_final_dist_to_first_goal_m,
completed_episodes, eval_time_s}`). This script reads those and writes:

| Output | Contents |
|--------|----------|
| `outputs/plots/dp_<Task>.png` | per hand: loss (log), success_rate, final dist — 1M vs 10M overlaid |
| `outputs/plots/dp_summary.png` | all hands; solid = 1M, dashed = 10M |
| `outputs/plots/dp_summary.md` | table of epochs done, grad steps, last/best success |

```bash
.venv/bin/python scripts/plot_diffusion_curves.py [--x steps|epoch]
```

**X-axis defaults to gradient steps, not epochs**, because the two variants are deliberately
sized for equal gradient steps (1M x 200 epochs vs 10M x 20 epochs, ~784k each). Plotting
against epochs would make matched-compute runs look 10x apart. `--x epoch` overrides.

Tolerates a torn final JSON line (training appends while plotting reads) and skips runs with
no rows yet, so it is safe to run mid-training.

### 6c. `scripts/select_experts.py` — `--variant` filter

`select_experts.py` scores every run dir under an experiment directory and takes the max. Once
a task is re-run on a longer schedule (rotation `_40k`), the re-run outscores the original and
gets selected — even mid-training. Added a filter:

```python
ap.add_argument("--variant", default="any",
                help="'any', 'none' for plain <timestamp>_seed<N> dirs only, "
                     "or an explicit suffix such as '40k'.")
...
variant = m.group(2)
if args.variant == "none" and variant is not None:
    continue
if args.variant not in ("any", "none") and variant != args.variant:
    continue
```

Manifest entries gained a `variant` field. Used to pin rotation demo collection to the
finished 10k sweep:

```bash
scripts/select_experts.py --only rotation --pick best --variant none \
    --out outputs/experts_rotation.json
```

### 6d. `slurm_jobs/train_diffusion.sbatch` — parameterised by task family

Added a `FAMILY` env var (`grasp` default, or `rotation`) selecting the `HANDS` array, instead
of duplicating the script. Array layout, epoch counts, eval cadence and checkpoint cadence are
identical for both families:

```bash
FAMILY=rotation sbatch slurm_jobs/train_diffusion.sbatch
```

### 6e. `scripts/subsample_dataset.py` — NEW

Builds a smaller demo dataset by copying whole episodes from a larger one, for the
data-scaling ablation.

```bash
python scripts/subsample_dataset.py --source <big.zarr> --output <small.zarr> \
    --target-steps 100000 [--overwrite] [--success-only]
```

Episodes are copied **whole** — never truncated mid-episode, because `DiffusionDataset`
builds obs/action windows within episode boundaries and a partial trailing episode would
generate windows running off the end. The greedy selector stops at whichever episode boundary
lands closest to the target (all 20 generated subsets came within 0.4%).

Copies `obs`/`action`/`reward`/`success` and preserves `obs_dim`, `action_dim`, `task`,
`checkpoint`; adds `subsampled_from`, `target_steps`, `source_n_steps`, `source_n_episodes`
to the store's `extra` attrs for provenance.

### 6f. `slurm_jobs/train_diffusion.sbatch` — arbitrary data scales

- `SIZES_LIST` env var (default `"1M 10M"`) replaces the hardcoded `SIZES=(1M 10M)`:
  `read -ra SIZES <<< "${SIZES_LIST:-1M 10M}"`. Array length is `5 * len(SIZES)`.
- Per-scale epochs / eval cadence / checkpoint cadence via a `case` on `$SIZE`:

```bash
case "$SIZE" in
  10k)  EPOCHS="${EPOCHS_10K:-20000}"; EVAL_EVERY=100; SAVE_EVERY=2000 ;;
  100k) EPOCHS="${EPOCHS_100K:-2000}"; EVAL_EVERY=10;  SAVE_EVERY=200  ;;
  1M)   EPOCHS="${EPOCHS_1M:-200}";    EVAL_EVERY=1;   SAVE_EVERY=20   ;;
  10M)  EPOCHS="${EPOCHS_10M:-20}";    EVAL_EVERY=1;   SAVE_EVERY=2    ;;
  *)    echo "unknown size $SIZE"; exit 1 ;;
esac
```

`EVAL_EVERY` is chosen so every scale evaluates every ~4k gradient steps. Eval-every-epoch
is not viable below 1M: at 10k an epoch is 39 steps, so it would mean ~20,000 evals
(~361 h) per 4.5 h run.

### 6g. `scripts/plot_diffusion_curves.py` — FIX: rotation runs plotted nothing

The two task families do not share eval metric names. Grasp reports `success_rate` and
`avg_final_dist_to_first_goal_m`; **rotation reports `avg_successes_before_drop`,
`drop_rate`, `avg_survival_time_s`, `avg_rot_dist` and has no `success_rate` at all.**
The plotter hardcoded `success_rate`, so every rotation run produced an all-NaN success
curve — silently, since `.get(key, np.nan)` never raises.

Added family detection and per-family metric selection:

```python
HEADLINE = {
    "grasp":    ("success_rate", "rollout success_rate", (-0.02, 1.02)),
    "rotation": ("avg_successes_before_drop", "avg successes before drop", None),
}
SECONDARY = {
    "grasp":    ("avg_final_dist_to_first_goal_m", "avg final dist to goal (m)"),
    "rotation": ("drop_rate", "drop rate"),
}
def family_of(task): return "rotation" if "Rotation" in task else "grasp"
```

Family is detected from the metric keys when loading and from the task name when
plotting. `set_ylim(0, 1)` is applied only for grasp — `avg_successes_before_drop` is an
unbounded count. Summaries are now emitted per family (`dp_summary_grasp.png`,
`dp_summary_rotation.png`) since the two headline metrics are not comparable.

Also generalised from the hardcoded `1M`/`10M` to `SIZE_ORDER = ("10k","100k","1M","10M")`,
and added a `tail10` column (mean of last 10 evals) to `dp_summary.md` — at small scales the
success curve swings enough between consecutive evals that `last` is not a stable estimate.

### 6h. `scripts/plot_diffusion_curves.py` — NEW `plot_scaling()`

`dp_scaling_<family>.png`: headline metric vs dataset size on a log x-axis, one line per
hand — the figure the ablation exists to produce. Circles are the mean of the last 10 evals,
triangles the best-of-run (showing how much headroom the noise hides), dotted horizontal
lines the RL expert reference loaded from `outputs/expert_eval/<Task>.json`.

### 6i. `scripts/eval_expert.py` — NEW (works around a broken `eval-policy`)

Needed for the expert reference lines. **`eval-policy` is broken for RL checkpoints** in
this repo version: `mjlab_hand.eval.base.run_eval` calls
``policy(_policy_input(obs, device))``, flattening the observation TensorDict before the
policy sees it. An rsl_rl policy indexes obs by group name and dies with::

    IndexError: too many indices for tensor of dimension 2
    (rsl_rl/models/mlp_model.py: obs_list = [obs[g] for g in self.obs_groups])

`collect_demos` already sidesteps this by calling ``policy(obs)`` on the raw observation and
carries a comment saying so.

This script duplicates run_eval's loop and passes raw obs. **Deliberately not fixed in
`run_eval` itself** — that function is on the hot path of ~30 running diffusion jobs, and
destabilising it for a reporting convenience was not worth it.

**Proper fix, for when nothing is mid-flight:** drop the pre-flattening in `run_eval` and
let each policy handle its own input. `DiffusionActionChunkPolicy.__call__` already calls
`_policy_input` internally, so the diffusion path would be unaffected; only `eval.py` and
`evaluate.py` call `run_eval`.

### 6j. `scripts/make_plot_index.py` — NEW

Writes `outputs/plots/INDEX.md`: what each of the ~35 figures shows and which script
regenerates it.

### 7. `scripts/plot_seed_curves.py` and `scripts/select_experts.py` — FIX: run-dir regex

Both matched run directories with `SEED_RE = re.compile(r"_seed(\d+)$")`, anchored at end of
string. The 40k rotation runs are named `<timestamp>_seed42_40k`, so **every one of them was
silently invisible** — no error, just missing from plots and from expert selection.

**Before:** `SEED_RE = re.compile(r"_seed(\d+)$")`
**After:**  `SEED_RE = re.compile(r"_seed(\d+)(?:_([A-Za-z0-9]+))?$")`

In `plot_seed_curves.discover()` the captured variant is folded into the group key so a longer
re-run plots as its own experiment rather than being mixed into the original:

```python
seed = int(m.group(1))
variant = m.group(2)
key = f"{exp_dir.name}_{variant}" if variant else exp_dir.name
prev = found[key].get(seed)
if prev is None or run_dir.name > prev.name:
    found[key][seed] = run_dir
```

### 8. `scripts/select_experts.py` — best-checkpoint selection

- New `checkpoint_iters(run) -> dict[int, Path]` and
  `best_checkpoint(run, tag, window=25) -> (Path, float, int)`. The latter smooths the
  headline curve with a 25-point moving average, finds its argmax, and snaps to the newest
  checkpoint at or before that iteration.
- New flags: `--pick {last,best}` (default `last`, preserving old behaviour) and
  `--only {all,grasp,rotation}`.
- Manifest entries gained `checkpoint_iter` and `pick_mode`.

Not cosmetic: `Grasp-Sharpa` peaks at iteration **2500** (0.2706) and decays to 0.2599 by
9999, so `--pick last` would have used measurably worse weights.

---

## 2026-08-24 — collection fixes

### 9. `src/mjlab_hand/diffusion/collect.py` — FIX: rotation success was never filtered

`collect_demos` gated success detection on `"pose" in command_manager.active_terms`, falling
back to `succ = True`. Verified on GPU (`slurm_jobs/check_terms.sh`) that the real terms are:

- `Grasp-*` → `['grasp_metrics', 'pose']` (filter worked)
- `InHand-Rotation-*` → `['rotation']` (filter did **not** work)

So every rotation episode was written and labelled successful, dropped objects included, and
`DiffusionDataset(success_only=True)` became a silent no-op for half the sweep.

**Added** after the `object_pos` helper:

```python
active_terms = env.unwrapped.command_manager.active_terms
has_pose = "pose" in active_terms
has_rotation = not has_pose and "rotation" in active_terms
if not (has_pose or has_rotation):
    print(f"[WARN] No 'pose' or 'rotation' command term in {active_terms}; "
          "every episode will be labelled successful.")

def rotation_success() -> torch.Tensor:
    return env.unwrapped.command_manager.get_term("rotation").metrics["episode_success"] > 0.5
```

**Added** to the per-step block:

```python
elif has_rotation:
    success_flag = success_flag | rotation_success()
```

**Changed** at both episode-write sites:

```python
-  succ = bool(success_flag[i].item()) if has_pose else True
+  succ = bool(success_flag[i].item()) if (has_pose or has_rotation) else True
```

**How to verify — the obvious check does not work.** With `keep_failures=False` only passing
episodes are ever written, so `n_success == n_episodes` holds *by construction* and proves
nothing. Use `--keep-failures`, which writes every episode with its real flag:
`slurm_jobs/verify_filter.sh` returned `n_episodes: 200, n_success: 197`.

### 10. `scripts/plot_training_curves.py` — stale tag names

`DEFAULT_TAGS` referenced `Loss/value_function`, `Episode/success`, `Episode/num_successes`,
none of which this mjlab version emits — it was silently dropping 3 of its 8 panels.
Replaced with `Loss/value`, `Metrics/grasp_metrics/object_height_beyond_table`,
`Metrics/rotation/episode_success`.

### 11. `scripts/plot_seed_curves.py` — NEW

Seed-aggregating plotter: per-experiment panels with mean ± 1 std across seeds plus faint
per-seed traces, and `summary_grasp.png` / `summary_rotation.png`. Seeds are interpolated onto
a shared step grid and truncated to their shortest common range, so it is safe to run
mid-training. Written because the pre-existing `plot_training_curves.py` emits one PNG per run
(30 disconnected figures for a 3-seed sweep) and hides the seed structure.

### 12. `scripts/watch_plots.sh` — status table bug

`err=$(grep -cE "..." "$e" 2>/dev/null || echo 0)` printed `0` twice: `grep -c` already prints
`0` and exits 1, so the `|| echo 0` fallback appended a second line. Dropped the fallback.

---

## Environment workarounds baked into every GPU job script

Not source changes, but required on this cluster and easy to lose:

```bash
# rlcompute (H200) nodes ship a stale /usr/local/cuda/compat/libcuda.so.555.42.06 that
# ldconfig prefers over the real 580.126.09 driver -> CUDA error 803 on every call.
export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Concurrent array tasks sharing one Warp kernel cache race and fail with a CCD-kernel
# FileNotFoundError. Give each task its own.
export WARP_CACHE_PATH="$JOBDIR/warp_cache/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

export MUJOCO_GL=egl   # headless rendering
```

Also: `srun` fails with `More processors requested than permitted` when invoked from inside an
existing interactive allocation. Use `sbatch`.
