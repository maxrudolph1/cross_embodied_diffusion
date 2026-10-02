# Code changes

Exact source edits made by agent sessions, so another agent can reproduce or revert them.
Newest first. Repo-relative paths. Artifacts and run registries live in
[`agent_logbook/RUNS.md`](agent_logbook/RUNS.md) /
[`agent_logbook/COLLECTIONS.md`](agent_logbook/COLLECTIONS.md); narrative in
[`agent_logbook/JOURNAL.md`](agent_logbook/JOURNAL.md). Numbered items are cited from code
comments ("See CHANGES.md item N") -- never renumber; append the next number.

---

## 2026-10-01 — plots for the ambient rotation sweep

### 62. `scripts/plot_ambient_rot_sweep.py` (NEW)

Reads `$SCRATCH/.../ambient_rot/*_sigma*_seed*`; per run uses `posthoc_eval.json` (fresh-seed)
when present, else in-training evals (titles then say PROVISIONAL). Writes to `outputs/plots/`:
`ambient_rot_by_hand.png` (one panel per target hand, score vs sigma for best_eval / best_val /
latest, seed dots), `ambient_rot_hands.png` (all hands, one checkpoint rule `--ckpt`, raw and
normalized to sigma=100, direct labels), `ambient_rot_control.png` (only if post-hoc control
evals exist), `ambient_rot_summary.csv` (every value; the table view). x axis: sigma 0..25
evenly spaced with a break before 100. Colours: dataviz reference categorical palette, fixed
order (validator not run: no Node.js on Vista; the palette is pre-validated, 3 series all-pairs,
5 as lines with mandatory direct labels).

---

## 2026-09-30 (night) — scheduling guidance for agents

### 61. `scripts/queue_wait_stats.sh` (NEW), `AGENTS.md`

`queue_wait_stats.sh [partition] [days] [time_limit] [sizes]`: partition node states,
submit->start wait of recently started jobs and age of pending jobs by node count (fine bins),
the user's usage vs the 40/20/96 caps, and `sbatch --test-only` projections per size (login
nodes only; prints the command elsewhere). AGENTS.md's Vista section rewritten (limits,
storage, running, minmax rule) plus a "Getting Vista jobs scheduled fast" procedure: measure,
project with `--test-only`, enumerate cap-feasible shapes, choose by expected completion
(wait + run time), tightest safe `--time`.

---

## 2026-09-30 (evening) — GPU job for reporting evals

### 60. `slurm_jobs/vista_eval_checkpoints.sbatch` (NEW)

Runs `scripts/eval_checkpoints.py` on a `gh` node for run dirs matching `RUNS` (globs relative
to `$SCRATCH/cross_embodied_diffusion/outputs`), `PARALLEL` (default 4) processes sharing the
GPU, optional `ALSO` control embodiment; array element a / process p = shard a*PARALLEL+p of
n_elements*PARALLEL. Needed because idev sessions can land on CPU-only `gg` nodes. Shell
logic (globs, sharding) dry-tested with the python call stubbed; not yet run on a GPU.

---

## 2026-09-30 — per-dimension obs/action reference

### 59. `scripts/dump_spaces.py` (NEW)

Builds each of the 10 task envs once and writes `outputs/analysis/spaces_reference.{json,md}`:
every policy-obs dimension labeled (term, joint name, xyz / quaternion wxyz component,
keypoint body) and every action dimension's target joint/tendon plus the action term's class
and settings (scale/offset, velocity limits, EMA). Dims match the demo zarrs for all 10 tasks.
Corrected the Allegro/LEAP "dimension j is the same quantity" note in ANALYSIS.md.

---

## 2026-09-29 (night) — why padded/pooled runs score 0: normalization, not pooling

### 58. `policy.py`, `pooling.py`, `train.py`, `cli/train_diffusion.py`

Diagnostic (JOURNAL 2026-09-30): same data and seed, plain specialist vs the same data through
the padded path. Plain 1M: 1.03 -> 1.56 -> 2.25 successes before drop; padded: 0.00 at every eval
(50k: 0.78-0.81 vs 0.00). The eval-side transform is exact (0.0 obs diff), so the model is the
problem: on the same held-out windows, first-executed-action MSE in raw units is 0.011 (plain),
0.50 (padded, as sampled), 0.056 (padded with the clamp loosened to +-5).

- **Sampler clamp** (`predict_action` clamped x0 to [-1, 1] unconditionally). That is exactly the
  data range under `LinearNormalizer`, but ~16% of GaussianNormalizer actions lie outside it (up to
  ~13), so every padded run executed actions truncated to mean +- 1 std. Now per-dim buffers
  `action_clip_low/high` (default +-1, so plain runs and old plain checkpoints are unchanged);
  `train.py` sets them to the training split's action range for padded runs
  (`_action_range`). Gaussian checkpoints saved before this load with +-5
  (`LEGACY_GAUSSIAN_CLIP`, warning printed). **Not sufficient on its own**: with the data-range
  clamp the padded 1M checkpoint still scored 0.03 (64 envs).
- **`--source-norm minmax`** (`pooling.fit_source_normalizer`, `TrainConfig.source_norm`, recorded
  in `extra.source_norm`): per-source min/max -> [-1, 1], stored as mean = midpoint, std =
  half-range so eval is unchanged. A single-source minmax pool equals the plain path's
  LinearNormalizer output to 2.4e-7, and its clip range is exactly +-1. Default stays `gaussian`
  until a minmax run is verified in closed loop (runs on idev 1031788, RUNS.md).

---

## 2026-09-29 (later) — ambient diffusion on padded multi-hand data

### 57. `dataset.py`, `train.py`, `cli/train_diffusion.py`, `cli/eval_diffusion.py`, `scripts/build_ambient_rotation_manifest.py` (NEW)

- **Ambient + padded is now supported** (was `NotImplementedError`, item 42).
  `DiffusionDataset.sample_ambient_batch` emits the per-episode `action_mask`, and its window
  gather is vectorized (arrays `_win_epi/_win_t/_win_start/_win_len` built in `__init__`):
  bitwise-identical batches to the old per-window loop for the same RNG (50 batches checked),
  ~5x faster on the login node. `ambient_tmin` stays one entry per source in dataset order.
  `train.py` now rejects `source_sample_mode != uniform` with ambient (it was silently ignored).
- **`--val-embodiment NAME`** (`TrainConfig.val_embodiment`): val loss, and so
  `policy_best_val.pt`, on one source of a pooled dataset only. Implemented as
  `DiffusionDataset(only_source=i)`, applied after the train/val split, so the held-out
  trajectories are the same as without it. `best_val.json` records `val_embodiment`.
- **`eval-diffusion --embodiment`** for scoring a pooled (padded) policy from the CLI.
- **`scripts/build_ambient_rotation_manifest.py`**: `--kind sweep` (320 runs: 5 target hands x
  sigma {0,1,2,3,4,5,6,8,10,12,14,16,18,20,25,100} x seeds 0-3; target 50k + other hands 1M,
  target first, `--ambient-tmin 0 s s s s`; eval + val on the target only; no epoch snapshots;
  50 epochs) and `--kind diagnostic` (8 Allegro runs). One run per manifest task; the 4 seeds
  of a config are consecutive, so PACK=4 puts one config per node.

Verified on the login node (CPU): gating (0 violations in 51,200 samples; only the target at
t < sigma), masks equal each source's real action dim, sigma=100 target-only, val filter =
exactly the target's episodes of the unfiltered val split, an ambient padded batch through
`compute_loss` + backward is finite, all 328 manifest runs parse through the real CLI with
existing datasets, sbatch node slots correct for `-N 2`. GPU smoke test (idev GH200, through
the sbatch with /tmp staging and `torch.compile`, 10k-per-hand pools, 2 epochs): ambient s=10
padded, pooled co-training and `--pool-sources` single-hand all trained, ran target evals, and
wrote `policy_{best_eval,best_val,latest}.pt` + `train_done.json`.
- **`scripts/eval_checkpoints.py`** (NEW): re-scores `policy_best_eval/best_val/latest.pt` of each
  run on its target with a fresh env seed (`1000 + training seed`, same for all three), 100
  envs = 100 episodes by default, `--also EMB` for control hands, `--shard i/n`; results in
  `<run>/posthoc_eval.json`, resumable. Smoke-tested on GPU.

---

## 2026-09-29 (later) — pool cross-embodiment datasets at load time

### 56. `src/mjlab_hand/diffusion/pooling.py` (NEW), `train.py`, `cli/train_diffusion.py`, `scripts/build_padded_dataset.py`, `scripts/check_pooling.py` (NEW), `slurm_jobs/vista_train_manifest.sbatch`

Multi-embodiment (padded) training no longer needs a pre-built `padded/*.zarr`.

- **`pooling.py`**: the padded scheme moved out of `build_padded_dataset.py` unchanged
  (`load_source`, `gather_padded`, per-source `GaussianNormalizer.fit` on all rows, zero-pad at
  the tail, concatenate whole episodes, `extra.sources` provenance) into `pool_padded(paths)`.
  One difference in mechanics, not output: sources are loaded one at a time, so peak RAM is one
  raw source + the pool. `pooled_store(paths)` wraps the result as `InMemoryTrajectoryStore`, a
  `TrajectoryStore` subclass over numpy arrays with the same attrs the builder writes. Because
  `DiffusionDataset`'s `np.asarray(store.data["obs"][:], float32)` is then a view, the train
  and val datasets share one copy (a zarr store decodes one copy each), so pooling in memory
  uses less RAM than reading a padded zarr. Embodiment name = source filename before
  `_expert` (unchanged; eval matches on it); duplicate names are refused. `task_family` is
  inferred (`Grasp-Allegro` -> `Grasp`) unless given.
- **`build_padded_dataset.py`**: now a thin writer over `pool_padded`; same CLI.
- **`train-diffusion --dataset A.zarr B.zarr ...`** (`nargs="+"`): >1 dataset pools in memory
  (`TrainConfig.dataset: Path | list[Path]`). `--pool-sources` with one dataset trains it
  through the padded path (per-source Gaussian stats, identity policy normalizers) -- the
  control for "is the padded path itself the problem". `--no-pool-sources` with several
  datasets is an error. `--task-family` overrides the inferred name. `train_config.json` /
  wandb record `dataset` as a list for pooled runs. `source_stats.json` is written exactly as
  for a padded zarr, so eval is unchanged. Manifests: `"dataset": [path, ...]`
  (`run_manifest_task.py` already expands lists).
- **sbatch staging** stages every path of a list-valued `dataset`.
- **`scripts/check_pooling.py`**: re-pools each `padded/*.zarr` from its recorded sources and
  compares (see JOURNAL.md for results).

Verified: on Vista, the pre-refactor builder, the new builder and `pooled_store` give
bitwise-identical arrays and metadata (rotation 10k x 5). Against the 14 zarrs built on the
old (x86) cluster: with the recorded normalizer stats the padding/concatenation is bitwise
identical; refitting on Vista differs from the recorded stats by float32 rounding only
(torch CPU reduction order; means off by <=2.4e-7, stds exact), i.e. ~1e-6 in normalized
values. `train-diffusion` setup (0 epochs) writes the same `source_stats.json` and val split
from `--dataset A..E` as from the equivalent zarr, and a pooled batch through 2 DataLoader
workers + `compute_loss(action_mask=...)` + backward is finite.

---

## 2026-09-29 — Vista storage: stage demos, outputs on `$SCRATCH`, promote script

### 55. `slurm_jobs/vista_train_manifest.sbatch`, `scripts/promote_outputs.sh` (NEW), `src/mjlab_hand/diffusion/train.py`

`$WORK` (Stockyard) is a 1 TB Lustre quota shared across all TACC systems and was nearly full
(item 54's incident), so Vista training no longer reads demos from it or writes checkpoints to it.

- **Data staging.** After picking its manifest tasks, each node rsyncs every dataset its runs
  use (`dataset` values under `data/mjlab_hand_demos/`, trailing slash ignored) to
  `DATA_STAGE=scratch` -> `$SCRATCH/cross_embodied_diffusion/mjlab_hand_demos/<same rel path>`
  (persistent mirror; later jobs only re-stat it) or `DATA_STAGE=tmp` -> `$JOBTMP/mjlab_hand_demos`
  (node-local, removed by the existing EXIT trap). `DATA_STAGE=auto` (default) is `tmp` for
  `SLURM_NNODES>1`, `scratch` otherwise; `none` reads in place. `rsync -a` is incremental and
  writes each file via temp+rename, so it repairs purged files and concurrent jobs staging the
  same dataset are safe. Datasets outside `data/mjlab_hand_demos` are read in place with a warning.
- **Outputs.** `OUT_ROOT` (default `$SCRATCH/cross_embodied_diffusion/outputs`).
- **Run root.** Manifests stay repo-relative: the runs execute from `$JOBTMP/root`, which
  symlinks every top-level repo entry back to the repo except `data/mjlab_hand_demos` (-> the
  staged copy; other `data/*` entries -> repo) and `outputs` (-> `OUT_ROOT`).
  `run_manifest_task.py` is called by absolute path with the manifest's absolute path.
  `train_config.json` therefore still records `data/mjlab_hand_demos/...` / `outputs/...`.
  Note: any manifest arg pointing at an existing `outputs/...` checkpoint now resolves on
  `$SCRATCH`, not Stockyard.
- `DRY_RUN=1` stages and passes `--dry-run` to `run_manifest_task.py`.
- **`train.py`** writes `train_done.json` (`num_epochs`, `best_loss`) as its last action.
- **`scripts/promote_outputs.sh RUN_OR_DIR...`** rsyncs run dirs (paths relative to, or under,
  the scratch outputs root; a directory is searched for `train_config.json`) to
  `$STOCKYARD/vista/cross_embodied_diffusion/outputs/<same rel path>` (the repo `outputs`
  symlink's target). Skips runs without `train_done.json` unless `--force` (needed for runs
  trained before this item). `--no-epoch-ckpts` drops `policy_epoch_*.pt`; `-n` dry-runs.
  Refuses if the bytes to transfer exceed free `/work` quota (`lfs quota`).

Verified on a login node with a fake `$SCRATCH`: scratch and tmp staging (byte-identical copy,
dedupe, `$JOBTMP` cleanup), dry-run commands, and promote on fake runs (skip/force/exclude,
quota check against the real `/work`). A real 1-epoch CPU `train-diffusion` through the run root
is recorded in JOURNAL.md. Not verified: a real `sbatch` on a GPU node.

---

## 2026-09-28 — Vista multi-node jobs

### 54. `slurm_jobs/vista_train_manifest.sbatch` — multi-node mode

The `qgh` QOS allows 96 running nodes per user but only 20 running jobs, so 1-node jobs can
never use more than 20 nodes (ANALYSIS.md "Vista job shape"). Now, submitted with `-N K`, the
batch step re-runs the script under
`srun --nodes=K --ntasks-per-node=1 --kill-on-bad-exit=0 bash slurm_jobs/vista_train_manifest.sbatch --node`
(the repo path, not `$0`: sbatch runs a spool copy that exists only on the first node). Each
node computes `NODE_SLOT = SLURM_ARRAY_TASK_ID * SLURM_NNODES + SLURM_NODEID` and runs
manifest tasks `[NODE_SLOT*PACK, (NODE_SLOT+1)*PACK)`, so K nodes of array element `a` are
equivalent to 1-node array elements `a*K .. a*K+K-1`. With `-N 1` (the default) the slot is
the array index, exactly as before. `#SBATCH --ntasks=1` became `--ntasks-per-node=1`.
`JOBTMP` gets a `_$SLURM_NODEID` suffix. The header documents the limits and the submit form.
Verified by running the batch path on 4 nodes in an idev allocation: each node took the right
4 tasks (8 runs) and all 32 runs completed; numbers in ANALYSIS.md.

## 2026-09-27 (later) — TACC Vista (GH200, aarch64) support; torch.compile

### 50. `pyproject.toml`, `uv.lock` — CUDA torch on linux-aarch64

PyPI's linux-aarch64 `torch` wheels are CPU-only (`Torch not compiled with CUDA enabled` on
Vista's GH200). Added an explicit `pytorch-cu128` index (`download.pytorch.org/whl/cu128`) and a
`tool.uv.sources` entry routing `torch` to it only under
`sys_platform == 'linux' and platform_machine == 'aarch64'`. `torch==2.10.0` is now a direct
dependency because `tool.uv.sources` only applies to direct deps; it pins the version already
locked, and without the pin uv picked 2.11.0 on aarch64. Relocked with
`uv lock --default-index https://pypi.org/simple` (a plain `uv lock` rewrites every URL to the
Aliyun mirror); the x86_64 resolution is unchanged (PyPI `torch 2.10.0`).

### 51. `scripts/run_manifest_task.py` `--parallel` + multiple task ids; `slurm_jobs/vista_train_manifest.sbatch` — NEW

Vista bills whole nodes (1 GH200 each), so one run per node wastes most of it. Measured on
Grasp-Allegro 50k: 1 run 7.0 s/epoch, 2 concurrent 9.2 s/epoch each, 4 concurrent 15.7 s/epoch
each (1.8x node throughput); full sweep 1-12 runs in `vista_train_manifest.sbatch`'s header (plateau ~1.9x from 6 runs). `run_manifest_task.py` now takes several task ids and, with
`--parallel`, starts every run at once, each with its own `WARP_CACHE_PATH/run<i>` subdir.
Without `--parallel` it behaves as before, so the existing sbatch files are unaffected. The
new sbatch packs `PACK` (default 2) manifest tasks per node, uses node-local
`/tmp/$USER/$SLURM_JOB_ID` for `TMPDIR` and the Warp cache (nodes are exclusive, so neither the
shared-`/tmp` ENOSPC nor the AF_UNIX path-length problem applies), and needs no
`LD_LIBRARY_PATH` workaround. Verified by running its body on a Vista compute node: 4
concurrent runs (2 tasks x 2 seeds) with env eval each epoch, all four wrote every checkpoint.

### 53. `torch.compile` for training — `--compile-mode` (`train.py`, `cli/train_diffusion.py`, `run_manifest_task.py`, `vista_train_manifest.sbatch`)

Opt-in `TrainConfig.compile_mode` / `--compile-mode`: `policy.noise_pred_net.compile(mode=...)`
in place (state_dict keys unchanged, so checkpoints load in eager eval exactly as before; the
`randn`/`randint` draws in `compute_loss`/`predict_action` stay eager). Default is eager.
`run_manifest_task.py --compile-mode` adds it to every run; the Vista sbatch passes
`COMPILE_MODE` (default `reduce-overhead`, `none` = eager) and sets `CC=gcc`.

Measured on Vista GH200, Grasp-Allegro 50k, full manifest config, s/epoch per run:

| mode | 1 run | 8 runs/node | node plateau |
|---|---|---|---|
| eager FP32 | 7.1 | 29.9 | 0.267 run-epochs/s |
| eager TF32 (what production eager does after its first eval) | 7.0 | 28.4 | 0.282 |
| `default` | 5.5 | – | – |
| `max-autotune-no-cudagraphs` | 5.3 | – | – |
| **`reduce-overhead`** (CUDA graphs) | **3.1** | **21.8** | **0.368** |

The single-run eager GPU was mostly idle on kernel-launch overhead; CUDA graphs remove it.
Correctness: per-epoch train loss matches eager to ~1e-4 over 8 epochs (same seed); a
150-epoch Grasp-Allegro 50k run scored 56.6% vs eager's 53.1% on the same 256 episodes
(within noise), val action loss 0.089 vs 0.087.

Two fixes it needed:
- TACC's modules set `CC=nvc`; Triton builds its launcher stub with `$CC` and fails
  (`InductorError: CalledProcessError`). `CC=gcc` fixes it.
- In-training env eval/render calls mjlab's `configure_torch_backends()`, which sets TF32
  through the new `fp32_precision` API. After that, reading the legacy `allow_tf32` flag
  raises "mix of the legacy and new APIs", and Inductor's `pad_mm` reads it on recompile:
  every compiled run with `--eval-*` died on its first step after the first eval.
  `_restore_backend_flags()` saves and restores the matmul/cudnn precision and cudnn
  benchmark/deterministic flags around each eval and render, for compiled runs only. Side
  finding: **eager runs switch from FP32 to TF32 at their first in-training eval** (and
  cudnn.benchmark turns on), which is pre-existing behaviour left unchanged. Compiled runs
  stay FP32 throughout.

### 52. `scripts/hf_sync.py` — FIX: `pull` wrote the Hub's `.gitattributes` into the repo root

`pull` mapped every non-`demos/` file to `--rl-root` (the repo root), so the dataset repo's own
LFS `.gitattributes` (`*.bin`, `*.pt`, ... `filter=lfs`) landed in the repo as an untracked
file; committing it would have turned on LFS for checkpoints. Now only `demos/` and `logs/`
paths are pulled.

---

## 2026-09-27 — Backfill for 2026-09-03 → 09-27, data paths, docs restructure

Items 41-47 were made between 2026-09-03 and 2026-09-26 and were recorded only in
`JOURNAL.md` at the time; this section was reconstructed on 2026-09-27 from the commit diffs
(hashes given) and cross-checked against the journal entries named in each item.

### 41. `src/mjlab_hand/diffusion/train.py`, `cli/train_diffusion.py` — optional WandB logging (`7afd5a7`, 09-03)

`TrainConfig.wandb_project/wandb_run_name/wandb_tags` (CLI `--wandb-project`,
`--wandb-run-name`, `--wandb-tags`). Off by default (`wandb_project=None`). Logs per-epoch
`train/loss`, per-eval `eval/<task>/<metric>`, and a `best_loss` summary. Any array job that
enables it needs a per-task `TMPDIR` (see CLAUDE.md / AGENTS.md env block).

### 42. Padded cross-embodiment scheme — NEW (`e8e044d`, 09-10/11; JOURNAL 2026-09-10)

N embodiments of different obs/action width pooled into one dataset:
- `scripts/build_padded_dataset.py` — NEW. Per source: fit a static `GaussianNormalizer` on that
  source alone, normalize, zero-pad obs/action to the max width (real dims front-packed),
  concatenate whole episodes. Writes `extra.padded=True` and `extra.sources[i]`
  (`n_steps`, real `obs_dim`/`action_dim`, mean/std).
- `diffusion/normalizer.py` — `GaussianNormalizer` (`fit`, `identity`, state dict).
- `diffusion/dataset.py` — `TrajectoryStore.source_real_dims()`; `DiffusionDataset` precomputes a
  per-episode `action_mask` (1 on the source's real action dims) and yields it from `__getitem__`.
- `diffusion/policy.py` — `DiffusionPolicyConfig.normalizer_type` (`"linear"` default,
  `"gaussian"` = identity normalizers); `compute_loss(..., action_mask=)` averages squared error
  over real dims only.
- `diffusion/train.py` — padded datasets use identity normalizers (data is already normalized
  upstream; refitting on the pool would mix scales), write `source_stats.json` into the output
  dir, and **refuse `ambient_tmin` + padded** (`sample_ambient_batch` emits no `action_mask`).
- `diffusion/evaluate.py` — `EmbodimentStats.load(source_stats.json, embodiment)`; `embodiment=`
  argument on eval/render normalizes live obs with that embodiment's stats, pads, and
  un-normalizes/slices the predicted action. Mutually exclusive with `onehot`.

### 43. Held-out validation loss (`a41a689`, 09-11; JOURNAL 2026-09-11)

- `diffusion/dataset.py` — `_train_val_split`: deterministic, **by whole trajectory**, stratified
  per source; `DiffusionDataset(split=, val_fraction=, val_seed=)`. `val_fraction=0` is a no-op.
- `diffusion/policy.py` — `action_reconstruction_loss`: full DDIM `predict_action` vs expert
  action in real units (respects `action_mask`), not the one-step noise loss.
- `train.py` / CLI — `--val-fraction` (default 0), `--val-seed`, `--val-every-epochs`,
  `--val-max-batches` (default 20; -1 = all). Normalizers are fit on the train split only.

### 44. Balanced per-source sampling + manifest runner (`ecd421f`, 09-12/13; JOURNAL 2026-09-12)

- `diffusion/dataset.py` — `source_id_per_window`, `source_sample_weights(mode)`.
- `train.py` / CLI — `--source-sample-mode {uniform,balanced}`; `balanced` uses a seeded
  `WeightedRandomSampler` giving each source equal total weight (for the scarce co-training
  pools). `uniform` is the old shuffling path, unchanged.
- `scripts/build_scarce_specialist_manifest.py` — NEW; writes the 80-run specialist/scarce
  manifest (2 runs packed per Slurm task).
- `scripts/run_manifest_task.py` — NEW; runs manifest[task_id]'s `train-diffusion` commands
  sequentially (`--dry-run` prints them).

### 45. Three-way checkpoint selection (`1fb9bf6`, 09-20/22; JOURNAL 2026-09-20/22)

`train.py` no longer writes `policy_best.pt` (it was selected by *training* loss and often never
saved). Instead: `policy_latest.pt` (unchanged); `policy_best_val.pt` + `best_val.json` (lowest
`val/action_loss`, only when `--val-fraction > 0`); `policy_best_eval.pt` + `best_eval.json`
(highest mean env-eval headline over `eval_specs`, copied from the weights just scored). Runs
trained before this change still have `policy_best.pt` on disk. Same commit:
`build_scarce_specialist_manifest.py` eval cadence `epochs // 4` -> `epochs // 10`, and
`--family`/`--kind`/`--out` arguments.

### 46. `scripts/prototype_vmap_seeds.py`, `scripts/prototype_torch_compile.py` — NEW (`1fb9bf6`; JOURNAL 2026-09-13)

Standalone throughput experiments; they monkeypatch in-process and change no library code.
Result: vmap-over-seeds and AMP give nothing at `batch_size=256` on an A40; `torch.compile` of
`noise_pred_net` gives 1.34x on `compute_loss` and 2.02x on `predict_action`. **Not wired into
`train.py`** — the JOURNAL entry has the proposed integration point and the constant-batch-shape
caveat.

### 47. `.gitignore` — ignore `tmp/` (`ed33c20`, 09-27)

Per-task Slurm `TMPDIR` scratch (`tmp/<job>_task_N`, 26k+ files on NFS) was untracked but not
ignored, so `git status`/`git add` walked it for minutes.

### 48. Repo-relative data paths + `scripts/hf_sync.py` — NEW (`75257e9`, 09-27; COLLECTIONS "Hugging Face mirror")

Every `/datastor2/mrudolph/mjlab_hand_demos` in `slurm_jobs/*.sbatch` and
`scripts/build_scarce_specialist_manifest.py` (and the untracked manifests) became
`data/mjlab_hand_demos`, a per-machine symlink. `scripts/hf_sync.py stage|push|pull` mirrors the
demo store (one tar per zarr) plus the source RL experts (from each 1M zarr's `checkpoint`
attribute) to the private HF dataset repo `maxrudolph/mjlab-hand-demos`. README "Data layout"
documents new-machine setup.

### 49. Agent docs restructure — `AGENTS.md` NEW; `CLAUDE.md`, `.cursor/rules/agent-logbook.mdc`, `agent_logbook/README.md` (09-27)

Documentation only. `AGENTS.md` is now the single tool-neutral source (repo summary, env vars,
data paths, logbook protocol covering all five log files, invariants, standing findings).
`CLAUDE.md` imports it (`@AGENTS.md`) and the Cursor rule points to it, so the two can no longer
drift. Fixed `CHANGES.md`/`ANALYSIS.md` links that assumed the logbook files sat beside them.
Supersedes item 40's description of `CLAUDE.md`'s contents.

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
see the verification note in [`agent_logbook/JOURNAL.md`](agent_logbook/JOURNAL.md) 2026-08-31.

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
