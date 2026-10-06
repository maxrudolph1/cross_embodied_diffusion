# Agent log book

> **How to update this log book.** Do this whenever the user says "update the log books", and at
> the end of any session that changed code, data, jobs or results. Do it in the same turn, without
> asking. (1) Add one entry for the session's work at the **top** of [Entries](#entries), because
> entries are listed newest first. Number it one higher than the current highest `A<N>`, never
> renumber, and date it `YYYY-MM-DD`. Start it with `<a id="aN"></a>` and the heading
> `### AN — YYYY-MM-DD — <title>`. Give it **Request** (what the user asked), **Done**
> (implementation, commands, paths, configs, install or environment steps, with enough detail that
> someone else could reproduce it), **Verified / not verified**, and **Pointers** (the
> [RUNS.md](RUNS.md) job IDs, [CHANGES.md](CHANGES.md) item numbers,
> [COLLECTIONS.md](COLLECTIONS.md) datasets and [experiment log book](experiment_log_book.md)
> entries `E<N>`). Put the detail of each code edit in a new numbered CHANGES.md item rather than
> here. (2) Add a row at the top of the [Table of contents](#table-of-contents): number, date, and a
> **single-sentence** description. (3) Rewrite [Current state](#current-state) so that a new agent
> knows what is running, what has finished, and what to do next. (4) Update the other docs it
> points to: every Slurm job launched or finished goes in `RUNS.md`, together with the exact
> command that recreates it; every code change gets a `CHANGES.md` item; every dataset built goes
> in `COLLECTIONS.md`; every result, comparison or new idea gets an entry in
> `experiment_log_book.md`, with its plots copied into `docs/plots/` and that experiment log book's
> own update steps followed. Record facts, not plans: do not invent results, and say what was not
> verified. Entries A1–A37 were rebuilt on 2026-10-04 from the old `agent_logbook/JOURNAL.md`,
> which is kept verbatim in [archive/](archive/JOURNAL_2026-08-22_to_2026-10-04.md) if you need
> the full original narrative.

**New session? Read [Current state](#current-state) and the top 3–5 entries, then [RUNS.md](RUNS.md)
for anything still running.** The rules for agents are in [`../AGENTS.md`](../AGENTS.md).

---

## Current state

*Last updated 2026-10-06 (A40).*

- **Repo location:** `/work/09312/rudolph/documents/cross_embodied_diffusion` (`~/documents/cross_embodied_diffusion`),
  branch `bundle-migration`. `~/cross_embodied_diffusion` is a compatibility symlink to it on Vista.
  The `/work` checkout is now level with `origin/bundle-migration` (A39 pulled the A38 docs restructure; `/work`
  was writable again on 2026-10-05).
- **Queued:** job `1052839` (submitted 2026-10-06 10:27), the rotation σ fine-grid × 3-seed sweep, 150 runs incl. σ 0/100 seeds 1–2
  ([E19](experiment_log_book.md#e19); exact command in the [RUNS](RUNS.md) row; ~8.7 h per run after the queue wait).
  When it finishes: `sbatch -A ASC26008 --array=0-14 --export=ALL,RUNS='diffusion/ambient_ta_r/*',WHICH=best_val
  slurm_jobs/vista_eval_checkpoints.sbatch` (already-scored runs are skipped), then `scripts/plot_ambient_sigma.py`.
- **Nothing else of this project is running.** The latest finished job, `1049683` (rotation ambient σ sweep, 55 runs), has
  finished and been re-scored: [E18](experiment_log_book.md#e18). Earlier: `1043385` / `1045839`
  ([E14](experiment_log_book.md#e14)–[E16](experiment_log_book.md#e16)).
- **Headline results:**
  - Co-training on all five hands, then fine-tuning on the 50k target, beats both co-training alone and
    target-only training in both families ([E15](experiment_log_book.md#e15)).
  - Rotation, ambient gating: σ 10–20 is best (mean 0.85 successes before drop vs 0.64 target-only, 0.43 full
    co-training; one seed) ([E18](experiment_log_book.md#e18)).
  - **Never evaluate at env seed 0:** the 1M demos were collected at seed 0, and the old `subsets_50k/` are the
    first ~100 episodes of the 1M stores. Target-only 50k grasp memorizes its ~100 starts
    ([E17](experiment_log_book.md#e17)). Reporting evals use seed 1234.
- **Next steps / open items**
  1. **Promote kept runs before `$SCRATCH` purges them:** `scripts/promote_outputs.sh diffusion/ambient_ta`,
     `diffusion/ambient_ta_ft`, `diffusion/ambient_ta_r` (135 runs, ~0.8 GB+ each; `/work` was at 707 GB of
     1 TB on 2026-10-04).
  2. Follow-ups suggested by E18, not yet decided by the user: more seeds (σ 0, 10, 20, 100), a finer σ grid
     in 1–20, a grasp sweep on random-draw subsets (grasp `50kr` subsets/pools not built yet), and fine-tuning
     from the σ 10–20 runs.
  3. Runs on the old `subsets_50k/` / `_K50k` stores have inflated first in-training evals and `best_rollout`
     (E17). Use `best_val` / `last0` for them, and `_K50kr` stores for new rotation work.
  4. Open discrepancy: under our frozen min/max, the normalized action std is 0.24/0.26, against Bundle's
     quoted 0.077 (CHANGES item 63). σ values may not map 1:1 onto Bundle's.
  5. `$SCRATCH/ced_docs` (the A38 clone used while `/work` was read-only) can be deleted.
  6. `main` was fast-forwarded to `bundle-migration` (`a72da5a`) on 2026-10-06 (A40).

## Environment & install (living reference)

- **Systems:** TACC Vista (GH200, aarch64; partition `gh`, project `ASC26008`) for all current work. The
  original x86 Slurm cluster (A40/H200, `slurm-node-*`, `/datastor2`) was used up to 2026-09-27.
- **Layout on Vista:** repo at `$WORK/documents/cross_embodied_diffusion`. Bulk storage lives in
  `$WORK/vista/cross_embodied_diffusion/`, symlinked in as `data/mjlab_hand_demos`, `logs`, `outputs`, and
  `.venv -> ~/.venvs/cross_embodied_diffusion -> $WORK/vista/cross_embodied_diffusion/venv`. Training
  outputs go to `$SCRATCH/cross_embodied_diffusion/outputs` (purged; promote with `scripts/promote_outputs.sh`).
- **Install:** `uv sync`. On aarch64 this needs the cu128 torch index pinned in `pyproject.toml`/`uv.lock`
  (CHANGES item 50). If the Aliyun mirror fails TLS, add `--default-index https://pypi.org/simple`.
  `huggingface_hub` is installed separately:
  `uv pip install --index-url https://pypi.org/simple huggingface_hub`.
- **The venv contains absolute paths**: script shebangs, `activate`, and the editable `mjlab_hand.pth` →
  `<repo>/src`. If the repo moves, rewrite them, as in A38, or `uv sync` again.
- **Data:** `uv run python scripts/hf_sync.py pull` restores the demos and RL experts from the private HF dataset
  `maxrudolph/mjlab-hand-demos` (README "Data layout", [COLLECTIONS.md](COLLECTIONS.md)). The term-aligned
  multi-hand stores are CPU builds: `scripts/padded_grid.py --family F --config C --build`. Val stores:
  `slurm_jobs/vista_collect_val.sbatch`, then `scripts/build_val_split.py`. Frozen normalizers:
  `configs/norm_{grasp,rotation}_minmax.json` (`scripts/build_family_normalizer.py`).
- **Jobs:** the recipe for every Slurm job, and the exact command that recreates it, is in [RUNS.md](RUNS.md).
  Agents in idev cannot `sbatch`, so they hand the user the exact command.

---

## Table of contents

| # | Date | Summary |
|---|---|---|
| [A40](#a40) | 2026-10-06 | Fast-forwarded `main`, benchmarked run packing on the current code (PACK=2 is fastest overall), and set up the 150-run rotation σ fine-grid × 3-seed sweep (+ σ 0/100 seeds 1–2). |
| [A39](#a39) | 2026-10-04 → 10-06 | Found that the demos were collected at seed 0 and the 50k subsets are prefixes (target-only grasp memorizes), rebuilt random-draw rotation subsets, ran and plotted the rotation σ 0–100 sweep (`1049683`). |
| [A38](#a38) | 2026-10-04 | Moved the repo to `~/documents/`, fixed the venv paths, and restructured all docs into `docs/` (agent and experiment log books, RUNS, plots). |
| [A37](#a37) | 2026-10-04 | Re-scored the 40 fine-tunes and cross-evaluated every generalist on all 5 hands: co-train then fine-tune wins on the target, and lr 1e-4 forgets the other hands. |
| [A36](#a36) | 2026-10-03 | Fresh-seed re-score of co-train vs target-only (`1043385`): grasp +0.38..+0.44, rotation −0.04..−0.15; fine-tunes `1045839` queued. |
| [A35](#a35) | 2026-10-03 | Job `1043385` finished 40/40 runs, and the provisional in-training evals replicate the grasp co-training gain. |
| [A34](#a34) | 2026-10-02 | Handoff: co-train vs target-only job `1043385` queued, with the step-by-step plan for when it finishes. |
| [A33](#a33) | 2026-10-02 | Merged the migration worktree back in; the main checkout now runs `bundle-migration`. |
| [A32](#a32) | 2026-10-01 | Re-implemented Bundle's design (term-aligned padding, frozen normalizer, noise-first, val store) from MIGRATION.md and verified it on CPU/GPU. |
| [A31](#a31) | 2026-10-01 | Ambient rotation sweep `1038730` finished 320/320: co-training works only for the column-aligned Allegro/LEAP pair. |
| [A30](#a30) | 2026-10-01 | Reporting evals of the 4 ambient minmax test runs: target improves with σ, and the gated LEAP control drops to 0.00 at σ=10. |
| [A29](#a29) | 2026-09-30 | Measured queue waits by job size, chose the 10×8-node sweep shape, and wrote the scheduling procedure into AGENTS.md. |
| [A28](#a28) | 2026-09-30 | Ambient minmax test runs `1036217` finished, so the pipeline works end to end; added the GPU eval sbatch. |
| [A27](#a27) | 2026-09-30 | `--source-norm minmax` makes padded runs match plain ones, so the sweep manifest switched to minmax. |
| [A26](#a26) | 2026-09-30 | Diagnostic: the padded path (Gaussian normalizer + clamp), not pooling, caused the zero scores of pooled runs. |
| [A25](#a25) | 2026-09-29 | Designed the 320-run ambient rotation sweep, three-checkpoint reporting and a diagnostic to run first. |
| [A24](#a24) | 2026-09-29 | Added in-memory multi-hand pooling (`--dataset A B ...`), bitwise-checked against the prebuilt padded zarrs. |
| [A23](#a23) | 2026-09-29 | The Vista sbatch now stages demos to `$SCRATCH`//tmp and writes outputs to `$SCRATCH`, with a promote script. |
| [A22](#a22) | 2026-09-28 | Multi-node Vista jobs plus a job-shape benchmark; a `$WORK` quota incident killed checkpoints. |
| [A21](#a21) | 2026-09-27 | Set up TACC Vista: CUDA torch on aarch64, data pulled from HF, packed runs per node, `torch.compile`. |
| [A20](#a20) | 2026-09-27 | Fixed git slowness, mirrored the demos and RL experts to Hugging Face, made data paths repo-relative, created AGENTS.md. |
| [A19](#a19) | 2026-09-26 | Relaunched the rotation specialists as `97798` with a shorter TMPDIR. |
| [A18](#a18) | 2026-09-26 | Audit: rotation specialist backfill `94864` produced nothing (AF_UNIX path too long). |
| [A17](#a17) | 2026-09-22 | Audited the scarce/specialist sweep, added three-way checkpoint selection, launched the rotation specialist backfill. |
| [A16](#a16) | 2026-09-13 | Prototyped vmap-over-seeds, bf16 and `torch.compile`: only compile helps at batch 256. |
| [A15](#a15) | 2026-09-13 | Fixed job `89262` (36/40 tasks dead from node-local /tmp exhaustion) with a per-task TMPDIR and resubmitted it as `90599`. |
| [A14](#a14) | 2026-09-12 | Built fresh 50k sets and scarce co-training pools, added balanced sampling, launched the 80-run specialist + scarce sweep. |
| [A13](#a13) | 2026-09-11 | Added a real validation loss: held-out episodes, DDIM-denoised action MSE. |
| [A12](#a12) | 2026-09-10 | Built 50k-scale pooled padded datasets and launched their training. |
| [A11](#a11) | 2026-09-10 | New cross-embodiment scheme: pad to the family max, static per-source normalization, masked loss. |
| [A10](#a10) | 2026-09-09 | Pinned the remaining grasp RL training to node-011 after finding more stale-driver nodes. |
| [A9](#a9) | 2026-09-09 | Root cause of the 7–14× RL slowdown: stale GPU driver on slurm-node-004. |
| [A8](#a8) | 2026-09-09 | Lesson: `train`'s printed elapsed time and ETA are wrong, so estimate from checkpoint mtimes. |
| [A7](#a7) | 2026-09-07 | Launched the RL expert completion pass for the 6 missing task/hand combos (`84348`). |
| [A6](#a6) | 2026-09-02 | First Slurm-trained BC rotation policies (LEAP 2.31, Allegro 1.62 successes before drop). |
| [A5](#a5) | 2026-09-01 | Fixed an ambient-diffusion sampling-order bug that starved low-noise training. |
| [A4](#a4) | 2026-09-01 | Rebuilt lost code from the logbook prose and verified that the RL pipeline still trains on GPU. |
| [A3](#a3) | 2026-08-25 → 08-31 | (Old cluster) DDIM sampler fix, data-scale tooling, mixed/ambient pipelines, separability and equivalence analyses. |
| [A2](#a2) | 2026-08-24 | Created the agent logbook and fixed the rotation-success filter in demo collection. |
| [A1](#a1) | 2026-08-22 → 08-23 | Install, first RL experts, first demos and the diffusion pipeline. |

---

## Entries

<a id="a40"></a>
### A40 — 2026-10-06 — `main` fast-forwarded; packing benchmark; σ fine-grid sweep set up

**Request.** Fast-forward `main` to `bundle-migration`. Set up ambient runs for rotation, 5 targets, 3 seeds,
σ ∈ {1,2,3,6,9,12,15,18,20}, with Slurm parameters chosen so everything finishes as fast as possible.

**Done.**
- `git push origin bundle-migration:main` (`2fa899e..a72da5a`, fast-forward checked with
  `git merge-base --is-ancestor`); local `main` moved to `a72da5a`.
- Manifest `slurm_jobs/ambient_rot_fine_manifest.json` (gitignored): `scripts/build_ambient_manifest.py
  --families InHand-Rotation --size 50kr --sigmas 1 2 3 6 9 12 15 18 20 --seeds 0 1 2 --root
  outputs/diffusion/ambient_ta_r`, then the 5 runs whose output dirs already exist (σ20 seed 0, from
  `1049683`) removed and the rest sorted by (seed, hand, σ): 130 runs. Recipe differs from E18 only in
  `ambient-tmin` and `output-dir`. DRY_RUN of node slots 0 and 64 at PACK=2: tasks 0 1 and 128 129, correct σ,
  seed and tmin order.
- On request, added the baselines σ 0 and 100 × seeds {1, 2} (20 runs, same builder flags, `--sigmas 0 100
  --seeds 1 2`) to the same manifest and re-sorted: 150 runs, all output dirs unique and new. DRY_RUN: slot 74
  = tasks 148 149 (Wuji σ20/σ100 seed 2), slot 75 exits with "no manifest tasks".
- **Packing benchmark** (idev c634-142, GH200, otherwise idle): the E18 rotation recipe on
  `padded_ta/InHand-Rotation_pad5_all_10k.zarr`, no in-training eval/val, 20 epochs (196 steps each), K
  copies run concurrently through `scripts/run_manifest_task.py --parallel`, timed between epochs 5 and 20 from
  per-line timestamps: K=1 33.4 steps/s per run; K=2 26.2 (52.5 per node); K=4 15.4 (61.6 per node). K=1 matches
  the full runs of `1049683` (~34 steps/s), so the small store is a fair proxy. Estimated full-run time
  (784k steps + 10 evals): 6.8 h / ~8.7 h / ~14.7 h.
- Shape (RUNS row): PACK=2 × 5 nodes × 15 jobs = 75 nodes in one wave, `-t 12:00:00`. PACK=1 would need 150
  nodes, i.e. two waves under the 96-node cap (~13.6 h + waits); PACK=4 one wave but ~14.7 h.
  `queue_wait_stats.sh gh 3`: median waits 10–16 h for every size 1–16 nodes.

**Verified / not verified.** Verified: the ff push, manifest contents, dry-run slot mapping, the benchmark.
Not verified: a full-length PACK=2 run (the 8.7 h is extrapolated, hence the 12 h limit); `sbatch --test-only`
projections (need a login node).

**Pointers.** RUNS: `1052839` · Experiments: [E19](experiment_log_book.md#e19) · AGENTS.md "Running
things" packing numbers updated.

<a id="a39"></a>
### A39 — 2026-10-04 → 10-06 — Eval-seed/memorization check, random-draw 50k rotation subsets, rotation σ sweep

**Request.**
1. Review the repo and plan new experiments now that co-training and fine-tuning are in.
2. Plan ambient σ runs at 10, 20, …, 90 for every target hand, 1 seed, `best_val` checkpoint.
3. The user wants general policies: find out whether the 50k policies memorize, find the collection seed
   (HF repo) or re-collect.
4. Rebuild the datasets, rerun σ 0 and 100 as well to confirm, rotation only.
5. When the jobs finish, plot performance per hand across σ for every metric, push, and update the log books.

**Done.**
- **Review finding:** the target-only grasp runs' first in-training eval (epoch 5) scored 0.56–0.84 and
  every later one ~0.3–0.4; the re-scored `best_rollout` (that epoch-5 checkpoint) was ~0.35. Re-evaluating
  it: seed 0 → 0.91, seed 1234 → 0.38, seed 7 → 0.46. In `evaluate.py` the first in-training eval builds a
  fresh env seeded with the train seed and later evals reuse it (`_ENV_CACHE`) without reseeding.
- **Collection seed:** not in the store attrs, the HF mirror (store tarballs + RL `params/*.yaml` only) or the
  sbatch files (`collect_1M*.sbatch` pass no `--seed`, so the default 0). Confirmed from the data
  (start-state nearest-neighbour matching, scratch script `start_overlap.py`, not tracked): seed 0 = 100%
  exact matches, seeds 1234/7 = 0%. `subsets_50k/` = the first ~100 episodes of each 1M store (bitwise).
  Rotation prefixes are biased to short episodes. Memorization evals (12 policies, seed 0 vs 1234):
  [E17](experiment_log_book.md#e17). Re-collection judged unnecessary (seed known, reporting seed disjoint).
- **Random-draw subsets (CHANGES 70):** `subsample_dataset.py --random-seed`, the `50kr` config in
  `padded_grid.py`, `--size` in `build_ambient_manifest.py`. Built the 5 rotation `subsets_50kr/` stores and 5
  `padded_ta/InHand-Rotation_pad5_scarce<Hand>_K50kr.zarr` pools ([COLLECTIONS](COLLECTIONS.md)).
- **Sweep:** manifest `slurm_jobs/ambient_rot_rand_manifest.json` (55 runs; differs from `1043385`'s recipe only
  in dataset and output dir), submitted by the user as `1049683` ([RUNS](RUNS.md)). The first 90-run plan
  (`slurm_jobs/ambient_sweep10_manifest.json`: σ 10–90, both families, old `_K50k` stores) was superseded by
  this and never submitted.
- **Scoring/plots:** a background waiter on idev c634-142 re-scored best_val for all 55 runs once every
  `selection.json` existed (4 shards, 01:43–03:06), then ran the new `scripts/plot_ambient_sigma.py`
  (CHANGES 71). Figures and CSV copied to `docs/plots/ambient_sigma_rotation{,_metrics}.png`,
  `ambient_sigma_rotation_summary.csv`. Results: [E18](experiment_log_book.md#e18).
- **Docs:** this session first wrote the old-layout logbook (`agent_logbook/`, `ANALYSIS.md`); the push was
  rejected because A38 had restructured the docs. That commit is kept on branch
  `backup/sigma-sweep-9b86259`, and its content was redone in the `docs/` layout (CHANGES renumbered
  69/70 → 70/71 because A38 took 69).

**Verified / not verified.**
- Verified: default `subsample_dataset.py` output is bitwise unchanged; `check_padded_dataset.py --rows 1000`
  OK on all five `_K50kr` pools; dry-run of manifest slots 0 and 54; 11/11 jobs COMPLETED, 55/55
  `selection.json`, 55 best_val rows, no tracebacks.
- Not verified: the categorical palette with the dataviz validator (node is not installed on Vista); seed
  variance of the E18 sweep (one seed); grasp on random-draw subsets.

**Pointers.** RUNS: `1049683`, idev rescore row · CHANGES 70, 71 · COLLECTIONS: 2026-10-05 section,
`subsets_50kr/`, `_K50kr` pools · Experiments: [E17](experiment_log_book.md#e17),
[E18](experiment_log_book.md#e18), update note on [E14](experiment_log_book.md#e14).

<a id="a38"></a>
### A38 — 2026-10-04 — Repo moved to `~/documents/`; docs restructured into `docs/`

**Request.** Move `cross_embodied_diffusion` to `~/documents/`. Restructure all project docs and agent
files into an agent log book, an experiment log book (with embedded plots) and a RUNS.md that records every
Slurm job and how to recreate it, all under `docs/`. Then push.

**Done.**
- Move: `mv /work/09312/rudolph/code/cross_embodied_diffusion /work/09312/rudolph/documents/` (a rename on
  the same filesystem; the now-empty `code/` dir was removed). `~/cross_embodied_diffusion` was re-pointed
  to `~/documents/cross_embodied_diffusion` so that old commands keep working. Stampede3's `$HOME` was not touched.
- Venv fix: the Vista venv held 72 files with absolute paths to the old location. Shebangs and
  `activate*` used `/home1/09312/rudolph/cross_embodied_diffusion`; `mjlab_hand.pth` and
  `direct_url.json` used `/work/.../code/cross_embodied_diffusion`. They were rewritten **in place** to
  the new paths (a Python read/replace/truncate; `sed -i` failed, see below). Verified: `.venv/bin/train --help`
  runs, and `import mjlab_hand` resolves to the new `src/`.
- **`/work` could not create files.** `lfs df -i /work` showed `work-MDT0000` at 100% inodes
  (filesystem-wide, not our quota: we use 1.99M of 3M files and 693 GB of 1 TB). Existing files can be
  edited in place, but `touch`, `mkdir` and `sed -i` fail with ENOSPC everywhere under
  `/work/09312/rudolph`. Because of that, the docs work was done in a clone at `$SCRATCH/ced_docs` and
  pushed from there. The `/work` checkout still has the old layout until it is pulled (see Current state).
- New docs layout (CHANGES item 69):
  - `docs/agent_log_book.md` (this file) and `docs/experiment_log_book.md` (E1–E16, rebuilt from the old
    JOURNAL and ANALYSIS) are new.
  - `docs/RUNS.md` was restructured: a "how to recreate" section, one row per Vista job with its exact
    `sbatch` line, and the legacy old-cluster tables kept below.
  - Moved: `CHANGES.md`, `MIGRATION.md`, `agent_logbook/COLLECTIONS.md` → `docs/`, and
    `media/figures/*` → `docs/plots/`.
  - Archived verbatim: `agent_logbook/JOURNAL.md` and `ANALYSIS.md` → `docs/archive/`.
  - Deleted: `agent_logbook/README.md`.
  - `AGENTS.md` now carries the new documentation map and logbook protocol, and points at `docs/`. The
    Cursor rule and `README.md` point to it.
  - Updated comments in the sbatch files and scripts that cited `agent_logbook/...` or `ANALYSIS.md`.
    Code comments that cite "CHANGES.md item N" now mean `docs/CHANGES.md`.

**Verified.** The venv works from the new path. Every relative link in the new docs was checked with a script
(all targets exist). Not verified: Stampede3 symlinks or venv (none found under `$WORK/stampede3`).

**Pointers.** CHANGES 69.

<a id="a37"></a>
### A37 — 2026-10-04 — Fine-tune results and cross-embodiment evals

**Done.**
- All 40 fine-tunes of `1045839` COMPLETED (each co-trained σ0 run, starting from its `policy_best_val.pt`,
  10 epochs on target-only gating, lr {1e-4, 1e-5}, 2 seeds).
- Re-scored 120/120 checkpoints on the user's gh idev (c619-132), 6 shards of `rescore_selected.py`
  (best_rollout / best_val / last0, 100 envs × 1500 steps, seed 1234) → `<run>/final_eval.jsonl`.
- Cross-embodiment evals: `scripts/eval_cross_embodiment.py --which last0 --skip-target --shard i/n`
  (new, CHANGES 68) on every co-trained (80) and fine-tuned (160) generalist, scored on all 5 hands
  → `<run>/cross_eval.jsonl`. 300 scores, finished 04:00.
- Plots: `scripts/plot_conditions.py` → `conditions_{overview,grasp,rotation}.png` and
  `cross_embodiment_{grasp,rotation}.png` + CSVs (copied to `docs/plots/`).
- Gotcha: a waiter that ran `pgrep -f rescore_selected` never fired, because the launching shell's own command
  line contained that string. Use a pid file instead.

**Pointers.** RUNS `1045839` + idev rescore/cross-eval rows; CHANGES 67, 68; [E15](experiment_log_book.md#e15),
[E16](experiment_log_book.md#e16).

<a id="a36"></a>
### A36 — 2026-10-03 — Co-train vs target-only: fresh-seed re-score

- Re-scored all 40 runs of `1043385` on the user's gh idev (4 shards, 13:54–16:41, 120/120, no errors). The
  duplicate batch eval job `1045799` was cancelled.
- Plots: `scripts/plot_cotrain_vs_target.py` (CHANGES 66) → `cotrain_vs_target_{overview,grasp,rotation}.png`
  + CSV. Also kept the provisional `ambient_rot_{by_hand,hands}.png` from the old tail-padded sweep.
- Queued the fine-tunes: `scripts/build_finetune_manifest.py --lrs 1e-4 1e-5 --out slurm_jobs/finetune_manifest.json`
  → job `1045839` (CHANGES 67 adds `--init-checkpoint`).

**Pointers.** [E14](experiment_log_book.md#e14); RUNS `1043385`, `1045839`.

<a id="a35"></a>
### A35 — 2026-10-03 — `1043385` done

All 10 array jobs COMPLETED (6h30–6h58 of 9 h); 40/40 runs have `selection.json`, with no errors. The
provisional in-training evals (32 envs) show grasp co-training gaining +0.43, which replicates Bundle's
+0.46. Rotation is about neutral (+0.075). See [E14](experiment_log_book.md#e14) for the final numbers.

<a id="a34"></a>
### A34 — 2026-10-02 — HANDOFF: co-train vs target-only queued

- `1043385` (`ct-vs-to`): 40 runs = {Grasp, InHand-Rotation} × 5 targets × σ {0 = co-train, 100 = target-only}
  × seeds {0,1}, on each target's `<Family>_pad5_scarce<Hand>_K50k` store, using the Bundle recipe. Array element a
  = one target; node i = task 4a+i. The queue median wait was ~10.5–11.3 h for every size from 1 to 16 nodes,
  so 1 run/node (~7 h) was chosen over packing (~15 h at 4/node).
- Run-time basis: rotation measured 7.5 min/epoch uncompiled; grasp speed had not been measured. The limit of
  a pending job can only be lowered (`scontrol update ... TimeLimit`).
- Prepared but on hold (the user said "don't focus on ambient σ not 0/100"):
  `diag_rot_ta_manifest.json` (Allegro × σ{0,2,100} × 3 seeds), the 320-run term-aligned sweep, and
  `cotrain_solo_manifest.json` (60 runs, superseded by 1043385's design).
- Open: the action std discrepancy (0.24/0.26 vs Bundle's 0.077), the old sweep 1038730 never re-scored (it
  needs a `vista-ambient-rotation` worktree), and the Rotation-Shadow val store has 19,541 steps (<20k).
- Session pattern: a Monitor loop (`squeue` state changes, error grep, hourly eval digest), re-armed every 30 min.

<a id="a33"></a>
### A33 — 2026-10-02 — Migration worktree consolidated

The migration had been built in a git worktree (`$WORK/code/ced-migrate`) so the running sweep was untouched.
The pending `vista-ambient-rotation` work was committed (`0687fc7`), the main checkout was switched to
`bundle-migration`, and the worktree was removed. To re-score old (pre-item-63) checkpoints:
`git worktree add ../ced-old vista-ambient-rotation`, symlink `data/mjlab_hand_demos`, `outputs`, `logs` and
`.venv` into it, submit `vista_eval_checkpoints.sbatch` from there (it puts that checkout's `src/` first),
then `git worktree remove`.

<a id="a32"></a>
### A32 — 2026-10-01 — Migrating to Bundle's design (MIGRATION.md), from the doc only

- The user pushed [MIGRATION.md](MIGRATION.md): make this repo ("Branch") behave like "Bundle" for diffusion
  training/model/eval. Bundle's code is unavailable, so everything was re-implemented from the doc on branch
  `bundle-migration`. Every change, plus the [reconstructed] choices and deviations, is in **CHANGES item 63**.
  Main deviations: the frozen normalizer is fit on the 1M stores, and the val stores come from fresh expert
  rollouts.
- Why: Bundle reports rotation scarce-50k σ0 0.655 → peak 1.137 at σ2, while our tail-padded, per-hand-normalized
  runs sat near 0.1 at σ0. Leading explanation: layout/normalization ([E13](experiment_log_book.md#e13)).
- Verified on CPU: store vs sources bitwise, frozen artifacts, padding plumbing and masked-loss gradients,
  noise-first sampler mass, the val-split collision filter, and manifest parsing. Verified on GPU (idev c610-072):
  padded eval on all 5 hands, env-vs-store column alignment (including a negative control), rescore, val stores
  built (0 collisions), and timing of 7.5 min/epoch uncompiled ≈ 7 h per 50-epoch run.
- Each sbatch now puts its own checkout's `src/` first on `PYTHONPATH`.

<a id="a31"></a>
### A31 — 2026-10-01 — Ambient rotation sweep `1038730` done

All 10 jobs COMPLETED (9h20–9h55 of 12 h); 320/320 runs. Provisional plots and the finding are in
[E13](experiment_log_book.md#e13): co-training works only for Allegro/LEAP, the one pair whose tail-padded
columns line up, which motivated A32. Never re-scored with fresh seeds.

<a id="a30"></a>
### A30 — 2026-10-01 — Reporting evals of the ambient minmax test runs (`1038574`)

`eval_checkpoints.py` (100 envs, seed 1000) on the 4 runs of `1036217`. Results are in
[E12](experiment_log_book.md#e12).

<a id="a29"></a>
### A29 — 2026-09-30 — Sweep job shape from queue evidence; scheduling procedure

`scripts/queue_wait_stats.sh gh 3` gave these median waits: 1 node 9.1 h, 2 nodes 2.6 h, 3–4 nodes 12.6 h,
5–8 nodes 5.8 h, 9–16 nodes 13.2 h. An earlier coarse "2–4 nodes = 2.8 h" recommendation was retracted. The
recommendation was 10 × 8-node jobs at PACK=4 (expected ~14 h to completion). The procedure is written into
AGENTS.md "Getting Vista jobs scheduled fast" (CHANGES 61).

<a id="a28"></a>
### A28 — 2026-09-30 — Ambient minmax test runs done; eval sbatch

`1036217` (Allegro target, 50k + 4×1M, minmax, σ 0/10/25/100, 1 run/node): all COMPLETED in 2h20–2h24,
and the pipeline works end to end. The session was on a CPU-only `gg` idev, so
`slurm_jobs/vista_eval_checkpoints.sbatch` (CHANGES 60) was written for the GPU evals.
[E12](experiment_log_book.md#e12).

<a id="a27"></a>
### A27 — 2026-09-30 — minmax fix verified; sweep manifest switched to minmax

Padded minmax now matches plain within noise ([E11](experiment_log_book.md#e11)).
`build_ambient_rotation_manifest.py` pooled runs pass `"source-norm": "minmax"`, and
`ambient_rot_manifest.json` was regenerated (320 runs). The user submitted `1036217`.

<a id="a26"></a>
### A26 — 2026-09-30 — Diagnostic: the padded path zeroed the pooled runs

On idev 1031788: plain specialists learn, while the same data through `--pool-sources` scores 0.00. Two causes:
the sampler's hard [-1,1] clamp, and the Gaussian normalizer's data scale. The fix was `--source-norm minmax`
(CHANGES 58). The Gaussian padded runs were killed, and the user was advised to cancel `1035261`.
[E11](experiment_log_book.md#e11).

<a id="a25"></a>
### A25 — 2026-09-29 — Ambient rotation sweep design

The user asked for ambient diffusion on mixed-embodiment rotation: per target hand, σ ∈
{0,1,2,3,4,5,6,8,10,12,14,16,18,20,25,100}, 4 seeds = 320 runs. σ convention: other hands are admitted at
t ≥ σ, where 0 = full co-training and 100 = target-only. The target has 50k and the others 1M each. Each run
reports three checkpoints (best in-training eval, best target val, latest), re-scored with fresh seeds.
Implemented in CHANGES 57. The diagnostic runs first.

<a id="a24"></a>
### A24 — 2026-09-29 — In-memory multi-hand pooling

`train-diffusion --dataset A.zarr B.zarr ...` pools in memory (`mjlab_hand.diffusion.pooling`, CHANGES 56).
On all 14 existing padded zarrs it is bitwise-identical to the prebuilt stores (refit stats differ by ≤2.4e-7).
Superseded by the term-aligned stores in A32 (pooling.py was deleted).

<a id="a23"></a>
### A23 — 2026-09-29 — Vista storage: stage demos, outputs on `$SCRATCH`

`vista_train_manifest.sbatch` stages only the datasets each node uses, to `$SCRATCH` (1-node) or node `/tmp`
(multi-node). Outputs go to `$SCRATCH/cross_embodied_diffusion/outputs`, and runs execute from a symlinked
run root so that paths stay repo-relative. `scripts/promote_outputs.sh` copies finished runs to `$WORK`
(CHANGES 55).

<a id="a22"></a>
### A22 — 2026-09-28 — Vista multi-node jobs, benchmark, `$WORK` quota scare

- Limits (`sacctmgr show qos qgh`): 20 running / 40 submitted jobs, 96 running nodes per user, 64 per job.
  `vista_train_manifest.sbatch` is multi-node aware with `-N K` (CHANGES 54).
  Benchmark: [E10](experiment_log_book.md#e10).
- Incident: the benchmark's first attempt wrote to `outputs/` on `$WORK`, which went over the 1 TB quota
  (`torch.save` "unexpected pos"). Rerun on `$SCRATCH`. On the user's instruction, deleted the epoch-5/10/15
  snapshots of job `1030560` (33 GB).

<a id="a21"></a>
### A21 — 2026-09-27 — TACC Vista set up as a second training server

- Bulk storage under `$WORK/vista/cross_embodied_diffusion/`, symlinked in. `hf_sync.py pull`: 85 files, 25 GB.
- `uv sync --frozen` installed CPU-only torch on aarch64, fixed with the cu128 index (CHANGES 50). Verified
  torch 2.10.0+cu128 + warp on GH200.
- Packing runs per node (CHANGES 51), `--compile-mode` (CHANGES 53; later removed in item 63). Verified that
  the RL expert eval (`scripts/eval_expert.py`) gives 100%, and that a 150-epoch Grasp-Allegro 50k policy gives
  59–84% depending on episode count. Headless EGL rendering works.

<a id="a20"></a>
### A20 — 2026-09-27 — Git slowness, HF mirror, repo-relative data paths, AGENTS.md

`tmp/` was added to `.gitignore` (26k untracked files made git hang). `scripts/hf_sync.py` mirrors the demos and
RL experts to the private HF dataset `maxrudolph/mjlab-hand-demos`. All data paths are now repo-relative with
per-machine symlinks. `AGENTS.md` became the single instruction source (CHANGES 47–49).

<a id="a19"></a>
### A19 — 2026-09-26 — Rotation specialists relaunched as `97798`

The shorter `TMPDIR=$PWD/tmp/rs_N` fixes the AF_UNIX 108-char limit. `--array=0-9%8`. As of 2026-09-27, tasks
0–7 COMPLETED with all three checkpoints. Wuji seed 1 was still running at that check, and its final outcome was not recorded. The evals were never tabulated.
[E9](experiment_log_book.md#e9).

<a id="a18"></a>
### A18 — 2026-09-26 — Audit: `94864` produced nothing

Tasks 0–7 hung 10–14 h on `OSError: AF_UNIX path too long` (the per-task TMPDIR was too long for the DataLoader
socket). Cancelled 2026-09-23. No rotation specialists existed at that point.

<a id="a17"></a>
### A17 — 2026-09-20/22 — Sweep audit, three-way checkpoint selection, rotation backfill

The scarce/specialist sweep had been cancelled on 09-14, not left pending. The grasp specialists were covered,
but rotation had 0/20 runs and scarce co-training trained nothing. New checkpoints `policy_latest.pt`,
`policy_best_val.pt` (DDIM action MSE) and `policy_best_eval.pt` replace the train-loss `policy_best.pt`
(CHANGES 45). Eval cadence changed to `epochs // 10`. Launched `94864`.

<a id="a16"></a>
### A16 — 2026-09-13 — Throughput prototypes: vmap, bf16, torch.compile

See [E8](experiment_log_book.md#e8). `scripts/prototype_vmap_seeds.py`, `prototype_torch_compile.py` (CHANGES 46).

<a id="a15"></a>
### A15 — 2026-09-13 — `89262`: node-local /tmp exhaustion

36/40 tasks died within seconds: wandb's service subprocess makes a temp dir under node-local `/tmp`, and
`/tmp` filled up. The fix is a per-task `TMPDIR` on shared storage (now in AGENTS.md). The failed indices were
resubmitted as `90599` (`--array=1,5-39%16`).

<a id="a14"></a>
### A14 — 2026-09-12 — Scarce co-training + specialist sweep (80 runs)

- `DiffusionDataset.source_sample_weights` (uniform/balanced) and `--source-sample-mode` (CHANGES 44).
- Datasets: 10 fresh 50k single-hand sets and 10 scarce pools (one hand at 50k, the others at 1M); see COLLECTIONS.
- Job `89262` (`train_scarce_specialist.sbatch`, `--array=0-39%20`): specialists (10 combos × {50k,1M} × 2
  seeds) and scarce co-training (2 families × 5 scarce hands × {uniform,balanced} × 2 seeds). It never
  completed (A15, A17).

<a id="a13"></a>
### A13 — 2026-09-11 — Real validation loss

Episode-level, per-source stratified train/val split (`--val-fraction`). `action_reconstruction_loss`
measures full DDIM sampling against the expert in real units. The normalizer is fit on the train split only
(CHANGES 43; replaced by the val store in item 63).

<a id="a12"></a>
### A12 — 2026-09-10 — 50k-scale pooled datasets

Each 1M set was subsampled to 10k, then pooled → `padded/{Grasp,InHand-Rotation}-AllHands_50k` (50,060 / 50,326
steps). Job `86050` (4000 epochs).

<a id="a11"></a>
### A11 — 2026-09-10 — Pad-to-max cross-embodiment scheme

`GaussianNormalizer` per source, `build_padded_dataset.py` (zero-pad at the tail to the family max), masked
loss, `CrossEmbodimentActionChunkPolicy` eval (CHANGES 42). `Grasp-AllHands` (5.0M steps, 189/28) and
`InHand-Rotation-AllHands` (4.9M, 89/22). Job `85823`. Later shown to fail for two reasons: misaligned columns
([E13](experiment_log_book.md#e13)) and normalization ([E11](experiment_log_book.md#e11)).

<a id="a10"></a>
### A10 — 2026-09-09 — RL training pinned to node-011

Nodes 002/003/004/005/007/008/009 have the stale 535 driver. Resubmitted the grasp experts as `85382`
(`train_rl_experts_node011.sbatch`), resuming with the *remaining* iteration budget, because resume adds
`max-iterations` on top of the loaded iteration. Lesson: `#SBATCH --output` must be on storage that compute
nodes can see.

<a id="a9"></a>
### A9 — 2026-09-09 — Stale driver on slurm-node-004

Driver 535 (CUDA 12.2) disables Warp conditional graphs, so grasp ran at 15–34 s/iter versus 2.5 s on node-011.
Check a node with `srun --jobid=<numeric JobId> --overlap nvidia-smi --query-gpu=driver_version --format=csv`.
Resubmitted as `85163` with node-004 excluded.

<a id="a8"></a>
### A8 — 2026-09-09 — Don't trust `train`'s printed elapsed time or ETA

The "Time elapsed" counter resets periodically, so its ETA was off by 10–25×. Estimate rate from two
checkpoints' mtimes and iteration numbers instead.

<a id="a7"></a>
### A7 — 2026-09-07 — RL expert completion pass (`84348`)

4/10 combos were already done; the other 6 (all grasp + Sharpa rotation) were restarted fresh with run-name
`slurm2` via `slurm_jobs/train_rl_experts.sbatch`. Mistake: it used `--agent.logger tensorboard`. Synced to
wandb afterwards with `wandb sync --sync-tensorboard`.

<a id="a6"></a>
### A6 — 2026-09-02 — First Slurm BC rotation policies

`dp-leap-rot-400k` (`80853`) and `dp-allegro-rot-400k` (`80854`). [E7](experiment_log_book.md#e7).

<a id="a5"></a>
### A5 — 2026-09-01 — Ambient sampling-order bug

Found by the user: sample the timestep first, then a valid tuple. Before this fix, sub-`t_min` training was
suppressed about 40×. `DiffusionDataset.sample_ambient_batch` was fixed (CHANGES 2026-09-01 section). This
means the 08-27 ambient sweep ([E4](experiment_log_book.md#e4)) used the buggy order.

<a id="a4"></a>
### A4 — 2026-09-01 — Reconstructed lost code; GPU verification

Code from another box existed only as logbook prose, and was rebuilt from CHANGES/ANALYSIS (DDIM sampler, ambient
gating, one-hot, multi-target eval, 19 scripts, 7 sbatch). Verified where possible (`check_sampler.py`
reproduced the documented MSEs). An RL "does it train" check was aborted at iter 64 because a done expert
already existed. Lesson: check RUNS.md first.

<a id="a3"></a>
### A3 — 2026-08-25 → 2026-08-31 — Old-cluster work (no journal entries; from CHANGES and ANALYSIS)

CHANGES items 1–40: the DDIM sampler fix (item 1), per-epoch eval and rendering, data-scale subsets and
`select_experts.py`, mixed-embodiment one-hot conditioning and the 40-cell grid, ambient per-source timestep
gating, separability/state-difference/state-equivalence scripts, and plot correctness fixes. Experiments
[E1](experiment_log_book.md#e1)–[E6](experiment_log_book.md#e6).

<a id="a2"></a>
### A2 — 2026-08-24 — Agent logbook added; collection fixes

Created `agent_logbook/` and the Cursor rule. CHANGES 9: `collect_demos` never filtered rotation success
(it now uses `RotationCommand.metrics["episode_success"]`).

<a id="a1"></a>
### A1 — 2026-08-22 → 2026-08-23 — Setup, RL training, demos, diffusion pipeline

- Install: `uv sync --group dev --default-index https://pypi.org/simple --system-certs`.
- RL experts: Grasp-Allegro on the local A40, and a Slurm array (`slurm_jobs/array_20260823_022315`) for the
  other 9 combos. Grasp-Shadow failed on the shared Warp cache; that is why the per-task `WARP_CACHE_PATH`
  exists.
- Diffusion pipeline (`collect-demos`, `train-diffusion`, `eval-diffusion`): the Allegro grasp smoke and full
  runs reached 0% success despite low loss. That was later explained by the invalid sampler (CHANGES 1).
