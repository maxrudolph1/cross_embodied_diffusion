# Slurm jobs (RUNS)

> **How to update this file.** Whenever a Slurm job (or an idev/interactive run that produces
> results) is launched, finishes, fails or is cancelled, and whenever the user says "update the log
> books", edit this file in the same turn, without asking. (1) Add one row per job at the **top** of
> the [job table](#job-table), newest first: job ID, the date submitted, a one-line purpose, the
> **exact submit command** (copy it from `sacct -X -j <id> -o SubmitLine%300`), the command that
> regenerates the manifest (manifests are gitignored), the output root, the status (`pending` /
> `running` / `done` / `failed` / `cancelled`, with dates and wall time), and links to the
> [experiment log book](experiment_log_book.md) entry `E<N>` and the [agent log book](agent_log_book.md)
> entry `A<N>`. (2) When the status changes, edit that row instead of adding a new one. (3) Record
> the evidence for the chosen job shape (queue stats, `--test-only`) in the row's notes. Narrative and
> debugging belong in the agent log book. Results belong in the experiment log book. This file records
> what ran and how to run it again.

## How to recreate a job (Vista)

All Vista training goes through one launcher plus a manifest. A manifest is a JSON list of tasks, and each
task is a list of `train-diffusion` argument dicts. Submit from a **login node** (sbatch is disabled in idev),
from the repo root:

```bash
cd ~/documents/cross_embodied_diffusion
mkdir -p slurm_jobs/vista_train_manifest/logs
# 1. regenerate the manifest (see the row's "manifest" column), e.g.
.venv/bin/python scripts/build_ambient_manifest.py --families Grasp InHand-Rotation \
    --sigmas 0 100 --seeds 0 1 --out slurm_jobs/cotrain_vs_target_manifest.json
# 2. dry run: stages data and prints the train-diffusion commands
sbatch -A ASC26008 -N 4 --array=0 --export=ALL,MANIFEST=<manifest>,PACK=1,DRY_RUN=1 slurm_jobs/vista_train_manifest.sbatch
# 3. submit with the row's exact command
```

- **Array sizing:** with `-N K` and `PACK` tasks per node, array element `a`, node `i` runs manifest tasks
  `[(a*K+i)*PACK, (a*K+i+1)*PACK)`. So `--array=0-(ceil(n_tasks/(K*PACK))-1)`.
- **Storage:** data is staged to `$SCRATCH` (or node `/tmp` when K>1). Outputs go to
  `$SCRATCH/cross_embodied_diffusion/outputs/<output-dir>`. Logs go to
  `slurm_jobs/vista_train_manifest/logs/task_<job>_<a>.{out,err}`. A run is complete when `selection.json`
  exists. Keep runs with `scripts/promote_outputs.sh <dir under outputs>`.
- **Reporting re-score** (fresh seeds; `<run>/final_eval.jsonl`):
  `sbatch -A ASC26008 --array=0-9 --export=ALL,RUNS='diffusion/<root>/*' slurm_jobs/vista_eval_checkpoints.sbatch`,
  or on a gh idev: `.venv/bin/python scripts/rescore_selected.py --run <dirs> --which best_rollout best_val last0
  --envs 100 --steps 1500 --eval-seed 1234 --shard i/n`.
- **Cross-embodiment eval** (`<run>/cross_eval.jsonl`):
  `.venv/bin/python scripts/eval_cross_embodiment.py --run <dirs> --which last0 --skip-target --shard i/n`.
- **Before a large submission**, follow AGENTS.md "Getting Vista jobs scheduled fast"
  (`scripts/queue_wait_stats.sh`, then `sbatch --test-only`).
- **Old-code jobs** (pre CHANGES item 63, i.e. `1030560`–`1038730`): their checkpoints do not load in the current
  code, and their manifest builders no longer exist on `bundle-migration`. Recreate them from a worktree of
  `vista-ambient-rotation` (commit `0687fc7`), as described in agent log [A33](agent_log_book.md#a33).

## Job table

Vista (TACC GH200, project ASC26008), newest first. Output roots are relative to
`$SCRATCH/cross_embodied_diffusion/outputs/` unless they say otherwise.

| Job | Submitted | Purpose | Exact submit command | Manifest (regenerate) | Output | Status | Log books |
|---|---|---|---|---|---|---|---|
| idev c639-092 (E21 evals) | 2026-10-08 10:08 | E21 evals run on the idev instead of waiting for the queued `1056925`–`1056932`: fine-tunes best_rollout/best_val/last0 at seed 4321, then 1234; fine-tunes best_val/last0 on the other 4 hands (seed 4321); sources σ 0/15/100 best_val on the other 4 hands (seed 4321). 8 processes on one GH200 | `slurm_jobs/vista_train_manifest/logs/rescore_e21/run.sh <p> 8` for p = 0..7 under `nohup` (gitignored wrapper: per stage `rescore_selected.py` / `eval_cross_embodiment.py --skip-target` with `--shard p/8`; logs `rescore_e21/p<p>.log`) | — | `ambient_ta_r_ft/*/final_eval.jsonl`, `cross_eval.jsonl`; `ambient_ta_r/*/cross_eval.jsonl` | running. The queued batch evals were left in the queue; both sides skip rows already written, so overlap only duplicates a few rows | [E21](experiment_log_book.md#e21), [A43](agent_log_book.md#a43) |
| `1056188` (train), `1056189` (ev-base4321), `1056925` (ev-basex), `1056926` (ev-ft1234), `1056927` (ev-ft4321), `1056932` (ev-ftx) | 2026-10-07 14:56 | Fine-tune the E19 rotation runs σ {0, 15, 100} × seeds {0,1,2} × 5 targets from `policy_best_val.pt`, target-only gating, lr 1e-5, 10 epochs, `_K50kr` stores: 45 runs. Plus evals: all 150 E19 runs best_val at seed 4321; E19 sources on the other 4 hands (seed 4321); fine-tunes best_rollout/best_val/last0 at seeds 1234 and 4321; fine-tunes best_val/last0 on the other 4 hands (seed 4321) | `bash slurm_jobs/submit_e21_finetune.sh` (login node). Train: `sbatch -A ASC26008 -p gh -N 5 --array=0-8 -t 02:30:00 --job-name=ft-amb-r --export=ALL,MANIFEST=slurm_jobs/finetune_r_manifest.json,PACK=1 slurm_jobs/vista_train_manifest.sbatch`; eval lines in the script (`vista_eval_checkpoints.sbatch` with `EVAL_SEED`/`CROSS`, `--dependency=afterany`) | `scripts/build_finetune_manifest.py --families InHand-Rotation --src-sigmas 0 15 100 --seeds 0 1 2 --lrs 1e-5 --size 50kr --src-root outputs/diffusion/ambient_ta_r --root outputs/diffusion/ambient_ta_r_ft --out slurm_jobs/finetune_r_manifest.json` | `diffusion/ambient_ta_r_ft/`; seed-4321 rows in `ambient_ta_r/*/final_eval.jsonl`, `cross_eval.jsonl` | **train `1056188` done** 2026-10-08 04:22–06:40, 9/9 COMPLETED, 45/45 runs with `selection.json`, 1h33–1h46 per job. **ev-base4321 `1056189` done** 04:40–06:29, 3/3 COMPLETED, 1h42–1h44 of the 1:45 limit (too tight; ~5.8 min/eval at 4 per GPU): best_val at seed 4321 for all 205 `ambient_ta_r` runs (E18 + E19; the glob took the whole dir). ev-basex/ev-ft1234/ev-ft4321 pending (Priority), ev-ftx (Dependency) at 2026-10-08 ~09:00. Submitted 2026-10-07 14:56. The script stopped at `ev-basex`: `--export` rejects the space-separated `RUNS` (fixed, CHANGES 75); the remaining 4 evals (ev-basex, ev-ft1234, ev-ft4321, ev-ftx) were submitted with `bash slurm_jobs/submit_e21_evals_rest.sh`, but only 3 were queued (`1056925`–`1056927`). `ev-ftx` was missing (cause unknown; 23 of the 40 job slots used) and was resubmitted on its own as `1056932`: `sbatch -A ASC26008 -p gh --array=0-5 -t 02:00:00 --job-name=ev-ftx --dependency=afterany:1056927 --export=ALL,RUNS='diffusion/ambient_ta_r_ft/*',WHICH=best_val:last0,EVAL_SEED=4321,CROSS=1 slurm_jobs/vista_eval_checkpoints.sbatch`. Shape: queue stats 2026-10-07 13:31 (7 d) show median waits ~10–13 h for every node-count bin, so shape barely changes the start; PACK=1 keeps the run short (~1h35, as `1045839`), 45 nodes + 18 in use ≤ 96, 25 queue slots + 4 ≤ 40. Eval sizing: ~5.5 min/eval at 4 per GH200 (idev rescore 2026-10-07). Dry run of slots 0/4 and 8/4 checked | [E21](experiment_log_book.md#e21), [A43](agent_log_book.md#a43) |
| idev rescore (c634-142 + c611-102) | 2026-10-07 | best_val re-score of the 150 new `ambient_ta_r` runs of `1052839` (100 envs x 1500 steps, seed 1234), 8 shards: 4 per GH200 on two idevs | `slurm_jobs/vista_train_manifest/logs/rescore_ambient_ta_r/run_shards.sh 0 3 8` on c634-142 and `... 4 7 8` on c611-102 (each runs `rescore_selected.py --run $SCRATCH/.../ambient_ta_r/* --which best_val --envs 100 --steps 1500 --eval-seed 1234 --shard i/8`; already-scored runs skipped) | — | `diffusion/ambient_ta_r/*/final_eval.jsonl` | done 2026-10-07 10:31–12:33, 205/205 rows, no errors | [E19](experiment_log_book.md#e19), [A42](agent_log_book.md#a42) |
| `1052839` | 2026-10-06 | Ambient σ fine grid, InHand-Rotation × 5 targets × σ {1,2,3,6,9,12,15,18,20} × seeds {0,1,2}, plus σ {0,100} × seeds {1,2}, on `_K50kr` stores: 150 runs (σ 0/20/100 seed 0 reused from `1049683`), Bundle recipe | `sbatch -A ASC26008 -p gh -N 5 --array=0-14 -t 12:00:00 --export=ALL,MANIFEST=slurm_jobs/ambient_rot_fine_manifest.json,PACK=2 slurm_jobs/vista_train_manifest.sbatch` | `scripts/build_ambient_manifest.py --families InHand-Rotation --size 50kr --sigmas 1 2 3 6 9 12 15 18 20 --seeds 0 1 2 --root outputs/diffusion/ambient_ta_r --out slurm_jobs/ambient_rot_fine_manifest.json`, then drop the 5 existing σ20 seed-0 runs; append `build_ambient_manifest.py ... --sigmas 0 100 --seeds 1 2` (same flags); sort by (seed, hand, σ) (A40) | `diffusion/ambient_ta_r/` | done 2026-10-07: 15/15 COMPLETED (8h32–9h03; started 10-06 18:37–19:26), 150/150 `selection.json`; best_val re-scored (idev row) |  Shape: 75 nodes in one wave (2 runs/node). Packing benchmark on this code (A40): ~8.7 h per run at PACK=2 vs 6.8 h alone and 14.7 h at PACK=4, so PACK=1 (130 nodes, 2 waves, ~13.6 h + waits) loses. Usage 10-06: 5 jobs / 19 nodes running → 15 jobs × 5 nodes fit the 15 free running-job slots and 77 free nodes exactly. Queue (`queue_wait_stats.sh gh 3`, 10-06): median start wait 10–16 h at every size 1–16 nodes. Limit 12 h = estimate +38% (extrapolated from the benchmark, not a measured full run) | [E19](experiment_log_book.md#e19), [A40](agent_log_book.md#a40) |
| idev rescore (c634-142) | 2026-10-06 | best_val re-score of the 55 `ambient_ta_r` runs (100 envs x 1500 steps, seed 1234), 4 shards on one GH200 | `python scripts/rescore_selected.py --run $SCRATCH/cross_embodied_diffusion/outputs/diffusion/ambient_ta_r/* --which best_val --envs 100 --steps 1500 --eval-seed 1234 --shard i/4` (i = 0..3, in parallel) | — | `diffusion/ambient_ta_r/*/final_eval.jsonl` | done 2026-10-06 01:43-03:06, 55/55 rows, no errors (logs `slurm_jobs/vista_train_manifest/logs/rescore_ambient_ta_r/`) | [E18](experiment_log_book.md#e18), [A39](agent_log_book.md#a39) |
| `1049683` | 2026-10-05 | Ambient σ sweep, InHand-Rotation × 5 targets × σ {0,10,…,100} × seed 0 on the random-draw `scarce<Hand>_K50kr` stores: 55 runs, Bundle recipe | `sbatch -A ASC26008 -p gh -N 5 --array=0-10 -t 09:00:00 --export=ALL,MANIFEST=slurm_jobs/ambient_rot_rand_manifest.json,PACK=1 slurm_jobs/vista_train_manifest.sbatch` | `scripts/build_ambient_manifest.py --families InHand-Rotation --size 50kr --sigmas 0 10 20 30 40 50 60 70 80 90 100 --seeds 0 --root outputs/diffusion/ambient_ta_r --out slurm_jobs/ambient_rot_rand_manifest.json` | `diffusion/ambient_ta_r/` | done 2026-10-06, 11/11 COMPLETED (6h39–6h59); submitted 10-05 01:42, started 16:51–18:48. Shape: 55 nodes + 34 already used/pending ≤ 96 cap, so all at once; queue median 3–5 h (`queue_wait_stats.sh gh 3`, 10-04 21:34) | [E18](experiment_log_book.md#e18), [A39](agent_log_book.md#a39) |
| idev c619-132 (cross-eval) | 2026-10-03/04 | Cross-embodiment eval of 80 co-trained + 160 fine-tuned generalists on all 5 hands | on the gh idev: `scripts/eval_cross_embodiment.py --run <dirs> --which last0 --skip-target --shard p/N` (4 shards co-train, 6 shards FT; logs `slurm_jobs/vista_train_manifest/logs/cross_{cotrain,ft}_idev_p*.log`) | — | `<run>/cross_eval.jsonl` | done 2026-10-04 04:00 | [E16](experiment_log_book.md#e16), [A37](agent_log_book.md#a37) |
| idev c619-132 (rescore FT) | 2026-10-03 | Fresh-seed re-score of the 40 fine-tunes | `scripts/rescore_selected.py --run diffusion/ambient_ta_ft/* --which best_rollout best_val last0 --envs 100 --steps 1500 --eval-seed 1234 --shard p/6` (logs `rescore_ft_idev_p*.log`) | — | `<run>/final_eval.jsonl` | done 2026-10-04 00:15 | [E15](experiment_log_book.md#e15), [A37](agent_log_book.md#a37) |
| `1045839` | 2026-10-03 | Fine-tune each co-trained σ0 run from `policy_best_val.pt`, target-only gating, lr {1e-4,1e-5}, 10 epochs: 40 runs | `sbatch -A ASC26008 -N 4 --array=0-9 --time=03:00:00 --job-name=ft-cotrain --export=ALL,MANIFEST=slurm_jobs/finetune_manifest.json,PACK=1 slurm_jobs/vista_train_manifest.sbatch` | `scripts/build_finetune_manifest.py --lrs 1e-4 1e-5 --out slurm_jobs/finetune_manifest.json` (needs the `ambient_ta` runs) | `diffusion/ambient_ta_ft/` | done 2026-10-03, 10/10 COMPLETED (~1h35 each) | [E15](experiment_log_book.md#e15), [A36](agent_log_book.md#a36) |
| `1045799` | 2026-10-03 | Batch re-score of `ambient_ta` (duplicate of the idev re-score) | `sbatch -A ASC26008 --array=0-9 --export=ALL,RUNS=diffusion/ambient_ta/* slurm_jobs/vista_eval_checkpoints.sbatch` | — | — | cancelled (the idev did it) | [A36](agent_log_book.md#a36) |
| idev c619-132 (rescore co-train) | 2026-10-03 | Fresh-seed re-score of the 40 `ambient_ta` runs | `scripts/rescore_selected.py --run diffusion/ambient_ta/* --which best_rollout best_val last0 --envs 100 --steps 1500 --eval-seed 1234 --shard p/4` (logs `rescore_ambient_ta_idev_p*.log`) | — | `<run>/final_eval.jsonl` | done 2026-10-03 16:41 | [E14](experiment_log_book.md#e14), [A36](agent_log_book.md#a36) |
| `1043385` | 2026-10-02 | Co-train (σ0) vs target-only (σ100), Grasp + Rotation × 5 targets × 2 seeds: 40 runs, Bundle recipe, term-aligned | `sbatch -A ASC26008 -N 4 --array=0-9 --time=09:00:00 --job-name=ct-vs-to --export=ALL,MANIFEST=slurm_jobs/cotrain_vs_target_manifest.json,PACK=1 slurm_jobs/vista_train_manifest.sbatch` | `scripts/build_ambient_manifest.py --families Grasp InHand-Rotation --sigmas 0 100 --seeds 0 1 --out slurm_jobs/cotrain_vs_target_manifest.json` | `diffusion/ambient_ta/` | done 2026-10-03, 10/10 COMPLETED (6h30–6h58) | [E14](experiment_log_book.md#e14), [A34](agent_log_book.md#a34) |
| idev 1037050 | 2026-10-02 | Collect val-store rollouts (10 tasks, 64 eps, seed 1000), then `build_val_split.py` | body of `slurm_jobs/vista_collect_val.sbatch` run directly; log `collect_val_idev1037050.log` | — | `data/mjlab_hand_demos/val/` | done | [A32](agent_log_book.md#a32) |
| `1038730` | 2026-09-30 | Ambient rotation sweep, tail-padded, 16 σ × 4 seeds × 5 targets: 320 runs (old code) | `sbatch -A ASC26008 -N 8 --array=0-9 --time=12:00:00 --job-name=amb-rot --export=ALL,MANIFEST=slurm_jobs/ambient_rot_manifest.json,PACK=4 slurm_jobs/vista_train_manifest.sbatch` | at `0687fc7`: `scripts/build_ambient_rotation_manifest.py --kind sweep --out slurm_jobs/ambient_rot_manifest.json` | `diffusion/ambient_rot/` | done 2026-10-01, 10/10 COMPLETED (9h20–9h55); never re-scored | [E13](experiment_log_book.md#e13), [A31](agent_log_book.md#a31) |
| `1038574` | 2026-10-01 | Reporting evals of the 4 ambient minmax test runs (+ LEAP control) | `sbatch -A ASC26008 --time=1:00:00 --export=ALL,RUNS=diffusion/diag_rot/*_minmax_seed0,ALSO=InHand-Rotation-LEAP slurm_jobs/vista_eval_checkpoints.sbatch` (old-code version of the sbatch) | — | `<run>/posthoc_eval.json` | done (23 min) | [E12](experiment_log_book.md#e12), [A30](agent_log_book.md#a30) |
| `1036217` | 2026-09-30 | Ambient minmax test, Allegro target, σ {0,10,25,100}, 1 run/node (old code) | `sbatch -A ASC26008 -N 1 --array=0-3 --time=8:00:00 --job-name=diag-amb-mm --export=ALL,MANIFEST=slurm_jobs/diag_rot_ambient_minmax_manifest.json,PACK=1 slurm_jobs/vista_train_manifest.sbatch` (limit later lowered to 5 h) | hand-edited from the diagnostic manifest; generator not recorded | `diffusion/diag_rot/*_minmax_seed0` | done 2026-09-30 (2h20–2h24) | [E12](experiment_log_book.md#e12), [A28](agent_log_book.md#a28) |
| `1036212` | 2026-09-30 | Same as `1036217`, PACK=4 single job | `sbatch -A ASC26008 -N 1 --time=24:00:00 --job-name=diag-amb-mm --export=ALL,MANIFEST=slurm_jobs/diag_rot_ambient_minmax_manifest.json,PACK=4 slurm_jobs/vista_train_manifest.sbatch` | — | — | cancelled (replaced by `1036217`) | |
| idev 1031788 | 2026-09-29/30 | Padded-path diagnostic: specialists plain vs `--pool-sources` (Gaussian), then minmax | on the idev: the sbatch body with `MANIFEST=slurm_jobs/diag_rot_idev_manifest.json`, then `diag_rot_idev2_manifest.json`; logs `idev_1031788_diag{,2,_posthoc}.log` | at `0687fc7`-era code (hand-built manifests) | `diffusion/diag_rot_idev/` | done (Gaussian runs killed) | [E11](experiment_log_book.md#e11), [A26](agent_log_book.md#a26) |
| `1035261` | 2026-09-29 | 8-run diagnostic (Gaussian padded) | `sbatch -A ASC26008 -N 2 --time=24:00:00 --job-name=diag-rot --export=ALL,MANIFEST=slurm_jobs/diag_rot_manifest.json,PACK=4 slurm_jobs/vista_train_manifest.sbatch` | at `0687fc7`: `build_ambient_rotation_manifest.py --kind diagnostic` | — | cancelled (superseded by the idev diagnostic) | [A26](agent_log_book.md#a26) |
| idev 1031790 (gh-dev, 4 nodes) | 2026-09-28 | Job-shape benchmark | scripts in `$SCRATCH/bench_multinode/` (purgeable) | — | `$SCRATCH/bench_multinode/` | done | [E10](experiment_log_book.md#e10), [A22](agent_log_book.md#a22) |
| `1030560` | 2026-09-27 | Scarce co-training sweep (Gaussian tail-padded pools), 40 runs, compiled | `sbatch -A ASC26008 --array=0-4 --export=ALL,MANIFEST=slurm_jobs/scarce_manifest.json,PACK=4 slurm_jobs/vista_train_manifest.sbatch` | `scripts/build_scarce_specialist_manifest.py` (deleted in item 63; `git show 1fb9bf6:scripts/build_scarce_specialist_manifest.py`) | `outputs/diffusion/scarce/` on `$WORK` | done (21h42); scores ~0, explained by [E11](experiment_log_book.md#e11) | [A22](agent_log_book.md#a22) |

Old-cluster jobs (2026-08-22 → 2026-09-27; partition `allnodes`, `slurm-node-*`) are below. They are kept
as originally recorded and are not runnable on Vista.

---

## Legacy registry — old cluster

### RL (rsl_rl / mjlab `train`)

| ID | Task | Run dir | Config | Latest ckpt | Status | Notes |
|----|------|---------|--------|-------------|--------|-------|
| rl-allegro-grasp-abort | Grasp-Allegro | `logs/rsl_rl/allegro_grasp/2026-08-22_10-32-05_initial` | 2048 envs, 10k iters, seed default, TB, run `initial` | `model_0.pt` | aborted | Stopped ~4 min in; superseded |
| rl-allegro-grasp | Grasp-Allegro | `logs/rsl_rl/allegro_grasp/2026-08-22_10-37-45_initial` | 2048 envs, 10k iters, seed 42, TB, run `initial`; local GPU A40 | `model_6600.pt` (~iter 6615/10000) | running / near-complete | Console: `logs/grasp_allegro_train.log`. Source of Allegro demos |
| rl-leap-grasp | Grasp-LEAP | `logs/rsl_rl/leap_grasp/2026-08-23_02-26-38_slurm` | 2048 envs, 10k iters, Slurm array task 0 | `model_5600.pt` | incomplete / stalled? | Job `array_20260823_022315` |
| rl-shadow-grasp | Grasp-Shadow | _(none)_ | Slurm array task 1 | — | **failed** | Warp cache `FileNotFoundError` on CCD kernel meta; see `slurm_jobs/array_20260823_022315/logs/task_72897_1.err` |
| rl-sharpa-grasp | Grasp-Sharpa | `logs/rsl_rl/sharpa_grasp/2026-08-23_02-26-38_slurm` | 2048 envs, 10k iters, Slurm task 2 | `model_3300.pt` | incomplete / stalled? | |
| rl-wuji-grasp | Grasp-Wuji | `logs/rsl_rl/wuji_grasp/2026-08-23_02-26-38_slurm` | 2048 envs, 10k iters, Slurm task 3 | `model_6700.pt` | incomplete / stalled? | |
| rl-allegro-rot | InHand-Rotation-Allegro | `logs/rsl_rl/allegro_inhand_rotation/2026-08-23_02-26-38_slurm` | 2048 envs, 10k iters, Slurm task 4 | `model_9999.pt` | done | |
| rl-leap-rot | InHand-Rotation-LEAP | `logs/rsl_rl/leap_inhand_rotation/2026-08-23_02-26-38_slurm` | 2048 envs, 10k iters, Slurm task 5 | `model_9999.pt` | done | |
| rl-shadow-rot | InHand-Rotation-Shadow | `logs/rsl_rl/shadow_inhand_rotation/2026-08-23_02-26-38_slurm` | 2048 envs, 10k iters, Slurm task 6 | `model_9999.pt` | done | |
| rl-sharpa-rot | InHand-Rotation-Sharpa | `logs/rsl_rl/sharpa_in_hand_rotation/2026-08-23_02-26-38_slurm` | 2048 envs, 10k iters, Slurm task 7 | `model_6000.pt` | incomplete / stalled? | |
| rl-wuji-rot | InHand-Rotation-Wuji | `logs/rsl_rl/wuji_inhand_rotation/2026-08-23_02-26-38_slurm` | 2048 envs, 10k iters, Slurm task 8 | `model_9999.pt` | done | |
| rl-leap-rot-gpu-verify | InHand-Rotation-LEAP | _(deleted)_ | 2048 envs, 10k iters, seed 42, run `gpu_verify`; local GPU A40 | `model_0.pt` | **aborted** | Launched to confirm the reconstructed pipeline trains end-to-end on real GPU (see `CHANGES.md`); killed at iter ~64 once it was pointed out `rl-leap-rot` (below, `done`, `model_9999.pt`) already exists -- redundant. Confirmed healthy first (reward 0.07->0.23 in 64 iters, ~6.1s/iter) before stopping; that's the useful outcome, not a trained checkpoint. Run dir, wandb run, and console log deleted |
| rl-allegro-grasp-v2 | Grasp-Allegro | `logs/rsl_rl/allegro_grasp/2026-09-07_22-54-28_slurm2` | 2048 envs, 10k iters, seed 42, run `slurm2` | `model_4900.pt` when moved | **resuming** on job `85163` task 0 (excl. node-004) | Started under array `84348` task 0 on `slurm-node-004` (bad driver, see JOURNAL 2026-09-09); cancelled at iter 4900 and resumed via `--agent.resume` on `85163`, node-004 excluded |
| rl-leap-grasp-v2 | Grasp-LEAP | `logs/rsl_rl/leap_grasp/2026-09-07_22-54-28_slurm2` | 2048 envs, 10k iters, seed 42, run `slurm2` | `model_4200.pt` when moved | **resuming** on job `85163` task 1 (excl. node-004) | Same story as Allegro above; cancelled at iter 4200 |
| rl-shadow-grasp-v2 | Grasp-Shadow | `logs/rsl_rl/shadow_grasp/2026-09-07_22-54-28_slurm2` | 2048 envs, 10k iters, seed 42, run `slurm2` | `model_4300.pt` when moved | **resuming** on job `85163` task 2 (excl. node-004) | First attempt (`rl-shadow-grasp` above) had failed on shared Warp cache; this attempt got the Warp-cache fix but landed on the bad-driver node. Cancelled at iter 4300 |
| rl-sharpa-grasp-v2 | Grasp-Sharpa | `logs/rsl_rl/sharpa_grasp/2026-09-07_22-54-28_slurm2` | 2048 envs, 10k iters, seed 42, run `slurm2` | `model_2400.pt` when moved | **resuming** on job `85163` task 3 (excl. node-004) | Was the slowest of the four on node-004 (~2x the others, `contact match overflow` warnings -- possibly unrelated to the driver issue, not yet investigated). Cancelled at iter 2400 |
| rl-wuji-grasp-v2 | Grasp-Wuji | `logs/rsl_rl/wuji_grasp/2026-09-08_02-26-59_slurm2` | 2048 envs, 10k iters, seed 42, run `slurm2`; Slurm array `84348` task 4 | `model_9999.pt` | **done** | Ran on `slurm-node-011` (good driver) -- 7h, 2.5s/iter. Synced to wandb: `https://wandb.ai/maxrudolph/mjlab/runs/xlek6ud9` |
| rl-sharpa-rot-v2 | InHand-Rotation-Sharpa | `logs/rsl_rl/sharpa_in_hand_rotation/2026-09-08_02-27-58_slurm2` | 2048 envs, 10k iters, seed 42, run `slurm2`; Slurm array `84348` task 5 | `model_9999.pt` | **done** | Ran on `slurm-node-011` -- 15h13m, 5.5s/iter. Synced to wandb by the (now-stopped) background watcher |

#### Slurm batch

- Array dir: `slurm_jobs/array_20260823_022315/`
- Submit: `sbatch slurm_jobs/array_20260823_022315/submission.sh`
- Resources: 1 GPU, 16 CPUs, 128GB, 36h, partition `allnodes`, array `0-8`
- Skipped Grasp-Allegro (already on local GPU)

#### Slurm batch — RL expert completion pass, `train_rl_experts.sbatch` (2026-09-07)

- Submit: `sbatch slurm_jobs/train_rl_experts.sbatch`
- Job `84348`, array `0-5`, one task per missing/incomplete combo (see table above)
- Resources: 1 GPU, 16 CPUs, 128GB, 36h, partition `allnodes`
- Per-task `WARP_CACHE_PATH` (fixes the original Grasp-Shadow failure), `LD_LIBRARY_PATH=/usr/lib64` (fixes CUDA error 803 on rlcompute H200 nodes)
- All 6 restarted from scratch (not resumed) at user's explicit choice, run-name `slurm2` so they land in fresh timestamped dirs alongside the earlier partial runs rather than overwriting them
- Once done, rerun `scripts/select_experts.py` to refresh `outputs/experts*.json` before using these as demo-collection sources
- **Logger mistake**: script used `--agent.logger tensorboard` (copied from the old array pattern) instead of `train`'s actual default (`wandb`) -- none of the 6 runs stream to wandb live. Fixed retroactively: `wandb sync --sync-tensorboard -p mjlab <rundir>` imports a tfevents dir as a wandb run after the fact. Grasp-Wuji (task 4, finished first) synced manually -> `https://wandb.ai/maxrudolph/mjlab/runs/xlek6ud9`. The other 5 are handled by a background watcher, `nohup`'d independent of any single tool call (survives the multi-hour training window): polls `sacct` every 10 min, syncs each task's run dir the moment it reaches `COMPLETED`/`FAILED`/etc. Script + logs (not checked in, scratch dir): `sync_rl_experts_wandb.sh`, `wandb_sync.log`, `sync_watcher.log`.

#### Plots / videos (RL)

- Curves: `outputs/plots/*.png` (+ `comparison_all_runs.png`)
- Videos: `outputs/videos/<Task>_model_<iter>/rl-video-step-0.mp4` for Allegro/LEAP/Sharpa/Wuji grasp and all five in-hand rotation hands (Shadow grasp missing)

### Diffusion policy

| ID | Task | Output dir | Dataset | Config highlights | Latest | Status | Notes |
|----|------|------------|---------|-------------------|--------|--------|-------|
| dp-allegro-smoke | Grasp-Allegro | `outputs/diffusion/grasp_allegro_smoke` | `data/demos/grasp_allegro_expert.zarr` | 20 epochs, smoke test | `policy_best.pt` (loss≈0.0149) | done | Eval success **0%** (16 eps). Log: `logs/train_diffusion_allegro_smoke.log` |
| dp-allegro-full | Grasp-Allegro | `outputs/diffusion/grasp_allegro_full` | `data/demos/grasp_allegro_expert_full.zarr` | 150 epochs, bs 256, obs_h=2, act_h=8, lr 1e-4, T=100, infer 16 | epoch ~138/150; `policy_epoch_0130.pt` | running / nearly done | Train log: `logs/train_diffusion_allegro_full.log`. Watch-eval: `logs/watch_eval_diffusion_allegro_full.log` → `eval_metrics.jsonl`. Periodic env eval still **0%** success (32 eps) through epoch 130; avg final dist ~0.3–0.4 m |
| dp-leap-rot-400k | InHand-Rotation-LEAP | `outputs/diffusion/InHand-Rotation-LEAP_400k` | `data/demos/InHand-Rotation-LEAP_expert_400k.zarr` | 500 epochs, bs 256, `EVAL_FINAL_ONLY=1` (eval only at epoch 500); Slurm job `80853`, `slurm-node-011` | `policy_epoch_0500.pt` / `policy_best.pt` (loss 0.006254) | **done** | 7h22m (09:00-16:22). Final eval (32 rollouts): **2.31** avg successes before drop, 100% drop rate, 24.9s survival. Logs: `slurm_jobs/train_diffusion/logs/task_80853_0.{out,err}` |
| dp-allegro-rot-400k | InHand-Rotation-Allegro | `outputs/diffusion/InHand-Rotation-Allegro_400k` | `data/demos/InHand-Rotation-Allegro_expert_400k.zarr` | 500 epochs, bs 256, `EVAL_FINAL_ONLY=1`; Slurm job `80854` | `policy_epoch_0500.pt` / `policy_best.pt` (loss 0.003741) | **done** | 7h51m (10:54-18:45), ran concurrently with `dp-leap-rot-400k` on a separate GPU allocation. Final eval: **1.62** avg successes before drop, 100% drop rate, 17.1s survival. Logs: `slurm_jobs/train_diffusion/logs/task_80854_0.{out,err}` |

#### Specialist + scarce co-training sweep, `train_scarce_specialist.sbatch` (2026-09-12/13)

- Manifest: `slurm_jobs/scarce_specialist_manifest.json` (built by
  `scripts/build_scarce_specialist_manifest.py`, 40 array-task entries / 80 total runs); each
  task's 2 runs (its config's 2 seeds) executed sequentially by `scripts/run_manifest_task.py`
- Tasks 0-19: specialist (10 task/embodiment combos x {50k, 1M} x 2 seeds)
- Tasks 20-39: scarce co-training (2 families x 5 scarce-hand choices x {uniform, balanced}
  sampling ratio x 2 seeds), pools from `COLLECTIONS.md`'s "scarce co-training pools" section
- All runs use `--val-fraction 0.1` (see JOURNAL.md's val-split entry); outputs under
  `outputs/diffusion/specialist/` and `outputs/diffusion/scarce/`; wandb project `mjlab`
- **Job `89262`** (`--array=0-39%20`, submitted 2026-09-12): 36/40 tasks **failed within
  seconds/minutes** on node-local `/tmp` exhaustion (wandb's service subprocess; `TMPDIR` wasn't
  pinned per-task the way `WARP_CACHE_PATH` was). Only tasks `0`, `2`, `3`, `4` started cleanly
  and are genuinely training (e.g. task `0` = Grasp-Allegro specialist 50k seed0, healthy at
  epoch 93/4000 as of 2026-09-13). See JOURNAL.md (2026-09-13 entry) for the root cause and the
  `TMPDIR` fix (now also in `CLAUDE.md`'s required-env-vars block).
- **Job `90599`** (`--array=1,5-39%16`, submitted 2026-09-13): resubmit of exactly the 36 failed
  indices against the same manifest, with the `TMPDIR` fix applied. Throttled to `%16` (not
  `%20`) so combined with `89262`'s 4 still-running tasks the total stays at the user's
  requested 20-concurrent cap.
- **Outcome (corrected 2026-09-22, was stale "pending" above): the sweep did not finish.**
  `90599` was cancelled 2026-09-14T10:31, most indices (5-6, 8, 10-22 confirmed, 23-39 bulk)
  never started. Final tally: specialist grasp fully covered (18/20 runs with a checkpoint, 2
  cut off early -- `Grasp-Sharpa_50k_seed1` stopped at epoch 1200/4000, `Grasp-Wuji_50k_seed1`
  at epoch 400/4000); specialist InHand-Rotation (10 configs, 20 runs) **zero** runs started;
  scarce co-training: only `source_stats.json`/`train_config.json` written for 4 configs, no
  runs trained at all. Do not treat this sweep as a source of rotation specialists.

#### InHand-Rotation specialist backfill, `train_rotation_specialist.sbatch` (2026-09-22)

- User asked for 1M + 50k specialist policies per embodiment for in-hand reorientation -- the
  gap identified above. Manifest builder (`build_scarce_specialist_manifest.py`) extended with
  `--family`/`--kind`/`--out` so it can emit a subset instead of only the full 80-run sweep.
- Manifest: `slurm_jobs/rotation_specialist_manifest.json`, built via
  `python3 scripts/build_scarce_specialist_manifest.py --family InHand-Rotation --kind specialist`
  -- 10 array tasks (5 hands x {50k, 1M}), 20 runs (2 seeds each).
- Also changed eval cadence project-wide in the builder: `eval_every = epochs // 10` (was
  `// 4`), at the user's request, so these runs eval 10x across training instead of 4x.
- **Job `94864`**, `--array=0-9%10` (10 tasks total, so `%10` is a no-op cap, kept explicit per
  user's "only launch 10 in parallel"). Submitted 2026-09-22, 4 tasks running immediately
  (`slurm-node-001`/`002`), rest pending on resources. `outputs/diffusion/specialist/InHand-Rotation-*`.
- Per-run checkpoints changed (see JOURNAL.md 2026-09-22 entry): `policy_latest.pt` (last),
  `policy_best_val.pt` (lowest `val/action_loss`, i.e. DDIM-sampled action MSE, not one-step
  noise-prediction loss), `policy_best_eval.pt` (highest mean env-eval headline across
  `eval_specs`). Old train-loss-based `policy_best.pt` removed -- nothing read it.
- **Outcome (audited 2026-09-26): failed, zero checkpoints.** Tasks 0-7 started, all hung
  from the first epoch until cancelled 2026-09-23T00:00:20 (8-9 never started). Every `.err`
  ends in DataLoader-worker `OSError: AF_UNIX path too long`: the per-task
  `TMPDIR="$PWD/tmp/rotation_specialist_task_N"` is long enough that multiprocessing's
  `pymp-*/listener-*` socket path exceeds the 108-char AF_UNIX limit. Output dirs contain only
  `train_config.json`. The new `policy_best_val.pt`/`policy_best_eval.pt` code is therefore
  still unexercised. Fix before relaunch: shorter TMPDIR (e.g. `/tmp/$USER/$SLURM_JOB_ID`
  under a quota'd node path, or `$JOBDIR` with a short root).
- **Relaunch: job `97798`** (2026-09-26), same manifest, `--array=0-9%8` (user: 8 in parallel).
  Fix: `TMPDIR="$PWD/tmp/rs_${SLURM_ARRAY_TASK_ID}"` (87-char socket path vs 109). Reproduced
  the old failure directly (`resource_sharer._start()` under the old `/scratch/...` path ->
  `AF_UNIX path too long`; new path OK). Note `$PWD` on the login node resolves to the shorter
  `/u/...` alias, so a login-node test without the explicit `/scratch` path passes falsely.
  Tasks 0-7 running on nodes 002/003/004/006, all past epoch 1 within 4 min; 8-9 pending on the
  cap. Rates: 50k ~12s/epoch (~13h/seed, ~26h/task for 2 seeds, near the 36h limit if eval or
  contention slows it), 1M ~3min/epoch (~10h/seed). `.err` files fill with harmless
  `OSError: [Errno 16] Device or resource busy: .../pymp-*` from NFS `.nfs*` files blocking
  multiprocessing's temp-dir cleanup at worker exit -- not fatal.
- **Status 2026-09-27 19:48:** no TIMEOUTs. Tasks 0-7 `COMPLETED` (13.3-24.3h wall each, limit
  36h); all 16 runs (Allegro/LEAP/Shadow/Sharpa x 50k/1M x seed0/1) reached their final epoch
  (4000 for 50k, 200 for 1M) and wrote `policy_latest.pt`, `policy_best_val.pt`,
  `policy_best_eval.pt` + `best_eval.json` -- first real exercise of the three-way selection
  code, and it worked. Tasks 8-9 (Wuji) `RUNNING`, seed0 done in both; seed1 at epoch 610/4000
  (50k, ~10s/epoch, ~9h left, ~20h total) and 109/200 (1M, ~127s/epoch, ~3h left, ~14h total) --
  both on track to finish well inside the limit. Eval results not yet tabulated.

#### Scripts

- `scripts/watch_eval_diffusion.py` — eval every N epochs while training
- Diffusion train/collect CLIs live under `src/mjlab_hand` (added Aug 23)


---

## Legacy registry text — Vista (as recorded at the time; the job table above supersedes it)

### Vista (TACC GH200) — 2026-09-28

| id | what | where | status | notes |
|---|---|---|---|---|
| `1030560` | scarce co-training manifest (`slurm_jobs/scarce_manifest.json`), 5 x 1-node, PACK=4 (8 runs/node, 40 runs, 50 epochs, compiled) | `outputs/diffusion/scarce/*` | running (started 2026-09-28 03:17) | Not launched by the 09-28 session; logged here because of the storage issue. On user instruction, deleted `policy_epoch_0005/0010/0015.pt` from all 40 runs (120 files, 33 GB) to keep `$WORK` under quota. `policy_epoch_0020/0025`, `policy_latest`, `policy_best_*` kept. At ~11 GB per 5-epoch snapshot round and 34 GB free afterwards, it will hit the quota again around epoch 35-40 unless more is freed. No save errors in its logs as of 13:00 |
| `1031790` (idev, gh-dev, 4 nodes) | job-shape benchmark: round 1 = 4 nodes x {4,8,12,16} runs; round 2 = one 4-node run of `vista_train_manifest.sbatch` (8 runs/node) | `$SCRATCH/bench_multinode/` | done | Grasp-Allegro 50k, 16 epochs, production flags. Results in ANALYSIS.md "Vista job shape". A first round-1 attempt wrote to `outputs/` on `$WORK`, pushed it over quota (`torch.save`: "unexpected pos"), and was killed and deleted |

### Vista — ambient rotation sweep (2026-09-29)

| id | what | where | status | notes |
|---|---|---|---|---|
| diag-rot `1035261` | 8-run diagnostic, `slurm_jobs/diag_rot_manifest.json`, `-N 2`, PACK=4, 24h, submitted by user 2026-09-29 | `$SCRATCH/cross_embodied_diffusion/outputs/diffusion/diag_rot/` | pending (Priority; gh partition full) | see JOURNAL 2026-09-29 "Ambient rotation sweep" |
| diag-rot-idev | diagnostic tasks 0-3 (the 4 specialists) run on idev `1031788`'s GPU while `1035261` waits; `slurm_jobs/diag_rot_idev_manifest.json` (same configs, `diag_rot_idev/` outdirs, `-idev` wandb names); log `slurm_jobs/vista_train_manifest/logs/idev_1031788_diag.log` | `.../outputs/diffusion/diag_rot_idev/` | running (started 2026-09-29; idev ends ~2026-09-30 13:27) | duplicates `1035261` node 0 unless that job is replaced by a 1-node tasks-4-7 job |
| diag-amb-mm `1036217` | ambient s=0/10/25/100 minmax, Allegro, `slurm_jobs/diag_rot_ambient_minmax_manifest.json`, `--array=0-3`, PACK=1, 5 h limit | `.../diag_rot/*_minmax_seed0` | **done** 2026-09-30, all COMPLETED in 2h20-2h24 (~2.7 min/epoch alone). In-training evals (32 envs, ep 5..50): s0 0.00->0.41 (rising at end), s10 0.09-0.53, s25 0.84->0.44-0.62, s100 0.72-0.97 then 0.59 | reporting evals pending (needs a gh node: `vista_eval_checkpoints.sbatch`) |
| diag-rot-idev2 | on idev `1031788`: `spec1M_padded_minmax`, `pool_cotrain_minmax` (`slurm_jobs/diag_rot_idev2_manifest.json`); the two gaussian padded runs of diag-rot-idev were killed at 1M ep ~60 / 50k ep ~950 (all evals 0.00) | `.../diag_rot_idev/` | running (started 2026-09-29 23:47) | tests CHANGES.md item 58 |
| ambient-rot `1038730` | 320 runs, `slurm_jobs/ambient_rot_manifest.json` (minmax), `-N 8 --array=0-9`, PACK=4, 12 h limit; submitted by user 2026-09-30 21:18 | `$SCRATCH/cross_embodied_diffusion/outputs/diffusion/ambient_rot/` | running: element 0 started 2026-10-01 10:53 (~13.5 h queue wait), all 10 running by ~11:45; ~11.2 min/epoch at 4 runs/node -> ~10 h of the 12 h limit | shape chosen from queue evidence (JOURNAL 2026-09-30 20:00); each 8-node job = half of one target hand |
| vista-eval `1038574` | reporting evals of the 4 `diag_rot/*_minmax_seed0` runs, ALSO=LEAP | `<run>/posthoc_eval.json` | done 2026-10-01 (23 min); results in JOURNAL 2026-10-01 02:45 | |

### Vista — co-training vs target-only, term-aligned (bundle-migration code), 2026-10-02

| id | what | where | status | notes |
|---|---|---|---|---|
| cotrain-vs-target `1043385` | 40 runs: Grasp + Rotation x 5 targets x sigma {0, 100} x seeds {0, 1}, `slurm_jobs/cotrain_vs_target_manifest.json`, `-N 4 --array=0-9`, PACK=1, 9 h | `$SCRATCH/cross_embodied_diffusion/outputs/diffusion/ambient_ta/` | **done** 2026-10-03: 10/10 COMPLETED (6h30-6h58), 40/40 selection.json; re-scored 2026-10-03 (100 eps): grasp +0.38..+0.44, rotation -0.04..-0.15, JOURNAL 2026-10-03 16:45 | queue 2026-10-02 15:15: median wait ~10.5-11.3 h for every size 1-16 nodes, so 1 run/node (~7 h, measured 7.5 min/epoch uncompiled) beats packing (~15 h at 4/node) |
| finetune `1045839` | 40 runs: fine-tune each co-trained (sigma0) run from its `policy_best_val.pt` on target-only data, lr {1e-4, 1e-5}, 10 epochs, `slurm_jobs/finetune_manifest.json`, `-N 4 --array=0-9`, PACK=1, 3 h | `$SCRATCH/cross_embodied_diffusion/outputs/diffusion/ambient_ta_ft/` | **done** 2026-10-03 (10/10 COMPLETED); re-scored 2026-10-04 00:15; results JOURNAL 2026-10-04 00:20 | ~1.5-2 h per run alone (10 x 7.5 min + 10 evals) |
