# Training runs

Status snapshot: **2026-08-24**. Artifact roots are gitignored; paths are relative to repo root.

## RL (rsl_rl / mjlab `train`)

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

### Slurm batch

- Array dir: `slurm_jobs/array_20260823_022315/`
- Submit: `sbatch slurm_jobs/array_20260823_022315/submission.sh`
- Resources: 1 GPU, 16 CPUs, 128GB, 36h, partition `allnodes`, array `0-8`
- Skipped Grasp-Allegro (already on local GPU)

### Slurm batch — RL expert completion pass, `train_rl_experts.sbatch` (2026-09-07)

- Submit: `sbatch slurm_jobs/train_rl_experts.sbatch`
- Job `84348`, array `0-5`, one task per missing/incomplete combo (see table above)
- Resources: 1 GPU, 16 CPUs, 128GB, 36h, partition `allnodes`
- Per-task `WARP_CACHE_PATH` (fixes the original Grasp-Shadow failure), `LD_LIBRARY_PATH=/usr/lib64` (fixes CUDA error 803 on rlcompute H200 nodes)
- All 6 restarted from scratch (not resumed) at user's explicit choice, run-name `slurm2` so they land in fresh timestamped dirs alongside the earlier partial runs rather than overwriting them
- Once done, rerun `scripts/select_experts.py` to refresh `outputs/experts*.json` before using these as demo-collection sources
- **Logger mistake**: script used `--agent.logger tensorboard` (copied from the old array pattern) instead of `train`'s actual default (`wandb`) -- none of the 6 runs stream to wandb live. Fixed retroactively: `wandb sync --sync-tensorboard -p mjlab <rundir>` imports a tfevents dir as a wandb run after the fact. Grasp-Wuji (task 4, finished first) synced manually -> `https://wandb.ai/maxrudolph/mjlab/runs/xlek6ud9`. The other 5 are handled by a background watcher, `nohup`'d independent of any single tool call (survives the multi-hour training window): polls `sacct` every 10 min, syncs each task's run dir the moment it reaches `COMPLETED`/`FAILED`/etc. Script + logs (not checked in, scratch dir): `sync_rl_experts_wandb.sh`, `wandb_sync.log`, `sync_watcher.log`.

### Plots / videos (RL)

- Curves: `outputs/plots/*.png` (+ `comparison_all_runs.png`)
- Videos: `outputs/videos/<Task>_model_<iter>/rl-video-step-0.mp4` for Allegro/LEAP/Sharpa/Wuji grasp and all five in-hand rotation hands (Shadow grasp missing)

## Diffusion policy

| ID | Task | Output dir | Dataset | Config highlights | Latest | Status | Notes |
|----|------|------------|---------|-------------------|--------|--------|-------|
| dp-allegro-smoke | Grasp-Allegro | `outputs/diffusion/grasp_allegro_smoke` | `data/demos/grasp_allegro_expert.zarr` | 20 epochs, smoke test | `policy_best.pt` (loss≈0.0149) | done | Eval success **0%** (16 eps). Log: `logs/train_diffusion_allegro_smoke.log` |
| dp-allegro-full | Grasp-Allegro | `outputs/diffusion/grasp_allegro_full` | `data/demos/grasp_allegro_expert_full.zarr` | 150 epochs, bs 256, obs_h=2, act_h=8, lr 1e-4, T=100, infer 16 | epoch ~138/150; `policy_epoch_0130.pt` | running / nearly done | Train log: `logs/train_diffusion_allegro_full.log`. Watch-eval: `logs/watch_eval_diffusion_allegro_full.log` → `eval_metrics.jsonl`. Periodic env eval still **0%** success (32 eps) through epoch 130; avg final dist ~0.3–0.4 m |
| dp-leap-rot-400k | InHand-Rotation-LEAP | `outputs/diffusion/InHand-Rotation-LEAP_400k` | `data/demos/InHand-Rotation-LEAP_expert_400k.zarr` | 500 epochs, bs 256, `EVAL_FINAL_ONLY=1` (eval only at epoch 500); Slurm job `80853`, `slurm-node-011` | `policy_epoch_0500.pt` / `policy_best.pt` (loss 0.006254) | **done** | 7h22m (09:00-16:22). Final eval (32 rollouts): **2.31** avg successes before drop, 100% drop rate, 24.9s survival. Logs: `slurm_jobs/train_diffusion/logs/task_80853_0.{out,err}` |
| dp-allegro-rot-400k | InHand-Rotation-Allegro | `outputs/diffusion/InHand-Rotation-Allegro_400k` | `data/demos/InHand-Rotation-Allegro_expert_400k.zarr` | 500 epochs, bs 256, `EVAL_FINAL_ONLY=1`; Slurm job `80854` | `policy_epoch_0500.pt` / `policy_best.pt` (loss 0.003741) | **done** | 7h51m (10:54-18:45), ran concurrently with `dp-leap-rot-400k` on a separate GPU allocation. Final eval: **1.62** avg successes before drop, 100% drop rate, 17.1s survival. Logs: `slurm_jobs/train_diffusion/logs/task_80854_0.{out,err}` |

### Specialist + scarce co-training sweep, `train_scarce_specialist.sbatch` (2026-09-12/13)

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
  requested 20-concurrent cap. Pending on cluster priority as of 2026-09-13.
- Once both jobs finish, expect all 80 runs to have completed across `89262` (4) + `90599` (up
  to 36, less any further failures) -- verify `outputs/diffusion/{specialist,scarce}/*/policy_latest.pt`
  count reaches 80 before treating the sweep as done.

### Scripts

- `scripts/watch_eval_diffusion.py` — eval every N epochs while training
- Diffusion train/collect CLIs live under `src/mjlab_hand` (added Aug 23)

## Template (copy for new runs)

```md
| id | Task | run/output dir | key config | latest artifact | status | notes |
```
