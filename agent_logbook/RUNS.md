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

### Slurm batch

- Array dir: `slurm_jobs/array_20260823_022315/`
- Submit: `sbatch slurm_jobs/array_20260823_022315/submission.sh`
- Resources: 1 GPU, 16 CPUs, 128GB, 36h, partition `allnodes`, array `0-8`
- Skipped Grasp-Allegro (already on local GPU)

### Plots / videos (RL)

- Curves: `outputs/plots/*.png` (+ `comparison_all_runs.png`)
- Videos: `outputs/videos/<Task>_model_<iter>/rl-video-step-0.mp4` for Allegro/LEAP/Sharpa/Wuji grasp and all five in-hand rotation hands (Shadow grasp missing)

## Diffusion policy

| ID | Task | Output dir | Dataset | Config highlights | Latest | Status | Notes |
|----|------|------------|---------|-------------------|--------|--------|-------|
| dp-allegro-smoke | Grasp-Allegro | `outputs/diffusion/grasp_allegro_smoke` | `data/demos/grasp_allegro_expert.zarr` | 20 epochs, smoke test | `policy_best.pt` (loss≈0.0149) | done | Eval success **0%** (16 eps). Log: `logs/train_diffusion_allegro_smoke.log` |
| dp-allegro-full | Grasp-Allegro | `outputs/diffusion/grasp_allegro_full` | `data/demos/grasp_allegro_expert_full.zarr` | 150 epochs, bs 256, obs_h=2, act_h=8, lr 1e-4, T=100, infer 16 | epoch ~138/150; `policy_epoch_0130.pt` | running / nearly done | Train log: `logs/train_diffusion_allegro_full.log`. Watch-eval: `logs/watch_eval_diffusion_allegro_full.log` → `eval_metrics.jsonl`. Periodic env eval still **0%** success (32 eps) through epoch 130; avg final dist ~0.3–0.4 m |

### Scripts

- `scripts/watch_eval_diffusion.py` — eval every N epochs while training
- Diffusion train/collect CLIs live under `src/mjlab_hand` (added Aug 23)

## Template (copy for new runs)

```md
| id | Task | run/output dir | key config | latest artifact | status | notes |
```
