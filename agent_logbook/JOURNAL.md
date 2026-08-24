# Agent journal

Newest entries first. Link run/collection IDs from `RUNS.md` / `COLLECTIONS.md`.

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
