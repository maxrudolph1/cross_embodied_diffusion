#!/bin/bash
#SBATCH --job-name=train-rl-experts-resume
#SBATCH --output=slurm_jobs/train_rl_experts/logs/resume_task_%A_%a.out
#SBATCH --error=slurm_jobs/train_rl_experts/logs/resume_task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=36:00:00
#SBATCH --array=0-3
#SBATCH --exclude=slurm-node-004
#
# Resubmission of the 4 grasp tasks from train_rl_experts.sbatch (array 84348)
# that got stuck on slurm-node-004. Diagnosed 2026-09-09: node-004 runs driver
# 535.216.03 (CUDA 12.2 max), so Warp's is_conditional_graph_supported() is
# False there and every job loses CUDA graph capture -- confirmed 7-14x
# slower (18-36s/iter) than the same task family on a modern-driver node
# (node-011, driver 580.173.02, 2.5s/iter for Grasp-Wuji). --exclude keeps
# these off node-004; if another old-driver node turns up, add it here too
# (check with: srun --jobid=<id> --overlap --cpu-bind=none nvidia-smi
# --query-gpu=driver_version --format=csv, or grep the task's .err for
# "CUDA version < 12.4 detected" once running).
#
# Resumes each task from its own latest checkpoint (2400-4900 iters already
# done under job 84348) rather than restarting, via --agent.resume with the
# default load-run/load-checkpoint regex (picks the alphabetically-latest
# matching run dir / checkpoint -- each hand only has one *_slurm2 dir, so
# this is unambiguous). Also switches to train's actual default logger
# (wandb) instead of the tensorboard override the original array used.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/train_rl_experts_resume_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

TASKS=(
  "Grasp-Allegro"
  "Grasp-LEAP"
  "Grasp-Shadow"
  "Grasp-Sharpa"
)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "[INFO] array_task=${SLURM_ARRAY_TASK_ID} task=${TASK} node=$(hostname) resuming"

stdbuf -oL -eL train "$TASK" \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 10000 \
  --agent.seed 42 \
  --agent.logger wandb \
  --agent.run-name slurm2 \
  --agent.resume True
