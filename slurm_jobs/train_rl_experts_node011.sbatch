#!/bin/bash
#SBATCH --job-name=train-rl-node011
#SBATCH --output=slurm_jobs/train_rl_experts/logs/node011_task_%A_%a.out
#SBATCH --error=slurm_jobs/train_rl_experts/logs/node011_task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --nodelist=slurm-node-011
#
# Pinned to slurm-node-011 ONLY, per explicit user decision (2026-09-09): most
# of the "allnodes" partition (confirmed so far: node-002/003/004/005/007/008/009)
# runs a stale NVIDIA driver (535.216.03, CUDA 12.2 max) that disables Warp's
# CUDA conditional graph capture, causing ~7-14x slower mujoco_warp physics
# stepping for contact-heavy grasp tasks. node-011 (driver 580.173.02, CUDA
# 13.0) is the only node confirmed to have the modern driver so far. This
# WILL queue if node-011 has no free GPU/CPU slot (it's often near-full with
# other jobs, including this user's own long-running "flock" jobs) -- that's
# the accepted tradeoff for guaranteed-fast training over immediate scheduling.
#
# Resumes the 3 remaining grasp tasks that were stuck on bad-driver nodes
# under job 85163 (Grasp-Sharpa, task 3 of that job, already lucked onto
# node-011 directly and is left running as-is). Iteration budgets below are
# REMAINING iterations to reach the standard 10000 total (not "10000 more" --
# see the resume+max-iterations overshoot bug noted in agent_logbook/JOURNAL.md,
# 2026-09-09): current checkpoints were Allegro=5800, LEAP=5200, Shadow=4900.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/train_rl_node011_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

TASKS=(
  "Grasp-Allegro"
  "Grasp-LEAP"
  "Grasp-Shadow"
)
REMAINING_ITERS=(
  4200
  4800
  5100
)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
MAX_ITERS="${REMAINING_ITERS[$SLURM_ARRAY_TASK_ID]}"

echo "[INFO] array_task=${SLURM_ARRAY_TASK_ID} task=${TASK} node=$(hostname) remaining_iters=${MAX_ITERS}"

stdbuf -oL -eL train "$TASK" \
  --env.scene.num-envs 2048 \
  --agent.max-iterations "$MAX_ITERS" \
  --agent.seed 42 \
  --agent.logger wandb \
  --agent.run-name slurm2 \
  --agent.resume True
