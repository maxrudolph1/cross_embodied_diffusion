#!/bin/bash
#SBATCH --job-name=scarce-specialist
#SBATCH --output=slurm_jobs/train_scarce_specialist/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/train_scarce_specialist/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=36:00:00
#SBATCH --array=0-39%20
#
# 40 array tasks, each running 2 train-diffusion runs sequentially (the 2
# seeds of one config, via scripts/run_manifest_task.py) -- 80 runs total,
# capped at 20 concurrent (%20). Manifest built by
# scripts/build_scarce_specialist_manifest.py:
#   - tasks 0-19:  specialist runs (10 task/embodiment combos x {50k, 1M})
#   - tasks 20-39: scarce co-training runs (2 families x 5 scarce-hand
#     choices x {uniform, balanced} sampling ratio)
# No node pin -- see train_cross_embodiment.sbatch: pure GPU training, only
# a small periodic --eval-spec rollout touches mujoco_warp.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/scarce_specialist_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

# wandb's service subprocess creates a tempfile.TemporaryDirectory() under
# $TMPDIR (default /tmp) on every train-diffusion invocation. With up to 20
# concurrent array tasks able to land on the same node, node-local /tmp can
# fill up and every wandb.init() after that point dies with ENOSPC (seen on
# slurm-node-005: 36/40 tasks failed in <1 minute this way). Point it at
# shared per-task storage instead, same rationale as WARP_CACHE_PATH.
export TMPDIR="$PWD/tmp/scarce_specialist_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

python3 scripts/run_manifest_task.py \
  slurm_jobs/scarce_specialist_manifest.json \
  "$SLURM_ARRAY_TASK_ID"
