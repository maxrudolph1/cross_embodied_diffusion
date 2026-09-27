#!/bin/bash
#SBATCH --job-name=rotation-specialist
#SBATCH --output=slurm_jobs/train_rotation_specialist/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/train_rotation_specialist/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=36:00:00
#SBATCH --array=0-9%8
#
# 10 array tasks (5 hands x {50k, 1M}), each running the 2 seeds of one config
# sequentially via scripts/run_manifest_task.py -- 20 InHand-Rotation
# specialist runs. Manifest: scripts/build_scarce_specialist_manifest.py
#   --family InHand-Rotation --kind specialist
# Checkpoints per run: policy_best_val.pt, policy_latest.pt, policy_best_eval.pt
# No node pin -- see train_cross_embodiment.sbatch: pure GPU training, only
# a small periodic --eval-spec rollout touches mujoco_warp.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/rotation_specialist_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

# wandb's service subprocess creates a tempfile.TemporaryDirectory() under
# $TMPDIR (default /tmp) on every train-diffusion invocation. With up to 20
# concurrent array tasks able to land on the same node, node-local /tmp can
# fill up and every wandb.init() after that point dies with ENOSPC (seen on
# slurm-node-005: 36/40 tasks failed in <1 minute this way). Point it at
# shared per-task storage instead, same rationale as WARP_CACHE_PATH.
# Keep this path SHORT: torch DataLoader workers open an AF_UNIX socket at
# $TMPDIR/pymp-XXXXXXXX/listener-XXXXXXXX, and sun_path caps at 107 chars.
# The old "$PWD/tmp/rotation_specialist_task_N" came to 109 and every task
# hung in epoch 1 on "OSError: AF_UNIX path too long" (job 94864).
export TMPDIR="$PWD/tmp/rs_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

python3 scripts/run_manifest_task.py \
  slurm_jobs/rotation_specialist_manifest.json \
  "$SLURM_ARRAY_TASK_ID"
