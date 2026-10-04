#!/bin/bash
#SBATCH --job-name=train-rl-experts
#SBATCH --output=slurm_jobs/train_rl_experts/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/train_rl_experts/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=36:00:00
#SBATCH --array=0-5
#
# Fills in the remaining RL expert checkpoints needed for "all embodiments,
# both tasks": the 4 rotation experts (Allegro/LEAP/Shadow/Wuji) already
# finished at model_9999.pt (see docs/RUNS.md) and are NOT redone
# here. This array trains the 6 combos that were missing or only partially
# trained, all fresh from scratch (user's explicit choice over resuming the
# partial checkpoints), each to the standard 10000 iterations / 2048 envs:
#
#   0  Grasp-Allegro            (had only model_6600.pt)
#   1  Grasp-LEAP                (had only model_6400.pt)
#   2  Grasp-Shadow              (previously failed: Warp cache FileNotFoundError,
#                                 fixed by giving every array task its own
#                                 WARP_CACHE_PATH below)
#   3  Grasp-Sharpa               (had only model_3800.pt)
#   4  Grasp-Wuji                 (had only model_7500.pt)
#   5  InHand-Rotation-Sharpa     (had only model_6800.pt)
#
# Run dirs land under logs/rsl_rl/<hand>_<task>/<timestamp>_slurm2/, alongside
# (not overwriting) the earlier partial runs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

# rlcompute (H200) nodes ship a stale /usr/local/cuda/compat libcuda that
# ldconfig prefers over the real driver -> CUDA error 803 on every call.
export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Concurrent array tasks sharing one Warp kernel cache race and fail with a
# CCD-kernel FileNotFoundError (this is what killed Grasp-Shadow last time).
export WARP_CACHE_PATH="$PWD/warp_cache/train_rl_experts_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

TASKS=(
  "Grasp-Allegro"
  "Grasp-LEAP"
  "Grasp-Shadow"
  "Grasp-Sharpa"
  "Grasp-Wuji"
  "InHand-Rotation-Sharpa"
)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "[INFO] array_task=${SLURM_ARRAY_TASK_ID} task=${TASK}"

stdbuf -oL -eL train "$TASK" \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 10000 \
  --agent.seed 42 \
  --agent.logger tensorboard \
  --agent.run-name slurm2
