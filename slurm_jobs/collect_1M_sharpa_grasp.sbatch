#!/bin/bash
#SBATCH --job-name=collect-1M-sharpa-grasp
#SBATCH --output=slurm_jobs/collect_1M/logs/sharpa_grasp_task_%A.out
#SBATCH --error=slurm_jobs/collect_1M/logs/sharpa_grasp_task_%A.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#
# Replaces array task 1 of collect_1M.sbatch (job 85711), cancelled before it
# started: Grasp-Sharpa training (job 85163 task 3) finished to its full
# 12500-iteration target while that array was still pending, so this uses the
# real final checkpoint (model_12499.pt) instead of the mid-training
# model_10800.pt the array had baked in at submit time.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/collect_1M_sharpa_grasp"
mkdir -p "$WARP_CACHE_PATH"

OUT_DIR="data/mjlab_hand_demos"
mkdir -p "$OUT_DIR"

CHECKPOINT="logs/rsl_rl/sharpa_grasp/2026-09-09_10-45-03_slurm2/model_12499.pt"
OUTPUT="${OUT_DIR}/Grasp-Sharpa_expert_1M.zarr"

echo "[INFO] task=Grasp-Sharpa checkpoint=${CHECKPOINT} episodes=2010 output=${OUTPUT}"

stdbuf -oL -eL collect-demos \
  --task Grasp-Sharpa \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --num-envs 256 \
  --num-episodes 2010 \
  --max-episode-steps 500
