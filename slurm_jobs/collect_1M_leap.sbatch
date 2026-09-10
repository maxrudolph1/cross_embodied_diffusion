#!/bin/bash
#SBATCH --job-name=collect-1M-leap
#SBATCH --output=slurm_jobs/collect_1M/logs/leap_task_%A.out
#SBATCH --error=slurm_jobs/collect_1M/logs/leap_task_%A.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --nodelist=slurm-node-011
#
# Follow-up: Grasp-LEAP finished training (model_9999.pt, job 85382 task 1).
# Same methodology as collect_1M.sbatch: 256 envs, 2010 episodes
# (ceil(1e6/~498)), output to /datastor2/mrudolph/mjlab_hand_demos/.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/collect_1M_leap"
mkdir -p "$WARP_CACHE_PATH"

OUT_DIR="/datastor2/mrudolph/mjlab_hand_demos"
mkdir -p "$OUT_DIR"

CHECKPOINT="logs/rsl_rl/leap_grasp/2026-09-09_21-37-04_slurm2/model_9999.pt"
OUTPUT="${OUT_DIR}/Grasp-LEAP_expert_1M.zarr"

echo "[INFO] task=Grasp-LEAP checkpoint=${CHECKPOINT} episodes=2010 output=${OUTPUT}"

stdbuf -oL -eL collect-demos \
  --task Grasp-LEAP \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --num-envs 256 \
  --num-episodes 2010 \
  --max-episode-steps 500
