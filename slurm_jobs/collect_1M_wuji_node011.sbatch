#!/bin/bash
#SBATCH --job-name=collect-1M-wuji-n011
#SBATCH --output=slurm_jobs/collect_1M/logs/wuji_node011_task_%A.out
#SBATCH --error=slurm_jobs/collect_1M/logs/wuji_node011_task_%A.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --nodelist=slurm-node-011
#
# Retry of array task 0 of collect_1M.sbatch (job 85711): Grasp-Wuji's
# collection landed on slurm-node-004 (stale driver) and TIMED OUT at 4h
# having written only 6.8MB of an expected ~600MB dataset -- confirms the
# driver issue is bad enough to break collection jobs too, not just
# multi-day training. Pinned to node-011 this time. Partial output at
# data/mjlab_hand_demos/Grasp-Wuji_expert_1M.zarr was deleted
# before resubmitting.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/collect_1M_wuji_node011"
mkdir -p "$WARP_CACHE_PATH"

OUT_DIR="data/mjlab_hand_demos"
mkdir -p "$OUT_DIR"

CHECKPOINT="logs/rsl_rl/wuji_grasp/2026-09-08_02-26-59_slurm2/model_9999.pt"
OUTPUT="${OUT_DIR}/Grasp-Wuji_expert_1M.zarr"

echo "[INFO] task=Grasp-Wuji checkpoint=${CHECKPOINT} episodes=2010 output=${OUTPUT} node=$(hostname)"

stdbuf -oL -eL collect-demos \
  --task Grasp-Wuji \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --num-envs 256 \
  --num-episodes 2010 \
  --max-episode-steps 500
