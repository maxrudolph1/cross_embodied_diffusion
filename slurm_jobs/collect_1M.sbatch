#!/bin/bash
#SBATCH --job-name=collect-1M
#SBATCH --output=slurm_jobs/collect_1M/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/collect_1M/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#
# Collects a ~1M-transition demo dataset per (task, embodiment) pair, but only
# for the pairs that already have a finished expert checkpoint as of
# 2026-09-09 (see agent_logbook/RUNS.md): Grasp-Wuji and all 5
# InHand-Rotation hands are "done" (model_9999.pt); Grasp-Sharpa is past its
# standard 10k budget (running to 12500, checkpoint below is whatever's
# latest at submit time) so treated as expert-ready too. Grasp-Allegro/LEAP/
# Shadow are NOT included -- still training (see job 85382) -- collect those
# separately once they finish.
#
# Episode counts are sized to land at/above 1M transitions using this
# project's established per-family measured rates (grasp caps at ~498
# steps/episode regardless of hand; rotation terminates early on
# object_dropped and varies by hand -- see agent_logbook/COLLECTIONS.md for
# the historical per-hand rates used below). Actual n_steps will vary a bit;
# that's expected and matches prior collections in this project.
#
# Output goes to data/mjlab_hand_demos/ per user request
# (NOT the gitignored data/demos/ used by earlier collections in this repo).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/collect_1M_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

OUT_DIR="data/mjlab_hand_demos"
mkdir -p "$OUT_DIR"

TASKS=(
  "Grasp-Wuji"
  "Grasp-Sharpa"
  "InHand-Rotation-Allegro"
  "InHand-Rotation-LEAP"
  "InHand-Rotation-Shadow"
  "InHand-Rotation-Sharpa"
  "InHand-Rotation-Wuji"
)
CHECKPOINTS=(
  "logs/rsl_rl/wuji_grasp/2026-09-08_02-26-59_slurm2/model_9999.pt"
  "logs/rsl_rl/sharpa_grasp/2026-09-09_10-45-03_slurm2/model_10800.pt"
  "logs/rsl_rl/allegro_inhand_rotation/2026-08-23_02-26-38_slurm/model_9999.pt"
  "logs/rsl_rl/leap_inhand_rotation/2026-08-23_02-26-38_slurm/model_9999.pt"
  "logs/rsl_rl/shadow_inhand_rotation/2026-08-23_02-26-38_slurm/model_9999.pt"
  "logs/rsl_rl/sharpa_in_hand_rotation/2026-09-08_02-27-58_slurm2/model_9999.pt"
  "logs/rsl_rl/wuji_inhand_rotation/2026-08-23_02-26-38_slurm/model_9999.pt"
)
# ceil(1_000_000 / measured_steps_per_episode)
NUM_EPISODES=(
  2010
  2010
  2065
  2059
  2175
  2132
  2107
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
EPISODES="${NUM_EPISODES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="${OUT_DIR}/${TASK}_expert_1M.zarr"

echo "[INFO] array_task=${SLURM_ARRAY_TASK_ID} task=${TASK} checkpoint=${CHECKPOINT} episodes=${EPISODES} output=${OUTPUT}"

stdbuf -oL -eL collect-demos \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --num-envs 256 \
  --num-episodes "$EPISODES" \
  --max-episode-steps 500
