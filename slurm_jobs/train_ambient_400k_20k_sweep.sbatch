#!/bin/bash
#SBATCH --job-name=ambient-400k-20k
#SBATCH --output=slurm_jobs/train_ambient_400k_20k_sweep/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/train_ambient_400k_20k_sweep/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --array=0-21%20

# Ambient-diffusion sigma* sweep: 400k of one embodiment ("target", admitted
# at every diffusion timestep) mixed with 20k of the other ("source", gated
# out of the loss below sigma*), for BOTH directions (LEAP-target /
# Allegro-target), swept over sigma* = 0, 10, 20, ..., 100 (11 points each).
# 2 directions x 11 sigmas = 22 array cells, throttled to 20 concurrent
# (%20) by user request.
#
# Datasets (unconditioned, --no-onehot, built ahead of time):
#   data/mixed_noc/InHand-Rotation_target-LEAP400k_source-Allegro20k.zarr
#   data/mixed_noc/InHand-Rotation_target-Allegro400k_source-LEAP20k.zarr
#
# sigma* is directly the t_min passed to --ambient-tmin (num_train_timesteps
# defaults to 100, so sigma-as-percentage == sigma-as-timestep). sigma*=0 is
# plain naive mixing (source admitted everywhere); sigma*=100 admits the
# source nowhere (t_min clamped to T-1 in practice), equivalent to solo
# target-only training -- both endpoints fall out of the same code path, no
# special-casing needed for a fresh sweep like this one.
#
# Both target and source are scored every eval (--eval-spec, unconditioned
# so onehot=null for each) -- the source-embodiment numbers are the sweep's
# control: they should fall as sigma* rises. Eval fires only at the final
# epoch (matches the just-completed solo 400k runs' EVAL_FINAL_ONLY convention).
#
# See CHANGES.md items 24-26 and the 2026-09-01/02 JOURNAL.md entries for the
# sampling-order fix this sweep exercises for the first time on a real run.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/train_ambient_400k_20k_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

SIGMAS=(0 10 20 30 40 50 60 70 80 90 100)
N_SIGMAS=${#SIGMAS[@]}

direction_idx=$(( SLURM_ARRAY_TASK_ID % 2 ))
sigma_idx=$(( SLURM_ARRAY_TASK_ID / 2 ))
SIGMA="${SIGMAS[$sigma_idx]}"

if [[ "$direction_idx" == "0" ]]; then
  TARGET_TASK="InHand-Rotation-LEAP"
  SOURCE_TASK="InHand-Rotation-Allegro"
  DATASET="data/mixed_noc/InHand-Rotation_target-LEAP400k_source-Allegro20k.zarr"
  TAG="target-LEAP400k_source-Allegro20k"
else
  TARGET_TASK="InHand-Rotation-Allegro"
  SOURCE_TASK="InHand-Rotation-LEAP"
  DATASET="data/mixed_noc/InHand-Rotation_target-Allegro400k_source-LEAP20k.zarr"
  TAG="target-Allegro400k_source-LEAP20k"
fi

OUTPUT_DIR="outputs/ambient/${TAG}_sigma${SIGMA}"
EPOCHS=500
SAVE_EVERY=50
EVAL_EVERY="$EPOCHS"   # eval only at the final epoch

echo "[INFO] direction=${TAG} sigma*=${SIGMA} dataset=${DATASET} output=${OUTPUT_DIR}"

EVAL_SPEC="[{\"task\": \"${TARGET_TASK}\", \"onehot\": null}, {\"task\": \"${SOURCE_TASK}\", \"onehot\": null}]"

train-diffusion \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --num-epochs "$EPOCHS" \
  --save-every-epochs "$SAVE_EVERY" \
  --latest-every-epochs "$SAVE_EVERY" \
  --eval-spec "$EVAL_SPEC" \
  --eval-every-epochs "$EVAL_EVERY" \
  --eval-num-envs 32 \
  --ambient-tmin 0 "$SIGMA" \
  --seed 0 \
  --wandb-project mjlab \
  --wandb-run-name "ambient_${TAG}_sigma${SIGMA}" \
  --wandb-tags ambient rotation "$TARGET_TASK" "sigma${SIGMA}"
