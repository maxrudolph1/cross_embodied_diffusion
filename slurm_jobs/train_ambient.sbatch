#!/bin/bash
#SBATCH --job-name=train-ambient
#SBATCH --output=slurm_jobs/train_ambient/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/train_ambient/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --exclude=sea112-rlcompute13
#SBATCH --array=0-11

# The ambient-diffusion sigma* sweep (interior points only -- sigma*=0 and
# sigma*=100 are scored from outputs/mixed_noc and outputs/diffusion
# respectively, see plot_ambient_sweep.py). Trains on a two-source mixed,
# UNCONDITIONED dataset (target = Allegro at budget N, source = LEAP fixed
# at 400k) with the source gated out of the loss below diffusion timestep
# sigma* (--ambient-tmin 0 <sigma>): the target is admitted at every t, the
# source only at t >= sigma*. Both embodiments are scored (--eval-spec),
# including LEAP -- the sweep's control: LEAP performance must FALL as
# sigma* rises, or the gating mechanism isn't doing what it claims.
#
#   FAMILY=grasp|rotation, SEED, EVAL_ENVS
#   TARGET_SIZES="10k 50k 100k 400k"   Allegro (target) data budgets
#   SIGMA_LIST="50 75 90"              interior gating thresholds
#
# The N=400k, sigma*=0 special case (needed because build_mixed.sbatch
# cannot build the diagonal) is NOT part of this array -- run it once,
# manually, on the build_ambient.sbatch dataset with --ambient-tmin 0 0
# and an "_amb0" output suffix.
#
# See CHANGES.md items 24, 25, 26, 28.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/train_ambient_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

FAMILY="${FAMILY:-grasp}"
if [[ "$FAMILY" == "rotation" ]]; then
  TASK_A="InHand-Rotation-Allegro"
  TASK_B="InHand-Rotation-LEAP"
  DISPLAY_FAMILY="InHand-Rotation"
else
  TASK_A="Grasp-Allegro"
  TASK_B="Grasp-LEAP"
  DISPLAY_FAMILY="Grasp"
fi

read -ra TARGETS <<< "${TARGET_SIZES:-10k 50k 100k 400k}"
read -ra SIGMAS <<< "${SIGMA_LIST:-50 75 90}"

N_SIGMAS=${#SIGMAS[@]}
n_idx=$(( SLURM_ARRAY_TASK_ID / N_SIGMAS ))
sigma_idx=$(( SLURM_ARRAY_TASK_ID % N_SIGMAS ))
N_BUDGET="${TARGETS[$n_idx]}"
SIGMA="${SIGMAS[$sigma_idx]}"

SEED="${SEED:-0}"
SEED_SUFFIX=""
[[ "$SEED" != "0" ]] && SEED_SUFFIX="_s${SEED}"

if [[ "$N_BUDGET" == "400k" ]]; then
  DATASET="data/mixed_noc/${DISPLAY_FAMILY}_A400k_L400k.zarr"
else
  DATASET="data/mixed_noc/${DISPLAY_FAMILY}_A${N_BUDGET}_L400k.zarr"
fi
OUTPUT_DIR="outputs/ambient/${DISPLAY_FAMILY}_A${N_BUDGET}_L400k_sigma${SIGMA}${SEED_SUFFIX}"

N_STEPS=$(.venv/bin/python -c "
from mjlab_hand.diffusion.dataset import TrajectoryStore
print(TrajectoryStore('${DATASET}', mode='r').n_steps)
")
BATCHES_PER_EPOCH=$(( N_STEPS / 256 ))
TARGET_STEPS=784000
EPOCHS=$(( (TARGET_STEPS + BATCHES_PER_EPOCH - 1) / BATCHES_PER_EPOCH ))
EVAL_EVERY=$(( EPOCHS / 2 ))
SAVE_EVERY=$(( EPOCHS / 10 ))
[[ "$SAVE_EVERY" -lt 1 ]] && SAVE_EVERY=1

echo "[INFO] ${FAMILY} N=${N_BUDGET} sigma*=${SIGMA}: dataset=$DATASET n_steps=$N_STEPS epochs=$EPOCHS"

EVAL_SPEC="[{\"task\": \"${TASK_A}\", \"onehot\": null}, {\"task\": \"${TASK_B}\", \"onehot\": null}]"

train-diffusion \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --num-epochs "$EPOCHS" \
  --save-every-epochs "$SAVE_EVERY" \
  --latest-every-epochs "$SAVE_EVERY" \
  --eval-spec "$EVAL_SPEC" \
  --eval-every-epochs "$EVAL_EVERY" \
  --eval-num-envs "${EVAL_ENVS:-32}" \
  --ambient-tmin 0 "$SIGMA" \
  --seed "$SEED"
