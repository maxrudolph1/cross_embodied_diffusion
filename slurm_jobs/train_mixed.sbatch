#!/bin/bash
#SBATCH --job-name=train-mixed
#SBATCH --output=slurm_jobs/train_mixed/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/train_mixed/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --exclude=sea112-rlcompute13
#SBATCH --array=0-19

# Trains a one-hot-conditioned mixed-embodiment policy for every ordered
# distinct-size pair built by build_mixed.sbatch (20 jobs per family).
#
# Epochs are DERIVED per run from the dataset, not tabulated -- mixed
# totals are irregular (10k+400k = 410k) -- so every run gets ~784000
# gradient steps (batch 256) regardless of its total step count.
#
#   FAMILY=grasp|rotation, SEED, EVAL_ENVS
#
# See CHANGES.md item 17.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/train_mixed_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

FAMILY="${FAMILY:-grasp}"
if [[ "$FAMILY" == "rotation" ]]; then
  TASK_A="InHand-Rotation-Allegro"
  TASK_B="InHand-Rotation-LEAP"
else
  TASK_A="Grasp-Allegro"
  TASK_B="Grasp-LEAP"
fi

SIZES=(10k 50k 100k 400k 1M)
PAIRS=()
for a in "${SIZES[@]}"; do
  for b in "${SIZES[@]}"; do
    [[ "$a" == "$b" ]] && continue
    PAIRS+=("${a}:${b}")
  done
done
PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
SIZE_A="${PAIR%%:*}"
SIZE_B="${PAIR##*:}"

SEED="${SEED:-0}"
SEED_SUFFIX=""
[[ "$SEED" != "0" ]] && SEED_SUFFIX="_s${SEED}"
EVAL_ENVS="${EVAL_ENVS:-32}"

DATASET="data/mixed/${FAMILY}_A${SIZE_A}_L${SIZE_B}.zarr"
OUTPUT_DIR="outputs/mixed/${FAMILY}_A${SIZE_A}_L${SIZE_B}${SEED_SUFFIX}"

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

echo "[INFO] ${FAMILY} A${SIZE_A}_L${SIZE_B}: n_steps=$N_STEPS epochs=$EPOCHS eval_every=$EVAL_EVERY"

EVAL_SPEC="[{\"task\": \"${TASK_A}\", \"onehot\": [1, 0]}, {\"task\": \"${TASK_B}\", \"onehot\": [0, 1]}]"

train-diffusion \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --num-epochs "$EPOCHS" \
  --save-every-epochs "$SAVE_EVERY" \
  --latest-every-epochs "$SAVE_EVERY" \
  --eval-spec "$EVAL_SPEC" \
  --eval-every-epochs "$EVAL_EVERY" \
  --eval-num-envs "$EVAL_ENVS" \
  --seed "$SEED"
