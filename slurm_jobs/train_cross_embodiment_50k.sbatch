#!/bin/bash
#SBATCH --job-name=train-cross-embod-50k
#SBATCH --output=slurm_jobs/train_cross_embodiment/logs/task50k_%A_%a.out
#SBATCH --error=slurm_jobs/train_cross_embodiment/logs/task50k_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#
# 50k-scale companion to train_cross_embodiment.sbatch (job 85823, the full
# ~5M-step runs). Same padded/masked/per-source-normalized cross-embodiment
# scheme, same task families, just the smaller pooled datasets built from
# scripts/build_padded_dataset.py over the *_10k per-embodiment subsets (see
# agent_logbook/COLLECTIONS.md, "Cross-embodiment padded datasets").
# No node pin -- see train_cross_embodiment.sbatch: this is pure GPU
# training plus a small periodic eval rollout, not worth restricting to
# node-011 the way continuous-simulation RL/collection jobs are.
#
#   0  Grasp-AllHands_50k            50,060 steps, obs 189, act 28
#   1  InHand-Rotation-AllHands_50k  50,326 steps, obs 89,  act 22
#
# Epoch count follows the same ~780k-total-gradient-steps convention as the
# rest of this project's scale ladder -- 4000 epochs at batch 256 is exactly
# this project's existing EPOCHS_50K default (slurm_jobs/train_diffusion.sbatch).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/train_cross_embodiment_50k_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

DATA_DIR="data/mjlab_hand_demos/padded"

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  FAMILY="Grasp"
  DATASET="${DATA_DIR}/Grasp-AllHands_50k.zarr"
  OUTPUT_DIR="outputs/diffusion/Grasp-AllHands_50k_padded"
  EVAL_SPEC=$(python3 -c "
import json
hands=['Allegro','LEAP','Shadow','Sharpa','Wuji']
print(json.dumps([{'task': f'Grasp-{h}', 'embodiment': f'Grasp-{h}'} for h in hands]))
")
else
  FAMILY="InHand-Rotation"
  DATASET="${DATA_DIR}/InHand-Rotation-AllHands_50k.zarr"
  OUTPUT_DIR="outputs/diffusion/InHand-Rotation-AllHands_50k_padded"
  EVAL_SPEC=$(python3 -c "
import json
hands=['Allegro','LEAP','Shadow','Sharpa','Wuji']
print(json.dumps([{'task': f'InHand-Rotation-{h}', 'embodiment': f'InHand-Rotation-{h}'} for h in hands]))
")
fi

echo "[INFO] array_task=${SLURM_ARRAY_TASK_ID} family=${FAMILY} dataset=${DATASET} output=${OUTPUT_DIR}"
echo "[INFO] eval_spec=${EVAL_SPEC}"

train-diffusion \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --num-epochs 4000 \
  --batch-size 256 \
  --save-every-epochs 400 \
  --latest-every-epochs 400 \
  --eval-spec "$EVAL_SPEC" \
  --eval-every-epochs 1000 \
  --eval-num-envs 16 \
  --eval-num-steps 800 \
  --num-workers 8 \
  --seed 0 \
  --wandb-project mjlab \
  --wandb-run-name "${FAMILY}-AllHands-50k-padded" \
  --wandb-tags cross-embodiment padded 50k
