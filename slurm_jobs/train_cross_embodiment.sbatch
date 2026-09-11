#!/bin/bash
#SBATCH --job-name=train-cross-embod
#SBATCH --output=slurm_jobs/train_cross_embodiment/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/train_cross_embodiment/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=36:00:00
#SBATCH --array=0-1
#
# Trains the two cross-embodiment diffusion policies from
# scripts/build_padded_dataset.py's pooled/padded/per-source-normalized
# datasets (data/COLLECTIONS.md, 2026-09-10): one policy per task family
# (Grasp, InHand-Rotation), each taking obs/action from an arbitrary one of
# its 5 embodiments via zero-padding to the family's max obs/action dim.
# No node pin: unlike RL training/demo collection, this job is almost all
# pure GPU tensor training; the stale-driver mujoco_warp slowdown (see
# agent_logbook/RUNS.md) only touches the periodic --eval-spec rollout
# (16 envs x 800 steps, every 10 epochs here), too small a fraction of
# runtime to justify restricting scheduling to node-011.
#
#   0  Grasp-AllHands            5,001,608 steps, obs 189, act 28
#   1  InHand-Rotation-AllHands  4,897,978 steps, obs 89,  act 22
#
# Epoch count follows this project's established total-gradient-steps
# convention (~780k steps regardless of dataset scale, see
# slurm_jobs/train_diffusion.sbatch's per-size EPOCHS table): both pooled
# sets are ~5x the 1M/200-epoch reference point, so 40 epochs at batch 256
# lands in the same ballpark (~19.5k steps/epoch * 40 ~= 780k).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/train_cross_embodiment_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

DATA_DIR="/datastor2/mrudolph/mjlab_hand_demos/padded"
HANDS=(Allegro LEAP Shadow Sharpa Wuji)

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  FAMILY="Grasp"
  DATASET="${DATA_DIR}/Grasp-AllHands.zarr"
  OUTPUT_DIR="outputs/diffusion/Grasp-AllHands_padded"
  EVAL_SPEC=$(python3 -c "
import json
hands=['Allegro','LEAP','Shadow','Sharpa','Wuji']
print(json.dumps([{'task': f'Grasp-{h}', 'embodiment': f'Grasp-{h}'} for h in hands]))
")
else
  FAMILY="InHand-Rotation"
  DATASET="${DATA_DIR}/InHand-Rotation-AllHands.zarr"
  OUTPUT_DIR="outputs/diffusion/InHand-Rotation-AllHands_padded"
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
  --num-epochs 40 \
  --batch-size 256 \
  --save-every-epochs 4 \
  --latest-every-epochs 4 \
  --eval-spec "$EVAL_SPEC" \
  --eval-every-epochs 10 \
  --eval-num-envs 16 \
  --eval-num-steps 800 \
  --num-workers 8 \
  --seed 0 \
  --wandb-project mjlab \
  --wandb-run-name "${FAMILY}-AllHands-padded" \
  --wandb-tags cross-embodiment padded
