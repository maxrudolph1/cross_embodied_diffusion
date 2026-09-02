#!/bin/bash
#SBATCH --job-name=train-diffusion
#SBATCH --output=slurm_jobs/train_diffusion/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/train_diffusion/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --array=0-9

# Trains one single-embodiment diffusion policy per (hand, size) array cell.
#
#   FAMILY=grasp|rotation           task family (default grasp)
#   HANDS_LIST="allegro leap ..."   override which embodiments the array
#                                   covers (default: all 5). Array length is
#                                   len(HANDS) * len(SIZES) -- derived, not
#                                   hardcoded, so a partial HANDS_LIST maps
#                                   indices correctly.
#   SIZES_LIST="10k 100k 1M 10M"    data scales to train (default "1M 10M")
#   SEED                            appends "_s<SEED>" to the output dir so
#                                   a second seed cannot overwrite the first
#                                   run's directory or its eval_metrics.jsonl
#   EVAL_ENVS                       rollout eval env count (default 32)
#   NUM_WORKERS                     DataLoader workers (default 8)
#   EPOCHS_10K/50K/100K/400K/1M/10M override the per-scale epoch count
#   EVAL_FINAL_ONLY=1                eval only at the last epoch (skips the
#                                    midpoint eval) -- for a quick "does this
#                                    policy work at all" check rather than a
#                                    learning curve
#
# Eval fires exactly at the midpoint and the end (EVAL_EVERY = EPOCHS / 2)
# by default, at every scale, unless EVAL_FINAL_ONLY=1. Checkpoint cadence
# (LATEST_EVERY) matches SAVE_EVERY.
#
# See CHANGES.md items 6d, 6f, 18, 27, 34.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

# rlcompute (H200) nodes ship a stale /usr/local/cuda/compat libcuda that
# ldconfig prefers over the real driver -> CUDA error 803 on every call.
export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Concurrent array tasks sharing one Warp kernel cache race and fail with a
# CCD-kernel FileNotFoundError. Give each task its own.
export WARP_CACHE_PATH="$PWD/warp_cache/train_diffusion_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

FAMILY="${FAMILY:-grasp}"
if [[ "$FAMILY" == "rotation" ]]; then
  TASK_PREFIX="InHand-Rotation"
else
  TASK_PREFIX="Grasp"
fi

DEFAULT_HANDS=(allegro leap shadow sharpa wuji)
read -ra HANDS <<< "${HANDS_LIST:-${DEFAULT_HANDS[*]}}"
read -ra SIZES <<< "${SIZES_LIST:-1M 10M}"

N_HANDS=${#HANDS[@]}
hand_idx=$(( SLURM_ARRAY_TASK_ID % N_HANDS ))
size_idx=$(( SLURM_ARRAY_TASK_ID / N_HANDS ))
HAND="${HANDS[$hand_idx]}"
SIZE="${SIZES[$size_idx]}"

case "$HAND" in
  allegro) HAND_DISPLAY="Allegro" ;;
  leap)    HAND_DISPLAY="LEAP" ;;
  shadow)  HAND_DISPLAY="Shadow" ;;
  sharpa)  HAND_DISPLAY="Sharpa" ;;
  wuji)    HAND_DISPLAY="Wuji" ;;
  *) echo "unknown hand $HAND" >&2; exit 1 ;;
esac
TASK="${TASK_PREFIX}-${HAND_DISPLAY}"

case "$SIZE" in
  10k)  EPOCHS="${EPOCHS_10K:-20000}"; SAVE_EVERY=2000 ;;
  50k)  EPOCHS="${EPOCHS_50K:-4000}";  SAVE_EVERY=400  ;;
  100k) EPOCHS="${EPOCHS_100K:-2000}"; SAVE_EVERY=200  ;;
  400k) EPOCHS="${EPOCHS_400K:-500}";  SAVE_EVERY=50   ;;
  1M)   EPOCHS="${EPOCHS_1M:-200}";    SAVE_EVERY=20   ;;
  10M)  EPOCHS="${EPOCHS_10M:-20}";    SAVE_EVERY=2    ;;
  *)    echo "unknown size $SIZE" >&2; exit 1 ;;
esac
if [[ "${EVAL_FINAL_ONLY:-0}" == "1" ]]; then
  EVAL_EVERY="$EPOCHS"
else
  EVAL_EVERY=$(( EPOCHS / 2 ))
fi
LATEST_EVERY="$SAVE_EVERY"

SEED="${SEED:-0}"
SEED_SUFFIX=""
[[ "$SEED" != "0" ]] && SEED_SUFFIX="_s${SEED}"
EVAL_ENVS="${EVAL_ENVS:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"

DATASET="data/demos/${TASK}_expert_${SIZE}.zarr"
OUTPUT_DIR="outputs/diffusion/${TASK}_${SIZE}${SEED_SUFFIX}"

echo "[INFO] task=$TASK size=$SIZE seed=$SEED epochs=$EPOCHS eval_every=$EVAL_EVERY"
echo "[INFO] dataset=$DATASET output=$OUTPUT_DIR"

train-diffusion \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --num-epochs "$EPOCHS" \
  --save-every-epochs "$SAVE_EVERY" \
  --latest-every-epochs "$LATEST_EVERY" \
  --eval-task "$TASK" \
  --eval-every-epochs "$EVAL_EVERY" \
  --eval-num-envs "$EVAL_ENVS" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED"
