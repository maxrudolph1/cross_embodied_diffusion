#!/bin/bash
#SBATCH --job-name=collect-val
#SBATCH --output=slurm_jobs/vista_train_manifest/logs/collect_val_%j.out
#SBATCH --error=slurm_jobs/vista_train_manifest/logs/collect_val_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#
# Fresh expert rollouts for the held-out validation stores (CHANGES.md item 63).
# Bundle built data/val/<family>_val_20k.zarr from its independent 10M
# collections, which are not on Vista; this collects a new independent set
# from the SAME expert checkpoints that collected each hand's 1M store (read
# from the store's `checkpoint` attr), same settings (256 envs, 500-step
# episodes, success-only) but seed VAL_SEED (default 1000; the 1M sets used 0).
# Then run scripts/build_val_split.py (CPU, any node) to drop episodes that
# also start a training episode and assemble the term-aligned val stores.
#
#   sbatch -A ASC26008 slurm_jobs/vista_collect_val.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate
# The shared venv's editable install points at one checkout's src/; put THIS
# checkout's src/ first so a job runs the code it was submitted from (e.g. a
# git worktree on another branch). CHANGES.md item 63.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
JOBTMP="/tmp/${USER}/${SLURM_JOB_ID}_val"
export TMPDIR="$JOBTMP/tmp" WARP_CACHE_PATH="$JOBTMP/warp"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"
trap 'rm -rf "$JOBTMP"' EXIT

VAL_SEED="${VAL_SEED:-1000}"
EPISODES="${EPISODES:-64}"   # ~30k steps per hand before filtering; 20k are kept
OUT=data/mjlab_hand_demos/val/raw
mkdir -p "$OUT"
for fam in Grasp InHand-Rotation; do
  for hand in Allegro LEAP Shadow Sharpa Wuji; do
    task="$fam-$hand"
    src="data/mjlab_hand_demos/${task}_expert_1M.zarr"
    ckpt=$(python3 -c "import zarr,sys; print(zarr.open_group(sys.argv[1], mode='r').attrs['checkpoint'])" "$src")
    echo "[INFO] $task <- $ckpt (seed $VAL_SEED, $EPISODES episodes)"
    stdbuf -oL -eL collect-demos --task "$task" --checkpoint "$ckpt" \
      --output "$OUT/${task}_val_seed${VAL_SEED}.zarr" \
      --num-envs 256 --num-episodes "$EPISODES" --max-episode-steps 500 --seed "$VAL_SEED"
  done
done
