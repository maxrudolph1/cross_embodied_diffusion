#!/bin/bash
#SBATCH --job-name=vista-eval
#SBATCH --output=slurm_jobs/vista_train_manifest/logs/eval_%A_%a.out
#SBATCH --error=slurm_jobs/vista_train_manifest/logs/eval_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#
# Fresh-seed reporting re-score (scripts/rescore_selected.py) of finished runs
# on $SCRATCH: policy_best_rollout, policy_best_val and the newest snapshot
# (last0), each on the run's own eval spec, 100 envs x 1500 steps, eval seed
# 1234 -> <run>/final_eval.jsonl. CHANGES.md items 60, 63.
#
#   RUNS      glob(s), relative to OUT_ROOT (default
#             $SCRATCH/cross_embodied_diffusion/outputs), quoted
#   PARALLEL  re-score processes sharing the node's GPU (default 4)
#   WHICH     checkpoints (default "best_rollout best_val last0")
#   (RUNS and WHICH may separate items with ":" instead of spaces: sbatch --export
#    rejects values with spaces. CHANGES.md item 75)
#   EVAL_SEED env seed of the eval (default 1234)
#   CROSS=1   score every hand of the family with scripts/eval_cross_embodiment.py
#             --skip-target -> <run>/cross_eval.jsonl (forgetting; items 68, 74)
#
# Each array element takes its share of the runs: process p of element a is
# shard (a * PARALLEL + p) of (n_elements * PARALLEL). Rows already in
# final_eval.jsonl are skipped, so a re-submit only does what is missing.
# Runs without selection.json (unfinished, or trained before item 63) are skipped.
#
#   sbatch -A ASC26008 --export=ALL,RUNS='diffusion/ambient_rot_ta/*' \
#     slurm_jobs/vista_eval_checkpoints.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate
# The shared venv's editable install points at one checkout's src/; put THIS
# checkout's src/ first so a job runs the code it was submitted from (e.g. a
# git worktree on another branch). CHANGES.md item 63.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

: "${RUNS:?set RUNS='glob relative to the outputs root' via --export}"
OUT_ROOT="${OUT_ROOT:-${SCRATCH:?}/cross_embodied_diffusion/outputs}"
PARALLEL="${PARALLEL:-4}"
read -ra WHICH_ARGS <<< "${WHICH:-best_rollout best_val last0}"; WHICH_ARGS=(${WHICH_ARGS[@]//:/ })
EVAL_SEED="${EVAL_SEED:-1234}"
SCRIPT=scripts/rescore_selected.py EXTRA=()
[[ "${CROSS:-0}" == 1 ]] && SCRIPT=scripts/eval_cross_embodiment.py EXTRA=(--skip-target)

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
JOBTMP="/tmp/${USER}/${SLURM_JOB_ID}_eval"
export TMPDIR="$JOBTMP/tmp" WARP_CACHE_PATH="$JOBTMP/warp"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"
trap 'rm -rf "$JOBTMP"' EXIT

shopt -s nullglob
read -ra PATTERNS <<< "${RUNS//:/ }"
DIRS=()
for p in "${PATTERNS[@]}"; do DIRS+=("$OUT_ROOT"/$p); done
(( ${#DIRS[@]} )) || { echo "[ERROR] no run dirs match $RUNS under $OUT_ROOT" >&2; exit 1; }

N_ELEM="${SLURM_ARRAY_TASK_COUNT:-1}"
ELEM="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL=$(( N_ELEM * PARALLEL ))
echo "[INFO] $(hostname) ${#DIRS[@]} run dirs, element $ELEM/$N_ELEM, $PARALLEL procs"
pids=()
for (( p = 0; p < PARALLEL; p++ )); do
  shard=$(( ELEM * PARALLEL + p ))
  WARP_CACHE_PATH="$JOBTMP/warp/p$p" python3 "$SCRIPT" \
    --shard "$shard/$TOTAL" --which "${WHICH_ARGS[@]}" --envs 100 --steps 1500 --eval-seed "$EVAL_SEED" \
    "${EXTRA[@]}" --run "${DIRS[@]}" &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit $fail
