#!/bin/bash
#SBATCH --job-name=vista-eval
#SBATCH --output=slurm_jobs/vista_train_manifest/logs/eval_%A_%a.out
#SBATCH --error=slurm_jobs/vista_train_manifest/logs/eval_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#
# Fresh-seed reporting evals (scripts/eval_checkpoints.py: policy_best_eval,
# policy_best_val, policy_latest on each run's target, 100 envs, env seed
# 1000 + training seed) for finished runs on $SCRATCH. CHANGES.md item 60.
#
#   RUNS      glob(s), relative to OUT_ROOT (default
#             $SCRATCH/cross_embodied_diffusion/outputs), quoted
#   PARALLEL  eval processes sharing the node's GPU (default 4)
#   ALSO      optional extra embodiment to score, e.g. InHand-Rotation-LEAP
#
# Each array element takes its share of the runs: process p of element a is
# shard (a * PARALLEL + p) of (n_elements * PARALLEL). Already-scored
# checkpoints are skipped, so a re-submit only does what is missing.
#
#   sbatch -A ASC26008 --export=ALL,RUNS='diffusion/diag_rot/*_minmax_seed0' \
#     slurm_jobs/vista_eval_checkpoints.sbatch
#   sbatch -A ASC26008 --array=0-9 --export=ALL,RUNS='diffusion/ambient_rot/*' \
#     slurm_jobs/vista_eval_checkpoints.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

: "${RUNS:?set RUNS='glob relative to the outputs root' via --export}"
OUT_ROOT="${OUT_ROOT:-${SCRATCH:?}/cross_embodied_diffusion/outputs}"
PARALLEL="${PARALLEL:-4}"
ALSO_ARGS=()
[[ -n "${ALSO:-}" ]] && ALSO_ARGS=(--also "$ALSO")

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CC=gcc
JOBTMP="/tmp/${USER}/${SLURM_JOB_ID}_eval"
export TMPDIR="$JOBTMP/tmp" WARP_CACHE_PATH="$JOBTMP/warp"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"
trap 'rm -rf "$JOBTMP"' EXIT

shopt -s nullglob
read -ra PATTERNS <<< "$RUNS"
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
  WARP_CACHE_PATH="$JOBTMP/warp/p$p" python3 scripts/eval_checkpoints.py \
    --shard "$shard/$TOTAL" "${ALSO_ARGS[@]}" "${DIRS[@]}" &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit $fail
