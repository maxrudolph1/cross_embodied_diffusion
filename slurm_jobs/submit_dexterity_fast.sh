#!/bin/bash
# GH-only fresh expert training for the eight non-Allegro task/hand combinations.
# Run from a Vista LOGIN node:
#   PROBE_ONLY=1 bash slurm_jobs/submit_dexterity_fast.sh  # scheduler projections
#   DRY_RUN=1 bash slurm_jobs/submit_dexterity_fast.sh    # commands only
#   bash slurm_jobs/submit_dexterity_fast.sh              # one submission
#
# Defaults: gh, eight one-node array jobs, one policy/GPU, cumulative evaluation
# stages at 1k/3k/10k iterations, 8 hours. Local Allegro runs are excluded.
# No bootstrap partition or dependency is used. SOURCE_REF pins committed code;
# submit_dexterity_experts.sh builds its immutable archive and stores the job ID.
# Overrides: TASKS, SOURCE_REF, RUN_TAG, NODES, PACK, STAGES, TIME_LIMIT, NUM_ENVS,
# SEED, EVAL_SEED, EVAL_ENVS, ACCOUNT, OUT_ROOT. PARTITION may only be gh.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${PARTITION:-gh}" != gh ]]; then
  echo '[ERROR] This launcher is gh-only; PARTITION must be gh.' >&2
  exit 1
fi
DRY_RUN="${DRY_RUN:-0}"
PROBE_ONLY="${PROBE_ONLY:-0}"
HOST_SHORT="$(hostname -s)"
if [[ "$DRY_RUN" != 1 && "$HOST_SHORT" =~ ^[ci][0-9]+-[0-9]+$ ]]; then
  echo "[ERROR] Vista disables sbatch (including --test-only) on compute node $HOST_SHORT." >&2
  echo 'Run this command from a login node. DRY_RUN=1 can be used here without submitting.' >&2
  exit 2
fi
ACCOUNT="${ACCOUNT:-ASC26008}"
NODES="${NODES:-1}"
PACK="${PACK:-1}"
STAGES="${STAGES:-1000 3000 10000}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
if [[ "$PROBE_ONLY" == 1 ]]; then
  result=0
  for nodes in 1 2 4 8; do
    command=(sbatch --test-only -A "$ACCOUNT" -p gh -N "$nodes" -t "$TIME_LIMIT" --wrap=true)
    printf 'Probe: '; printf '%q ' "${command[@]}"; printf '\n'
    if [[ "$DRY_RUN" != 1 ]]; then
      if ! "${command[@]}"; then result=1; fi
    fi
  done
  exit "$result"
fi
TASKS="${TASKS:-Baoding-LEAP Baoding-Shadow Baoding-Sharpa Baoding-Wuji Pen-Spinning-LEAP Pen-Spinning-Shadow Pen-Spinning-Sharpa Pen-Spinning-Wuji}"
SOURCE_REVISION="$(git rev-parse "${SOURCE_REF:-HEAD}^{commit}")"
RUN_TAG="${RUN_TAG:-dexterity_gh_$(date -u +%Y%m%d_%H%M%S)}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo '[ERROR] Invalid RUN_TAG' >&2; exit 1; }
COMMAND=(env "DRY_RUN=$DRY_RUN" "ACCOUNT=$ACCOUNT" "SOURCE_REF=$SOURCE_REVISION"
  "TASKS=$TASKS" PARTITION=gh "NODES=$NODES" "PACK=$PACK" "STAGES=$STAGES"
  "TIME_LIMIT=$TIME_LIMIT" "RUN_TAG=$RUN_TAG" DEPENDENCY= RESUME_ROOT=
  bash slurm_jobs/submit_dexterity_experts.sh)
printf 'GH training: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
"${COMMAND[@]}"
if [[ "$DRY_RUN" != 1 ]]; then
  JOB_ID="$(cat "slurm_jobs/vista_dexterity_experts/submissions/$RUN_TAG.job_id")"
  [[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo '[ERROR] Invalid submitted job ID.' >&2; exit 1; }
  echo "Submitted gh training array $JOB_ID from source $SOURCE_REVISION"
fi
