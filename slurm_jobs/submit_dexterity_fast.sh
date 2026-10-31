#!/bin/bash
# One-command bootstrap -> continuation for the eight hands/tasks not already
# running as local Allegro pilots. Run from a Vista LOGIN node:
#
#   bash slurm_jobs/submit_dexterity_fast.sh
#
# First inspect scheduler estimates without submitting:
#   PROBE_ONLY=1 bash slurm_jobs/submit_dexterity_fast.sh
# On any node, review both submission commands and mappings without submitting:
#   DRY_RUN=1 bash slurm_jobs/submit_dexterity_fast.sh
#
# Defaults: gh-dev 4 nodes x PACK=2, 500 iterations, 2 hours; then gh eight
# one-node jobs, PACK=1, 1k/3k/10k cumulative iterations, 8 hours. The continuation
# uses afterany so a timed-out bootstrap can continue from its latest intact
# checkpoint (ZIP/CRC validation skips an interrupted save). If none is valid,
# the continuation fails explicitly, never restarting from random weights.
# All commands pin the same committed SOURCE_REF. The manifest and source archive
# are created by submit_dexterity_experts.sh; job IDs are read from its .job_id file.
#
# Overrides: TASKS, SOURCE_REF, RUN_TAG, NUM_ENVS, SEED, EVAL_SEED, EVAL_ENVS,
# BOOT_NODES, BOOT_PACK, BOOT_STAGES, BOOT_TIME, FULL_NODES, FULL_PACK,
# FULL_STAGES, FULL_TIME, ACCOUNT, OUT_ROOT. Measure queue/runtime before changing
# shapes. gh-dev allows only one running job/user and eight nodes/job.
set -euo pipefail
cd "$(dirname "$0")/.."
DRY_RUN="${DRY_RUN:-0}"
PROBE_ONLY="${PROBE_ONLY:-0}"
HOST_SHORT="$(hostname -s)"
if [[ "$DRY_RUN" != 1 && "$HOST_SHORT" =~ ^[ci][0-9]+-[0-9]+$ ]]; then
  echo "[ERROR] Vista disables sbatch (including --test-only) on compute node $HOST_SHORT." >&2
  echo 'Run this command from a login node. DRY_RUN=1 can be used here without submitting.' >&2
  exit 2
fi
ACCOUNT="${ACCOUNT:-ASC26008}"
BOOT_NODES="${BOOT_NODES:-4}"
BOOT_PACK="${BOOT_PACK:-2}"
BOOT_STAGES="${BOOT_STAGES:-500}"
BOOT_TIME="${BOOT_TIME:-02:00:00}"
FULL_NODES="${FULL_NODES:-1}"
FULL_PACK="${FULL_PACK:-1}"
FULL_STAGES="${FULL_STAGES:-1000 3000 10000}"
FULL_TIME="${FULL_TIME:-08:00:00}"
if [[ "$PROBE_ONLY" == 1 ]]; then
  result=0
  for partition in gh-dev gh; do
    if [[ "$partition" == gh-dev ]]; then sizes=(2 4 8); limit="$BOOT_TIME"
    else sizes=(1 2 4 8); limit="$FULL_TIME"; fi
    for nodes in "${sizes[@]}"; do
      command=(sbatch --test-only -A "$ACCOUNT" -p "$partition" -N "$nodes" -t "$limit" --wrap=true)
      printf 'Probe: '; printf '%q ' "${command[@]}"; printf '\n'
      if [[ "$DRY_RUN" != 1 ]]; then
        if ! "${command[@]}"; then result=1; fi
      fi
    done
  done
  exit "$result"
fi
TASKS="${TASKS:-Baoding-LEAP Baoding-Shadow Baoding-Sharpa Baoding-Wuji Pen-Spinning-LEAP Pen-Spinning-Shadow Pen-Spinning-Sharpa Pen-Spinning-Wuji}"
.venv/bin/python - "$BOOT_NODES" "$BOOT_PACK" "$BOOT_STAGES" "$FULL_STAGES" "$TASKS" <<'PY'
import re, sys
nodes, pack = map(int, sys.argv[1:3])
boot, full = (list(map(int, text.split())) for text in sys.argv[3:5])
tasks = re.split(r'[\s,]+', sys.argv[5].strip())
if not 1 <= nodes <= 8 or pack < 1:
    raise SystemExit('Bootstrap requires 1..8 gh-dev nodes and positive PACK')
if (len(tasks) + nodes * pack - 1) // (nodes * pack) > 3:
    raise SystemExit('Bootstrap shape exceeds gh-dev three-submitted-job cap before existing jobs')
if not boot or not full or min(boot + full) <= 0 or any(b <= a for a,b in zip(boot,boot[1:])) or any(b <= a for a,b in zip(full,full[1:])):
    raise SystemExit('BOOT_STAGES/FULL_STAGES must be positive increasing cumulative counts')
if full[0] <= boot[-1]:
    raise SystemExit('First FULL_STAGES count must exceed the last BOOT_STAGES count')
PY
SOURCE_REVISION="$(git rev-parse "${SOURCE_REF:-HEAD}^{commit}")"
RUN_TAG="${RUN_TAG:-dexterity_fast_$(date -u +%Y%m%d_%H%M%S)}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo '[ERROR] Invalid RUN_TAG' >&2; exit 1; }
BOOT_TAG="${RUN_TAG}_boot"
FULL_TAG="${RUN_TAG}_full"
SUBMISSIONS=slurm_jobs/vista_dexterity_experts/submissions
for tag in "$BOOT_TAG" "$FULL_TAG"; do
  for suffix in json tar.gz sbatch job_id; do
    if [[ -e "$SUBMISSIONS/$tag.$suffix" ]]; then
      echo "[ERROR] Existing submission artifact $SUBMISSIONS/$tag.$suffix; choose a new RUN_TAG." >&2
      exit 1
    fi
  done
done
COMMON=(env "DRY_RUN=$DRY_RUN" "ACCOUNT=$ACCOUNT" "SOURCE_REF=$SOURCE_REVISION" "TASKS=$TASKS")
BOOT_COMMAND=("${COMMON[@]}" PARTITION=gh-dev "NODES=$BOOT_NODES" "PACK=$BOOT_PACK"
  "STAGES=$BOOT_STAGES" "TIME_LIMIT=$BOOT_TIME" "RUN_TAG=$BOOT_TAG" DEPENDENCY= RESUME_ROOT=
  bash slurm_jobs/submit_dexterity_experts.sh)
printf 'Bootstrap: '; printf '%q ' "${BOOT_COMMAND[@]}"; printf '\n'
# set -e stops here if bootstrap preparation/submission fails.
"${BOOT_COMMAND[@]}"
if [[ "$DRY_RUN" == 1 ]]; then
  BOOT_JOB_ID='<BOOTSTRAP_JOB_ID>'
  echo '[DRY RUN] No bootstrap was submitted; continuation dependency below is a placeholder.'
else
  BOOT_JOB_ID="$(cat "$SUBMISSIONS/$BOOT_TAG.job_id")"
  [[ "$BOOT_JOB_ID" =~ ^[0-9]+$ ]] || { echo '[ERROR] Invalid bootstrap job ID; stopping.' >&2; exit 1; }
fi
FULL_COMMAND=("${COMMON[@]}" PARTITION=gh "NODES=$FULL_NODES" "PACK=$FULL_PACK"
  "STAGES=$FULL_STAGES" "TIME_LIMIT=$FULL_TIME" "RUN_TAG=$FULL_TAG"
  "DEPENDENCY=afterany:$BOOT_JOB_ID" "RESUME_ROOT=outputs/dexterity/$BOOT_TAG"
  bash slurm_jobs/submit_dexterity_experts.sh)
printf 'Continuation: '; printf '%q ' "${FULL_COMMAND[@]}"; printf '\n'
"${FULL_COMMAND[@]}"
if [[ "$DRY_RUN" != 1 ]]; then
  FULL_JOB_ID="$(cat "$SUBMISSIONS/$FULL_TAG.job_id")"
  [[ "$FULL_JOB_ID" =~ ^[0-9]+$ ]] || { echo '[ERROR] Invalid continuation job ID.' >&2; exit 1; }
  echo "Submitted bootstrap $BOOT_JOB_ID and continuation $FULL_JOB_ID from source $SOURCE_REVISION"
fi
