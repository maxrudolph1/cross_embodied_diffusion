#!/bin/bash
# Prepare immutable source/manifest, print exact command and submit from LOGIN.
# Defaults: ten tasks, seed 0, one task/GH200, 1k/3k/10k cumulative iterations.
# Examples (pick nodes/time using queue_wait_stats + sbatch --test-only first):
#   DRY_RUN=1 NODES=2 bash slurm_jobs/submit_dexterity_experts.sh
#   NODES=5 TIME_LIMIT=08:00:00 bash slurm_jobs/submit_dexterity_experts.sh
#   TASKS='Baoding-LEAP Pen-Spinning-LEAP' NODES=1 bash slurm_jobs/submit_dexterity_experts.sh
# NODES=1/2/5/10 yields 10/5/2/1 array elements at PACK=1 for all ten tasks.
# TASKS may list any subset (space- or comma-separated), e.g. omit local Allegro.
# STAGES='1000 3000 10000', NUM_ENVS=1024, SEED=0, EVAL_SEED=1234, PACK=1,
# RUN_TAG=<unique>, OUT_ROOT=<scratch outputs root>, ACCOUNT=ASC26008 override.
# PARTITION=gh-dev uses a 2-hour default; PARTITION=gh uses 12 hours by default.
# For bootstrap -> continuation, submit both from identical SOURCE_REF=<commit>:
#   TASKS='...' PARTITION=gh-dev NODES=4 PACK=2 STAGES=500 RUN_TAG=boot ...
#   TASKS='...' PARTITION=gh NODES=1 DEPENDENCY=afterany:<jobid> \
#     RESUME_ROOT=outputs/dexterity/boot RUN_TAG=long ...
# Continuation selects the latest intact per-task checkpoint after its dependency;
# ZIP/CRC validation skips a save interrupted by a time limit. If none is intact,
# it fails explicitly. Never silently restarts from random weights.
# DRY_RUN creates only reproducibility files and prints every slot; no submission.
set -euo pipefail
cd "$(dirname "$0")/.."
NODES="${NODES:-1}"
PACK="${PACK:-1}"
PARTITION="${PARTITION:-gh}"
case "$PARTITION" in
  gh) TIME_LIMIT="${TIME_LIMIT:-12:00:00}" ;;
  gh-dev) TIME_LIMIT="${TIME_LIMIT:-02:00:00}" ;;
  *) echo '[ERROR] PARTITION must be gh or gh-dev' >&2; exit 1 ;;
esac
ACCOUNT="${ACCOUNT:-ASC26008}"
SOURCE_REF="${SOURCE_REF:-HEAD}"
export SOURCE_REVISION="$(git rev-parse "${SOURCE_REF}^{commit}")"
RUN_TAG="${RUN_TAG:-dexterity_$(date -u +%Y%m%d_%H%M%S)}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo '[ERROR] Invalid RUN_TAG' >&2; exit 1; }
export NODES PACK RUN_TAG
export NUM_ENVS="${NUM_ENVS:-1024}" SEED="${SEED:-0}" EVAL_SEED="${EVAL_SEED:-1234}"
export EVAL_ENVS="${EVAL_ENVS:-100}" STAGES="${STAGES:-1000 3000 10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-250}" KEEP_LAST="${KEEP_LAST:-3}"
export TASKS="${TASKS:-}"
export RESUME_ROOT="${RESUME_ROOT:-}"
mkdir -p slurm_jobs/vista_dexterity_experts/{logs,submissions}
BASE="slurm_jobs/vista_dexterity_experts/submissions/$RUN_TAG"
export DEXTERITY_MANIFEST="$BASE.json"
export DEXTERITY_SOURCE_ARCHIVE="$BASE.tar.gz"
export DEXTERITY_LAUNCHER="$BASE.sbatch"
if [[ -e "$DEXTERITY_MANIFEST" || -e "$DEXTERITY_SOURCE_ARCHIVE" ]]; then
  echo "[ERROR] RUN_TAG already exists: $RUN_TAG; use a new tag" >&2; exit 1
fi
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  git diff --quiet "$SOURCE_REVISION" -- src scripts slurm_jobs pyproject.toml uv.lock || {
    echo '[ERROR] Commit source/launcher edits before submission so source is reproducible.' >&2; exit 1;
  }
  git cat-file -e "$SOURCE_REVISION:scripts/run_dexterity_batch_node.py" || {
    echo '[ERROR] Commit the new launcher before submission.' >&2; exit 1;
  }
fi
git archive --format=tar.gz -o "$DEXTERITY_SOURCE_ARCHIVE" "$SOURCE_REVISION" \
  src scripts slurm_jobs configs asset pyproject.toml uv.lock README.md
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  cp slurm_jobs/vista_dexterity_experts.sbatch "$DEXTERITY_LAUNCHER"
else
  git show "$SOURCE_REVISION:slurm_jobs/vista_dexterity_experts.sbatch" > "$DEXTERITY_LAUNCHER"
fi
.venv/bin/python - "$DEXTERITY_MANIFEST" <<'PY'
import json, os, re, sys
from pathlib import Path
hands = ('Allegro', 'LEAP', 'Shadow', 'Sharpa', 'Wuji')
all_tasks = [f'{family}-{hand}' for family in ('Baoding', 'Pen-Spinning') for hand in hands]
tasks = re.split(r'[\s,]+', os.environ['TASKS'].strip()) if os.environ['TASKS'].strip() else all_tasks
if len(set(tasks)) != len(tasks) or any(t not in all_tasks for t in tasks):
    raise SystemExit('TASKS must contain unique registered dexterity task names')
nodes, pack = int(os.environ['NODES']), int(os.environ['PACK'])
if not 1 <= nodes <= 64 or pack < 1:
    raise SystemExit('NODES must be 1..64 and PACK must be positive')
stages = list(map(int, os.environ['STAGES'].split()))
if not stages or stages[0] <= 0 or any(b <= a for a,b in zip(stages,stages[1:])):
    raise SystemExit('STAGES must be positive increasing cumulative counts')
seed, eval_seed = int(os.environ['SEED']), int(os.environ['EVAL_SEED'])
if eval_seed == 0 or seed == eval_seed:
    raise SystemExit('Use a nonzero held-out EVAL_SEED different from SEED')
if not re.fullmatch(r'[A-Za-z0-9_-]+', os.environ['RUN_TAG']):
    raise SystemExit('RUN_TAG may contain only letters, digits, underscore and hyphen')
resume_root = os.environ['RESUME_ROOT']
if resume_root and (Path(resume_root).is_absolute() or '..' in Path(resume_root).parts or not resume_root.startswith('outputs/')):
    raise SystemExit('RESUME_ROOT must be repository-relative under outputs/')
manifest = dict(version=1, source_revision=os.environ['SOURCE_REVISION'], resume_root=resume_root,
                output_prefix=f"dexterity/{os.environ['RUN_TAG']}", stages=stages,
                eval_seed=eval_seed, eval_envs=int(os.environ['EVAL_ENVS']),
                save_interval=int(os.environ['SAVE_INTERVAL']), keep_last=int(os.environ['KEEP_LAST']),
                tasks=[dict(task=t,seed=seed,num_envs=int(os.environ['NUM_ENVS'])) for t in tasks])
if min(manifest['eval_envs'],manifest['save_interval'],manifest['keep_last'],manifest['tasks'][0]['num_envs']) < 1:
    raise SystemExit('Environment/checkpoint counts must be positive')
Path(sys.argv[1]).write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Tasks={len(tasks)} nodes/job={nodes} PACK={pack} array_elements={(len(tasks)+nodes*pack-1)//(nodes*pack)}')
PY
COUNT=$(.venv/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"]))' "$DEXTERITY_MANIFEST")
JOBS=$(( (COUNT + NODES * PACK - 1) / (NODES * PACK) ))
if (( JOBS > 40 )); then echo '[ERROR] Shape exceeds 40 submitted jobs before existing queue usage' >&2; exit 1; fi
COMMAND=(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -N "$NODES" --array="0-$((JOBS-1))"
  --time="$TIME_LIMIT" --job-name=dexterity-rl
  --export="ALL,PACK=$PACK,DEXTERITY_MANIFEST=$DEXTERITY_MANIFEST,DEXTERITY_SOURCE_ARCHIVE=$DEXTERITY_SOURCE_ARCHIVE,DEXTERITY_LAUNCHER=$DEXTERITY_LAUNCHER"
  slurm_jobs/vista_dexterity_experts.sbatch)
if [[ -n "${DEPENDENCY:-}" ]]; then
  COMMAND=("${COMMAND[@]:0:${#COMMAND[@]}-1}" "--dependency=$DEPENDENCY" "${COMMAND[-1]}")
fi
printf 'Submit: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for ((a=0; a<JOBS; a++)); do
    for ((n=0; n<NODES; n++)); do
      .venv/bin/python scripts/run_dexterity_batch_node.py --manifest "$DEXTERITY_MANIFEST" \
        --array-index "$a" --nodes "$NODES" --node-id "$n" --pack "$PACK" --dry-run
    done
  done
  exit 0
fi
REPLY=$("${COMMAND[@]}")
printf '%s\n' "$REPLY"
# TACC prepends a banner even with --parsable. Match only a job-id response line.
JOB_ID=$(printf '%s\n' "$REPLY" | sed -nE \
  -e 's/^([0-9]+)(;[^[:space:]]+)?$/\1/p' \
  -e 's/^Submitted batch job ([0-9]+)$/\1/p' | tail -n 1)
if [[ -z "$JOB_ID" ]]; then
  echo '[ERROR] Submission returned no parseable job ID; inspect output before retrying.' >&2; exit 1
fi
printf '%s\n' "$JOB_ID" > "$BASE.job_id"
echo "Submitted job $JOB_ID; manifest=$DEXTERITY_MANIFEST"
