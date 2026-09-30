#!/bin/bash
#SBATCH --job-name=vista-manifest
#SBATCH --output=slurm_jobs/vista_train_manifest/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/vista_train_manifest/logs/task_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#
# TACC Vista (GH200, aarch64) launcher for a train-diffusion manifest built by
# scripts/build_scarce_specialist_manifest.py. Vista allocates (and bills)
# whole nodes, one GPU each, so each array task packs PACK manifest tasks
# (= 2*PACK runs, both seeds of each config) onto its node and runs them all
# concurrently. Measured 2026-09-27, full manifest config (8 workers, val
# every epoch), Grasp-Allegro 50k, runs per node -> s/epoch per run, node
# throughput vs 1 run, GPU util:
#   1 -> 7.1s 1.0x 35% | 2 -> 8.9s 1.6x 79% | 4 -> 15.8s 1.8x 85%
#   6 -> 22.7s 1.9x 93% | 8 -> 29.9s 1.9x 94% | 12 -> 45.1s 1.9x 95%
# Throughput plateaus at ~6 runs (GPU-bound). Limits on N: a 50k run is 4000
# epochs, so N=8 -> ~33h and N=12 -> ~50h (over the 48h wall); host RAM for
# the largest pool (padded/Grasp-AllHands) is ~12.5 GB/run (N=8 -> 137 of
# 212 GB). So PACK=2 (4 runs, ~18h for 50k) for turnaround, PACK=4 (8 runs,
# same SU/run within 5%, half the nodes) when the 20-running-job cap binds.
#
# COMPILE_MODE (default reduce-overhead; "none" = eager) torch.compiles the
# UNet (CHANGES.md item 53). With it the same sweep gives 1 -> 3.1s | 2 ->
# 5.9s | 4 -> 10.8s | 8 -> 21.8s per epoch: 2.3x faster alone, node
# throughput plateau 1.38x eager's. 50k runs: ~12h at 4/node, ~24h at 8/node.
#
# Submit from a LOGIN node (sbatch is disabled on compute/idev nodes), from
# the repo root, with the array sized to ceil(n_manifest_tasks / PACK):
#
#   mkdir -p slurm_jobs/vista_train_manifest/logs
#   sbatch -A <project> --array=0-4 \
#     --export=ALL,MANIFEST=slurm_jobs/rotation_specialist_manifest.json,PACK=2 \
#     slurm_jobs/vista_train_manifest.sbatch
#
# qgh limits: 48h wall, 20 running / 40 submitted jobs per user (each array
# element counts as one job), but 96 nodes running per user and 64 per job.
# So 1-node jobs cap you at 20 nodes; multi-node jobs reach 96.
#
# MULTI-NODE (CHANGES.md item 54): submit with -N K and every node runs its
# own PACK manifest tasks, like K separate 1-node jobs that count as ONE job.
# Node i of array element a takes the node slot a*K + i, i.e. manifest tasks
# [(a*K+i)*PACK, (a*K+i+1)*PACK). Size the array to
# ceil(n_manifest_tasks / (PACK*K)). Example: 80 tasks, PACK=4, K=10 -> 2 jobs:
#
#   sbatch -A <project> -N 10 --array=0-1 \
#     --export=ALL,MANIFEST=slurm_jobs/scarce_manifest.json,PACK=4 \
#     slurm_jobs/vista_train_manifest.sbatch
#
# The job holds (and bills) all K nodes until its slowest node finishes, so
# put equal-length runs in the same job. A node whose runs fail does not stop
# the others (--kill-on-bad-exit=0). Measured trade-offs: ANALYSIS.md
# "Vista job shape".
#
# STORAGE (CHANGES.md item 55): demos are staged to $SCRATCH (1-node) or node
# /tmp (multi-node) and outputs land in $SCRATCH/cross_embodied_diffusion/
# outputs; override with DATA_STAGE=scratch|tmp|none and OUT_ROOT=... in
# --export. Promote finished runs to Stockyard with scripts/promote_outputs.sh.
# DRY_RUN=1 stages and prints the train-diffusion commands without running.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"

# Batch step of a multi-node job: fan out one copy of this script per node.
# Re-run from the repo path, not $0 (sbatch runs a spool copy that exists
# only on the first node).
if [[ "${1:-}" != "--node" && "${SLURM_NNODES:-1}" -gt 1 ]]; then
  echo "[INFO] multi-node job: ${SLURM_NNODES} nodes ${SLURM_JOB_NODELIST}"
  exec srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=0 \
    bash slurm_jobs/vista_train_manifest.sbatch --node
fi
source .venv/bin/activate

: "${MANIFEST:?set MANIFEST=path/to/manifest.json via --export}"
PACK="${PACK:-2}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
COMPILE_ARGS=()
[[ "$COMPILE_MODE" != "none" ]] && COMPILE_ARGS=(--compile-mode "$COMPILE_MODE")

export MUJOCO_GL=egl
# TACC's default modules set CC=nvc; Triton (torch.compile) needs gcc to build
# its launcher stub, else InductorError: CalledProcessError on nvc.
export CC=gcc
export PYOPENGL_PLATFORM=egl
# Nodes are exclusive, so node-local /tmp is ours alone (no shared-/tmp ENOSPC
# from concurrent wandb services, unlike the shared-node cluster) and the path
# stays far below the 107-char AF_UNIX limit for DataLoader sockets.
# run_manifest_task.py --parallel gives each run its own Warp cache subdir.
JOBTMP="/tmp/${USER}/${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
export TMPDIR="$JOBTMP/tmp"
export WARP_CACHE_PATH="$JOBTMP/warp"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"
trap 'rm -rf "$JOBTMP"' EXIT

N_TASKS=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$MANIFEST")
NODE_SLOT=$(( ${SLURM_ARRAY_TASK_ID:-0} * ${SLURM_NNODES:-1} + ${SLURM_NODEID:-0} ))
FIRST=$(( NODE_SLOT * PACK ))
TASK_IDS=()
for (( t = FIRST; t < FIRST + PACK && t < N_TASKS; t++ )); do TASK_IDS+=("$t"); done
if (( ${#TASK_IDS[@]} == 0 )); then
  echo "[WARN] node slot $NODE_SLOT has no manifest tasks (N_TASKS=$N_TASKS, PACK=$PACK)"
  exit 0
fi
echo "[INFO] $(hostname) manifest=$MANIFEST tasks=${TASK_IDS[*]} compile=$COMPILE_MODE"

# --- Stage data, redirect outputs (CHANGES.md item 55) ----------------------
# $WORK (Stockyard) is a shared, ~full Lustre quota, so runs no longer read
# demos from or write checkpoints to it. Before training, every dataset this
# node's runs use is rsync'd from data/mjlab_hand_demos to
#   DATA_STAGE=scratch: $SCRATCH/cross_embodied_diffusion/mjlab_hand_demos
#                       (persistent mirror; later jobs only re-stat it)
#   DATA_STAGE=tmp:     node-local /tmp, deleted when the job ends
#   DATA_STAGE=none:    no copy, read in place (the old behaviour)
# The default (auto) is tmp for multi-node jobs and scratch for 1-node jobs.
# rsync is incremental and writes each file via temp+rename, so it also
# repairs files the $SCRATCH purge removed, and concurrent jobs staging the
# same dataset don't corrupt each other. Datasets outside
# data/mjlab_hand_demos are read in place.
#
# Outputs go to OUT_ROOT (default $SCRATCH/cross_embodied_diffusion/outputs).
# $SCRATCH is purged: copy runs worth keeping to Stockyard with
# scripts/promote_outputs.sh.
#
# Manifests stay repo-relative: the runs execute from a per-node run root
# whose entries symlink back to the repo, except data/mjlab_hand_demos ->
# the staged copy and outputs -> OUT_ROOT. So train_config.json still records
# "data/mjlab_hand_demos/..." and "outputs/...", as before.
: "${SCRATCH:?SCRATCH not set}"
REPO="$PWD"
SCRATCH_ROOT="$SCRATCH/cross_embodied_diffusion"
OUT_ROOT="${OUT_ROOT:-$SCRATCH_ROOT/outputs}"
DATA_STAGE="${DATA_STAGE:-auto}"
[[ "$DATA_STAGE" == auto ]] && DATA_STAGE=$( (( ${SLURM_NNODES:-1} > 1 )) && echo tmp || echo scratch )
SRC_DATA="$(realpath "$REPO/data/mjlab_hand_demos")"
case "$DATA_STAGE" in
  scratch) STAGE_DIR="$SCRATCH_ROOT/mjlab_hand_demos" ;;
  tmp)     STAGE_DIR="$JOBTMP/mjlab_hand_demos" ;;
  none)    STAGE_DIR="$SRC_DATA" ;;
  *) echo "[ERROR] DATA_STAGE must be auto|scratch|tmp|none, got '$DATA_STAGE'" >&2; exit 1 ;;
esac
MANIFEST_ABS="$(realpath "$MANIFEST")"
mapfile -t DATASETS < <(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
ds = lambda d: d if isinstance(d, list) else [d]  # list = pooled sources
print("\n".join(sorted({p.rstrip("/") for t in sys.argv[2:] for r in m[int(t)] for p in ds(r["dataset"])})))
' "$MANIFEST_ABS" "${TASK_IDS[@]}")

if [[ "$DATA_STAGE" != none ]]; then
  for ds in "${DATASETS[@]}"; do
    ds="${ds%/}"
    if [[ "$ds" != data/mjlab_hand_demos/* ]]; then
      echo "[WARN] $ds is not under data/mjlab_hand_demos; reading it in place"
      continue
    fi
    rel="${ds#data/mjlab_hand_demos/}"
    t0=$SECONDS
    mkdir -p "$STAGE_DIR/$rel"
    rsync -a "$SRC_DATA/$rel/" "$STAGE_DIR/$rel/"
    echo "[INFO] staged $ds -> $STAGE_DIR/$rel ($(du -sh "$STAGE_DIR/$rel" | cut -f1), $((SECONDS - t0))s)"
  done
fi

RUNROOT="$JOBTMP/root"
mkdir -p "$RUNROOT/data" "$OUT_ROOT"
for e in "$REPO"/*; do
  case "${e##*/}" in data|outputs) ;; *) ln -s "$e" "$RUNROOT/" ;; esac
done
for e in "$REPO"/data/*; do
  if [[ -e "$e" && "${e##*/}" != mjlab_hand_demos ]]; then ln -s "$e" "$RUNROOT/data/"; fi
done
ln -s "$STAGE_DIR" "$RUNROOT/data/mjlab_hand_demos"
ln -s "$OUT_ROOT" "$RUNROOT/outputs"
echo "[INFO] data=$STAGE_DIR ($DATA_STAGE) outputs=$OUT_ROOT"
cd "$RUNROOT"

DRY_ARGS=()
[[ -n "${DRY_RUN:-}" ]] && DRY_ARGS=(--dry-run)
python3 "$REPO/scripts/run_manifest_task.py" --parallel "${DRY_ARGS[@]}" "${COMPILE_ARGS[@]}" \
  "$MANIFEST_ABS" "${TASK_IDS[@]}"
