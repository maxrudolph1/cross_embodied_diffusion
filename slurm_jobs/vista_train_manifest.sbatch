#!/bin/bash
#SBATCH --job-name=vista-manifest
#SBATCH --output=slurm_jobs/vista_train_manifest/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/vista_train_manifest/logs/task_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
# element counts as one job).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
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
JOBTMP="/tmp/${USER}/${SLURM_JOB_ID}"
export TMPDIR="$JOBTMP/tmp"
export WARP_CACHE_PATH="$JOBTMP/warp"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"
trap 'rm -rf "$JOBTMP"' EXIT

N_TASKS=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$MANIFEST")
FIRST=$(( SLURM_ARRAY_TASK_ID * PACK ))
TASK_IDS=()
for (( t = FIRST; t < FIRST + PACK && t < N_TASKS; t++ )); do TASK_IDS+=("$t"); done
if (( ${#TASK_IDS[@]} == 0 )); then
  echo "[WARN] array index $SLURM_ARRAY_TASK_ID has no manifest tasks (N_TASKS=$N_TASKS, PACK=$PACK)"
  exit 0
fi
echo "[INFO] $(hostname) manifest=$MANIFEST tasks=${TASK_IDS[*]} compile=$COMPILE_MODE"

python3 scripts/run_manifest_task.py --parallel "${COMPILE_ARGS[@]}" "$MANIFEST" "${TASK_IDS[@]}"
