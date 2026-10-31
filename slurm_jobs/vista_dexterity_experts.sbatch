#!/bin/bash
#SBATCH --job-name=dexterity-rl
#SBATCH --output=slurm_jobs/vista_dexterity_experts/logs/job_%A_%a.out
#SBATCH --error=slurm_jobs/vista_dexterity_experts/logs/job_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#
# Submit with submit_dexterity_experts.sh from a login node. Every node executes
# PACK independent task workflows on its one GH200 (default PACK=1). Node slot
# array_index*NODES+node_id owns task indices slot*PACK through (slot+1)*PACK-1.
# Source is frozen at submission, copied to node-local /tmp, and PYTHONPATH
# selects that snapshot rather than the editable venv's original checkout.
# Outputs stay repo-relative in metadata via outputs -> scratch outputs.
# DRY_RUN=1 prints commands and does not construct an environment or use a GPU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Run through sbatch or set SLURM_SUBMIT_DIR for DRY_RUN}"
: "${DEXTERITY_MANIFEST:?Set DEXTERITY_MANIFEST using submit_dexterity_experts.sh}"
PACK="${PACK:-1}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exec .venv/bin/python scripts/run_dexterity_batch_node.py \
    --manifest "$DEXTERITY_MANIFEST" --pack "$PACK" --dry-run
fi
: "${DEXTERITY_SOURCE_ARCHIVE:?Submit through submit_dexterity_experts.sh}"
: "${SCRATCH:?SCRATCH must be set on Vista}"
if [[ "${1:-}" != --node && "${SLURM_NNODES:-1}" -gt 1 ]]; then
  : "${DEXTERITY_LAUNCHER:?Missing immutable launcher copy}"
  exec srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=0 \
    bash "$DEXTERITY_LAUNCHER" --node
fi
REPO="$PWD"
MANIFEST_ABS="$(realpath "$DEXTERITY_MANIFEST")"
ARCHIVE_ABS="$(realpath "$DEXTERITY_SOURCE_ARCHIVE")"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/cross_embodied_diffusion/outputs}"
export DEXTERITY_NODE_TMP="/tmp/dex_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}_${SLURM_NODEID:-0}"
RUNROOT="$DEXTERITY_NODE_TMP/root"
mkdir -p "$RUNROOT" "$OUT_ROOT"
tar -xzf "$ARCHIVE_ABS" -C "$RUNROOT"
ln -s "$(realpath "$REPO/.venv")" "$RUNROOT/.venv"
ln -s "$OUT_ROOT" "$RUNROOT/outputs"
export PYTHONPATH="$RUNROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$RUNROOT"
echo "[INFO] host=$(hostname) job=$SLURM_JOB_ID node=${SLURM_NODEID:-0} snapshot=$ARCHIVE_ABS outputs=$OUT_ROOT"
exec .venv/bin/python scripts/run_dexterity_batch_node.py \
  --manifest "$MANIFEST_ABS" --pack "$PACK"
