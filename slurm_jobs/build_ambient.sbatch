#!/bin/bash
#SBATCH --job-name=build-ambient
#SBATCH --output=slurm_jobs/build_ambient/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/build_ambient/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --array=0-1

# The 2 A400k_L400k diagonal datasets build_mixed.sbatch structurally
# cannot make (it skips a == b). Needed because the ambient sweep holds the
# foreign-data partner (LEAP) fixed at 400k while sweeping the target
# (Allegro) budget N, and N=400k is the one cell where target and partner
# coincide. Built unconditioned (--no-onehot), matching the rest of
# data/mixed_noc/: the ambient sweep does not re-litigate conditioning,
# already shown capacity-neutral in both seeds by the mixed-matrix work.
#
# See CHANGES.md item 28 / ANALYSIS.md "Ambient-sweep diagonal datasets".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

FAMILIES=(grasp rotation)
FAMILY="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"

if [[ "$FAMILY" == "rotation" ]]; then
  TASK_A="InHand-Rotation-Allegro"
  TASK_B="InHand-Rotation-LEAP"
  DISPLAY_FAMILY="InHand-Rotation"
else
  TASK_A="Grasp-Allegro"
  TASK_B="Grasp-LEAP"
  DISPLAY_FAMILY="Grasp"
fi

SOURCE_A="data/demos/${TASK_A}_expert_400k.zarr"
SOURCE_B="data/demos/${TASK_B}_expert_400k.zarr"
OUTPUT="data/mixed_noc/${DISPLAY_FAMILY}_A400k_L400k.zarr"

echo "[INFO] ${FAMILY}: A=${SOURCE_A} L=${SOURCE_B} -> ${OUTPUT}"
.venv/bin/python scripts/build_mixed_dataset.py \
  --source-a "$SOURCE_A" \
  --source-b "$SOURCE_B" \
  --output "$OUTPUT" \
  --no-onehot \
  --overwrite
