#!/bin/bash
#SBATCH --job-name=build-mixed
#SBATCH --output=slurm_jobs/build_mixed/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/build_mixed/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --array=0-19

# CPU-only array. Builds every ORDERED distinct-size pair from
# {10k,50k,100k,400k,1M} for one task family = 20 mixtures, one-hot
# conditioned. Both orderings (A10k_L1M and A1M_L10k) are built so hand
# identity is not confounded with data budget.
#
#   FAMILY=grasp|rotation  (default grasp)
#
# See CHANGES.md item 17.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this via sbatch}"
source .venv/bin/activate

FAMILY="${FAMILY:-grasp}"
if [[ "$FAMILY" == "rotation" ]]; then
  TASK_A="InHand-Rotation-Allegro"
  TASK_B="InHand-Rotation-LEAP"
else
  TASK_A="Grasp-Allegro"
  TASK_B="Grasp-LEAP"
fi

SIZES=(10k 50k 100k 400k 1M)
PAIRS=()
for a in "${SIZES[@]}"; do
  for b in "${SIZES[@]}"; do
    [[ "$a" == "$b" ]] && continue  # diagonal is never run
    PAIRS+=("${a}:${b}")
  done
done

PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
SIZE_A="${PAIR%%:*}"
SIZE_B="${PAIR##*:}"

SOURCE_A="data/demos/${TASK_A}_expert_${SIZE_A}.zarr"
SOURCE_B="data/demos/${TASK_B}_expert_${SIZE_B}.zarr"
OUTPUT="data/mixed/${FAMILY}_A${SIZE_A}_L${SIZE_B}.zarr"

echo "[INFO] ${FAMILY}: A=${SOURCE_A} L=${SOURCE_B} -> ${OUTPUT}"
.venv/bin/python scripts/build_mixed_dataset.py \
  --source-a "$SOURCE_A" \
  --source-b "$SOURCE_B" \
  --output "$OUTPUT" \
  --overwrite
