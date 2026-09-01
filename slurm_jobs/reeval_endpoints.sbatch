#!/bin/bash
#SBATCH --job-name=reeval-endpoints
#SBATCH --output=slurm_jobs/reeval_endpoints/logs/task_%A_%a.out
#SBATCH --error=slurm_jobs/reeval_endpoints/logs/task_%A_%a.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0:30:00
#SBATCH --array=0-13

# Re-scores the ambient sweep's endpoint runs (sigma*=0 naive mixing, and
# sigma*=100 target-only) at 100 rollouts from their saved
# policy_latest.pt, so the sweep's interior (sigma*=50/75/90, also scored
# at 100 rollouts) isn't compared against noisier 32-rollout endpoint
# numbers.
#
# Writes to eval_metrics_100.jsonl, NEVER appending to the original
# eval_metrics.jsonl -- no already-published number is mutated. Truncates
# its own output file on start, so a requeue cannot double-append.
#
#   FAMILY_LIST="grasp rotation", SEEDS="0 1", ROLLOUTS (default 100)
#
# See CHANGES.md item 33.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="$PWD/warp_cache/reeval_endpoints_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

read -ra FAMILIES <<< "${FAMILY_LIST:-grasp rotation}"
read -ra SEEDS <<< "${SEEDS:-0 1}"
N_BUDGETS=(10k 50k 100k 400k)
ROLLOUTS="${ROLLOUTS:-100}"

# Flat cell list: (family, seed, N_budget, endpoint in {0, 100}).
CELLS=()
for FAMILY in "${FAMILIES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for N in "${N_BUDGETS[@]}"; do
      CELLS+=("${FAMILY}:${SEED}:${N}:0")
      CELLS+=("${FAMILY}:${SEED}:${N}:100")
    done
  done
done

CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
IFS=':' read -r FAMILY SEED N_BUDGET ENDPOINT <<< "$CELL"

if [[ "$FAMILY" == "rotation" ]]; then
  TASK="InHand-Rotation-Allegro"
  DISPLAY_FAMILY="InHand-Rotation"
else
  TASK="Grasp-Allegro"
  DISPLAY_FAMILY="Grasp"
fi

SEED_SUFFIX=""
[[ "$SEED" != "0" ]] && SEED_SUFFIX="_s${SEED}"

if [[ "$ENDPOINT" == "100" ]]; then
  RUN_DIR="outputs/diffusion/${TASK}_${N_BUDGET}${SEED_SUFFIX}"
elif [[ "$N_BUDGET" == "400k" ]]; then
  RUN_DIR="outputs/ambient/${DISPLAY_FAMILY}_A400k_L400k_amb0${SEED_SUFFIX}"
else
  RUN_DIR="outputs/mixed_noc/${DISPLAY_FAMILY}_A${N_BUDGET}_L400k${SEED_SUFFIX}"
fi

POLICY="${RUN_DIR}/policy_latest.pt"
OUT_JSONL="${RUN_DIR}/eval_metrics_100.jsonl"

echo "[INFO] cell=${CELL} task=${TASK} run_dir=${RUN_DIR} rollouts=${ROLLOUTS}"

if [[ ! -f "$POLICY" ]]; then
  echo "[WARN] Missing ${POLICY}, skipping" >&2
  exit 0
fi

: > "$OUT_JSONL"  # truncate: a requeue must not double-append

.venv/bin/python - "$RUN_DIR" "$TASK" "$ROLLOUTS" "$OUT_JSONL" <<'PYEOF'
import json
import sys
from pathlib import Path

from mjlab_hand.diffusion.evaluate import evaluate_diffusion_policy

run_dir, task, rollouts, out_jsonl = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
run_dir = Path(run_dir)
cfg = json.loads((run_dir / "train_config.json").read_text())
epoch = cfg["num_epochs"]

metrics = evaluate_diffusion_policy(
    task=task,
    policy_path=run_dir / "policy_latest.pt",
    num_envs=rollouts,
    num_steps=5000,
    seed=0,
)
row = {"epoch": epoch, "eval_task": task, "onehot": None, "metrics": metrics}
with out_jsonl.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"[INFO] Wrote {out_jsonl}: {metrics}")
PYEOF
