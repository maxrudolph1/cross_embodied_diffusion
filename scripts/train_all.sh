#!/usr/bin/env bash
# Train all hand grasp and in-hand rotation policies sequentially on one GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UV_FLAGS=(--default-index https://pypi.org/simple)
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERS="${MAX_ITERS:-10000}"
LOGGER="${LOGGER:-tensorboard}"

TASKS=(
  "Grasp-Allegro"
  "Grasp-LEAP"
  "Grasp-Shadow"
  "Grasp-Sharpa"
  "Grasp-Wuji"
  "InHand-Rotation-Allegro"
  "InHand-Rotation-LEAP"
  "InHand-Rotation-Shadow"
  "InHand-Rotation-Sharpa"
  "InHand-Rotation-Wuji"
)

for task in "${TASKS[@]}"; do
  echo "=== Training ${task} ==="
  uv run "${UV_FLAGS[@]}" train "$task" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.max-iterations "$MAX_ITERS" \
    --agent.logger "$LOGGER" \
    --agent.run-name "batch-$(date +%Y%m%d)"
done
