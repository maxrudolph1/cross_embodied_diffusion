#!/usr/bin/env bash
# Record rollout videos for the latest checkpoint of each trained experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UV_FLAGS=(--default-index https://pypi.org/simple)
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl}"
OUT_ROOT="${OUT_ROOT:-outputs/videos}"

mapfile -t EXP_DIRS < <(find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

for exp_dir in "${EXP_DIRS[@]}"; do
  exp_name="$(basename "$exp_dir")"
  latest_run="$(find "$exp_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
  [[ -n "$latest_run" ]] || continue

  latest_ckpt="$(find "$latest_run" -name 'model_*.pt' | sort -V | tail -1)"
  [[ -n "$latest_ckpt" ]] || continue

  case "$exp_name" in
    allegro_grasp) task_id="Grasp-Allegro" ;;
    leap_grasp) task_id="Grasp-LEAP" ;;
    shadow_grasp) task_id="Grasp-Shadow" ;;
    sharpa_grasp) task_id="Grasp-Sharpa" ;;
    wuji_grasp) task_id="Grasp-Wuji" ;;
    allegro_inhand_rotation) task_id="InHand-Rotation-Allegro" ;;
    leap_inhand_rotation) task_id="InHand-Rotation-LEAP" ;;
    shadow_inhand_rotation) task_id="InHand-Rotation-Shadow" ;;
    sharpa_in_hand_rotation) task_id="InHand-Rotation-Sharpa" ;;
    wuji_inhand_rotation) task_id="InHand-Rotation-Wuji" ;;
    *) echo "Skipping unknown experiment dir: $exp_name"; continue ;;
  esac

  out_dir="$OUT_ROOT/$exp_name"
  mkdir -p "$out_dir"

  echo "=== Recording ${task_id} from ${latest_ckpt} ==="
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
    uv run "${UV_FLAGS[@]}" play "$task_id" \
      --checkpoint-file "$latest_ckpt" \
      --video \
      --video-length 400 \
      --num-envs 1 \
      --viewer native \
      --log-root "$LOG_ROOT" || true

  if compgen -G "$latest_run/videos/play/*.mp4" > /dev/null; then
    cp "$latest_run"/videos/play/*.mp4 "$out_dir/" || true
  fi
done
