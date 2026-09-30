#!/bin/bash
# Copy finished training runs from $SCRATCH (where vista_train_manifest.sbatch
# writes them; $SCRATCH is purged) to Stockyard, where the repo's `outputs`
# symlink points. See CHANGES.md item 55.
#
# Usage: scripts/promote_outputs.sh [options] RUN_OR_DIR...
#
# Each argument is a path relative to the scratch outputs root (e.g.
# diffusion/scarce/Grasp-Scarce-Allegro_uniform_seed0) or an absolute path
# under it. A directory that is not itself a run is searched for runs (dirs
# with train_config.json), so `diffusion/scarce` promotes every finished run
# in that sweep. The relative layout is kept, so the run lands at the same
# repo-relative outputs/... path it had during training.
#
# Only runs with train_done.json (written by train-diffusion at the very end)
# are copied; unfinished runs are skipped unless --force. Runs trained before
# that marker existed need --force.
#
# Options:
#   -n, --dry-run        list what would be copied, copy nothing
#   --no-epoch-ckpts     skip policy_epoch_*.pt (keeps latest/best_val/best_eval;
#                        ~0.27 GB each, usually most of a run's size)
#   --force              also copy runs without train_done.json
#   --src DIR            scratch outputs root
#                        (default $SCRATCH/cross_embodied_diffusion/outputs)
#   --dst DIR            Stockyard outputs root
#                        (default $STOCKYARD/vista/cross_embodied_diffusion/outputs)
set -euo pipefail

SRC="${SCRATCH:?SCRATCH not set}/cross_embodied_diffusion/outputs"
DST="${STOCKYARD:?STOCKYARD not set}/vista/cross_embodied_diffusion/outputs"
DRY=0 FORCE=0 NO_EPOCH=0
ARGS=()
while (( $# )); do
  case "$1" in
    -n|--dry-run) DRY=1 ;;
    --force) FORCE=1 ;;
    --no-epoch-ckpts) NO_EPOCH=1 ;;
    --src) SRC="$2"; shift ;;
    --dst) DST="$2"; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
(( ${#ARGS[@]} )) || { echo "usage: $0 [options] RUN_OR_DIR... (--help)" >&2; exit 2; }
SRC="$(realpath -m "$SRC")"

RUNS=()
for a in "${ARGS[@]}"; do
  a="${a%/}"
  [[ "$a" == /* ]] || a="$SRC/$a"
  a="$(realpath -m "$a")"
  if [[ "$a" != "$SRC"/* || ! -d "$a" ]]; then
    echo "[ERROR] not a directory under $SRC: $a" >&2; exit 1
  fi
  mapfile -t found < <(find "$a" -name train_config.json -printf '%h\n' | sort)
  (( ${#found[@]} )) || echo "[WARN] no runs (train_config.json) under $a" >&2
  RUNS+=("${found[@]}")
done

RSYNC_OPTS=(-a)
(( NO_EPOCH )) && RSYNC_OPTS+=(--exclude 'policy_epoch_*.pt')
TODO=()
for r in "${RUNS[@]}"; do
  if [[ ! -f "$r/train_done.json" ]] && (( ! FORCE )); then
    echo "[SKIP] unfinished (no train_done.json): ${r#"$SRC"/}"
    continue
  fi
  TODO+=("$r")
done
(( ${#TODO[@]} )) || { echo "nothing to promote"; exit 0; }

# Bytes that would actually be transferred (rsync skips files already there).
NEED=0
for r in "${TODO[@]}"; do
  b=$(rsync "${RSYNC_OPTS[@]}" --dry-run --stats "$r/" "$DST/${r#"$SRC"/}/" 2>/dev/null \
      | awk -F': ' '/Total transferred file size/ {gsub(/[^0-9]/, "", $2); print $2}')
  NEED=$(( NEED + ${b:-0} ))
done
gb() { awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1e9 }'; }
echo "[INFO] ${#TODO[@]} run(s), $(gb "$NEED") to copy -> $DST"

# $WORK quota (1 TB, shared with all projects/systems): refuse rather than let
# the copy die partway through.
if FS=$(df --output=target "$DST" 2>/dev/null | tail -1) && [[ "$FS" == /work* ]]; then
  read -r USED LIMIT < <(lfs quota -u "$USER" /work | awk '$1 == "/work" {gsub(/\*/, ""); print $2, $4}')
  if [[ -n "${LIMIT:-}" && "$LIMIT" -gt 0 ]]; then
    FREE=$(( (LIMIT - USED) * 1024 ))
    echo "[INFO] /work quota: $(gb $((USED * 1024))) used, $(gb "$FREE") free"
    if (( NEED > FREE )); then
      echo "[ERROR] not enough /work quota; free space or use --no-epoch-ckpts" >&2; exit 1
    fi
  fi
fi

for r in "${TODO[@]}"; do
  rel="${r#"$SRC"/}"
  if (( DRY )); then
    echo "[DRY] $rel"
    continue
  fi
  mkdir -p "$DST/$rel"
  rsync "${RSYNC_OPTS[@]}" "$r/" "$DST/$rel/"
  echo "[OK] $rel"
done
