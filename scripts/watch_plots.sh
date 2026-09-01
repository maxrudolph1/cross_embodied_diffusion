#!/usr/bin/env bash
# Status table across in-flight training/eval logs: staleness since last
# write, error-line counts, and the last line of each log.
#
# Do not trust Slurm's `State=RUNNING` (see CHANGES.md item 35 for the
# full lesson) -- a task can sit at State RUNNING for tens of minutes doing
# no work. Detect hangs from log-file mtime staleness (the "stale(s)"
# column here) rather than the scheduler's reported state, and confirm
# with `sacct` AveCPU vs ElapsedRaw if a log looks stale.
#
# See CHANGES.md item 12: `err=$(grep -cE ... "$f" 2>/dev/null || echo 0)`
# printed "0" twice -- `grep -c` already prints 0 (and exits 1) on no
# match, so the `|| echo 0` fallback appended a second line. Fixed to
# `|| true`, which only swallows the nonzero exit (needed under
# `set -e`/`pipefail`) without duplicating grep's own output.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIRS=(logs slurm_jobs)
now=$(date +%s)

printf "%-60s %8s %6s %s\n" "log" "stale(s)" "errs" "last line"
for dir in "${LOG_DIRS[@]}"; do
  [[ -d "$dir" ]] || continue
  while IFS= read -r -d '' f; do
    mtime=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || echo "$now")
    stale=$(( now - mtime ))
    err=$(grep -cE "Error|Traceback|CUDA error|FileNotFoundError" "$f" 2>/dev/null || true)
    last=$(tail -n 1 "$f" 2>/dev/null | cut -c1-90)
    printf "%-60s %8d %6d %s\n" "$f" "$stale" "${err:-0}" "$last"
  done < <(find "$dir" \( -name "*.log" -o -name "*.out" -o -name "*.err" \) -print0 2>/dev/null)
done
