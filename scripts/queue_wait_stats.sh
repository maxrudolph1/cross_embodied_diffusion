#!/bin/bash
# Which job shape will start soonest on a TACC Vista partition? Run this
# before submitting anything large, and pick the shape from its output, not
# from memory (AGENTS.md "Getting Vista jobs scheduled fast").
#
#   scripts/queue_wait_stats.sh [partition=gh] [days=3] [time_limit=12:00:00] [sizes="1 2 4 8 16"]
#
# Prints:
#   1. partition node states (idle nodes => anything small starts now);
#   2. submit->start wait of jobs that STARTED in the last <days>, by node
#      count (all users; biased low -- jobs still waiting are excluded);
#   3. age of jobs still PENDING, by node count (biased the other way);
#   4. your own queue usage against the per-user caps (40 submitted /
#      20 running jobs, 96 running nodes; every array element is a job);
#   5. the scheduler's own projected start time per size via
#      `sbatch --test-only` (nothing is submitted). Only works on a login
#      node -- sbatch is disabled on compute/idev nodes; there it prints the
#      command for the user to run instead.
set -uo pipefail
PART="${1:-gh}"
DAYS="${2:-3}"
TLIM="${3:-12:00:00}"
SIZES="${4:-1 2 4 8 16}"

echo "== $PART node states ($(date '+%F %H:%M'))"
sinfo -p "$PART" -h -o "%T %D" | sort | sed 's/^/   /'

bucket='def bucket(n):
    for hi, k in ((1, "1"), (2, "2"), (4, "3-4"), (8, "5-8"), (16, "9-16"), (32, "17-32")):
        if n <= hi:
            return k
    return "33+"
ORDER = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33+"]
def q(w, f):
    return w[min(len(w) - 1, int(len(w) * f))]'

echo
echo "== Jobs STARTED in the last $DAYS days: submit->start wait (h) by nodes"
squeue -p "$PART" -h -t R -o "%D|%V|%S" | python3 -c "
import sys, datetime as D, collections
$bucket
now = D.datetime.now(); b = collections.defaultdict(list)
for line in sys.stdin:
    n, sub, st = line.strip().split('|')
    st, sub = D.datetime.fromisoformat(st), D.datetime.fromisoformat(sub)
    if (now - st).total_seconds() <= $DAYS * 86400:
        b[bucket(int(n))].append((st - sub).total_seconds() / 3600)
for k in ORDER:
    w = sorted(b[k])
    if w:
        print(f'   {k:>5} nodes  n={len(w):4d}  median {q(w,.5):6.1f}  p25 {q(w,.25):6.1f}  p75 {q(w,.75):6.1f}')
"

echo
echo "== Jobs PENDING now: age so far (h) by nodes"
squeue -p "$PART" -h -t PD -o "%D|%V|%r" | python3 -c "
import sys, datetime as D, collections
$bucket
now = D.datetime.now(); b = collections.defaultdict(list); r = collections.defaultdict(collections.Counter)
for line in sys.stdin:
    n, sub, reason = line.strip().split('|')
    k = bucket(int(n)); r[k][reason] += 1
    b[k].append((now - D.datetime.fromisoformat(sub)).total_seconds() / 3600)
for k in ORDER:
    w = sorted(b[k])
    if w:
        print(f'   {k:>5} nodes  n={len(w):4d}  median age {q(w,.5):6.1f}  ({dict(r[k].most_common(2))})')
"

echo
echo "== Your queue usage (caps: 40 submitted, 20 running jobs, 96 running nodes)"
mine=$(squeue -u "$USER" -h -r -o "%T %D")
echo "   submitted jobs: $(echo -n "$mine" | grep -c .)   running jobs: $(echo "$mine" | grep -c '^RUNNING')   running nodes: $(echo "$mine" | awk '$1=="RUNNING"{s+=$2} END{print s+0}')"

echo
echo "== Scheduler projection (sbatch --test-only, time limit $TLIM; nothing is submitted)"
probe=$(sbatch --test-only -p "$PART" -N 1 -t "$TLIM" --wrap=true 2>&1)
if echo "$probe" | grep -qi "not available on compute nodes"; then
    echo "   sbatch is disabled on this (compute/idev) node. Run on a login node:"
    echo "   for n in $SIZES; do echo -n \"\$n nodes: \"; sbatch --test-only -A <PROJECT> -p $PART -N \$n -t $TLIM --wrap=true 2>&1 | tail -1; done"
else
    for n in $SIZES; do
        echo "   $n nodes: $(sbatch --test-only -p "$PART" -N "$n" -t "$TLIM" --wrap=true 2>&1 | tail -1)"
    done
fi
