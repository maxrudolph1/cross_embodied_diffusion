#!/bin/bash
# E21: fine-tune the E19 rotation ambient runs (sources sigma 0 / 15 / 100, seeds 0-2, policy_best_val.pt)
# on their target only, lr 1e-5, 10 epochs, on the _K50kr stores; then score everything on a fresh
# test seed (4321) as well as the selection seed (1234). CHANGES.md item 74. Run from the repo root on a
# Vista LOGIN node (sbatch is disabled on compute/idev nodes):
#
#   python scripts/build_finetune_manifest.py --families InHand-Rotation --src-sigmas 0 15 100 \
#       --seeds 0 1 2 --lrs 1e-5 --size 50kr --src-root outputs/diffusion/ambient_ta_r \
#       --root outputs/diffusion/ambient_ta_r_ft --out slurm_jobs/finetune_r_manifest.json
#   bash slurm_jobs/submit_e21_finetune.sh            # DRY=1 prints the sbatch lines only
#
# Jobs (25 queue slots; peak 45 + 3 nodes). Evals: ~5.5 min each with 4 per GPU (idev rescore 2026-10-07):
#   train   45 runs, PACK=1, 9 x 5-node jobs, ~1h35 each (E15 fine-tunes), limit 2:30
#   base    all 150 E19 runs, best_val, own target, seed 4321 (held-out sigma curve)    now, 3 nodes ~70 min
#   basex   E19 sigma 0/15/100 sources, best_val, other 4 hands, seed 4321 (forgetting)  after base, 3 x ~80 min
#   ft1234  fine-tunes, best_rollout best_val last0, own target, seed 1234 (comparable to E20) after train, 2 x ~95 min
#   ft4321  same at seed 4321 (the reported test number)                                after train, 2 x ~95 min
#   ftx     fine-tunes, best_val last0, other 4 hands, seed 4321 (forgetting)           after ft4321, 6 x ~80 min
# Evals are resumable (rows already written are skipped): resubmit the same line if one times out.
set -euo pipefail
cd "$(dirname "$0")/.."
A=(-A ASC26008 -p gh)
R=diffusion/ambient_ta_r
SRC="$R/*_sigma0_seed* $R/*_sigma15_seed* $R/*_sigma100_seed*"
FT="diffusion/ambient_ta_r_ft/*"
EVAL=slurm_jobs/vista_eval_checkpoints.sbatch

sub() {  # sub <name> <sbatch args...>; echoes the job id
  if [[ -n "${DRY:-}" ]]; then echo "sbatch --parsable $*" >&2; echo "DRY_$1"; return; fi
  shift
  sbatch --parsable "$@"
}

[[ -f slurm_jobs/finetune_r_manifest.json ]] || { echo "build the manifest first (see header)" >&2; exit 1; }
TRAIN=$(sub train "${A[@]}" -N 5 --array=0-8 -t 02:30:00 --job-name=ft-amb-r \
  --export=ALL,MANIFEST=slurm_jobs/finetune_r_manifest.json,PACK=1 slurm_jobs/vista_train_manifest.sbatch)
BASE=$(sub base "${A[@]}" --array=0-2 -t 01:45:00 --job-name=ev-base4321 \
  --export=ALL,RUNS="$R/*",WHICH=best_val,EVAL_SEED=4321 $EVAL)
BASEX=$(sub basex "${A[@]}" --array=0-2 -t 02:00:00 --job-name=ev-basex --dependency=afterany:$BASE \
  --export=ALL,RUNS="$SRC",WHICH=best_val,EVAL_SEED=4321,CROSS=1 $EVAL)
FT1=$(sub ft1234 "${A[@]}" --array=0-1 -t 02:30:00 --job-name=ev-ft1234 --dependency=afterany:$TRAIN \
  --export=ALL,RUNS="$FT",EVAL_SEED=1234 $EVAL)
FT2=$(sub ft4321 "${A[@]}" --array=0-1 -t 02:30:00 --job-name=ev-ft4321 --dependency=afterany:$TRAIN \
  --export=ALL,RUNS="$FT",EVAL_SEED=4321 $EVAL)
FTX=$(sub ftx "${A[@]}" --array=0-5 -t 02:00:00 --job-name=ev-ftx --dependency=afterany:$FT2 \
  --export=ALL,RUNS="$FT",WHICH="best_val last0",EVAL_SEED=4321,CROSS=1 $EVAL)
echo "train=$TRAIN base=$BASE basex=$BASEX ft1234=$FT1 ft4321=$FT2 ftx=$FTX"
