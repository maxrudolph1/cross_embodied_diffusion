#!/bin/bash
# E21: the 4 eval arrays that slurm_jobs/submit_e21_finetune.sh failed to submit on 2026-10-07 (the space in
# RUNS made sbatch --export fail; CHANGES.md item 75). Chains onto train 1056188 and ev-base4321 1056189.
set -euo pipefail
cd "$(dirname "$0")/.."
A=(-A ASC26008 -p gh)
R=diffusion/ambient_ta_r
FT="diffusion/ambient_ta_r_ft/*"
EVAL=slurm_jobs/vista_eval_checkpoints.sbatch
BASEX=$(sbatch --parsable "${A[@]}" --array=0-2 -t 02:00:00 --job-name=ev-basex --dependency=afterany:1056189 \
  --export=ALL,RUNS="$R/*_sigma0_seed*:$R/*_sigma15_seed*:$R/*_sigma100_seed*",WHICH=best_val,EVAL_SEED=4321,CROSS=1 $EVAL)
FT1=$(sbatch --parsable "${A[@]}" --array=0-1 -t 02:30:00 --job-name=ev-ft1234 --dependency=afterany:1056188 \
  --export=ALL,RUNS="$FT",EVAL_SEED=1234 $EVAL)
FT2=$(sbatch --parsable "${A[@]}" --array=0-1 -t 02:30:00 --job-name=ev-ft4321 --dependency=afterany:1056188 \
  --export=ALL,RUNS="$FT",EVAL_SEED=4321 $EVAL)
FTX=$(sbatch --parsable "${A[@]}" --array=0-5 -t 02:00:00 --job-name=ev-ftx --dependency=afterany:$FT2 \
  --export=ALL,RUNS="$FT",WHICH=best_val:last0,EVAL_SEED=4321,CROSS=1 $EVAL)
echo "basex=$BASEX ft1234=$FT1 ft4321=$FT2 ftx=$FTX"
