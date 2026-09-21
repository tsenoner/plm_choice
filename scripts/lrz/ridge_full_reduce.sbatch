#!/bin/bash
# M-8, full cohort: reduce $DSS/ridge_full/*/dist_<arm>.parquet (75,849,972 pairs x 15 arms,
# 11.6 GB) to per-arm histograms + exact quantiles + dip tests small enough to draw locally.
#
#     cd ~/plm_choice_ridge && git pull && mkdir -p logs
#     sbatch --array=0-14%8 scripts/lrz/ridge_full_reduce.sbatch
#
# REQUIRES scripts/lrz/ridge_identical.sbatch to have run first.  The --identical-root
# below is an INPUT and that job is the only thing that writes it; without it
# ridge_full_reduce.py falls back to the `distance == 0` rule, the exclusion stops being
# the same rows in every arm, and n_used -- the pair count in Figure 2's title -- moves.
#
# One array task per arm so a single slow arm does not hold the others.  Each task holds one
# arm's 75.8M float64 distances (~607 MB) plus the sorted copies diptest makes of it, which is
# what sets --mem: three live copies at 607 MB each plus polars' read buffers stays well under
# 32G, and the job is single-threaded on purpose (np.percentile and the dip routine do not
# thread, and OMP oversubscription on a shared login-adjacent partition is antisocial).
#
# OUTPUT GOES TO $DSS.  The cluster HOME quota filled up on 2026-09-17 and silently truncated
# five jobs' output; nothing here may write into ~ except the slurm logs.
#SBATCH -J ridge-reduce
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu                    # MANDATORY: lrz-cpu rejects the default QOS
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export POLARS_MAX_THREADS=2

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
IN="$DSS/ridge_full"
OUT="$DSS/ridge_full_summary"
PY="$HOME/plm_choice/.venv/bin/python"

ARMS=(ankh_base ankh_large clean esm1b esm2_150m esm2_35m esm2_3b esm2_650m esm2_8m
      esm3_open esmc_300m esmc_600m prott5 prottucker random_1024)
IDX=${SLURM_ARRAY_TASK_ID:?set --array=0-14}
[ "$IDX" -lt "${#ARMS[@]}" ] || { echo "index $IDX out of range"; exit 2; }
ARM=${ARMS[$IDX]}

mkdir -p "$OUT"
cd "$HOME/plm_choice_ridge"
export PYTHONPATH="$HOME/plm_choice_ridge/src"

srun "$PY" scripts/ridge_full_reduce.py \
    --identical-root "$DSS/ridge_identical" \
    --dist-root "$IN" \
    --out-dir "$OUT" \
    --arm "$ARM"

echo "done: $ARM"
