#!/bin/bash
# Figure 3 (the fingerprint matrix) over the FULL corrected cohort: reduce
# $DSS/ridge_full/*/dist_<arm>.parquet (75,849,972 pairs x 14 pLM arms) to the two
# 14x14 triangles, as a ~100 KB JSON the laptop draws from.
#
#     cd ~/plm_choice_figs && git pull && mkdir -p logs
#     sbatch scripts/lrz/fingerprint_full.sbatch
#
# ONE job, not an array: Spearman's rho is a statistic OF A PAIR of arms, so every
# arm's rank vector has to be live at the same time.  That sets the memory, and it is
# worth stating how: 14 arms x 74.5M pairs is 8.3 GB of float64 ranks plus 8.3 GB of
# sorted float32 copies for the two Wasserstein normalisations, plus one arm's raw
# column and the ~2.4 GB the sort transiently needs.  ~19 GB live, so 64G leaves room
# for polars' read buffers without pretending to a precision we do not need.
#
# Single-threaded on purpose: the work is one np.argsort per arm and 91 dot products,
# none of which thread usefully, and a 1-core job backfills where a 16-core one queues.
#
# OUTPUT GOES TO $DSS.  The cluster HOME quota filled up on 2026-09-17 and silently
# truncated five jobs' output; nothing here writes into ~ except the slurm logs.
#SBATCH -J fingerprint
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu                    # MANDATORY: lrz-cpu rejects the default QOS
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export POLARS_MAX_THREADS=2

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
OUT="$DSS/fingerprint_full"
PY="$HOME/plm_choice/.venv/bin/python"

mkdir -p "$OUT"
cd "$HOME/plm_choice_figs"
export PYTHONPATH="$HOME/plm_choice_figs/src"

srun /usr/bin/time -v "$PY" scripts/fingerprint_full_reduce.py \
    --dist-root "$DSS/ridge_full" \
    --identical-root "$DSS/ridge_identical" \
    --out-dir "$OUT"

echo "done"
