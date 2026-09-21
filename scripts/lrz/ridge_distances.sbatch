#!/bin/bash
# E7/M-8: per-pair embedding distances for all 15 arms over the corrected
# sprot_pre2024_e1_sub10 train split (5,958,720 pairs), in ONE CPU job.
#
#   sbatch scripts/lrz/ridge_distances.sbatch   # copied to ~/ridge_2026_09_18/ on the cluster
#
# One job, arms processed sequentially: each arm's embedding matrix is loaded into
# RAM once (esm2_3b is the largest at 2560 dims -> ~3.1 GiB for the 324k proteins
# this pair set touches) and the pairs are evaluated vectorised in chunks. There is
# no per-pair HDF5 lookup anywhere.
#
# Peak memory is set by esm2_3b: the core driver holds the 5.74 GiB file while the
# 3.1 GiB matrix is built from it, plus ~1.5 GiB of chunk buffers -> ~10 GiB.
# Re-running is safe and cheap: an arm whose parquet AND manifest entry both exist
# is skipped, so a walltime kill resumes rather than restarts.
#SBATCH -J ridge-dist
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
unset VIRTUAL_ENV

# One core, so one thread: numpy sizes its BLAS pools from the NODE's core count,
# not the cgroup's allocation, and a 1-CPU task running 20 threads starves its
# neighbours. Asking for one core is not a compromise here -- the inner loop is a
# strided gather, an elementwise subtract and an einsum reduction, none of which
# numpy threads -- and a 1-core/24G/3h job backfills where a 4-core/32G/8h one queues.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
EMB="$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k"
PAIRS="$DSS/plm_choice_data/data/processed/sprot_pre2024_e1_sub10/sets/train.parquet"
OUT="$HOME/ridge_2026_09_18/dist"

cd "$HOME/plm_choice"
/usr/bin/time -v ./.venv/bin/python "$HOME/ridge_2026_09_18/ridge_pair_distances.py" \
  --pairs "$PAIRS" --emb-dir "$EMB" --out-dir "$OUT" --chunk 100000
echo "done"
