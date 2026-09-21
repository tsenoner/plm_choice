#!/bin/bash
# M-8: mark the pair-table rows whose two accessions carry the SAME sequence, so every
# arm's reduction drops the same rows.
#
#     cd ~/plm_choice_ridge && mkdir -p logs && sbatch scripts/lrz/ridge_identical.sbatch
#
# RUN THIS BEFORE scripts/lrz/ridge_full_reduce.sbatch. That job passes
# --identical-root "$DSS/ridge_identical" and this is the only thing that fills it.
# Without it ridge_full_reduce.py falls back to the `distance == 0` rule its own
# docstring documents as both circular and inaccurate -- it misses the float16 arms'
# identical pairs, which land at 0.0004-0.0056 rather than 0.0, while over-dropping in
# a compressed-range arm like ankh_large. The exclusion would then differ per arm, and
# n_used -- the pair count Figure 2's title states -- would change with it.
#
# The inputs are the same two the rest of this set already names: the cohort FASTA that
# random_pair_distances.sbatch hashes, and the corrected pair tables that
# ridge_distances_full.sbatch computes distances over. They have to be those two: the
# mask is positional, so it is only valid against the pair tables the distance parquets
# were written from, in the same split order.
#
# Memory: one dict of ~570k accessions -> int and one of ~570k SHA-1 digests -> int,
# plus one split's query/target columns and their two int64 index arrays at a time
# (train is the big one at ~68M rows). 32G is the same envelope as the sibling jobs.
# One core: the work is a dict lookup per row and an elementwise compare, neither of
# which threads.
#SBATCH -J ridge-identical
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu                    # MANDATORY: lrz-cpu rejects the default QOS
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export POLARS_MAX_THREADS=2

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
FASTA="$DSS/plm_choice_data/data/raw/sprot_2024/sprot.fasta"
SETS="$DSS/plm_choice_data/data/processed/sprot_pre2024_e1/sets"
OUT="$DSS/ridge_identical"
PY="$HOME/plm_choice/.venv/bin/python"

# The checkout this branch lives in. The default matches the other five sbatch files
# in this directory; once feat/ridge-recompute merges, the throwaway worktree is gone
# and this becomes $HOME/plm_choice, hence the override.
CHECKOUT="${PLM_CHOICE_CHECKOUT:-$HOME/plm_choice_ridge}"

mkdir -p "$OUT"
cd "$CHECKOUT"
export PYTHONPATH="$CHECKOUT/src"

srun "$PY" scripts/ridge_identical_pairs.py \
    --fasta "$FASTA" \
    --sets-dir "$SETS" \
    --out-dir "$OUT"

echo "done: wrote <split>_identical.parquet and <split>_identical_keyed.parquet to $OUT"
