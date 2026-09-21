#!/bin/bash
# E7/M-8, full cohort: per-pair embedding distances for every arm over ALL of the corrected
# Swiss-Prot pair set — train + val + test, 75,849,972 pairs — one array task per arm.
#
#     cd ~/plm_choice_ridge && mkdir -p logs
#     sbatch --array=0-14%8 scripts/lrz/ridge_distances_full.sbatch
#
# ---------------------------------------------------------------------------
# WHY THE FULL SET. The first rebuild used the 10% training sample (5,958,720 pairs) so the
# figure and the retrained probe table would describe the same data. That sample is already
# dense enough that the density curve IS the population curve — but the quartiles quoted in
# the manuscript should come from the whole pair set, and having both lets us say so rather
# than assume it. Both results are kept: the subset under ridge_2026-09-18/, this one under
# $DSS/ridge_full/.
#
# WHAT "ALL OF SWISS-PROT" MEANS HERE. These are the pairs MMseqs2 and Foldseek actually
# found (corrected HFSP, deduplicated, ≤1022-residue cohort), not a true all-vs-all — a real
# all-vs-all over 526,871 proteins would be 1.4e11 pairs. The figure title must say so; the
# published caption's "all-vs-all" is wrong for this data.
#
# IDENTICAL SEQUENCES. 23,369 pairs (0.39% of the subset) sit at exactly distance 0 in twelve
# of fifteen arms; all eight spot-checked are the same sequence under two accessions. They
# are the spike at 0 and they pin the min-max minimum. The distances are computed here and
# dropped at plotting time, so the exclusion stays visible and reversible.
#
# Memory: the full pair set touches ~520k proteins, so esm2_3b needs ~5.3 GiB for the matrix
# on top of the ~5.7 GiB core-driver read — 32G leaves headroom. One core on purpose: the
# inner loop is a gather, a subtract and an einsum, none of which numpy threads.
# ---------------------------------------------------------------------------
#SBATCH -J ridge-full
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu                    # MANDATORY: lrz-cpu rejects the default QOS
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
SETS="$DSS/plm_choice_data/data/processed/sprot_pre2024_e1/sets"
EMB="$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k"
OUT="$DSS/ridge_full"
PY="$HOME/plm_choice/.venv/bin/python"

ARMS=(ankh_base ankh_large clean esm1b esm2_150m esm2_35m esm2_3b esm2_650m esm2_8m
      esm3_open esmc_300m esmc_600m prott5 prottucker random_1024)
IDX=${SLURM_ARRAY_TASK_ID:?set --array=0-14}
[ "$IDX" -lt "${#ARMS[@]}" ] || { echo "index $IDX out of range"; exit 2; }
ARM=${ARMS[$IDX]}

cd "$HOME/plm_choice_ridge"
export PYTHONPATH="$HOME/plm_choice_ridge/src"

for split in train val test; do
    mkdir -p "$OUT/$split"
    echo "=== $ARM / $split ==="
    srun "$PY" scripts/ridge_pair_distances.py \
        --pairs "$SETS/$split.parquet" \
        --emb-dir "$EMB" \
        --out-dir "$OUT/$split" \
        --arms "$ARM"
done

echo "done: $ARM"
