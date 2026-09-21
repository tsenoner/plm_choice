#!/bin/bash
# Embedding distances over a uniform random sample of ALL cohort pairs, for Figures 2 and 3.
#
#     cd ~/plm_choice_ridge && mkdir -p logs && sbatch scripts/lrz/random_pair_distances.sbatch
#
# The filtered pair set the figures use today is a biased slice of protein space: between-model
# rank correlation is 0.77 on it against 0.34 on random pairs (job 5798086). If the claim is
# about protein space, the figure must be drawn on a uniform sample of it. 1.39e11 pairs exist;
# 5,000,000 of them put the sampling error of a rank correlation near 4.5e-4, three orders of
# magnitude below anything the figures report, and --convergence shows that rather than asserts it.
#
# Memory: the largest arm (esm2_3b, 2560-d over 526,871 proteins) is ~5.4 GiB as float32, plus
# the 5M-row frame; 64G is comfortable. One core -- the inner loop is a gather and an einsum.
#SBATCH -J random-pairs
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
unset VIRTUAL_ENV
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
cd "$HOME/plm_choice_ridge"
export PYTHONPATH="$HOME/plm_choice_ridge/src"
srun "$HOME/plm_choice/.venv/bin/python" scripts/random_pair_distances.py \
  --emb-dir "$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k" \
  --fasta "$DSS/plm_choice_data/data/raw/sprot_2024/sprot.fasta" \
  --out-dir "$DSS/random_pairs" \
  --n-pairs 5000000 --seed 42 --convergence
