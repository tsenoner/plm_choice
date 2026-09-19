#!/bin/bash
# Reduce the full corrected 75.8M-pair cohort to the supplementary fingerprint figure.
#
# Output is the hexbin cache that src/visualization/pairwise_embedding_comparison.py
# reads with --hexbin_json, so the figure itself is drawn locally from a few MB of
# counts rather than from ~30 GB of per-arm distance files.
#
#   sbatch scripts/lrz/fingerprint_hexbin.sbatch                       # published min-max axes
#   P99_LIMIT=1.2 OUT_NAME=fingerprint_hexbin_p99 \
#       sbatch scripts/lrz/fingerprint_hexbin.sbatch                   # ridge-figure axes
#SBATCH -J fp-hexbin
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
unset VIRTUAL_ENV
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
cd "$HOME/plm_choice_suppfigs"
mkdir -p logs
export PYTHONPATH="$HOME/plm_choice_suppfigs/src"
srun "$HOME/plm_choice/.venv/bin/python" scripts/fingerprint_hexbin_reduce.py \
  --dist-dir "$DSS/ridge_full" \
  --identical-dir "$DSS/ridge_identical" \
  --splits train val test \
  --gridsize 50 \
  --verify-pairs 6 \
  --p99-limit "${P99_LIMIT:-0}" \
  --out "$DSS/supp_figures/${OUT_NAME:-fingerprint_hexbin}"
