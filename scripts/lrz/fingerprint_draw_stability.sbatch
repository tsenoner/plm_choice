#!/bin/bash
# Is the random-pair fingerprint stable across INDEPENDENT draws, and does a small all-vs-all
# reproduce the published rho = 0.38 for CLEAN against ESM-1b?
#SBATCH -J fp-stability
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
unset VIRTUAL_ENV
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
cd "$HOME/plm_choice_ridge"
export PYTHONPATH="$HOME/plm_choice_ridge/src"
srun "$HOME/plm_choice/.venv/bin/python" scripts/fingerprint_draw_stability.py \
  --emb-dir "$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k" \
  --out "$DSS/fingerprint_stability" --n-pairs 1000000 --draws 5
