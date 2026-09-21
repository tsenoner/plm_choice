#!/bin/bash
#SBATCH -J fp-population
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
unset VIRTUAL_ENV
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
cd "$HOME/plm_choice_ridge"
export PYTHONPATH="$HOME/plm_choice_ridge/src"
srun "$HOME/plm_choice/.venv/bin/python" scripts/fingerprint_pair_population.py \
  --emb-dir "$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k" \
  --filtered "$DSS/ridge_full/train" \
  --n-random 2000000 --out "$DSS/fingerprint_population"
