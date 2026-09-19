#!/bin/bash
# Redraw the supplementary pair-filtering figure from the corrected pipeline.
#
# --plots-only stops before the merge, so the canonical merged pair table is read
# from but never rewritten; --force-plots is what makes the rerun actually replace
# the panels instead of finding the old PNGs and skipping them.
#
# This runs on the cluster because the two inputs are the raw all-vs-all search
# tables (102.5M MMseqs2 rows, 276.9M Foldseek rows) and the deduplication sorts
# them whole.
#
#   sbatch scripts/lrz/filtering_figure.sbatch
#SBATCH -J supp-filtering
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
unset VIRTUAL_ENV
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
cd "$HOME/plm_choice_suppfigs"
mkdir -p logs
export PYTHONPATH="$HOME/plm_choice_suppfigs/src"
export MPLBACKEND=Agg
srun "$HOME/plm_choice/.venv/bin/python" src/data_preparation/merge_datasets.py \
  --dataset sprot_pre2024 \
  --data-dir "$DSS/plm_choice_data/data" \
  --output-dir "$DSS/supp_figures/filtering" \
  --plots-only --force-plots
