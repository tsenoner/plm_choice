#!/bin/bash
# Redraw the supplementary pair-filtering figure from the corrected pipeline.
#
# --plots-only stops before the merge, so the canonical merged pair table is read
# from but never rewritten. Redrawing every panel is the DEFAULT: the opposite
# flag, --reuse-plots, keeps whatever PNGs are already in the output directory,
# which is how the published funnel outlived two corrections to its own data.
#
# This runs on the cluster because the two inputs are the raw all-vs-all search
# tables (102.5M MMseqs2 rows, 276.9M Foldseek rows) and the deduplication sorts
# them whole.
#
#     cd ~/plm_choice && mkdir -p logs
#     sbatch scripts/lrz/filtering_figure.sbatch
#
# REPO names the checkout to run, for driving this from a branch worktree before
# the branch is merged:
#
#     REPO=$HOME/plm_choice_suppfigs sbatch scripts/lrz/filtering_figure.sbatch
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
REPO="${REPO:-$HOME/plm_choice}"
PY="$HOME/plm_choice/.venv/bin/python"   # a worktree has no .venv of its own
[ -d "$REPO/src" ] || { echo "repo missing: $REPO"; exit 2; }
cd "$REPO"
mkdir -p logs
export PYTHONPATH="$REPO/src"
export MPLBACKEND=Agg
srun "$PY" src/data_preparation/merge_datasets.py \
  --dataset sprot_pre2024 \
  --data-dir "$DSS/plm_choice_data/data" \
  --output-dir "$DSS/supp_figures/filtering" \
  --plots-only
