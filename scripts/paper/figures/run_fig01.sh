#!/bin/bash
# Figure 1: |Spearman rho| of the FNN probe and of the raw Euclidean distance, per
# target, for the twelve foundation models (clean and prottucker excluded, random
# baseline dropped).
#
# Grid: the main probe grid, dataset full100_p10 (every pair of the post-E1 splits,
# early-stopping patience 10), read from probe_2026-10-08/probe_metrics.csv.
#
# Environment:
#   REPO             checkout of this repository (required)
#   PAPER_ARTEFACTS  directory holding the measured outputs (required)
#   FIG01_OUT        output directory: PNGs in fig01/, log in fig01_run.log (required)
#   MANUSCRIPT_DIR   the manuscript's bib_2026 directory (optional); when set, the
#                    figure is installed as figures/fig01_spearman_rho.png
set -euo pipefail

W=${REPO:?set REPO to a checkout of this repository}
B=${PAPER_ARTEFACTS:?set PAPER_ARTEFACTS to the directory holding the measured outputs}
OUT=${FIG01_OUT:?set FIG01_OUT to the output directory}
PY=${REPO}/.venv/bin/python
CSV=$B/probe_2026-10-08/probe_metrics.csv

mkdir -p "$OUT/fig01"
OUT=$(cd "$OUT" && pwd)  # absolute, since the plotter runs from $W
cd "$W"

PYTHONPATH=src "$PY" src/visualization/create_performance_summary_plots.py \
  --metrics_csv "$CSV" \
  --dataset full100_p10 \
  --model_types fnn euclidean \
  --exclude_plms clean prottucker \
  --ignore-random \
  --output "$OUT/fig01" 2>&1 | tee "$OUT/fig01_run.log"

FIG=$OUT/fig01/absolute_spearman_rho.png

if [ -n "${MANUSCRIPT_DIR:-}" ]; then
  cp "$FIG" "$MANUSCRIPT_DIR/figures/fig01_spearman_rho.png"
  echo "FIG01 DONE: installed $MANUSCRIPT_DIR/figures/fig01_spearman_rho.png"
else
  echo "FIG01 DONE: $FIG (MANUSCRIPT_DIR unset, nothing installed)"
fi
