#!/bin/bash
# Figure 1, redrawn 2026-09-21 from the REFRESHED probe grid.
#
# Why this rerun exists (two independent reasons):
#
#  1. The metrics CSV was one cell stale. It was collected 2026-09-20 18:26, fifty
#     minutes BEFORE LRZ job 5799723_21 (ESM-2 3B / alntmscore) finished at 19:16, so
#     alntmscore carried 14 of 15 arms and panel B drew an "n/a" band for ESM-2 3B.
#     Re-collected as LRZ job 5800647 (NOT on the login node -- a login-node run wedged
#     in cxiWaitEventWait for 35 minutes with 684 reads and zero writes).
#
#  2. Panel C was titled "Function - HFSP". HFSP is alignment-derived and correlates
#     with sequence identity at r=0.920; the Discussion calls that panel a sequence
#     panel. The title is now "HFSP" alone, fixed in PARAMETER_TITLES in the plotting
#     code, not by hand on the PNG.
#
# Branch: feat/refresh-figures, worktree ../plm_choice_figs.
set -euo pipefail

W=${REPO:?set REPO to a checkout of this repository}
B=${PAPER_ARTEFACTS:?set PAPER_ARTEFACTS to the directory holding the measured outputs}
R=$B/final_figures_2026-09-21
PY=${REPO}/.venv/bin/python
M=${MANUSCRIPT_DIR:?set MANUSCRIPT_DIR to the manuscript's bib_2026 directory}/figures
CSV=$B/probe_e1_2026-09-18/probe_metrics.csv

mkdir -p "$R/fig01"
cd "$W"

# Same arguments as the 2026-09-20 run; only the CSV underneath has changed.
PYTHONPATH=src "$PY" src/visualization/create_performance_summary_plots.py \
  --metrics_csv "$CSV" \
  --dataset sprot_pre2024_e1_sub10 \
  --model_types fnn euclidean \
  --exclude_plms clean prottucker \
  --ignore-random \
  --output "$R/fig01" 2>&1 | tee "$R/fig01_run.log"

# --- install ------------------------------------------------------------------
cp "$R/fig01/pearson_r2.png" "$M/fig01_pearson_r2.png"

echo "FIG01 DONE"
md5 -q "$M/fig01_pearson_r2.png"
