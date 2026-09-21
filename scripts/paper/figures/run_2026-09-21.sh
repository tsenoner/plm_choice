#!/bin/bash
# Figure 2's axis + Figure 3's colour scale, 2026-09-21.
#
# Everything below runs from the tracked repo on branch feat/ridge-recompute. Nothing
# here monkey-patches matplotlib: the diverging correlation scale and the shared
# Wasserstein top that redraw_fig03_diverging.py used to inject at runtime are now
# arguments of plot_combined_wasserstein_correlation, and the 1st/99th-percentile
# ticks and the legend placement are arguments of plot_ridge_distributions.
#
# Inputs (all outside git):
#   final_figures_2026-09-20/ridge/summaries   5,000,000 uniformly random pairs
#   ridge_full_2026-09-19/summaries            75,849,972 aligner-found pairs
#   final_figures_2026-09-20/fp/out/fingerprint_full.json   random-pair fingerprint
#   figures_2026-09-19/fig03/fingerprint_full.json          alignable fingerprint
#
# Installed from this run:
#   random/ridge_minmax.png    -> manuscript/bib_2026/figures/fig02.png
#   alignable/ridge_minmax.png -> manuscript/bib_2026/figures/som_figS05.png
# fig03.png and som_figS06.png were NOT reinstalled: fig03/fingerprint_*.png come out
# byte-identical (md5 7c61cb65... and bba25ce8...) to the versions already installed,
# which is the point -- the repo now reproduces them.
set -euo pipefail

W=${REPO:?set REPO to a checkout of this repository}
B=${PAPER_ARTEFACTS:?set PAPER_ARTEFACTS to the directory holding the measured outputs}
R=$B/final_figures_2026-09-21
PY=${REPO}/.venv/bin/python
M=${MANUSCRIPT_DIR:?set MANUSCRIPT_DIR to the manuscript's bib_2026 directory}/figures

RANDOM_SUM=$B/final_figures_2026-09-20/ridge/summaries
ALIGN_SUM=$B/ridge_full_2026-09-19/summaries

mkdir -p "$R/random" "$R/alignable" "$R/compare" "$R/fig03"
cd "$W"

# --- Figure 2 and its supplementary twin, three axes each ---------------------
PYTHONPATH=src "$PY" scripts/ridge_figure.py \
  --summary-dir "$RANDOM_SUM" --out-dir "$R/random" --norm minmax p99 median \
  --title-suffix "{pairs} uniformly random pairs of the 526,871-protein cohort"

PYTHONPATH=src "$PY" scripts/ridge_figure.py \
  --summary-dir "$ALIGN_SUM" --out-dir "$R/alignable" --norm minmax p99 median \
  --title-suffix "{pairs} MMseqs2/Foldseek-alignable pairs of the same cohort"

# --- the three axes side by side, on the numbers ------------------------------
PYTHONPATH=src "$PY" scripts/ridge_axis_comparison.py \
  --summary-dir "$RANDOM_SUM" --out-dir "$R/compare" --tag _random \
  --population "5,000,000 uniformly random pairs of the 526,871-protein cohort"

PYTHONPATH=src "$PY" scripts/ridge_axis_comparison.py \
  --summary-dir "$ALIGN_SUM" --out-dir "$R/compare" --tag _alignable \
  --population "75,849,972 MMseqs2/Foldseek-alignable pairs of the same cohort"

# --- Figure 3 and its twin, from the repo instead of the monkey patch ---------
PYTHONPATH=src "$PY" scripts/fingerprint_figure.py \
  --payload random="$B/final_figures_2026-09-20/fp/out/fingerprint_full.json" \
            alignable="$B/figures_2026-09-19/fig03/fingerprint_full.json" \
  --out-dir "$R/fig03"

# --- the number the axis choice argues about, re-measured on THIS population -----
# Reads the 600 MB pair table, so it is the slow step (~3 min); everything above runs
# off the reduced summaries.
PYTHONPATH=src "$PY" scripts/ridge_divisor_stability.py \
  --pairs-parquet "$B/final_figures_2026-09-20/random_pairs/random_pairs_distances.parquet" \
  --out "$R/compare/divisor_stability_random.json"

# --- install ------------------------------------------------------------------
cp "$R/random/ridge_minmax.png"    "$M/fig02.png"
cp "$R/alignable/ridge_minmax.png" "$M/som_figS05.png"

echo "ALL DONE"
