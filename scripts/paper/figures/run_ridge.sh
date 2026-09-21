#!/bin/bash
# Job 3a: reduce the 5M uniform random-pair sample to per-arm ridge summaries,
# then draw Figure 2 (p99, the published scaling) and the min-max variant.
#
# The reducer and the plotter are the tracked scripts from ../plm_choice_ridge,
# called unmodified: ridge_full_reduce.py already accepts a flat --pairs-parquet
# carrying dist_<arm> columns, which is exactly the random-pair table's layout.
# No --identical-root is passed because the sampler already removed self-pairs and
# identical sequences; the fallback (drop distance == 0) is a no-op here, verified:
# every arm has zero exact-zero distances.
set -euo pipefail

W=${REPO:?set REPO to a checkout of this repository}
R=${PAPER_ARTEFACTS:?set PAPER_ARTEFACTS to the directory holding the measured outputs}/final_figures_2026-09-20
PY=${REPO}/.venv/bin/python
PAIRS=$R/random_pairs/random_pairs_distances.parquet
SUM=$R/ridge/summaries
FIG=$R/ridge/figures

mkdir -p "$SUM" "$FIG"
cd "$W"

ARMS="ankh_base ankh_large clean esm1b esm2_8m esm2_35m esm2_150m esm2_650m esm2_3b esm3_open esmc_300m esmc_600m prott5 prottucker random_1024"

for arm in $ARMS; do
  echo "=== $arm ==="
  PYTHONPATH=src "$PY" scripts/ridge_full_reduce.py \
    --pairs-parquet "$PAIRS" --arm "$arm" --out-dir "$SUM"
done

echo "=== ridge figure: p99 (Figure 2 scaling) + minmax variant ==="
PYTHONPATH=src "$PY" scripts/ridge_figure.py \
  --summary-dir "$SUM" --out-dir "$FIG" --norm p99 minmax

echo "ALL DONE"
