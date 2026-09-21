#!/bin/bash
# Job 3b: Figure 3 (the fingerprint matrix) on the 5M uniform random-pair sample.
#
# fingerprint_full_reduce.py is the tracked reducer and is called UNMODIFIED. It wants
# the cluster's split layout (<root>/<split>/dist_<arm>.parquet) plus a positional
# identical-sequence mask, so the flat random-pair table is materialised into that
# layout as a single split named "all", with an all-False mask: the sampler already
# dropped self-pairs and identical sequences (keep = (a != b) & (h[a] != h[b])), so
# masking again would be double-counting. Same estimator, same population rule, same
# 14 arms (random_1024 excluded by design), so the new matrix is comparable cell for
# cell with the filtered-pair one.
set -euo pipefail

W=${REPO:?set REPO to a checkout of this repository}
R=${PAPER_ARTEFACTS:?set PAPER_ARTEFACTS to the directory holding the measured outputs}/final_figures_2026-09-20
PY=${REPO}/.venv/bin/python
PAIRS=$R/random_pairs/random_pairs_distances.parquet
FP=$R/fp

mkdir -p "$FP/dist/all" "$FP/identical" "$FP/out" "$FP/plot"

echo "=== materialising the split layout the reducer expects ==="
"$PY" - "$PAIRS" "$FP" <<'PYEOF'
import sys
from pathlib import Path
import polars as pl

pairs, fp = Path(sys.argv[1]), Path(sys.argv[2])
ARMS = ("ankh_base", "ankh_large", "clean", "esm1b", "esm2_8m", "esm2_35m",
        "esm2_150m", "esm2_650m", "esm2_3b", "esm3_open", "esmc_300m",
        "esmc_600m", "prott5", "prottucker")
n = None
for arm in ARMS:
    col = f"dist_{arm}"
    df = pl.read_parquet(pairs, columns=[col])
    n = df.height if n is None else n
    assert df.height == n, f"{arm}: {df.height} != {n}"
    df.write_parquet(fp / "dist" / "all" / f"{col}.parquet")
    print(f"  wrote {col}.parquet ({df.height:,} rows)", flush=True)
# All-False: identical-sequence pairs were removed by the sampler, by SHA-1 of the
# sequence, before any distance was computed.
pl.DataFrame({"identical": [False] * n}).write_parquet(
    fp / "identical" / "all_identical.parquet"
)
print(f"  wrote all_identical.parquet ({n:,} rows, all False)")
PYEOF

echo "=== reduce: Spearman + Wasserstein over 14 arms ==="
cd "$W"
PYTHONPATH=src "$PY" scripts/fingerprint_full_reduce.py \
  --dist-root "$FP/dist" --identical-root "$FP/identical" \
  --out-dir "$FP/out" --splits all

echo "=== draw Figure 3 (Wasserstein upper, Spearman lower) ==="
PYTHONPATH=src "$PY" src/visualization/pairwise_embedding_comparison.py \
  --precomputed "$FP/out/fingerprint_full.json" \
  --precomputed_output "$R/fig03_randompairs.png" \
  --output_dir "$FP/plot"

echo "ALL DONE"
