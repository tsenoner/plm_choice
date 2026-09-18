#!/bin/bash
# E7/M-8: everything that runs on the Mac once the cluster job's per-arm distance
# parquets have been rsynced into $RES/dist/.
#
#   bash run_ridge_rebuild.sh
#
# Writes only into the results dir (outside git) -- never into the main repo.
set -euo pipefail

REPO=/Users/tsenoner/Documents/projects/plm_choice
WT=/Users/tsenoner/Documents/projects/plm_choice_ridge
RES=/Users/tsenoner/Documents/projects/_plm_choice_internal_backup/ridge_2026-09-18
PY="$REPO/.venv/bin/python"
PAIRS="$REPO/data/processed/sprot_pre2024_e1_sub10/sets/train.parquet"
RANKING="$REPO/out/sprot_pre2024_subset/plm_ranking_by_spearman.csv"

mkdir -p "$RES/new_data" "$RES/figure"

# 1. merge the per-arm parquets into one pairwise table
"$PY" - <<'EOF'
import polars as pl, glob, os
RES="/Users/tsenoner/Documents/projects/_plm_choice_internal_backup/ridge_2026-09-18"
PAIRS="/Users/tsenoner/Documents/projects/plm_choice/data/processed/sprot_pre2024_e1_sub10/sets/train.parquet"
out=f"{RES}/new_data/pairwise_distances_e1_sub10_train.parquet"
base=pl.read_parquet(PAIRS)
print("base", base.shape)
for p in sorted(glob.glob(f"{RES}/dist/dist_*.parquet")):
    arm=os.path.basename(p)[len("dist_"):-len(".parquet")]
    d=pl.read_parquet(p)
    assert d.height==base.height, (p, d.height, base.height)
    # attach positionally -- so check EVERY id, not a sample: a reordered parquet
    # would otherwise pair each distance with the wrong protein pair, silently.
    assert d["query"].equals(base["query"]), f"{p}: query column reordered"
    assert d["target"].equals(base["target"]), f"{p}: target column reordered"
    base=base.with_columns(d[f"dist_{arm}"])
    print("merged", arm)
base.write_parquet(out, compression="zstd")
print("wrote", out, base.shape, base.columns)
EOF

MERGED="$RES/new_data/pairwise_distances_e1_sub10_train.parquet"

# 2. quartiles under every grid / estimator combination, plus the two caches
"$PY" "$WT/scripts/ridge_quartile_analysis.py" \
  --parquet "$MERGED" \
  --out-json "$RES/new_data/quartiles_new_e1_sub10.json" \
  --cache-500 "$RES/new_data/cache500_new.json" \
  --cache-200 "$RES/new_data/cache200_new.json" \
  --label new_e1_sub10 \
  --columns dist_ankh_base dist_ankh_large dist_clean dist_esm1b dist_esm2_8m \
            dist_esm2_35m dist_esm2_150m dist_esm2_650m dist_esm2_3b dist_esm3_open \
            dist_esmc_300m dist_esmc_600m dist_prott5 dist_prottucker \
  2>&1 | tee "$RES/new_data/quartiles.log"

# 3. the real figure, through the unmodified current code path.
#
# Two orderings, because the published figure and the published caption disagree:
# the caption says "Rows are ordered by overall performance ranking (best-performing
# models at top)", but the published PNG is in the default family-then-size order
# (Ankh Base at top, Prot Tucker at bottom), i.e. it was produced WITHOUT
# --ranking_csv. `figure/` reproduces the published layout; `figure_ranked/` is what
# the caption describes.
cd "$WT"
PYTHONPATH=src "$PY" src/visualization/pairwise_embedding_comparison.py \
  --data_path "$MERGED" \
  --output_dir "$RES/figure" \
  --visualizations distribution_normalized_ridge dip_test \
  --force_recompute 2>&1 | tee "$RES/figure/regen.log"

mkdir -p "$RES/figure_ranked"
PYTHONPATH=src "$PY" src/visualization/pairwise_embedding_comparison.py \
  --data_path "$MERGED" \
  --output_dir "$RES/figure_ranked" \
  --visualizations distribution_normalized_ridge \
  --ranking_csv "$RANKING" \
  --force_recompute 2>&1 | tee "$RES/figure_ranked/regen.log"

echo "DONE"
