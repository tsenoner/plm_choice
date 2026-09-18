#!/usr/bin/env python3
"""E7/M-8: independent check that the recomputed distances reproduce the published ones.

The corrected pair set and the pre-rebuild `train_ext.parquet` are different pair
sets, but they overlap: any (query, target) present in both must get the same
distance, because only the pair *selection* changed, not the embeddings or the
metric.  Comparing on that overlap validates the whole recompute chain -- the
cohort2k HDF5 files, the (1, D) flattening, the chunked L2, the 4-decimal
rounding -- against a value that was produced years earlier by completely
different code.

Read-only on the main repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, type=Path, help="merged new distances")
    ap.add_argument("--old", required=True, type=Path, help="train_ext.parquet")
    ap.add_argument("--sample", type=int, default=2_000_000,
                    help="rows of the new table to look up in the old one")
    args = ap.parse_args()

    new = pl.read_parquet(args.new).head(args.sample)
    dist_cols = [c for c in new.columns if c.startswith("dist_")]
    print(f"new sample rows={len(new):,} cols={dist_cols}")

    old_cols = pl.scan_parquet(args.old).collect_schema().names()
    shared = [c for c in dist_cols if c in old_cols]
    print(f"shared distance columns: {shared}")

    keys = new.select("query", "target")
    old = (
        pl.scan_parquet(args.old)
        .select(["query", "target", *shared])
        .join(keys.lazy(), on=["query", "target"], how="semi")
        .collect()
    )
    print(f"overlapping pairs found in train_ext: {len(old):,}")
    if not len(old):
        print("NO OVERLAP -- cannot cross-check this way")
        return

    merged = old.join(new.select(["query", "target", *shared]), on=["query", "target"],
                      suffix="_new")
    print(f"joined rows: {len(merged):,}")

    print(f"\n{'column':22s} {'n':>10s} {'n both finite':>14s} {'max |diff|':>12s} {'exact':>7s}")
    for c in shared:
        a = merged[c].to_numpy()
        b = merged[f"{c}_new"].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        d = np.abs(a[both] - b[both])
        print(f"{c:22s} {len(a):10,d} {int(both.sum()):14,d} "
              f"{(d.max() if d.size else float('nan')):12.6f} "
              f"{str(bool(d.size and d.max() == 0)):>7s}")


if __name__ == "__main__":
    main()
