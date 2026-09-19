#!/usr/bin/env python3
"""Does the pair POPULATION explain the fingerprint change, or the deduplication?

The published Figure 3 reports CLEAN correlating with its parent ESM-1b at rho=0.38; on our
corrected pair table it is 0.834, and 90 of 91 cells rose. Deduplication cannot be the cause:
a duplicated pair carries the SAME embedding distance in both orientations, so collapsing it
leaves every rank untouched. The candidate explanation is the population. Our pairs are the
ones MMseqs2/Foldseek could align -- related proteins with a real similarity gradient -- while
the published caches carry the fingerprint of a small all-vs-all table, where most pairs are
unrelated and every model is merely reporting "far apart" in its own way.

This script settles it by measuring the same statistic on both populations:
  A) RANDOM pairs drawn uniformly from the cohort (an all-vs-all sample), and
  B) the FILTERED pairs the paper uses,
with identical code, arms and sample size.

    python scripts/fingerprint_pair_population.py --emb-dir <cohort2k> --filtered <dist parquet> \
        --n-random 2000000 --out <dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy.stats import spearmanr

ARMS = ["clean", "esm1b", "prott5", "prottucker", "ankh_base", "ankh_large", "esmc_300m", "esm2_8m"]


def load_matrix(h5_path: Path, ids: list[str]) -> np.ndarray:
    """Embeddings for `ids`, one row each, float32. Handles the (1, D) arms."""
    with h5py.File(h5_path, "r") as h:
        first = np.asarray(h[ids[0]]).ravel()
        out = np.empty((len(ids), first.size), dtype=np.float32)
        for i, pid in enumerate(ids):
            out[i] = np.asarray(h[pid]).ravel()
    return out


def pair_distances(mat: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.empty(a.size, dtype=np.float64)
    for s in range(0, a.size, chunk):
        e = min(s + chunk, a.size)
        d = mat[a[s:e]].astype(np.float64) - mat[b[s:e]].astype(np.float64)
        out[s:e] = np.sqrt(np.einsum("ij,ij->i", d, d))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dir", required=True, type=Path)
    ap.add_argument("--filtered", required=True, type=Path, help="parquet with dist_<arm> columns")
    ap.add_argument("--n-random", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # The id list every arm shares: use the smallest arm's keys (clean/esm1b are the 1022 cohort).
    with h5py.File(args.emb_dir / "clean.h5") as h:
        ids = sorted(h.keys())
    print(f"{len(ids):,} cohort proteins", flush=True)

    # A) random pairs, i.e. a sample of the all-vs-all population
    a = rng.integers(0, len(ids), size=args.n_random)
    b = rng.integers(0, len(ids), size=args.n_random)
    keep = a != b
    a, b = a[keep], b[keep]
    print(f"{a.size:,} random pairs", flush=True)

    rand_d: dict[str, np.ndarray] = {}
    for arm in ARMS:
        mat = load_matrix(args.emb_dir / f"{arm}.h5", ids)
        rand_d[arm] = pair_distances(mat, a, b)
        del mat
        print(f"  random distances done: {arm}", flush=True)

    # B) the filtered pairs the paper uses, same arms, same sample size
    cols = [f"dist_{arm}" for arm in ARMS]
    filt = pl.read_parquet(args.filtered, columns=cols)
    mask = np.ones(filt.height, dtype=bool)
    for c in cols:
        v = filt[c].to_numpy()
        mask &= np.isfinite(v) & (v > 0)  # > 0 also drops identical-sequence pairs
    idx = np.flatnonzero(mask)
    if idx.size > a.size:
        idx = rng.choice(idx, a.size, replace=False)
    filt_d = {arm: filt[f"dist_{arm}"].to_numpy()[idx] for arm in ARMS}
    print(f"{idx.size:,} filtered pairs", flush=True)

    rows = []
    for x, y in itertools.combinations(ARMS, 2):
        rows.append({
            "arm_a": x, "arm_b": y,
            "rho_random_pairs": float(spearmanr(rand_d[x], rand_d[y]).statistic),
            "rho_filtered_pairs": float(spearmanr(filt_d[x], filt_d[y]).statistic),
        })
    df = pl.DataFrame(rows).with_columns(
        (pl.col("rho_filtered_pairs") - pl.col("rho_random_pairs")).alias("difference")
    )
    df.write_csv(args.out / "fingerprint_pair_population.csv")
    summary = {
        "n_random_pairs": int(a.size),
        "n_filtered_pairs": int(idx.size),
        "n_proteins": len(ids),
        "mean_rho_random": float(df["rho_random_pairs"].mean()),
        "mean_rho_filtered": float(df["rho_filtered_pairs"].mean()),
        "clean_esm1b_random": float(df.filter((pl.col("arm_a") == "clean") & (pl.col("arm_b") == "esm1b"))["rho_random_pairs"][0]),
        "clean_esm1b_filtered": float(df.filter((pl.col("arm_a") == "clean") & (pl.col("arm_b") == "esm1b"))["rho_filtered_pairs"][0]),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(df.sort("difference"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
