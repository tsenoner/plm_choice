#!/usr/bin/env python3
"""E7/M-8: quantify what moved in the ridge figure.

Three effects are confounded between the published figure and a rerun:

  1. the density grid: the published cache holds 200 points, every surviving code
     path emits 500;
  2. the percentile estimator: the published cache carries no ``median`` key, so
     the published median is a KDE-CDF grid estimate like Q25/Q75.  Current code
     writes the exact ``np.median`` and ``plot_ridge_distributions`` prefers it;
  3. the pair set: pre-rebuild 113,186,256 pairs vs the corrected, deduplicated
     10% sample of 5,958,720.

This script computes each arm's quartiles under every combination so the three
can be read apart, and additionally reports the exact empirical quartiles, which
depend on neither the grid nor the KDE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


def normalize_distribution(x: pl.Series) -> np.ndarray:
    """Byte-for-byte the plotting code's normaliser (min-max over the column)."""
    x_clean = x.drop_nulls().to_numpy()
    if len(x_clean) == 0:
        return x_clean
    x_clean = x_clean[np.isfinite(x_clean)]
    if len(x_clean) == 0:
        return x_clean
    if np.all(x_clean == x_clean[0]):
        return np.full_like(x_clean, 0.5)
    return MinMaxScaler().fit_transform(x_clean.reshape(-1, 1)).ravel()


def kde_entry(data_clean: np.ndarray, n_grid: int) -> dict:
    """The cache entry compute_distribution_data writes, at a chosen grid size."""
    x_range = np.linspace(0, 1, n_grid)
    if len(data_clean) > 1:
        kernel = stats.gaussian_kde(data_clean)
        density = kernel(x_range)
        peak_idx = int(np.argmax(density))
        peak_x, peak_y = float(x_range[peak_idx]), float(density[peak_idx])
    else:
        density = np.zeros_like(x_range)
        peak_x = peak_y = 0.0
    return {
        "x_range": x_range.tolist(),
        "density": density.tolist(),
        "peak_x": peak_x,
        "peak_y": peak_y,
        "min": float(data_clean.min()) if len(data_clean) else 0.0,
        "max": float(data_clean.max()) if len(data_clean) else 0.0,
        "median": float(np.median(data_clean)) if len(data_clean) else None,
    }


def cdf_quartiles(entry: dict, use_exact_median: bool) -> dict:
    """plot_ridge_distributions' percentile estimator, both estimator variants."""
    x_range = np.array(entry["x_range"])
    density = np.array(entry["density"])
    dx = x_range[1] - x_range[0]
    cumulative = np.cumsum(density / (np.sum(density) * dx)) * dx
    out = {}
    for name, p in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
        if name == "median" and use_exact_median and entry.get("median") is not None:
            out[name] = float(entry["median"])
            continue
        i = int(np.searchsorted(cumulative, p))
        out[name] = float(x_range[i]) if i < len(x_range) else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cache-500", type=Path, help="also write the 500-pt cache JSON")
    ap.add_argument("--cache-200", type=Path)
    ap.add_argument("--columns", nargs="*", default=None)
    args = ap.parse_args()

    lf = pl.scan_parquet(args.parquet)
    schema = lf.collect_schema().names()
    cols = args.columns or [c for c in schema if c.startswith("dist_")]
    n_rows = lf.select(pl.len()).collect().item()
    print(f"{args.label}: {args.parquet} rows={n_rows:,} cols={cols}")

    results = {"label": args.label, "parquet": str(args.parquet), "n_rows": n_rows,
               "arms": {}}
    cache500 = {"metadata": {"normalized": True}, "distributions": {}}
    cache200 = {"metadata": {"normalized": True}, "distributions": {}}

    for col in cols:
        s = lf.select(col).collect().get_column(col)
        raw = s.drop_nulls().to_numpy()
        raw = raw[np.isfinite(raw)]
        norm = normalize_distribution(s)
        del s

        e500 = kde_entry(norm, 500)
        e200 = kde_entry(norm, 200)
        cache500["distributions"][col] = e500
        cache200["distributions"][col] = e200

        results["arms"][col] = {
            "n_valid": int(len(norm)),
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
            "empirical": {
                "q25": float(np.quantile(norm, 0.25)),
                "median": float(np.median(norm)),
                "q75": float(np.quantile(norm, 0.75)),
            },
            # published convention: KDE-CDF for all three percentiles
            "kde200_cdf": cdf_quartiles(e200, use_exact_median=False),
            "kde500_cdf": cdf_quartiles(e500, use_exact_median=False),
            # current convention: KDE-CDF for Q25/Q75, exact np.median for median
            "kde200_current": cdf_quartiles(e200, use_exact_median=True),
            "kde500_current": cdf_quartiles(e500, use_exact_median=True),
            "peak_x_500": e500["peak_x"],
            "peak_x_200": e200["peak_x"],
        }
        print(f"  {col}: n={len(norm):,} raw=[{raw.min():.4f},{raw.max():.4f}] "
              f"emp_med={np.median(norm):.4f} kde500_med={results['arms'][col]['kde500_cdf']['median']:.4f}")
        del raw, norm

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))
    if args.cache_500:
        args.cache_500.parent.mkdir(parents=True, exist_ok=True)
        args.cache_500.write_text(json.dumps(cache500))
    if args.cache_200:
        args.cache_200.parent.mkdir(parents=True, exist_ok=True)
        args.cache_200.write_text(json.dumps(cache200))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
