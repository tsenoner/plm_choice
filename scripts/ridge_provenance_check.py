#!/usr/bin/env python3
"""E7/M-8: can the published ridge statistics be reproduced from anything that survives?

The premise of E7 is that the published figure used a 200-point density grid where
all current code emits 500, so a rerun moves the quartiles.  That is true, but it is
not the whole story, and this script is the check that says so.

Method.  Run the *published* estimator (200-point KDE on the min-max normalised
column, percentiles read off its CDF) over every surviving table that carries
``dist_<arm>`` columns, and compare three things to the published CSV:

  * the median, which min-max scaling moves;
  * the IQR, which min-max scaling also moves;
  * ``r = (median - q25) / (q75 - q25)``, which it CANNOT move.  ``r`` is invariant
    under every affine map, and min-max scaling is affine, so if ``r`` disagrees the
    underlying raw distances differ in shape -- no choice of normalisation anchors
    can reconcile them.

Read-only.  Writes one JSON of the comparison to --out-json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

LABEL_TO_KEY = {
    "Ankh Base": "ankh_base", "Ankh Large": "ankh_large", "CLEAN": "clean",
    "ESM1b": "esm1b", "ESM 1b": "esm1b", "ESM2 8M": "esm2_8m",
    "ESM2 35M": "esm2_35m", "ESM2 150M": "esm2_150m", "ESM2 650M": "esm2_650m",
    "ESM2 3B": "esm2_3b", "ESM3": "esm3_open", "ESM C 300M": "esmc_300m",
    "ESM C 600M": "esmc_600m", "Prot T5": "prott5", "ProtT5": "prott5",
    "Prot Tucker": "prottucker", "ProtTucker": "prottucker",
}


def shape_r(q25: float, med: float, q75: float) -> float:
    return (med - q25) / (q75 - q25)


def published(path: Path) -> dict[str, tuple[float, float, float]]:
    out = {}
    for row in csv.DictReader(path.open()):
        key = LABEL_TO_KEY.get(row["plm_name"].replace("\n", " "), row["plm_name"])
        out[f"dist_{key}"] = (
            float(row["q25"]), float(row["median"]), float(row["q75"])
        )
    return out


def published_estimator(col: np.ndarray, n_grid: int = 200) -> tuple[float, float, float]:
    """Exactly what produced the published CSV: 200-pt KDE, percentiles off its CDF."""
    a = col[np.isfinite(col)]
    a = MinMaxScaler().fit_transform(a.reshape(-1, 1)).ravel()
    x = np.linspace(0, 1, n_grid)
    d = stats.gaussian_kde(a)(x)
    dx = x[1] - x[0]
    cum = np.cumsum(d / (np.sum(d) * dx)) * dx
    return tuple(float(x[int(np.searchsorted(cum, p))]) for p in (0.25, 0.5, 0.75))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published-csv", required=True, type=Path)
    ap.add_argument("--candidate", action="append", required=True,
                    help="name=/path/to.parquet, repeatable")
    ap.add_argument("--head", type=int, default=None,
                    help="only the first N rows of each candidate")
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    pub = published(args.published_csv)
    report = {"published_csv": str(args.published_csv), "head": args.head,
              "candidates": {}}

    for spec in args.candidate:
        name, _, path = spec.partition("=")
        lf = pl.scan_parquet(path)
        have = [c for c in lf.collect_schema().names() if c in pub]
        n_rows = lf.select(pl.len()).collect().item()
        print(f"\n=== {name}  rows={n_rows:,}  shared arms={len(have)}")
        print(f"{'arm':16s} {'pub med':>8s} {'cand med':>9s} "
              f"{'pub IQR':>8s} {'cand IQR':>9s} {'pub r':>6s} {'cand r':>7s} {'dr':>7s}")
        rows, drs = {}, []
        for c in sorted(have):
            s = lf.select(c).head(args.head).collect() if args.head else lf.select(c).collect()
            q25, med, q75 = published_estimator(s.get_column(c).to_numpy())
            pq, pm, p7 = pub[c]
            rp, rc = shape_r(pq, pm, p7), shape_r(q25, med, q75)
            drs.append(rp - rc)
            rows[c] = {"pub": {"q25": pq, "median": pm, "q75": p7, "r": rp},
                       "candidate": {"q25": q25, "median": med, "q75": q75, "r": rc},
                       "d_median": pm - med, "d_r": rp - rc}
            print(f"{c:16s} {pm:8.4f} {med:9.4f} {p7 - pq:8.4f} {q75 - q25:9.4f} "
                  f"{rp:6.3f} {rc:7.3f} {rp - rc:+7.3f}")
        report["candidates"][name] = {
            "path": path, "n_rows": n_rows, "arms": rows,
            "mean_abs_d_r": float(np.mean(np.abs(drs))),
            "max_abs_d_r": float(np.max(np.abs(drs))),
            "mean_abs_d_median": float(
                np.mean([abs(v["d_median"]) for v in rows.values()])
            ),
        }
        print(f"mean |dr| = {report['candidates'][name]['mean_abs_d_r']:.3f}  "
              f"max |dr| = {report['candidates'][name]['max_abs_d_r']:.3f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
