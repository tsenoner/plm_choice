#!/usr/bin/env python3
"""How much does each candidate ridge divisor move when the pairs are resampled?

The manuscript's case for the /p99 axis rests on a measured number -- resampling 10%
of the pairs moves the min-max divisor (the single most distant pair) by up to 13.7%
and the 99th-percentile divisor by at most 0.094%.  That was measured on the 75.5M
aligner-found pairs.  Figure 2 is now drawn on 5,000,000 uniformly random pairs, a
different population with different tails, so the number has to be re-measured there
before it is quoted for or against either axis.

Each divisor is a statistic of the same column, so they are compared on the same
draws: one resample, three divisors.  Deviation is reported against the full-sample
value of that divisor, in percent, as the worst and the median over the draws.

    python scripts/ridge_divisor_stability.py \
        --pairs-parquet <random_pairs_distances.parquet> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

#: The three anchors the ridge axis can be built on, as percentile levels.  ``None``
#: means the sample maximum, which is what min-max divides by.
DIVISORS = {"max": None, "p99.9": 99.9, "p99": 99.0}

ARMS = (
    "ankh_base", "ankh_large", "clean", "esm1b", "esm2_8m", "esm2_35m", "esm2_150m",
    "esm2_650m", "esm2_3b", "esm3_open", "esmc_300m", "esmc_600m", "prott5",
    "prottucker",
)


def value(x: np.ndarray, level: float | None) -> float:
    return float(x.max()) if level is None else float(np.percentile(x, level))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-parquet", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    rows = []
    header = "arm".ljust(12) + "".join(f"{k + ' dev %':>14s}" for k in DIVISORS)
    print(f"{args.draws} draws of {args.fraction:.0%} of the pairs, worst deviation")
    print(header)
    for arm in args.arms:
        x = (
            pl.read_parquet(args.pairs_parquet, columns=[f"dist_{arm}"])
            .to_series()
            .to_numpy()
            .astype(np.float64)
        )
        n_sub = int(round(args.fraction * x.size))
        full = {k: value(x, lvl) for k, lvl in DIVISORS.items()}
        devs = {k: [] for k in DIVISORS}
        for _ in range(args.draws):
            s = rng.choice(x, n_sub, replace=False)
            for k, lvl in DIVISORS.items():
                devs[k].append(abs(value(s, lvl) - full[k]) / full[k] * 100)
        row = {"arm": arm, "n_pairs": int(x.size), "n_resample": n_sub}
        for k in DIVISORS:
            row[f"{k}_full"] = full[k]
            row[f"{k}_worst_dev_pct"] = float(np.max(devs[k]))
            row[f"{k}_median_dev_pct"] = float(np.median(devs[k]))
        rows.append(row)
        print(arm.ljust(12) + "".join(f"{row[f'{k}_worst_dev_pct']:14.3f}" for k in DIVISORS))

    worst = {k: max(r[f"{k}_worst_dev_pct"] for r in rows) for k in DIVISORS}
    print("\nworst over arms: " + ", ".join(f"{k} {v:.2f}%" for k, v in worst.items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": str(args.pairs_parquet),
                "fraction": args.fraction,
                "draws": args.draws,
                "seed": args.seed,
                "worst_over_arms_pct": worst,
                "per_arm": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
