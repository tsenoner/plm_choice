#!/usr/bin/env python3
"""Does the 10% subset give the same ridge statistics as the full cohort?

The probes were retrained on a 10% sample of the corrected pairs, so the manuscript may
want to quote quartiles from the same data the probe table describes.  That is only
allowed if the two agree, and "they agree" is a measurement, not an assumption.  This
puts the two side by side per arm and reports the largest relative disagreement.

Everything compared here is scale-free or raw, never the normalised axis: the raw
quartiles, and the quartile coefficient of dispersion (q75-q25)/(q75+q25), which is
invariant under any positive rescaling and is therefore the statistic that can settle
"is this arm broader than that one" without first choosing a normalisation.

    python scripts/ridge_subset_vs_full.py --full <dir> --subset <dir> --out <csv>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def load(summary_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(summary_dir.glob("*.json"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", required=True, type=Path)
    ap.add_argument("--subset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    full, sub = load(args.full), load(args.subset)
    rows = []
    for arm in sorted(set(full) & set(sub)):
        f, s = full[arm], sub[arm]
        row = {"plm_name": arm, "n_full": f["n_used"], "n_sub10": s["n_used"]}
        for stat in ("q25", "median", "q75", "iqr", "qcd"):
            row[f"{stat}_full"] = f[stat]
            row[f"{stat}_sub10"] = s[stat]
            row[f"{stat}_rel_diff"] = (
                (s[stat] - f[stat]) / f[stat] if f[stat] else float("nan")
            )
        row["p99_full"] = f["quantiles"]["p99"]
        row["p99_sub10"] = s["quantiles"]["p99"]
        row["max_full"] = f["max"]
        row["max_sub10"] = s["max"]
        # Min-max's anchor is the single most distant pair, so this column is the one
        # that moves between cohorts: a 10% sample simply may not contain it.
        row["max_rel_diff"] = (s["max"] - f["max"]) / f["max"]
        rows.append(row)

    df = pl.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.out)

    def worst(col: str) -> tuple[str, float]:
        sub = df.select("plm_name", pl.col(col).abs().alias("a")).sort("a", descending=True)
        return sub[0, "plm_name"], sub[0, "a"]

    print(f"wrote {args.out}  ({len(df)} arms)")
    for stat in ("q25", "median", "q75", "iqr", "qcd"):
        arm, val = worst(f"{stat}_rel_diff")
        print(f"  {stat:>7s}: max |relative difference| = {100 * val:.3f}%  ({arm})")
    arm, val = worst("max_rel_diff")
    print(f"  {'max':>7s}: max |relative difference| = {100 * val:.3f}%  ({arm})"
          "   <- the min-max divisor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
