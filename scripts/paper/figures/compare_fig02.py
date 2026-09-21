#!/usr/bin/env python3
"""Figure 2, side by side: uniform random pairs vs the filtered (alignable) pairs.

Both columns come from the SAME estimator (ridge_full_reduce.py -> ridge_figure.py)
under the SAME p99 normalisation, so every difference below is a difference in which
pairs are described, not in how they are measured.

Quartiles are quoted on the p99-normalised axis the figure actually draws (q25/median/
q75), with the raw-scale values and the p99 divisor alongside, because the divisor is
itself a property of the population and moves between the two.
"""

from __future__ import annotations

# These paths were absolute to one machine. They are environment variables now, so an
# unset one fails here by name rather than as a FileNotFoundError further down.
import os
from pathlib import Path

import polars as pl


def _need(var: str) -> str:
    """The value of `var`, or a message naming what to set."""
    try:
        return os.environ[var]
    except KeyError:
        raise SystemExit(f"set {var} before running this script") from None


ARTEFACTS = _need("PAPER_ARTEFACTS")

B = Path(ARTEFACTS)
NEW = B / "final_figures_2026-09-20/ridge/figures/ridge_p99_statistics.csv"
OLD = B / "ridge_full_2026-09-19/figures/ridge_p99_statistics.csv"
OUT = B / "final_figures_2026-09-20/fig02_random_vs_filtered.csv"

KEYS = [
    ("q25", "q25 (/p99)"),
    ("median", "median (/p99)"),
    ("q75", "q75 (/p99)"),
    ("iqr", "IQR (/p99)"),
    ("qcd", "QCD"),
    ("raw_q25", "raw q25"),
    ("raw_median", "raw median"),
    ("raw_q75", "raw q75"),
    ("raw_p99", "raw p99 (divisor)"),
    ("raw_max", "raw max"),
]


def main() -> None:
    new = pl.read_csv(NEW).sort("plm_name")
    old = pl.read_csv(OLD).sort("plm_name")
    assert new["plm_name"].to_list() == old["plm_name"].to_list(), "arm sets differ"

    rows = []
    for arm, disp in zip(new["plm_name"], new["plm_display_name"], strict=True):
        n = new.filter(pl.col("plm_name") == arm)
        o = old.filter(pl.col("plm_name") == arm)
        for key, label in KEYS:
            nv, ov = float(n[key][0]), float(o[key][0])
            rows.append({
                "arm": arm,
                "display": disp,
                "statistic": label,
                "random_5M": nv,
                "filtered_alignable": ov,
                "delta": nv - ov,
                "pct_change": (nv - ov) / ov * 100 if ov else float("nan"),
            })
        rows.append({
            "arm": arm, "display": disp, "statistic": "n pairs",
            "random_5M": float(n["n_used"][0]),
            "filtered_alignable": float(o["n_used"][0]),
            "delta": float(n["n_used"][0] - o["n_used"][0]),
            "pct_change": float("nan"),
        })

    df = pl.DataFrame(rows)
    df.write_csv(OUT)

    # --- printed table: the four numbers the caption quotes --------------------
    hdr = (f"{'arm':<12} {'q25 /p99':>19} {'median /p99':>19} "
           f"{'q75 /p99':>19} {'QCD':>19}")
    print("Figure 2, p99-normalised quartiles.  random 5M  ->  filtered (alignable)")
    print(hdr)
    print("-" * len(hdr))
    for arm in new["plm_name"]:
        n = new.filter(pl.col("plm_name") == arm)
        o = old.filter(pl.col("plm_name") == arm)
        cells = []
        for key in ("q25", "median", "q75", "qcd"):
            cells.append(f"{float(n[key][0]):.3f} -> {float(o[key][0]):.3f}")
        print(f"{arm:<12} " + " ".join(f"{c:>19}" for c in cells))

    print()
    hdr2 = f"{'arm':<12} {'raw median':>23} {'raw p99 (divisor)':>25} {'raw max':>25}"
    print("Raw scale (same two populations)")
    print(hdr2)
    print("-" * len(hdr2))
    for arm in new["plm_name"]:
        n = new.filter(pl.col("plm_name") == arm)
        o = old.filter(pl.col("plm_name") == arm)
        c = []
        for key in ("raw_median", "raw_p99", "raw_max"):
            c.append(f"{float(n[key][0]):.4g} -> {float(o[key][0]):.4g}")
        print(f"{arm:<12} {c[0]:>23} {c[1]:>25} {c[2]:>25}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
