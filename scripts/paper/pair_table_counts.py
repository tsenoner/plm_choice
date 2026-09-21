#!/usr/bin/env python3
"""Count, per split, how many pairs carry each similarity target.

Why this exists. Supplementary Methods states six per-split counts -- pairs with PIDE and
HFSP, and pairs with a TM-score, for train/val/test -- and the Results opener restates
them as percentages ("99% carry a TM-score and 48% PIDE and HFSP"). An audit on
2026-09-21 could not trace any of the six to a file or a script: they were written down
once and never recomputed. A number in a paper that nothing regenerates is a number
nobody can check, including us.

This recomputes them from the pair tables themselves and writes the result as JSON, so
the claim has an artefact behind it and the deposit can carry it.

    python scripts/paper/pair_table_counts.py --sets-dir <dataset>/sets --out counts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

# The three targets, and the columns that carry them. HFSP is only defined where PIDE is,
# so "PIDE and HFSP" is one population rather than two.
PIDE_COLS = ("fident", "pide")
HFSP_COLS = ("hfsp",)
TM_COLS = ("alntmscore", "tmscore", "alntm")


def _first_present(schema: list[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in schema:
            return name
    return None


def count_split(path: Path) -> dict[str, int | str]:
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema().names()
    pide, hfsp, tm = (
        _first_present(schema, PIDE_COLS),
        _first_present(schema, HFSP_COLS),
        _first_present(schema, TM_COLS),
    )
    if pide is None or tm is None:
        raise SystemExit(f"{path}: expected a PIDE and a TM-score column, found {schema}")

    exprs = [pl.len().alias("n_pairs"), pl.col(tm).is_not_null().sum().alias("n_tmscore")]
    # PIDE and HFSP together, because that is the population the manuscript names.
    both = pl.col(pide).is_not_null()
    if hfsp is not None:
        both = both & pl.col(hfsp).is_not_null()
    exprs.append(both.sum().alias("n_pide_hfsp"))

    row = lf.select(exprs).collect().row(0, named=True)
    return {"split": path.stem, "columns": {"pide": pide, "hfsp": hfsp, "tmscore": tm}, **row}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    rows = [count_split(args.sets_dir / f"{s}.parquet") for s in ("train", "val", "test")]
    total = {k: sum(int(r[k]) for r in rows) for k in ("n_pairs", "n_tmscore", "n_pide_hfsp")}
    total["pct_tmscore"] = round(100 * total["n_tmscore"] / total["n_pairs"], 2)
    total["pct_pide_hfsp"] = round(100 * total["n_pide_hfsp"] / total["n_pairs"], 2)

    report = {"sets_dir": str(args.sets_dir), "per_split": rows, "total": total}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    for r in rows:
        print(f"{r['split']:6} {r['n_pairs']:>12,}  PIDE+HFSP {r['n_pide_hfsp']:>12,}  TM {r['n_tmscore']:>12,}")
    print(f"{'total':6} {total['n_pairs']:>12,}  PIDE+HFSP {total['n_pide_hfsp']:>12,} "
          f"({total['pct_pide_hfsp']}%)  TM {total['n_tmscore']:>12,} ({total['pct_tmscore']}%)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
