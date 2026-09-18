#!/usr/bin/env python3
"""E7/M-8: build the before/after quartile tables for the rebuilt ridge figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

ORDER = [
    "dist_ankh_base", "dist_ankh_large", "dist_clean", "dist_esm1b",
    "dist_esm2_8m", "dist_esm2_35m", "dist_esm2_150m", "dist_esm2_650m",
    "dist_esm2_3b", "dist_esm3_open", "dist_esmc_300m", "dist_esmc_600m",
    "dist_prott5", "dist_prottucker",
]
#: Two-line figure labels, verbatim from visualization/plm_constants.py, plus the
#: older spellings the *published* statistics CSV used ("ESM1b" where the table now
#: says "ESM\n1b").  The published CSV keys its rows by label, so mapping back to
#: the embedding key needs both spellings.
_DISPLAY = {
    "ankh_base": "Ankh\nBase", "ankh_large": "Ankh\nLarge", "clean": "CLEAN",
    "esm1b": "ESM\n1b", "esm2_8m": "ESM2\n8M", "esm2_35m": "ESM2\n35M",
    "esm2_150m": "ESM2\n150M", "esm2_650m": "ESM2\n650M", "esm2_3b": "ESM2\n3B",
    "esm3_open": "ESM3", "esmc_300m": "ESM C\n300M", "esmc_600m": "ESM C\n600M",
    "prott5": "Prot\nT5", "prottucker": "Prot\nTucker",
}
_LEGACY = {"esm1b": "ESM1b"}
LABEL = {f"dist_{k}": v.replace("\n", " ") for k, v in _DISPLAY.items()}


def read_published(path: Path) -> dict[str, dict]:
    """Map a ridge statistics CSV to embedding keys, old format or new.

    The current writer puts the embedding key in ``plm_name``; the published CSV
    put the two-line display label there instead.
    """
    df = pl.read_csv(path)
    inv: dict[str, str] = {}
    for key, disp in _DISPLAY.items():
        for spelling in (key, disp, disp.replace("\n", " "), disp.replace("\n", "")):
            inv[spelling] = f"dist_{key}"
    for key, legacy in _LEGACY.items():
        inv[legacy] = f"dist_{key}"

    out = {}
    for row in df.iter_rows(named=True):
        name = row["plm_name"]
        key = inv.get(name) or inv.get(name.replace("\n", " "))
        if key is None:
            raise SystemExit(f"cannot map plm_name {name!r} from {path} to a key")
        out[key] = {"q25": row["q25"], "median": row["median"], "q75": row["q75"]}
    return out


def md_table(header: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def f(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published-csv", required=True, type=Path)
    ap.add_argument("--new-json", required=True, type=Path)
    ap.add_argument("--old-json", type=Path, default=None)
    ap.add_argument("--new-stats-csv", type=Path, default=None,
                    help="statistics CSV the real plotting run emitted")
    ap.add_argument("--out-md", required=True, type=Path)
    args = ap.parse_args()

    pub = read_published(args.published_csv)
    new = json.loads(args.new_json.read_text())
    old = json.loads(args.old_json.read_text()) if args.old_json else None
    shipped = read_published(args.new_stats_csv) if args.new_stats_csv else None

    parts: list[str] = []

    # --- headline before/after -------------------------------------------------
    rows = []
    for col in ORDER:
        p = pub.get(col, {})
        n = new["arms"][col]["kde500_current"]
        rows.append([
            LABEL[col],
            f(p.get("q25")), f(p.get("median")), f(p.get("q75")),
            f(n["q25"]), f(n["median"]), f(n["q75"]),
            f(n["q25"] - p["q25"], 4) if p else "-",
            f(n["median"] - p["median"], 4) if p else "-",
            f(n["q75"] - p["q75"], 4) if p else "-",
        ])
    parts.append("### Published vs rebuilt (the numbers that would go in the caption)\n\n"
                 "Published = 200-point grid, all three percentiles read off the KDE CDF, "
                 "over an input that no longer exists and is NOT `train_ext.parquet` "
                 "(see the provenance check).  Rebuilt = current code path (500-point grid, "
                 "exact `np.median`) over the corrected 5,958,720-pair set.  The delta is "
                 "therefore a TOTAL delta: it cannot be decomposed against the published "
                 "side, because that side's input is unrecoverable.\n\n"
                 + md_table(
                     ["pLM", "pub Q25", "pub med", "pub Q75",
                      "new Q25", "new med", "new Q75",
                      "dQ25", "dmed", "dQ75"], rows))

    # --- decomposition ---------------------------------------------------------
    rows = []
    for col in ORDER:
        a = new["arms"][col]
        o = old["arms"][col] if old else None
        rows.append([
            LABEL[col],
            f(pub[col]["median"]),
            f(o["kde200_cdf"]["median"]) if o else "-",
            f(o["kde500_cdf"]["median"]) if o else "-",
            f(a["kde200_cdf"]["median"]),
            f(a["kde500_cdf"]["median"]),
            f(a["kde500_current"]["median"]),
            f(a["empirical"]["median"]),
        ])
    parts.append("### Median: grid change vs estimator change vs data change\n\n"
                 "`old` = pre-rebuild `train_ext.parquet` (NOT the published figure's "
                 "input -- it is included as the only surviving pre-rebuild distance "
                 "table), `new` = corrected `sprot_pre2024_e1_sub10` train split.  "
                 "`cdf` = the published estimator (percentile read off the KDE CDF); "
                 "`current` = what `plot_ridge_distributions` emits today (exact "
                 "`np.median`).  The clean, well-defined isolations are new/200 vs "
                 "new/500 (pure grid effect) and new/500/cdf vs new/500/current "
                 "(pure estimator effect).\n\n"
                 + md_table(
                     ["pLM", "published", "old/200/cdf", "old/500/cdf",
                      "new/200/cdf", "new/500/cdf", "new/500/current",
                      "new empirical"], rows))

    for q in ("q25", "q75"):
        rows = []
        for col in ORDER:
            a = new["arms"][col]
            o = old["arms"][col] if old else None
            rows.append([
                LABEL[col], f(pub[col][q]),
                f(o["kde200_cdf"][q]) if o else "-",
                f(o["kde500_cdf"][q]) if o else "-",
                f(a["kde200_cdf"][q]), f(a["kde500_cdf"][q]),
                f(a["empirical"][q]),
            ])
        parts.append(f"### {q.upper()}: same decomposition\n\n" + md_table(
            ["pLM", "published", "old/200/cdf", "old/500/cdf",
             "new/200/cdf", "new/500/cdf", "new empirical"], rows))

    # --- raw scale -------------------------------------------------------------
    rows = []
    for col in ORDER:
        a = new["arms"][col]
        o = old["arms"][col] if old else None
        rows.append([
            LABEL[col], f"{a['n_valid']:,}",
            f(a["raw_min"], 3), f(a["raw_max"], 3),
            f(o["raw_min"], 3) if o else "-", f(o["raw_max"], 3) if o else "-",
        ])
    parts.append("### Raw (un-normalised) Euclidean scale, and what min-max is anchored on\n\n"
                 "The figure min-max-scales each column, so its x-axis is pinned to the "
                 "single closest and single furthest pair in that column.  Those anchors "
                 "move when the pair set changes.\n\n"
                 + md_table(["pLM", "new n valid", "new raw min", "new raw max",
                             "old raw min", "old raw max"], rows))

    if shipped:
        rows = []
        for col in ORDER:
            s, n = shipped.get(col, {}), new["arms"][col]["kde500_current"]
            rows.append([LABEL[col], f(s.get("q25")), f(s.get("median")),
                         f(s.get("q75")),
                         f(abs(s["q25"] - n["q25"]), 6) if s else "-",
                         f(abs(s["median"] - n["median"]), 6) if s else "-",
                         f(abs(s["q75"] - n["q75"]), 6) if s else "-"])
        parts.append("### Cross-check: the statistics CSV the real plotting run wrote\n\n"
                     + md_table(["pLM", "Q25", "median", "Q75",
                                 "|d| Q25", "|d| med", "|d| Q75"], rows))

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {args.out_md}")
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
