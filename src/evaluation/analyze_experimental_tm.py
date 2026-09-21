#!/usr/bin/env python3
"""
Task B6 / reviewer R2.2 analysis.

Reads the parquet written by ``src/data_preparation/pdb_tmscore.py`` and asks:
how much does the AlphaFold/Foldseek ``alntmscore`` we train on actually track
EXPERIMENTAL structural similarity, and how much of experimental structural
similarity is just sequence identity?

Writes stats.json into the given results directory and prints the same JSON.

NOTE on normalisation, which the whole comparison hinges on:
  * Foldseek ``alntmscore`` is normalised by the ALIGNMENT SPAN,
    ``min(qEndPos-qStartPos, dbEndPos-dbStartPos)`` (foldseek
    ``structureconvertalis.cpp``). It is therefore not a whole-chain TM-score
    and is biased upward whenever the alignment covers only part of a chain.
  * US-align ``TM1``/``TM2`` are normalised by the full length of chain 1 / 2.
    We report three derived quantities:
      tmscore_exp        = min(TM1, TM2)  -> normalised by the LONGER chain
      tmscore_exp_short  = max(TM1, TM2)  -> normalised by the SHORTER chain
      tmscore_exp_avg    = length-weighted mean (approximates US-align -a T)
    ``tmscore_exp_short`` is the closest in spirit to alntmscore's short
    normalisation length, so the headline comparison is reported for all three.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

EXP_VARIANTS = [
    ("tmscore_exp", "min(TM1,TM2), normalised by the LONGER chain"),
    ("tmscore_exp_short", "max(TM1,TM2), normalised by the SHORTER chain"),
    ("tmscore_exp_avg", "length-weighted mean (approx. US-align -a T)"),
]


# A CATH homologous-superfamily code is four dot-separated integers, "1.10.8.10".
_CATH_CODE_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def load_cath(path: Path, wanted: set[tuple[str, str]]) -> dict[tuple[str, str], set[str]]:
    """
    (pdb_id, chain) -> set of CATH homologous-superfamily codes (C.A.T.H).

    Expects ``cath-b-newest-all.gz``, whose rows are
    ``<domain> <version> <C.A.T.H> <boundaries>`` -- field 3 is the full
    four-level code. ``cath-domain-list.txt`` parses here just as happily, but
    ITS field 3 is the single architecture digit: every pair would still get a
    same/different label and the whole stratification would be meaningless with
    no error anywhere. Hence the shape check on the code.
    """
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    n_rows = n_bad_code = 0
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 3:
                continue
            n_rows += 1
            if not _CATH_CODE_RE.match(p[2]):
                n_bad_code += 1
                continue
            dom = p[0]
            if len(dom) < 6:
                continue
            key = (dom[:4].lower(), dom[4:-2])
            if key in wanted:
                out[key].add(p[2])
    if n_rows and n_bad_code > n_rows // 2:
        raise SystemExit(
            f"{path}: {n_bad_code} of {n_rows} rows carry no C.A.T.H code in field 3. "
            "--cath expects cath-b-newest-all.gz (CATH daily release), not "
            "cath-domain-list.txt, whose field 3 is the architecture digit."
        )
    return dict(out)


def corr_block(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 10:
        return {"n": int(len(x))}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    d = y - x  # predicted minus experimental
    return {
        "n": int(len(x)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "pearson_r2": float(pr**2),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
        "mean_signed_diff": float(d.mean()),
        "median_signed_diff": float(np.median(d)),
        "sd_diff": float(d.std(ddof=1)),
        "loa_lower": float(d.mean() - 1.96 * d.std(ddof=1)),
        "loa_upper": float(d.mean() + 1.96 * d.std(ddof=1)),
        "mean_exp": float(x.mean()),
        "mean_pred": float(y.mean()),
        "frac_pred_higher": float((d > 0).mean()),
    }


def cluster_bootstrap_r(
    df: pl.DataFrame, xcol: str, ycol: str, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """
    Percentile CI for Pearson r, resampling PROTEINS (not pairs), because pairs
    that share a protein are not independent observations.
    """
    rng = np.random.default_rng(seed)
    prots = np.array(sorted(set(df["query"].to_list()) | set(df["target"].to_list())))
    idx = {p: i for i, p in enumerate(prots)}
    qi = np.array([idx[p] for p in df["query"].to_list()])
    ti = np.array([idx[p] for p in df["target"].to_list()])
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    out = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(prots), len(prots))
        mult = np.bincount(draw, minlength=len(prots))
        w = mult[qi] * mult[ti]          # pair appears mult_q * mult_t times
        if w.sum() < 50:
            continue
        sel = np.repeat(np.arange(len(w)), w)
        if len(np.unique(x[sel])) < 3:
            continue
        out.append(stats.pearsonr(x[sel], y[sel])[0])
    if len(out) < 50:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="B6 / R2.2: does alntmscore track EXPERIMENTAL structural similarity?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--parquet", type=Path, required=True,
                    help="tmscore_exp parquet written by data_preparation/pdb_tmscore.py.")
    ap.add_argument("--results_dir", type=Path, required=True,
                    help="stats.json is written here.")
    ap.add_argument("--cath", type=Path, default=None,
                    help="cath-b-newest-all.gz from the CATH daily release (rows: "
                         "'<domain> <version> <C.A.T.H> <boundaries>'). NOT "
                         "cath-domain-list.txt. Optional: enables the same- vs "
                         "different-superfamily stratification. No downloader ships "
                         "with this repo.")
    ap.add_argument("--attrition", type=Path, default=None,
                    help="attrition.json from the same run; copied into stats.json so "
                         "the funnel and the source stamps travel with the numbers.")
    args = ap.parse_args()

    df = pl.read_parquet(args.parquet)
    total = df.height
    scored = df.filter(pl.col("tmscore_exp").is_not_null())
    both = scored.filter(pl.col("alntmscore").is_not_null())

    S: dict = {
        "n_rows_in_parquet": total,
        "n_scored_by_usalign": scored.height,
        "n_scored_with_alntmscore": both.height,
    }

    # ---- CATH -----------------------------------------------------------
    have_cath = bool(args.cath and args.cath.exists())
    if have_cath:
        wanted = set(
            zip(scored["q_pdb"].to_list(), scored["q_chain"].to_list(), strict=True)
        ) | set(zip(scored["t_pdb"].to_list(), scored["t_chain"].to_list(), strict=True))
        cath = load_cath(args.cath, wanted)
        S["cath_chains_assigned"] = len(cath)
        S["cath_chains_wanted"] = len(wanted)

        lab = []
        for qp, qc, tp, tc in zip(
            both["q_pdb"], both["q_chain"], both["t_pdb"], both["t_chain"], strict=True
        ):
            a, b = cath.get((qp, qc)), cath.get((tp, tc))
            lab.append(None if not a or not b else ("same" if a & b else "different"))
        both = both.with_columns(pl.Series("cath_rel", lab, dtype=pl.Utf8))
        S["cath_pairs_labelled"] = int(both["cath_rel"].is_not_null().sum())
        S["cath_same"] = int((both["cath_rel"] == "same").sum())
        S["cath_different"] = int((both["cath_rel"] == "different").sum())

    # ---- headline agreement --------------------------------------------
    S["agreement"] = {}
    for col, desc in EXP_VARIANTS:
        sub = both.filter(pl.col(col).is_not_null())
        blk = corr_block(sub[col].to_numpy(), sub["alntmscore"].to_numpy())
        blk["description"] = desc
        if col == "tmscore_exp":
            lo, hi = cluster_bootstrap_r(sub, col, "alntmscore")
            blk["pearson_r_cluster_boot_ci95"] = [lo, hi]
        S["agreement"][col] = blk

    # ---- how much of experimental TM is sequence identity ---------------
    S["sequence_identity_explains"] = {}
    fid = both.filter(pl.col("fident").is_not_null())
    S["n_with_fident"] = fid.height
    for col, desc in EXP_VARIANTS + [("alntmscore", "AFDB/Foldseek predicted-structure TM")]:
        sub = fid.filter(pl.col(col).is_not_null())
        x = sub["fident"].to_numpy()
        y = sub[col].to_numpy()
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
        S["sequence_identity_explains"][col] = {
            "n": int(len(x)),
            "pearson_r": float(pr[0]),
            "r2_linear": float(pr[0] ** 2),
            "spearman_rho": float(sr[0]),
            "r2_spearman": float(sr[0] ** 2),
            "description": desc,
        }

    # ---- by sequence-identity bin ---------------------------------------
    # MEASURED 2026-09-18: fident in sets/test.parquet has a hard floor of 0.30
    # and is NULL for 39.7% of all test pairs -- the MMseqs all-vs-all only
    # reports pairs it could align. The requested "0-20%" bin is therefore
    # empty by construction, not by sampling. The NULL stratum is reported
    # separately as "unalignable by MMseqs (fident NULL)", which is the real
    # low-identity stratum.
    bins = [(0.0, 0.20, "0-20% (empty: fident floor is 0.30)"),
            (0.20, 0.50, "20-50% (effectively 30-50%)"),
            (0.50, 1.0001, "50-100%")]
    S["by_fident_bin"] = {}
    for lo, hi, name in bins:
        sub = fid.filter((pl.col("fident") >= lo) & (pl.col("fident") < hi))
        blk = corr_block(sub["tmscore_exp"].to_numpy(), sub["alntmscore"].to_numpy())
        if blk["n"] >= 10:
            blk["r2_fident_vs_exp"] = float(
                stats.pearsonr(sub["fident"].to_numpy(), sub["tmscore_exp"].to_numpy())[0] ** 2
            )
        S["by_fident_bin"][name] = blk

    # The pairs MMseqs could not align at all -- the genuine low-identity
    # stratum, invisible to any fident-binned analysis.
    nullf = both.filter(pl.col("fident").is_null())
    S["by_fident_bin"]["fident NULL (MMseqs found no alignment)"] = corr_block(
        nullf["tmscore_exp"].to_numpy(), nullf["alntmscore"].to_numpy()
    )

    # ---- by CATH relation -----------------------------------------------
    if have_cath:
        S["by_cath"] = {}
        for rel in ("same", "different"):
            sub = both.filter(pl.col("cath_rel") == rel)
            S["by_cath"][rel] = corr_block(
                sub["tmscore_exp"].to_numpy(), sub["alntmscore"].to_numpy()
            )

    # ---- coverage / normalisation confound ------------------------------
    # CAVEAT, do not read this as the direct test. The confound is that Foldseek
    # normalises alntmscore by ITS OWN alignment span, and the pair table
    # (query, target, fident, hfsp, alntmscore) carries no Foldseek span columns,
    # so that quantity is not available here. len_ali/len1/len2 are US-ALIGN's
    # numbers on the experimental structures. Pairs that US-align aligns nearly
    # end to end are the ones where a span-normalised and a chain-normalised
    # score can least diverge, so this is a proxy, not the measurement.
    cov = both.with_columns(
        (pl.col("len_ali") / pl.min_horizontal("len1", "len2")).alias("ali_frac")
    )
    S["alignment_coverage"] = {
        "median_len_ali_over_shorter_chain": float(cov["ali_frac"].median()),
        "frac_pairs_ali_ge_0.9_of_shorter": float((cov["ali_frac"] >= 0.9).mean()),
        "caveat": (
            "US-align's alignment span on the experimental structures, NOT "
            "Foldseek's span behind alntmscore -- the pair table has no Foldseek "
            "span columns. Proxy for the normalisation confound, not a direct test."
        ),
    }
    near_full = cov.filter(pl.col("ali_frac") >= 0.9)
    S["near_full_length_subset"] = corr_block(
        near_full["tmscore_exp"].to_numpy(), near_full["alntmscore"].to_numpy()
    )

    # ---- structure provenance -------------------------------------------
    meth = (
        pl.concat([scored.select(pl.col("q_method").alias("m")),
                   scored.select(pl.col("t_method").alias("m"))])
        ["m"].value_counts(sort=True)
    )
    S["method_counts_over_pair_slots"] = {r[0]: r[1] for r in meth.iter_rows()}
    res = pl.concat([scored.select(pl.col("q_resolution").alias("r")),
                     scored.select(pl.col("t_resolution").alias("r"))])["r"].drop_nulls()
    S["resolution"] = {
        "n_with_resolution": int(res.len()),
        "median": float(res.median()),
        "max": float(res.max()),
    }
    S["whole_chain_fallback_pair_slots"] = int(
        scored["q_whole_chain_fallback"].sum() + scored["t_whole_chain_fallback"].sum()
    )

    if args.attrition and args.attrition.exists():
        S["attrition"] = json.loads(args.attrition.read_text())

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "stats.json").write_text(json.dumps(S, indent=2))
    print(json.dumps(S, indent=2))


if __name__ == "__main__":
    main()
