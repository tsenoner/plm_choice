#!/usr/bin/env python3
"""How unevenly do proteins share the test pairs, and what does a per-protein cap change?

What it measures. On one test split of pairs (query, target, fident=PIDE, hfsp, alntmscore=
TM-score), for the two populations a probe is scored on -- pairs with PIDE and HFSP (the
two are always jointly present) and pairs with a TM-score:

1. per-protein pair counts (a protein counts once per pair it is in): n proteins, median /
   mean / p90 / p99 / max, Gini, the share of pair-memberships and of pairs held by the top
   1% / 10% / 43% of proteins, and the smallest share of proteins that holds 70% of the
   pair-memberships;
2. a greedy random per-protein cap (pairs visited in a seeded random order, a pair kept only
   while both its proteins are below k), with the pairs retained and how the PIDE, HFSP and
   TM-score label distributions move; the retained (query, target) lists are written so the
   saved per-pair probe predictions can be re-scored on them;
3. the TM-score population stratified by PIDE: no PIDE (no MMseqs2 alignment that passed
   the reliability filters), [0.3, 0.5), [0.5, 0.7), [0.7, 0.9), [0.9, 1.0].

Why it exists. Reviewer R1.3 (task D14) asked how many pairs each
protein appears in and whether the results hold when pairs are capped per protein; R1.6
asked for the low-homology end of the TM-score target, binned by sequence identity. The
manuscript's "43.3% of queries ... contribute 68.7% of all pairs" is a statement about the
raw MMseqs2 search (queries at the 300-target cap, every directional hit counted once by its
query), not about the filtered, deduplicated test split the probes are scored on -- this
measures the latter.

Cohort. Probes are scored after a load-time filter that keeps pairs whose two proteins are
both <=1022 residues (shared.protein_cohort; the 526,871-protein cohort is exactly the
<=1022 entries of the deposit's sprot.fasta.fai). Pass --fai to apply it; every number is
then reported both before and after the filter.

Samples. If the input is a uniform random sample of a split's pairs (the 10% grid's
sprot_pre2024_e1_sub10 is a seed-42 10% sample of the full post-E1 split), pass
--sample-fraction. Label distributions and stratum shares are then unbiased for the full
split and counts are also given scaled by 1/fraction, but per-protein counts are thinned
(a protein's count is ~Binomial(full count, fraction)) and low-count proteins drop out, so
the concentration statistics are NOT those of the full split. To show which way they move,
they are recomputed on further random thinnings of the input. A per-protein cap of k on a
10% sample corresponds roughly to a cap of 10k on the full split. Run on the full split for
the paper numbers:

    python scripts/paper/pair_redundancy.py \
        --pairs <sprot_pre2024_e1/sets/test.parquet> --fai <sprot.fasta.fai> \
        --out-counts <dir>/pair_counts --out-caps <dir>/pair_caps --tag test_full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

POPULATIONS = {"pide_hfsp": ("fident", "hfsp"), "tmscore": ("alntmscore",)}
LABELS = ("fident", "hfsp", "alntmscore")
TOP_SHARES = (0.01, 0.10, 0.43)
MEMBERSHIP_TARGETS = (0.687, 0.70)
# Lower edges; the last bin is closed at 1.0. The reliability filter leaves no PIDE < 0.3.
PIDE_EDGES = (0.3, 0.5, 0.7, 0.9)
PIDE_BIN_NAMES = ("[0.3,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def population(df: pl.DataFrame, name: str) -> pl.DataFrame:
    cond = pl.lit(True)
    for col in POPULATIONS.get(name, ()):  # "all": every pair, whichever targets it carries
        cond = cond & pl.col(col).is_not_null()
    return df.filter(cond)


def degrees(df: pl.DataFrame) -> pl.DataFrame:
    """Pairs per protein, highest first, ties broken by accession so ranks are reproducible."""
    acc = pl.concat([df.select(pl.col("query").alias("acc")), df.select(pl.col("target").alias("acc"))])
    return acc.group_by("acc").len("n_pairs").sort(["n_pairs", "acc"], descending=[True, False])


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype("float64"))
    n = x.size
    return float((2.0 * np.sum(np.arange(1, n + 1) * x)) / (n * x.sum()) - (n + 1) / n)


def concentration(df: pl.DataFrame) -> dict:
    deg = degrees(df)
    d = deg["n_pairs"].to_numpy()
    n_prot, n_pairs = d.size, df.height
    members = int(d.sum())
    assert members == 2 * n_pairs, (members, n_pairs)
    cum = np.cumsum(d) / members
    out = {
        "n_pairs": n_pairs,
        "n_proteins": n_prot,
        "median": float(np.median(d)),
        "mean": float(d.mean()),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "max": int(d.max()),
        "gini": gini(d),
        "share_proteins_with_1_pair": float((d == 1).mean()),
    }
    for share in TOP_SHARES:
        k = math.ceil(share * n_prot)
        top = deg["acc"].head(k).to_list()
        touched = df.select((pl.col("query").is_in(top) | pl.col("target").is_in(top)).sum()).item()
        out[f"top{share:.0%}_n_proteins"] = k
        out[f"top{share:.0%}_share_memberships"] = float(cum[k - 1])
        out[f"top{share:.0%}_share_pairs_touched"] = touched / n_pairs
    for target in MEMBERSHIP_TARGETS:
        k = int(np.searchsorted(cum, target) + 1)
        out[f"share_proteins_holding_{target:.1%}_memberships"] = k / n_prot
    return out


def label_summary(df: pl.DataFrame, col: str) -> dict:
    x = df[col].drop_nulls().to_numpy().astype("float64")
    if x.size == 0:
        return {"n": 0}
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "q25": float(q25), "median": float(q50), "q75": float(q75)}


def greedy_cap(df: pl.DataFrame, k: int, seed: int) -> np.ndarray:
    """Boolean mask of the pairs kept: random visiting order, keep while both ends are below k."""
    codes = pl.concat([df["query"], df["target"]]).rank("dense").to_numpy().astype("int64") - 1
    q, t = codes[: df.height], codes[df.height :]
    order = np.random.default_rng(seed).permutation(df.height)
    count = [0] * (int(codes.max()) + 1)
    keep = np.zeros(df.height, dtype=bool)
    qo, to = q[order].tolist(), t[order].tolist()
    for pos, a, b in zip(order.tolist(), qo, to, strict=True):
        if count[a] < k and count[b] < k:
            count[a] += 1
            count[b] += 1
            keep[pos] = True
    return keep


def pide_bin(col: str = "fident") -> pl.Expr:
    x = pl.col(col)
    expr = pl.when(x.is_null()).then(pl.lit("no_pide"))
    for lo, hi, name in zip(PIDE_EDGES, (*PIDE_EDGES[1:], None), PIDE_BIN_NAMES, strict=True):
        cond = (x >= lo) & (x < hi) if hi is not None else (x >= lo) & (x <= 1.0)
        expr = expr.when(cond).then(pl.lit(name))
    return expr.otherwise(pl.lit("outside_bins")).alias("pide_bin")


def tm_strata(df: pl.DataFrame, frac: float | None) -> list[dict]:
    tm = population(df, "tmscore").with_columns(pide_bin())
    rows = []
    for name in ("no_pide", *PIDE_BIN_NAMES, "outside_bins"):
        sub = tm.filter(pl.col("pide_bin") == name)
        if name == "outside_bins" and sub.height == 0:
            continue
        s = label_summary(sub, "alntmscore")
        rows.append({
            "pide_bin": name,
            "n_pairs": sub.height,
            "n_pairs_scaled_to_full": round(sub.height / frac) if frac else None,
            "share_of_tm_pairs": sub.height / tm.height,
            "n_proteins": degrees(sub).height if sub.height else 0,
            "tm_mean": s.get("mean"), "tm_sd": s.get("sd"), "tm_q25": s.get("q25"),
            "tm_median": s.get("median"), "tm_q75": s.get("q75"),
            "share_tm_ge_0.5": float((sub["alntmscore"] >= 0.5).mean()) if sub.height else None,
        })
    return rows


def hfsp(fident: pl.Expr, ungapped: pl.Expr) -> pl.Expr:
    """Mahlich et al. 2018 Eq. 4, exactly as merge_datasets._compute_hfsp_scores computes it."""
    return (pl.when(ungapped <= 11).then(fident * 100 - 100)
            .when(ungapped <= 450)
            .then(fident * 100 - 770 * ungapped.cast(pl.Float64).pow(-0.33 * (1 + (-ungapped / 1000).exp())))
            .otherwise(fident * 100 - 28.4))


def realign(pairs: pl.DataFrame, fasta: Path, mmseqs: str, workdir: Path) -> pl.DataFrame:
    """Align exactly the given pairs with MMseqs2 (no prefilter, so no 300-target cap and no
    E-value cut) and return PIDE, coverage and HFSP per pair, as the pair table defines them."""
    import subprocess

    from Bio import SeqIO

    workdir.mkdir(parents=True, exist_ok=True)
    index = SeqIO.index(str(fasta), "fasta")
    ids = sorted(set(pairs["query"].to_list()) | set(pairs["target"].to_list()))
    with (workdir / "seqs.fasta").open("w") as fh:
        for acc in ids:
            fh.write(f">{acc}\n{index[acc].seq}\n")

    def run(*cmd: str) -> None:
        subprocess.run([mmseqs, *cmd, "-v", "1"], check=True)

    run("createdb", str(workdir / "seqs.fasta"), str(workdir / "seqDB"))
    lookup = pl.read_csv(workdir / "seqDB.lookup", separator="\t", has_header=False,
                         new_columns=["key", "acc", "file"])
    key = dict(zip(lookup["acc"].to_list(), lookup["key"].to_list(), strict=True))
    # A prefilter result that lists only these pairs: align computes exactly them.
    edges = sorted((key[q], key[t]) for q, t in zip(pairs["query"].to_list(), pairs["target"].to_list(), strict=True))
    (workdir / "pref.tsv").write_text("".join(f"{a}\t{b}\t0\t0\n" for a, b in edges))
    run("tsv2db", str(workdir / "pref.tsv"), str(workdir / "prefDB"), "--output-dbtype", "7")
    # The pair table's search used --alignment-mode 3 with -a; keep those, drop every cut-off.
    run("align", str(workdir / "seqDB"), str(workdir / "seqDB"), str(workdir / "prefDB"), str(workdir / "alnDB"),
        "-a", "--alignment-mode", "3", "-e", "1e10", "-c", "0", "--min-seq-id", "0")
    cols = ["query", "target", "fident", "alnlen", "nident", "mismatch", "qcov", "tcov", "evalue", "bits"]
    run("convertalis", str(workdir / "seqDB"), str(workdir / "seqDB"), str(workdir / "alnDB"),
        str(workdir / "aln.tsv"), "--format-output", ",".join(cols))
    aln = pl.read_csv(workdir / "aln.tsv", separator="\t", has_header=False, new_columns=cols,
                      schema_overrides={"evalue": pl.Float64})
    aln = aln.unique(["query", "target"], keep="first").with_columns(
        pl.min_horizontal("qcov", "tcov").alias("min_cov"),
        hfsp(pl.col("fident"), pl.col("nident") + pl.col("mismatch")).alias("hfsp_realigned"),
    ).rename({"fident": "fident_realigned"})
    return pairs.join(aln.select("query", "target", "fident_realigned", "min_cov", "hfsp_realigned", "evalue"),
                      on=["query", "target"], how="left")


def realign_no_pide(base: pl.DataFrame, n: int, seed: int, fasta: Path, mmseqs: str, workdir: Path) -> tuple[dict, pl.DataFrame]:
    """Is the no-PIDE stratum low identity, or pairs the capped search never reported?

    A random n of the TM-score pairs without PIDE are aligned directly. Those that would pass
    the pair table's own MMseqs2 filters (min coverage >= 0.8, PIDE >= 0.3, HFSP >= 0) are
    homologous at the table's own standard and lack PIDE only because the search did not
    return them. A calibration sample of pairs WITH PIDE checks that direct alignment
    reproduces the recorded value.
    """
    tm = population(base, "tmscore")
    no_pide = tm.filter(pl.col("fident").is_null()).sample(n=min(n, tm.height), seed=seed, shuffle=True)
    calib = population(base, "pide_hfsp").sample(n=min(n // 2, tm.height), seed=seed, shuffle=True)
    res = realign(no_pide.select("query", "target", "alntmscore"), fasta, mmseqs, workdir / "no_pide")
    cal = realign(calib.select("query", "target", "fident"), fasta, mmseqs, workdir / "calibration")
    d = (cal["fident_realigned"] - cal["fident"]).abs()
    passes = ((pl.col("min_cov") >= 0.8) & (pl.col("fident_realigned") >= 0.3) & (pl.col("hfsp_realigned") >= 0))
    res = res.with_columns(passes.fill_null(False).alias("passes_table_filters"),
                           pide_bin("fident_realigned").alias("realigned_pide_bin"))
    # If the 300-target cap is why a homologous pair has no PIDE, its proteins should sit in
    # large families, i.e. carry many PIDE pairs themselves.
    deg = degrees(population(base, "pide_hfsp"))
    for side in ("query", "target"):
        res = res.join(deg.rename({"acc": side, "n_pairs": f"{side}_n_pide_pairs"}), on=side, how="left")
    res = res.with_columns(pl.min_horizontal(pl.col("query_n_pide_pairs").fill_null(0),
                                             pl.col("target_n_pide_pairs").fill_null(0)).alias("min_n_pide_pairs"))
    out = {
        "n_sampled": res.height, "n_aligned": int(res["fident_realigned"].is_not_null().sum()),
        "calibration_n": cal.height,
        "calibration_share_identical_at_3_decimals": float((d <= 0.0015).mean()),
        "calibration_share_abs_diff_le_0.02": float((d <= 0.02).mean()),
        "share_passing_table_filters": float(res["passes_table_filters"].mean()),
        "share_passing_filters_and_evalue_le_1e-3": float(res.select(
            (pl.col("passes_table_filters") & (pl.col("evalue") <= 1e-3)).mean()).item()),
        "share_fident_lt_0.3": float((res["fident_realigned"] < 0.3).mean()),
        "share_min_cov_lt_0.8": float((res["min_cov"] < 0.8).mean()),
        "realigned_fident_quartiles": [float(x) for x in np.nanpercentile(res["fident_realigned"].to_numpy(), [25, 50, 75])],
        "tm_mean_passing": float(res.filter(pl.col("passes_table_filters"))["alntmscore"].mean()),
        "tm_mean_failing": float(res.filter(~pl.col("passes_table_filters"))["alntmscore"].mean()),
        "tm_mean_fident_lt_0.3": float(res.filter(pl.col("fident_realigned") < 0.3)["alntmscore"].mean()),
        "median_min_n_pide_pairs_passing": float(res.filter(pl.col("passes_table_filters"))["min_n_pide_pairs"].median()),
        "median_min_n_pide_pairs_failing": float(res.filter(~pl.col("passes_table_filters"))["min_n_pide_pairs"].median()),
        "median_min_n_pide_pairs_all_pide_pairs": float(population(base, "pide_hfsp").join(
            deg.rename({"acc": "query", "n_pairs": "a"}), on="query").join(
            deg.rename({"acc": "target", "n_pairs": "b"}), on="target").select(
            pl.min_horizontal("a", "b")).to_series().median()),
        "n_by_realigned_bin_among_passing":dict(res.filter(pl.col("passes_table_filters"))
                                                 .group_by("realigned_pide_bin").len().iter_rows()),
    }
    return out, res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, type=Path, help="test split parquet")
    ap.add_argument("--fai", type=Path, help="sprot.fasta.fai; applies the <=--max-len cohort filter")
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--sample-fraction", type=float, help="the input is a uniform pair sample at this fraction")
    ap.add_argument("--caps", type=int, nargs="+", default=[10, 50, 100])
    ap.add_argument("--thin", type=float, nargs="*", default=[0.5, 0.25],
                    help="extra random thinnings of the input, to show how concentration moves with sampling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True, help="prefix of the pair-list files, e.g. test_full")
    ap.add_argument("--out-counts", required=True, type=Path)
    ap.add_argument("--out-caps", required=True, type=Path)
    ap.add_argument("--n-proteins-full", type=int,
                    help="with --sample-fraction: the full split's protein count (all pairs, before the "
                         "cohort filter), to anchor an estimate of the full split's concentration")
    ap.add_argument("--realign", type=int, default=0,
                    help="directly align this many random no-PIDE TM-score pairs (needs --fasta)")
    ap.add_argument("--fasta", type=Path, help="sequences for --realign")
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--workdir", type=Path, help="scratch for --realign")
    args = ap.parse_args(argv)
    frac = args.sample_fraction

    raw = pl.read_parquet(args.pairs, columns=["query", "target", *LABELS])
    unordered = raw.select(pl.min_horizontal("query", "target").alias("a"),
                           pl.max_horizontal("query", "target").alias("b")).unique().height
    report: dict = {
        "input": {"pairs": str(args.pairs), "sha256": sha256(args.pairs), "rows": raw.height,
                  "unique_unordered_pairs": unordered,
                  "self_pairs": int((raw["query"] == raw["target"]).sum()),
                  "sample_fraction": frac, "seed": args.seed},
    }

    frames = {"pre_cohort": raw}
    if args.fai:
        fai = pl.read_csv(args.fai, separator="\t", has_header=False, columns=[0, 1],
                          new_columns=["acc", "length"])
        cohort = fai.filter(pl.col("length") <= args.max_len)["acc"].to_list()
        known = fai["acc"].to_list()
        in_pairs = pl.concat([raw["query"], raw["target"]]).unique()
        post = raw.filter(pl.col("query").is_in(cohort) & pl.col("target").is_in(cohort))
        frames["post_cohort"] = post
        report["cohort"] = {"fai": str(args.fai), "max_len": args.max_len, "cohort_size": len(cohort),
                            "proteins_in_pairs_missing_from_fai": int((~in_pairs.is_in(known)).sum()),
                            "rows_removed": raw.height - post.height,
                            "share_rows_removed": (raw.height - post.height) / raw.height}

    # --- 1. per-protein pair counts -------------------------------------------------------
    conc_rows, report["concentration"] = [], {}
    for fname, frame in frames.items():
        for pop in (*POPULATIONS, "all"):
            sub = population(frame, pop)
            c = concentration(sub)
            report["concentration"][f"{fname}/{pop}"] = c
            conc_rows.append({"filter": fname, "population": pop, "input_thinning": 1.0,
                              "fraction_of_full_split": frac or 1.0, **c})
            for r in args.thin:
                rng = np.random.default_rng(args.seed)
                thinned = sub.filter(pl.Series(rng.random(sub.height) < r))
                conc_rows.append({"filter": fname, "population": pop, "input_thinning": r,
                                  "fraction_of_full_split": (frac or 1.0) * r, **concentration(thinned)})
        if frac:
            # High counts thin almost proportionally, so p90/p99/max scale; the low end does not.
            report["concentration"][f"{fname}/scaled_upper_tail_estimate"] = {
                pop: {k: report["concentration"][f"{fname}/{pop}"][k] / frac for k in ("p90", "p99", "max")}
                for pop in POPULATIONS}

    if frac and args.n_proteins_full:
        # Proteins a sample misses are the low-count ones, and they hold few memberships, so
        # ranking the sampled proteins and dividing by the full split's protein count gives a
        # usable estimate of the full split's shares (biased slightly towards concentration,
        # because ranking on a noisy count favours proteins that drew high).
        d = degrees(raw)["n_pairs"].to_numpy()
        cum, n_full = np.cumsum(d) / d.sum(), args.n_proteins_full
        report["concentration"]["pre_cohort/all_pairs_anchored_estimate"] = {
            "n_proteins_full_assumed": n_full, "n_proteins_in_sample": int(d.size),
            **{f"top{s:.0%}_share_memberships": float(cum[min(math.ceil(s * n_full), d.size) - 1]) for s in TOP_SHARES},
            **{f"share_proteins_holding_{t:.1%}_memberships": float((np.searchsorted(cum, t) + 1) / n_full)
               for t in MEMBERSHIP_TARGETS},
        }

    base = frames.get("post_cohort", raw)
    per_protein =degrees(population(base, "pide_hfsp")).rename({"n_pairs": "n_pairs_pide_hfsp"}).join(
        degrees(population(base, "tmscore")).rename({"n_pairs": "n_pairs_tmscore"}), on="acc", how="full",
        coalesce=True).join(degrees(base).rename({"n_pairs": "n_pairs_any"}), on="acc", how="full",
                            coalesce=True).fill_null(0).sort("n_pairs_any", descending=True)

    # --- 2. per-protein caps ----------------------------------------------------------------
    args.out_caps.mkdir(parents=True, exist_ok=True)
    cap_rows = []
    for pop in POPULATIONS:
        sub = population(base, pop)
        for k in [None, *args.caps]:
            kept = sub if k is None else sub.filter(pl.Series(greedy_cap(sub, k, args.seed)))
            if k is not None:
                path = args.out_caps / f"{args.tag}_{pop}_cap{k}_seed{args.seed}.parquet"
                kept.select("query", "target").write_parquet(path)
            deg = degrees(kept)["n_pairs"].to_numpy()
            row = {"population": pop, "cap": k if k is not None else "none", "n_pairs": kept.height,
                   "share_retained": kept.height / sub.height, "n_proteins": deg.size,
                   "max_pairs_per_protein": int(deg.max()), "gini": gini(deg),
                   "share_no_pide": float(kept["fident"].is_null().mean())}
            for col in (POPULATIONS[pop] if pop == "pide_hfsp" else ("alntmscore", "fident")):
                for stat, v in label_summary(kept, col).items():
                    row[f"{col}_{stat}"] = v
            cap_rows.append(row)

    # --- 3. TM-score by PIDE bin ------------------------------------------------------------
    strata_rows = []
    for fname, frame in frames.items():
        for r in tm_strata(frame, frac):
            strata_rows.append({"filter": fname, **r})
    report["no_pide_share"] = {
        fname: {"of_all_rows": float(frame["fident"].is_null().mean()),
                "of_tm_pairs": float(population(frame, "tmscore")["fident"].is_null().mean()),
                "rows_with_neither_target": int((frame["fident"].is_null() & frame["alntmscore"].is_null()).sum())}
        for fname, frame in frames.items()}

    args.out_counts.mkdir(parents=True, exist_ok=True)
    if args.realign:
        if not (args.fasta and args.workdir):
            raise SystemExit("--realign needs --fasta and --workdir")
        report["no_pide_realigned"], realigned = realign_no_pide(
            base, args.realign, args.seed, args.fasta, args.mmseqs, args.workdir)
        realigned.write_csv(args.out_counts / "no_pide_realigned_sample.csv")
    pl.DataFrame(conc_rows).write_csv(args.out_counts / "degree_concentration.csv")
    pl.DataFrame(cap_rows).write_csv(args.out_counts / "cap_label_shift.csv")
    pl.DataFrame(strata_rows).write_csv(args.out_counts / "tm_by_pide_bin.csv")
    per_protein.write_parquet(args.out_counts / "per_protein_pair_counts.parquet")
    hist = pl.concat([
        degrees(population(base, pop)).group_by("n_pairs").len("n_proteins").with_columns(pl.lit(pop).alias("population"))
        for pop in POPULATIONS]).sort(["population", "n_pairs"])
    hist.write_csv(args.out_counts / "degree_histogram.csv")
    report["caps"] = cap_rows
    report["tm_by_pide_bin"] = strata_rows
    (args.out_counts / "pair_redundancy_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("input", "cohort", "no_pide_share") if k in report}, indent=2))
    print(f"wrote {args.out_counts} and {args.out_caps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
