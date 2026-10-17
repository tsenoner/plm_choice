#!/usr/bin/env python3
"""Who is in the pair set? Taxonomy, length, function and search truncation, by split and by pair (R1.3).

What it measures. For the 526,871-protein cohort, each probe split (train / validation / test
proteins, from the cluster split), the test pairs the probes are scored on (the PIDE/HFSP and
the TM-score populations, after the <=1022 cohort filter) and the 5,000,000 uniformly random
pairs behind Figures 2 and 3, the share falling in each

1. superkingdom (first element of the UniProt lineage) and each of the cohort's ten largest
   organisms (UniProt organism, i.e. strain level where UniProt names a strain);
2. length quartile, with edges taken over the cohort (a length equal to an edge falls in the
   lower quartile, as in build_go_cohort.py);
3. function: >=1 fully specified EC number with ECO:0000269 evidence, the filter of the EC
   cohort before its Gene3D and stratum caps; >=1 Molecular Function term under the six CAFA
   experimental codes, non-obsolete and not the root, the filter of the GO-MF cohort
   (build_go_cohort.scoreable_mf_terms, imported rather than restated);
4. truncation: the protein's own query reached the search's target cap (MMseqs2 300,
   Foldseek 1000), read from the raw hit tables, so the remote hits of that protein's search
   were cut.

Protein columns count proteins. Pair columns count pair memberships (each pair counts both of
its proteins), so they read on the same scale as the protein columns and show which proteins
the pairs are made of. composition.csv also gives, per pair set, the share of pairs whose two
proteins share a superkingdom or an organism (not in the table: no protein column has one).

Why it exists. PNAS reviewer R1.3, quoted in Yana Bromberg's comment Y2 (task D14, plan C11),
asked for a breakdown of the pair set by species, function, length and truncation; the
per-protein pair counts of the same comment are pair_redundancy.py.

Inputs. The organism, lineage and annotations come from the UniProt JSON dump that the GO arm
was exported from (export_go_annotations.iter_sprot_entries); lengths and the cohort come from
the deposit's sprot.fasta.fai, so a sequence changed in a later release cannot move a protein
between quartiles. The split lists are split_dataset.py's {train,val,test}_sequences.txt. The
EC cohort freeze is only a check: every one of its proteins must pass the EC filter here.

    PYTHONPATH=src python scripts/paper/organism_map.py \\
        --sprot-json data/raw/sprot_2024/sprot.json --fai data/raw/sprot_2024/sprot.fasta.fai \\
        --obo data/reference/go/go-basic.obo --splits <data_split_mmseqs/splits> \\
        --test-pairs <sprot_pre2024_e1/sets/test.parquet> \\
        --random-pairs <random_pairs/random_pairs_distances.parquet> \\
        --mmseqs-hits <mmseqs/sprot_all_vs_all.parquet> \\
        --foldseek-hits <foldseek/afdb_swissprot_v4_all_vs_all.parquet> \\
        --ec-freeze <c1_strat_2026-09-15/ec_strat_v2_freeze.json> --out <dir> \\
        --extra-pairs cap10_pide_hfsp=<pair_caps/test_full_pide_hfsp_cap10_seed42.parquet> ...

Outputs: organism_map.parquet (one row per cohort protein), composition.csv (one row per
column x category), composition_summary.json (inputs with sha256, counts, checks) and
composition_table.md (the Supplement's Pandoc grid table). --extra-pairs adds pair sets to the CSV,
e.g. the per-protein capped lists of pair_redundancy.py, to show how far a cap moves the composition.
--map reuses an organism_map.parquet and skips the 7.4 GB pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _cohort import COHORT_SIZE  # noqa: E402
from build_go_cohort import scoreable_mf_terms  # noqa: E402

from data_preparation.export_go_annotations import (  # noqa: E402
    go_annotations,
    iter_sprot_entries,
    read_fai_lengths,
)
from data_preparation.go_semantic_similarity import CAFA_CORE6, parse_obo  # noqa: E402
from shared.protein_cohort import load_excluded_proteins  # noqa: E402

MAX_LENGTH = 1022
SPLITS = ("train", "val", "test")
SUPERKINGDOMS = ("Bacteria", "Eukaryota", "Archaea", "Viruses")
TOP_N = 10
EC_FULL = re.compile(r"\d+\.\d+\.\d+\.n?\d+")  # four fields, no "-"; the EC cohort counts 2.5.1.n9 too
EC_EXPERIMENTAL = "ECO:0000269"
CAPS = {"mmseqs": 300, "foldseek": 1000}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _names(desc: dict):
    """Every name block of a protein description, including those of its domains and chains."""
    yield desc.get("recommendedName") or {}
    yield from desc.get("alternativeNames", ())
    yield from desc.get("submissionNames", ())
    for part in (*desc.get("includes", ()), *desc.get("contains", ())):
        yield from _names(part)


def experimental_ec(entry: dict) -> bool:
    for name in _names(entry.get("proteinDescription", {})):
        for ec in name.get("ecNumbers", ()):
            if EC_FULL.fullmatch(ec.get("value", "")) and any(
                    e.get("evidenceCode") == EC_EXPERIMENTAL for e in ec.get("evidences", ())):
                return True
    return False


def superkingdom(entry: dict) -> str:
    lineage = entry.get("organism", {}).get("lineage", ())
    return lineage[0] if lineage else "unclassified"


def map_entries(json_path: Path, lengths: dict[str, int], go_terms: dict) -> tuple[pl.DataFrame, dict]:
    """One row per cohort protein, from one pass over the dump.

    An accession that is secondary in the dump (merged since the deposit) is matched through
    secondaryAccessions, but an entry that has it as its primary accession wins; one the dump
    does not hold at all keeps null annotations and is counted.
    """
    primary, secondary, n_entries = {}, {}, 0
    for entry in iter_sprot_entries(json_path):
        n_entries += 1
        acc = entry["primaryAccession"]
        others = [a for a in entry.get("secondaryAccessions", ()) if a in lengths]
        if acc not in lengths and not others:
            continue
        org = entry.get("organism", {})
        go_rows, _, _ = go_annotations(entry)
        mf = {term for _, term, aspect, code in go_rows if aspect == "F" and code in CAFA_CORE6}
        scoreable, _ = scoreable_mf_terms({acc: mf}, go_terms)
        record = {"taxon_id": org.get("taxonId"), "organism": org.get("scientificName"),
                  "superkingdom": superkingdom(entry), "json_length": entry.get("sequence", {}).get("length"),
                  "exp_ec": experimental_ec(entry), "exp_go_mf": bool(scoreable)}
        if acc in lengths:
            primary[acc] = record
        for a in others:
            secondary.setdefault(a, record)
    rows = secondary | primary
    missing = sorted(set(lengths) - set(rows))
    df = pl.DataFrame([{"accession": a, "length": lengths[a], **rows.get(a, {})} for a in sorted(lengths)],
                      schema_overrides={"taxon_id": pl.Int64, "json_length": pl.Int64})
    stats = {"dump_entries": n_entries, "cohort_matched": len(rows),
             "matched_via_secondary": len(set(secondary) - set(primary)),
             "cohort_missing_from_dump": len(missing), "missing_examples": missing[:20]}
    return df, stats


def capped_queries(path: Path, cap: int, deposit: set[str]) -> tuple[set[str], dict]:
    """Accessions whose own query returned the cap's number of hits, with the table's totals."""
    counts = pl.scan_parquet(path).group_by("query").len("n_hits").collect(engine="streaming")
    n = counts["n_hits"]
    if int(n.max()) > cap:
        raise SystemExit(f"{path}: a query has {int(n.max())} hits, above the stated cap {cap}")
    at_cap = set(counts.filter(pl.col("n_hits") == cap)["query"].to_list())
    # AFDB's Swiss-Prot set is a little newer than the deposit, so a few queries are proteins the
    # cohort cannot contain; they are counted. Mostly-foreign ids would mean a different id scheme.
    stray = at_cap - deposit
    if len(stray) > 0.01 * len(at_cap):
        raise SystemExit(f"{path}: {len(stray)} capped query ids are not deposit accessions, e.g. {sorted(stray)[:3]}")
    return at_cap - stray, {"queries": counts.height, "queries_at_cap": len(at_cap),
                            "capped_outside_deposit": sorted(stray),
                            "share_queries_at_cap": len(at_cap) / counts.height, "hits": int(n.sum()),
                            "share_hits_from_capped": int(n.filter(n == cap).sum()) / int(n.sum())}


def read_ids(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def memberships(pairs: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([pairs.select(pl.col("query").alias("accession")),
                      pairs.select(pl.col("target").alias("accession"))])


def counts_by(frame: pl.DataFrame, col: str) -> dict:
    return dict(frame.group_by(col).len().iter_rows())


def categories(m: pl.DataFrame, top: list[tuple[int, str]], edges: np.ndarray) -> dict[tuple[str, str], int]:
    """Counts per (block, category) for a frame of proteins (or of pair memberships)."""
    out: dict[tuple[str, str], int] = {}
    sk = counts_by(m, "superkingdom")
    for k in SUPERKINGDOMS:
        out[("superkingdom", k)] = sk.pop(k, 0)
    out[("superkingdom", "other")] = sum(sk.values())  # incl. a protein the dump does not hold
    tax = counts_by(m, "taxon_id")
    for t, _ in top:
        out[("organism", str(t))] = tax.get(t, 0)
    out[("organism", "top10")] = sum(tax.get(t, 0) for t, _ in top)
    q = np.searchsorted(edges, m["length"].to_numpy(), side="left")
    for i, c in enumerate(np.bincount(q, minlength=4)):
        out[("length", f"Q{i + 1}")] = int(c)
    for col in ("exp_ec", "exp_go_mf", "mmseqs_capped", "foldseek_capped"):
        out[("annotation", col)] = int(m[col].fill_null(False).sum())
    return out


def largest(frame: pl.DataFrame, n: int = TOP_N) -> pl.DataFrame:
    return frame.group_by("taxon_id").len().sort(["len", "taxon_id"], descending=[True, False]).head(n)


def pair_level(pairs: pl.DataFrame, attrs: pl.DataFrame) -> dict[tuple[str, str], int]:
    a = attrs.select("accession", "superkingdom", "taxon_id")
    j = (pairs.select("query", "target")
         .join(a.rename(lambda c: f"q_{c}"), left_on="query", right_on="q_accession")
         .join(a.rename(lambda c: f"t_{c}"), left_on="target", right_on="t_accession"))
    assert j.height == pairs.height, (j.height, pairs.height)
    return {("pair", "same_superkingdom"): int((j["q_superkingdom"] == j["t_superkingdom"]).sum()),
            ("pair", "same_organism"): int((j["q_taxon_id"] == j["t_taxon_id"]).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sprot-json", type=Path, required=True)
    ap.add_argument("--fai", type=Path, required=True, help="the deposit's sprot.fasta.fai")
    ap.add_argument("--obo", type=Path, required=True, help="go-basic.obo of the GO cohort")
    ap.add_argument("--splits", type=Path, required=True, help="dir with {train,val,test}_sequences.txt")
    ap.add_argument("--test-pairs", type=Path, required=True)
    ap.add_argument("--random-pairs", type=Path, required=True)
    ap.add_argument("--mmseqs-hits", type=Path, required=True)
    ap.add_argument("--foldseek-hits", type=Path, required=True)
    ap.add_argument("--ec-freeze", type=Path, required=True, help="EC cohort freeze, checked against")
    ap.add_argument("--extra-pairs", nargs="*", default=[], metavar="NAME=PATH",
                    help="further pair sets, e.g. pair_redundancy.py's capped lists (composition.csv only)")
    ap.add_argument("--map", type=Path, help="reuse an organism_map.parquet instead of reading the dump")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    extra = dict(spec.split("=", 1) for spec in args.extra_pairs)

    deposit = read_fai_lengths(args.fai)
    lengths = {k: v for k, v in deposit.items() if v <= MAX_LENGTH}
    if len(lengths) != COHORT_SIZE:
        raise SystemExit(f"{len(lengths):,} proteins of <= {MAX_LENGTH} residues, expected {COHORT_SIZE:,}")
    if overlap := set(lengths) & load_excluded_proteins():
        raise SystemExit(f"{len(overlap)} cohort proteins are on the exclusion freeze, e.g. {sorted(overlap)[:3]}")
    summary: dict = {"inputs": {k: {"path": str(p), "sha256": sha256(p)} for k, p in [
        ("fai", args.fai), ("test_pairs", args.test_pairs), ("random_pairs", args.random_pairs),
        ("mmseqs_hits", args.mmseqs_hits), ("foldseek_hits", args.foldseek_hits), ("ec_freeze", args.ec_freeze),
        ("obo", args.obo), *[(f"{s}_sequences", args.splits / f"{s}_sequences.txt") for s in SPLITS],
        *[(name, Path(path)) for name, path in extra.items()]]}}

    if args.map:
        attrs = pl.read_parquet(args.map)
        prev = json.loads((args.map.parent / "composition_summary.json").read_text())
        summary["map"], summary["inputs"]["sprot_json"] = prev["map"], prev["inputs"]["sprot_json"]
    else:
        attrs, summary["map"] = map_entries(args.sprot_json, lengths, parse_obo(args.obo))
        summary["inputs"]["sprot_json"] = {"path": str(args.sprot_json), "sha256": sha256(args.sprot_json)}
    print(f"map: {summary['map']}", file=sys.stderr)

    for tool, cap in CAPS.items():
        capped, summary[f"{tool}_hits"] = capped_queries(getattr(args, f"{tool}_hits"), cap, set(deposit))
        attrs = attrs.with_columns(pl.col("accession").is_in(sorted(capped)).alias(f"{tool}_capped"))
        print(f"{tool}: {summary[f'{tool}_hits']}", file=sys.stderr)

    split_of = {}
    for s in SPLITS:
        for a in read_ids(args.splits / f"{s}_sequences.txt"):
            if split_of.setdefault(a, s) != s:
                raise SystemExit(f"{a} is in two splits")
    attrs = attrs.with_columns(pl.col("accession").replace_strict(split_of, default=None).alias("split"))
    if attrs["split"].null_count():
        raise SystemExit(f"{attrs['split'].null_count()} cohort proteins are in no split")
    attrs.write_parquet(args.out / "organism_map.parquet")

    # Checks: the EC filter must admit the whole EC cohort; JSON and deposit lengths should agree.
    ec_ids = set(json.loads(args.ec_freeze.read_text())["ids"])
    ec_ok = set(attrs.filter(pl.col("exp_ec"))["accession"].to_list())
    summary["checks"] = {
        "ec_cohort_size": len(ec_ids), "ec_cohort_failing_filter": sorted(ec_ids - ec_ok),
        "json_length_differs_from_deposit": int((attrs["json_length"] != attrs["length"]).sum()),
    }

    in_cohort = pl.col("query").is_in(sorted(lengths)) & pl.col("target").is_in(sorted(lengths))
    test = pl.read_parquet(args.test_pairs, columns=["query", "target", "fident", "alntmscore"])
    n_raw = test.height
    test = test.filter(in_cohort)
    rnd = pl.read_parquet(args.random_pairs, columns=["query", "target"])
    if rnd.filter(~in_cohort).height:
        raise SystemExit("random pairs reach outside the cohort")
    pair_sets = {"test_pide_hfsp": test.filter(pl.col("fident").is_not_null()),
                 "test_tmscore": test.filter(pl.col("alntmscore").is_not_null()),
                 "random": rnd}
    for name, path in extra.items():
        pair_sets[name] = pl.read_parquet(path, columns=["query", "target"])
        if pair_sets[name].filter(~in_cohort).height:
            raise SystemExit(f"{path}: pairs reach outside the cohort")
    summary["pairs"] = {"test_rows": n_raw, "test_rows_in_cohort": test.height,
                        **{k: v.height for k, v in pair_sets.items()}}

    edges = np.quantile(attrs["length"].to_numpy(), [0.25, 0.5, 0.75])
    names = dict(attrs.select("taxon_id", "organism").unique("taxon_id").iter_rows())
    top = [(int(t), names[t]) for t in largest(attrs)["taxon_id"].to_list()]
    summary["length_quartile_edges"] = edges.tolist()
    summary["top_organisms"] = [{"taxon_id": t, "organism": o} for t, o in top]

    columns: dict[str, tuple[pl.DataFrame, pl.DataFrame | None]] = {"cohort": (attrs, None)}
    columns |= {s: (attrs.filter(pl.col("split") == s), None) for s in SPLITS}
    columns |= {k: (memberships(v).join(attrs, on="accession"), v) for k, v in pair_sets.items()}
    rows, top_by_column = [], {}
    for col, (frame, pairs) in columns.items():
        counts = categories(frame, top, edges)
        total = frame.height
        if pairs is not None:
            assert total == 2 * pairs.height, (col, total, pairs.height)
            for key, c in pair_level(pairs, attrs).items():
                rows.append({"column": col, "block": key[0], "category": key[1], "count": c,
                             "total": pairs.height, "share": c / pairs.height})
        for (block, cat), c in counts.items():
            rows.append({"column": col, "block": block, "category": cat, "count": c, "total": total,
                         "share": c / total})
        top_by_column[col] = largest(frame)["taxon_id"].to_list()
    summary["top10_matches_cohort"] = {c: set(t) == {x for x, _ in top} for c, t in top_by_column.items()}
    summary["organisms_per_column"] = {c: frame["taxon_id"].n_unique() for c, (frame, _) in columns.items()}
    # Which proteins are truncated: the cap rate within each superkingdom of the cohort.
    summary["cap_rate_by_superkingdom"] = {
        r["superkingdom"]: {"proteins": r["n"], "mmseqs_capped": r["mm"], "foldseek_capped": r["fs"]}
        for r in attrs.group_by("superkingdom").agg(pl.len().alias("n"), pl.col("mmseqs_capped").mean().alias("mm"),
                                                    pl.col("foldseek_capped").mean().alias("fs")).sort("superkingdom").iter_rows(named=True)}
    summary["top10_by_column"] = top_by_column
    comp = pl.DataFrame(rows)
    comp.write_csv(args.out / "composition.csv")
    (args.out / "composition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "composition_table.md").write_text(table_md(comp, columns, top, edges, min(lengths.values())))
    print(json.dumps({k: summary[k] for k in ("pairs", "checks", "top10_matches_cohort")}, indent=1))
    print(f"wrote {args.out}/organism_map.parquet, composition.csv, composition_summary.json, composition_table.md")
    return 0


HEAD = {"cohort": "Swiss-Prot ≤1022", "train": "Train", "val": "Validation", "test": "Test",
        "test_pide_hfsp": "Test, PIDE/HFSP", "test_tmscore": "Test, TM-score", "random": "Random"}
ANNOTATION = {"exp_ec": "EC number (ECO:0000269)", "exp_go_mf": "GO-MF (CAFA codes)",
              "mmseqs_capped": "MMseqs2, 300 targets", "foldseek_capped": "Foldseek, 1000 targets"}


def short_name(organism: str) -> str:
    """Binomial only for the table (organism_map.parquet keeps UniProt's name, strain included)."""
    return organism.split(" (")[0]


def pct(share: float) -> str:
    """Percent to one decimal; an exact zero prints as 0 so it cannot be read as rounding."""
    return "0" if share == 0 else "<0.1" if share < 0.0005 else f"{100 * share:.1f}"


def table_md(comp: pl.DataFrame, columns: dict, top: list[tuple[int, str]], edges: np.ndarray,
             min_length: int) -> str:
    """Pandoc grid table: rows are categories; columns the protein sets, then the pair sets."""
    share = {(r["column"], r["block"], r["category"]): r["share"] for r in comp.iter_rows(named=True)}
    cols = [c for c in columns if c in HEAD]  # --extra-pairs sets go to composition.csv only
    # Quartile i holds lengths above i edges, so the integer bounds are floor(edge) and floor(edge) + 1.
    lo = [min_length, *(int(np.floor(x)) + 1 for x in edges)]
    hi = [*(int(np.floor(x)) for x in edges), MAX_LENGTH]
    n = [f"{(frame if pairs is None else pairs).height:,}" for frame, pairs in (columns[c] for c in cols)]
    body: list[tuple[str, list[str]]] = [("Proteins or pairs", n)]

    def block(title: str, items: list[tuple[str, str, str]]) -> None:
        body.append((f"**{title}**", [""] * len(cols)))
        for label, blk, cat in items:
            body.append((label, [pct(share[(c, blk, cat)]) if (c, blk, cat) in share else "--" for c in cols]))

    block("Superkingdom", [(k, "superkingdom", k) for k in SUPERKINGDOMS])
    block("Organism", [(f"*{short_name(o)}*", "organism", str(t)) for t, o in top]
          + [("Ten largest together", "organism", "top10")])
    block("Length (residues)", [(f"{a}--{b}", "length", f"Q{i + 1}") for i, (a, b) in enumerate(zip(lo, hi, strict=True))])
    block("Experimental function", [(ANNOTATION[k], "annotation", k) for k in ("exp_ec", "exp_go_mf")])
    block("Search reached its cap", [(ANNOTATION[k], "annotation", k) for k in ("mmseqs_capped", "foldseek_capped")])
    # The pair-level shares (same superkingdom / organism) have no protein-column value; composition.csv has them.

    w = [max(len(lbl) for lbl, _ in body)] + [max(len(HEAD[c]), *(len(v[i]) for _, v in body))
                                               for i, c in enumerate(cols)]
    k = 1 + len(SPLITS)  # protein columns
    span = [sum(w[1:1 + k]) + 3 * (k - 1), sum(w[1 + k:]) + 3 * (len(cols) - k - 1)]

    def row(cells: list[str], widths: list[int]) -> str:
        # A cell wider than its column breaks the grid, so this refuses one.
        assert all(len(c) <= n for c, n in zip(cells, widths, strict=True)), cells
        return "| " + " | ".join(c.ljust(n) for c, n in zip(cells, widths, strict=True)) + " |"

    def rule(widths: list[int], ch: str = "-", blank_first: bool = False) -> str:
        return "+" + "+".join((" " if blank_first and i == 0 else ch) * (n + 2) for i, n in enumerate(widths)) + "+"

    out = [rule([w[0]] + span), row(["", "Proteins", "Pair memberships"], [w[0]] + span),
           rule(w, blank_first=True), row([""] + [HEAD[c] for c in cols], w), rule(w, "=")]
    for label, cells in body:
        out += [row([label] + cells, w), rule(w)]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
