#!/usr/bin/env python3
"""Build the GO Molecular Function cohort, and the EC+GO union for the identity control.

**Why a cohort, not every annotated protein.** Swiss-Prot's experimentally
annotated proteins are dominated by a few large superfamilies (Rossmann folds,
P-loop NTPases, kinases). Scored over all of them, a nearest-neighbour transfer
number would mostly describe how well a model separates those families from each
other. The EC arm fixed this with a CATH-superfamily cap (EC v2,
``ec_strat_v2_freeze.json``); this script applies the same recipe to GO-MF so the
two arms are comparable:

* **filter** -- >=1 CAFA core-6 (EXP, IDA, IPI, IMP, IGI, IEP) MF annotation whose
  term is in ``go-basic.obo`` (non-obsolete, MF namespace, not the MF root
  GO:0003674); <=1022 residues by the FASTA index; not in
  ``freeze/embedding_excluded_proteins.json``; >=1 Gene3D cross-reference.
  The length cut is applied from the ``.fai`` and not only via the exclusion
  freeze, because proteins >2000 aa were removed physically and the freeze never
  lists them.
* **strata** -- CATH superfamily x length quartile, at most ``--cap`` proteins per
  cell, sampled with ``--seed``.

**Documented choices** (the EC v2 recipe does not pin them, so they are fixed here):

* A protein with several Gene3D superfamilies is assigned to the
  **lexicographically smallest** id (plain string order, so ``3.40.50.10140`` sorts
  before ``3.40.50.720``). This is arbitrary but deterministic, independent of
  cross-reference order in the dump, and is consistent with the EC v2 cohort:
  under that assignment no EC v2 (superfamily, EC class) group exceeds 4 x 25.
* Length quartile edges are ``numpy.quantile(..., [0.25, 0.5, 0.75])`` over the
  filtered population (before capping); a protein's quartile is the number of
  edges strictly below its length, so a length equal to an edge falls in the
  lower quartile.
* Over-cap cells are sampled with one ``numpy.random.default_rng(seed)``,
  visiting cells in sorted (superfamily, quartile) order and drawing only for
  cells above the cap; member ids are sorted before drawing.
* A core-6 annotation to a term that is obsolete in (or absent from) the OBO is
  dropped, and a protein left with no term is not in the population. With the
  2026-03-25 ``go-basic.obo`` this is what reproduces the previously measured
  40,460 core-6 MF proteins (36,929 at <=1022 aa) from the 40,764 in the dump.
* Evidence is matched on the code column of ``export_go_annotations``. Its
  summary checks that this equals a CONTAINS match on the raw ``GoEvidenceType``
  field (0 disagreeing MF rows on the 2024 dump).

**Outputs** (to ``--out-dir``): ``go_mf_cohort_freeze.json`` (``ids``, ``n``,
``seed``, ``set_name``, ``filter``, ``strata`` as in the EC v2 freeze, plus the
funnel and superfamily statistics), ``go_mf_labels.tsv`` (``protein_id``,
``GO_term``) and ``go_mf_cath_labels.tsv`` (``Entry``, ``Gene3D``, the layout of
``cath_labels_v2.tsv``).

With ``--ec-freeze`` it also writes ``union_ids.txt`` (EC v2 ids | GO ids) and
``union.fasta``; with ``--run-mmseqs`` it runs the local all-vs-all MMseqs2
search the identity-control variants need and writes ``union_allvsall.m8``
(headerless; columns query, target, fident, evalue, alnlen, qcov, tcov; self
hits removed) plus a ``.summary.json`` with hit counts and runtime.

Usage::

    PYTHONPATH=src python scripts/build_go_cohort.py \\
        --annotations RESULTS/go_annotations_sprot2024.tsv \\
        --gene3d RESULTS/gene3d_sprot2024.tsv \\
        --obo data/reference/go/go-basic.obo \\
        --fai data/raw/sprot_2024/sprot.fasta.fai \\
        --out-dir RESULTS \\
        --ec-freeze .../ec_strat_v2_freeze.json --fasta data/raw/sprot_2024/sprot.fasta \\
        --run-mmseqs
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_preparation.export_go_annotations import read_fai_lengths  # noqa: E402
from data_preparation.go_semantic_similarity import CAFA_CORE6, GOTerm, parse_obo  # noqa: E402

# The same two constants the scorer uses, imported rather than restated: if the cohort and
# evaluation.go_similarity_matrix disagreed on what "MF" or "the root" is, the cohort would
# contain proteins the scorer has no scoreable term for.
from evaluation.go_similarity_matrix import MF_NAMESPACE, MF_ROOT  # noqa: E402
from shared.protein_cohort import content_hash, load_excluded_proteins  # noqa: E402

SET_NAME = "go_mf_core6_sfcap"

#: MMseqs2 settings of the identity control (design item 3). Fixed here rather
#: than exposed as flags: the three neighbour-eligibility variants are defined by
#: this search, so changing a setting changes the experiment.
MMSEQS_ARGS: tuple[str, ...] = (
    "-s", "7.5",
    "-e", "10",
    "--max-seqs", "10000",
    "--format-output", "query,target,fident,evalue,alnlen,qcov,tcov",
)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def load_core6_mf(annotations: Path | str, codes: Iterable[str] = CAFA_CORE6) -> dict[str, set[str]]:
    """protein -> set of GO ids with aspect F and an evidence code in ``codes``."""
    df = pl.read_csv(annotations, separator="\t", schema_overrides={"evidence": pl.Utf8})
    df = df.filter((pl.col("aspect") == "F") & pl.col("evidence").is_in(sorted(codes)))
    out: dict[str, set[str]] = {}
    for pid, term in zip(df["protein_id"].to_list(), df["GO_term"].to_list(), strict=True):
        out.setdefault(pid, set()).add(term)
    return out


def scoreable_mf_terms(
    raw: Mapping[str, set[str]], go_terms: Mapping[str, GOTerm]
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Keep the terms a GO score can use; count what was dropped and why.

    ``parse_obo`` already omits obsolete terms, so "not in ``go_terms``" covers both
    obsolete and unknown ids. The root is dropped because every MF term propagates
    to it: as a scored term it would make every pair share a label. A term filed as
    aspect F but living in another namespace would be a dump/OBO mismatch; it is
    counted separately so it cannot hide inside the obsolete count.
    """
    kept: dict[str, set[str]] = {}
    counts = Counter()
    for pid, terms in raw.items():
        good = set()
        for term in terms:
            if term == MF_ROOT:
                counts["annotations_root_dropped"] += 1
            elif term not in go_terms:
                counts["annotations_obsolete_or_unknown_dropped"] += 1
            elif go_terms[term].namespace != MF_NAMESPACE:
                counts["annotations_wrong_namespace_dropped"] += 1
            else:
                good.add(term)
        if good:
            kept[pid] = good
    for key in (
        "annotations_root_dropped",
        "annotations_obsolete_or_unknown_dropped",
        "annotations_wrong_namespace_dropped",
    ):
        counts.setdefault(key, 0)
    return kept, dict(counts)


def load_gene3d(path: Path | str) -> dict[str, list[str]]:
    """Entry -> sorted Gene3D ids, from the ``Entry``/``Gene3D`` TSV."""
    df = pl.read_csv(path, separator="\t", schema_overrides={"Gene3D": pl.Utf8})
    return {
        pid: sorted({x for x in value.split(";") if x})
        for pid, value in zip(df["Entry"].to_list(), df["Gene3D"].to_list(), strict=True)
        if value
    }


def obo_data_version(obo: Path | str) -> str | None:
    with open(obo) as handle:
        for line in handle:
            if line.startswith("data-version:"):
                return line.split(":", 1)[1].strip()
            if line.startswith("[Term]"):
                return None
    return None


# --------------------------------------------------------------------------- #
# Stratification
# --------------------------------------------------------------------------- #


def assign_superfamily(gene3d_ids: Iterable[str]) -> str:
    """The lexicographically smallest id: deterministic and order-independent."""
    ids = [x for x in gene3d_ids if x]
    if not ids:
        raise ValueError("protein has no Gene3D id to assign")
    return min(ids)


def length_quartile_edges(lengths: Iterable[int]) -> np.ndarray:
    return np.quantile(np.asarray(list(lengths), dtype=float), [0.25, 0.5, 0.75])


def length_quartile(length: int, edges: np.ndarray) -> int:
    """0..3 = number of edges strictly below ``length`` (ties go to the lower bin)."""
    return int(np.searchsorted(edges, length, side="left"))


def stratified_cap(
    cells: Mapping[tuple, Iterable[str]], cap: int, seed: int
) -> tuple[list[str], int]:
    """Keep every cell of <= ``cap`` members whole; sample ``cap`` from the others.

    Returns ``(sorted selected ids, number of cells that were capped)``. Cells are
    visited in sorted key order and members sorted before drawing, so the result
    depends only on the cell contents and the seed, never on dict or file order.
    """
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    n_capped = 0
    for key in sorted(cells):
        members = sorted(cells[key])
        if len(members) > cap:
            idx = rng.choice(len(members), size=cap, replace=False)
            members = [members[i] for i in sorted(idx)]
            n_capped += 1
        selected.extend(members)
    return sorted(selected), n_capped


def superfamily_stats(assigned: Mapping[str, str], gene3d: Mapping[str, list[str]]) -> dict:
    """Superfamily count and top-5 share, by assignment and by membership.

    ``top5_superfamily_share`` is the fraction of proteins whose ASSIGNED
    superfamily is one of the five most frequent assignments -- the quantity the
    cap acts on. The membership variant counts a protein under every superfamily
    it carries (share = proteins carrying any of the five most carried ones),
    which is the stricter view of how much one family dominates.
    """
    def top(counter: Counter) -> list[tuple[str, int]]:
        # After capping many superfamilies tie at 4 x cap; break ties by id so the
        # reported names do not depend on dict insertion order.
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

    n = len(assigned)
    by_assignment = Counter(assigned.values())
    top5 = top(by_assignment)
    membership = Counter(sf for pid in assigned for sf in gene3d[pid])
    top5_members = [sf for sf, _ in top(membership)]
    carrying = sum(1 for pid in assigned if set(gene3d[pid]) & set(top5_members))
    return {
        "n": n,
        "n_superfamilies": len(by_assignment),
        "top5_superfamily_share": round(sum(c for _, c in top5) / n, 6) if n else None,
        "top5_superfamilies": dict(top5),
        "n_superfamilies_any_membership": len(membership),
        "top5_superfamily_share_any_membership": round(carrying / n, 6) if n else None,
        "top5_superfamilies_any_membership": {sf: membership[sf] for sf in top5_members},
    }


def build_cohort(
    raw_core6: Mapping[str, set[str]],
    go_terms: Mapping[str, GOTerm],
    gene3d: Mapping[str, list[str]],
    fai_lengths: Mapping[str, int],
    excluded: frozenset[str],
    *,
    max_length: int = 1022,
    cap: int = 25,
    seed: int = 42,
) -> tuple[dict, dict[str, set[str]]]:
    """Apply the filters, stratify, cap. Returns ``(manifest, labels of cohort proteins)``."""
    def short(p: str) -> bool:
        return p in fai_lengths and fai_lengths[p] <= max_length

    scoreable, dropped = scoreable_mf_terms(raw_core6, go_terms)
    # Two views of the same filters. The first follows the raw dump; the second is
    # the previously measured anchor (40,460 / 36,929 on the 2024 dump), which
    # counted only proteins left with a term the ontology still has.
    funnel: dict[str, int] = {
        "core6_mf_proteins": len(raw_core6),
        f"core6_mf_proteins_le_{max_length}_aa": sum(1 for p in raw_core6 if short(p)),
        "scoreable_core6_mf_proteins": len(scoreable),
        f"scoreable_le_{max_length}_aa": sum(1 for p in scoreable if short(p)),
    }
    not_excluded = {p for p in scoreable if short(p)} - excluded
    funnel["scoreable_short_not_in_embedding_exclusion"] = len(not_excluded)
    population = sorted(p for p in not_excluded if gene3d.get(p))
    funnel["gene3d_annotated"] = len(population)

    edges = length_quartile_edges(fai_lengths[p] for p in population)
    assigned = {p: assign_superfamily(gene3d[p]) for p in population}
    cells: dict[tuple[str, int], list[str]] = {}
    for p in population:
        cells.setdefault((assigned[p], length_quartile(fai_lengths[p], edges)), []).append(p)
    ids, n_capped = stratified_cap(cells, cap, seed)
    funnel["after_cap"] = len(ids)

    manifest = {
        "ids": ids,
        "n": len(ids),
        "seed": seed,
        "set_name": SET_NAME,
        "filter": (
            f"CAFA core-6 ({', '.join(sorted(CAFA_CORE6))}) Molecular Function, evidence "
            f"code match (== CONTAINS on GoEvidenceType), >=1 term in go-basic.obo "
            f"(non-obsolete, MF, not root {MF_ROOT}), <={max_length} aa (fai), not in "
            f"embedding exclusion freeze, Gene3D-annotated"
        ),
        "strata": f"CATH superfamily x length quartile, cap {cap}/cell",
        "cap": cap,
        "superfamily_assignment": "lexicographically smallest Gene3D id (string order)",
        "length_quartile_edges": [float(x) for x in edges],
        "length_quartile_rule": "quartile = number of edges strictly below length",
        "n_cells": len(cells),
        "n_cells_capped": n_capped,
        "funnel": funnel,
        "annotation_drops": dropped,
        "cohort_stats": superfamily_stats({p: assigned[p] for p in ids}, gene3d),
        "population_stats": superfamily_stats(assigned, gene3d),
        "content_sha256": content_hash(ids),
    }
    return manifest, {p: scoreable[p] for p in ids}


# --------------------------------------------------------------------------- #
# Union FASTA + MMseqs2
# --------------------------------------------------------------------------- #


def iter_fasta_records(
    fasta: Path | str, fai: Path | str, ids: Iterable[str]
) -> Iterator[tuple[str, str]]:
    """``(id, sequence)`` for each id, read by seeking via the ``.fai`` offsets.

    Raises ``KeyError`` for an id missing from the index: a union FASTA that
    silently lacks a protein would drop it from every identity-control variant.
    """
    index: dict[str, tuple[int, int, int, int]] = {}
    with open(fai) as handle:
        for line in handle:
            name, length, offset, line_bases, line_bytes = line.rstrip("\n").split("\t")[:5]
            index[name] = (int(length), int(offset), int(line_bases), int(line_bytes))
    with open(fasta, "rb") as handle:
        for pid in ids:
            if pid not in index:
                raise KeyError(f"{pid} is not in {fai}")
            length, offset, line_bases, line_bytes = index[pid]
            n_lines = -(-length // line_bases) if length else 0
            handle.seek(offset)
            chunk = handle.read(length + n_lines * (line_bytes - line_bases))
            seq = chunk.replace(b"\n", b"").replace(b"\r", b"").decode("ascii")[:length]
            if len(seq) != length:
                raise ValueError(f"{pid}: read {len(seq)} residues, index says {length}")
            yield pid, seq


def write_union(
    ec_ids: Iterable[str], go_ids: Iterable[str], fasta: Path | str, fai: Path | str, out_dir: Path
) -> dict:
    union = sorted(set(ec_ids) | set(go_ids))
    (out_dir / "union_ids.txt").write_text("".join(f"{p}\n" for p in union))
    with open(out_dir / "union.fasta", "w") as handle:
        for pid, seq in iter_fasta_records(fasta, fai, union):
            handle.write(f">{pid}\n{seq}\n")
    ec, go = set(ec_ids), set(go_ids)
    return {"n_ec": len(ec), "n_go": len(go), "n_overlap": len(ec & go), "n_union": len(union)}


def drop_self_hits(raw_m8: Path | str, out_m8: Path | str) -> dict:
    """Copy an MMseqs2 tab file without query == target rows; count what was seen."""
    n_raw = n_self = 0
    queries: set[str] = set()
    with open(raw_m8) as src, open(out_m8, "w") as dst:
        for line in src:
            n_raw += 1
            query, target = line.split("\t", 2)[:2]
            if query == target:
                n_self += 1
                continue
            queries.add(query)
            dst.write(line)
    return {
        "hits_raw": n_raw,
        "self_hits_dropped": n_self,
        "hits": n_raw - n_self,
        "queries_with_non_self_hit": len(queries),
    }


def run_mmseqs_allvsall(
    fasta: Path, out_m8: Path, mmseqs: str, threads: int | None, tmp_root: Path | None
) -> dict:
    tmp_dir = Path(tempfile.mkdtemp(prefix="mmseqs_union_", dir=tmp_root))
    raw_m8 = out_m8.with_name(out_m8.name + ".raw")
    cmd = [mmseqs, "easy-search", str(fasta), str(fasta), str(raw_m8), str(tmp_dir), *MMSEQS_ARGS]
    if threads:
        cmd += ["--threads", str(threads)]
    version = subprocess.run([mmseqs, "version"], capture_output=True, text=True).stdout.strip()
    started = time.monotonic()
    try:
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    seconds = time.monotonic() - started
    counts = drop_self_hits(raw_m8, out_m8)
    raw_m8.unlink()
    return {
        "command": " ".join(cmd),
        "mmseqs_version": version,
        "columns": MMSEQS_ARGS[-1].split(","),
        "seconds": round(seconds, 1),
        **counts,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the CATH-capped GO-MF cohort (and the EC+GO union + MMseqs2 all-vs-all).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotations", type=Path, required=True,
                        help="export_go_annotations TSV (protein_id, GO_term, aspect, evidence).")
    parser.add_argument("--gene3d", type=Path, required=True,
                        help="export_go_annotations --gene3d_output TSV (Entry, Gene3D).")
    parser.add_argument("--obo", type=Path, required=True, help="go-basic.obo")
    parser.add_argument("--fai", type=Path, required=True, help="sprot.fasta.fai (lengths).")
    parser.add_argument("--exclusion-freeze", type=Path, default=None,
                        help="Embedding exclusion freeze (default: freeze/embedding_excluded_proteins.json).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--cap", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ec-freeze", type=Path, default=None,
                        help="EC v2 freeze JSON; writes union_ids.txt + union.fasta.")
    parser.add_argument("--fasta", type=Path, default=None, help="sprot.fasta (needed with --ec-freeze).")
    parser.add_argument("--run-mmseqs", action="store_true",
                        help="Run the all-vs-all MMseqs2 search on union.fasta.")
    parser.add_argument("--mmseqs", default="/opt/homebrew/bin/mmseqs")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Parent dir for MMseqs2 temp files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_mmseqs and args.ec_freeze is None:
        print("--run-mmseqs needs --ec-freeze (the search runs on the union)", file=sys.stderr)
        return 2
    if args.ec_freeze is not None and args.fasta is None:
        print("--ec-freeze needs --fasta", file=sys.stderr)
        return 2
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    go_terms = parse_obo(args.obo)
    raw = load_core6_mf(args.annotations)
    gene3d = load_gene3d(args.gene3d)
    fai_lengths = read_fai_lengths(args.fai)
    excluded = load_excluded_proteins(args.exclusion_freeze)

    manifest, labels = build_cohort(
        raw, go_terms, gene3d, fai_lengths, excluded,
        max_length=args.max_length, cap=args.cap, seed=args.seed,
    )
    manifest["sources"] = {
        "annotations": str(args.annotations),
        "gene3d": str(args.gene3d),
        "obo": str(args.obo),
        "obo_data_version": obo_data_version(args.obo),
        "fai": str(args.fai),
        "exclusion_freeze_n_excluded": len(excluded),
    }

    (out_dir / "go_mf_cohort_freeze.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with open(out_dir / "go_mf_labels.tsv", "w") as handle:
        handle.write("protein_id\tGO_term\n")
        for pid in manifest["ids"]:
            for term in sorted(labels[pid]):
                handle.write(f"{pid}\t{term}\n")
    with open(out_dir / "go_mf_cath_labels.tsv", "w") as handle:
        handle.write("Entry\tGene3D\n")
        for pid in manifest["ids"]:
            handle.write(f"{pid}\t{';'.join(gene3d[pid])}\n")

    report = {k: v for k, v in manifest.items() if k != "ids"}
    report["n_label_rows"] = sum(len(labels[p]) for p in manifest["ids"])

    if args.ec_freeze is not None:
        ec_ids = json.loads(args.ec_freeze.read_text())["ids"]
        report["union"] = write_union(ec_ids, manifest["ids"], args.fasta, args.fai, out_dir)
        if args.run_mmseqs:
            summary = run_mmseqs_allvsall(
                out_dir / "union.fasta", out_dir / "union_allvsall.m8",
                args.mmseqs, args.threads, args.tmp_dir,
            )
            summary["union"] = report["union"]
            (out_dir / "union_allvsall.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            report["mmseqs"] = summary

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
