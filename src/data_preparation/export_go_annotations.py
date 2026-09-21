#!/usr/bin/env python3
"""Export Swiss-Prot GO annotations, with their evidence codes, from the UniProt JSON dump.

**Why this exists.** The GO arm answers reviewer R2.1 only if its labels were
assigned by experiment (see ``go_semantic_similarity.CAFA_CORE6``). That filter
needs the evidence code of every annotation, and the GO columns of a UniProt TSV
export do not carry one: they list terms, and the evidence is lost. The JSON dump
keeps it per cross-reference, so it is the source of truth here.

**What it writes.** A 4-column TSV, one row per (protein, GO term, evidence code)::

    protein_id  GO_term     aspect  evidence
    A0A009IHW8  GO:0003953  F       IDA

which ``go_semantic_similarity.load_annotations_tsv`` reads with evidence
filtering enabled. Every evidence code is exported, including IEA: filtering is
the loader's job, so the choice of code set stays visible at the point of use
instead of being baked into an intermediate file. Optionally it also writes the
Gene3D (CATH superfamily) cross-references, in the ``Entry``/``Gene3D`` layout of
the UniProt export that ``evaluation.label_adapters.load_cath_labels`` reads, so
the GO cohort can be stratified without a second pass over 7.4 GB.

**JSON layout.** ``GoTerm`` holds ``"<aspect>:<term name>"`` and
``GoEvidenceType`` holds ``"<code>:<assigned by>"`` (e.g. ``"IDA:UniProtKB"``,
``"IEA:UniProtKB-EC"``). The evidence code is the part before the colon. The
summary also counts proteins whose evidence field merely *contains* a core-6
code, and how many rows the two definitions disagree on, so a cohort defined by
the CONTAINS match can be checked against this column rather than assumed equal.

Usage::

    PYTHONPATH=src python -m data_preparation.export_go_annotations \\
        --sprot_json data/raw/sprot_2024/sprot.json \\
        --output go_annotations_sprot2024.tsv \\
        --gene3d_output gene3d_sprot2024.tsv \\
        --fai data/raw/sprot_2024/sprot.fasta.fai
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from data_preparation.go_semantic_similarity import CAFA_CORE6

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#: Header of the annotation TSV. ``load_annotations_tsv`` skips it (its GO column
#: does not start with ``GO:``); pandas/polars read it as column names.
GO_TSV_HEADER: tuple[str, ...] = ("protein_id", "GO_term", "aspect", "evidence")

#: Header of the Gene3D TSV: the UniProt-export column names, so the file is a
#: drop-in for ``label_adapters.load_cath_labels`` and the EC v2 ``cath_labels_v2.tsv``.
GENE3D_TSV_HEADER: tuple[str, ...] = ("Entry", "Gene3D")

VALID_ASPECTS: frozenset[str] = frozenset({"F", "P", "C"})

# The dump is the REST ``{"results": [ ... ]}`` stream with one entry per line.
# These are the only non-entry lines it contains.
_WRAPPER_LINES: frozenset[str] = frozenset({'{"results": [', "]}", "]", "[", "{"})


def iter_sprot_entries(json_path: Path | str) -> Iterator[dict]:
    """Yield UniProtKB entries one at a time from the REST JSON dump.

    ``json.load`` on the 7.4 GB file would need several times that in RAM. The dump
    writes one entry per line (``{"results": [``, then ``{...},`` per entry, then
    ``]}``), so parsing line by line keeps memory flat at one entry and runs at
    ~270 MB/s. A line that does not parse raises with its line number: a dump that
    was re-serialised with a different layout must fail loudly, not yield nothing.
    """
    with open(json_path, encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line in _WRAPPER_LINES:
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{json_path}:{lineno}: expected one UniProtKB entry per line "
                    f"({exc.msg}). Was the dump re-formatted?"
                ) from exc


def _properties(xref: dict) -> dict[str, str]:
    return {p.get("key", ""): p.get("value", "") for p in xref.get("properties", ())}


def go_annotations(entry: dict) -> tuple[list[tuple[str, str, str, str]], list[str], int]:
    """GO rows of one entry, plus the raw evidence fields and a malformed-row count.

    Returns ``(rows, raw_evidence, n_malformed)``: ``rows`` are unique
    ``(protein_id, GO_term, aspect, evidence_code)`` tuples in entry order;
    ``raw_evidence`` is the untouched ``GoEvidenceType`` field of each kept row
    (same order), which the CONTAINS cross-check needs; ``n_malformed`` counts GO
    cross-references skipped because the aspect or the evidence code could not be
    read. Those are counted rather than raised on: one odd record must not abort
    a 542k-entry export, but it must not vanish silently either.
    """
    accession = entry["primaryAccession"]
    rows: list[tuple[str, str, str, str]] = []
    raw_evidence: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    malformed = 0
    for xref in entry.get("uniProtKBCrossReferences", ()):
        if xref.get("database") != "GO":
            continue
        go_id = xref.get("id", "")
        props = _properties(xref)
        term = props.get("GoTerm", "")
        evidence_field = props.get("GoEvidenceType", "")
        aspect = term[:1] if term[1:2] == ":" else ""
        code = evidence_field.split(":", 1)[0].strip().upper()
        if not go_id.startswith("GO:") or aspect not in VALID_ASPECTS or not code.isalpha():
            malformed += 1
            continue
        row = (accession, go_id, aspect, code)
        if row in seen:
            continue
        seen.add(row)
        rows.append(row)
        raw_evidence.append(evidence_field)
    return rows, raw_evidence, malformed


def gene3d_ids(entry: dict) -> list[str]:
    """Sorted, unique Gene3D (CATH superfamily) ids of one entry, e.g. ``3.40.50.720``."""
    return sorted(
        {
            xref["id"]
            for xref in entry.get("uniProtKBCrossReferences", ())
            if xref.get("database") == "Gene3D" and xref.get("id")
        }
    )


def read_fai_lengths(fai_path: Path | str) -> dict[str, int]:
    """Sequence length per id from a samtools ``.fai`` index (columns: name, length, ...)."""
    lengths: dict[str, int] = {}
    with open(fai_path) as handle:
        for line in handle:
            name, length = line.split("\t", 2)[:2]
            lengths[name] = int(length)
    return lengths


def _contains_any(field: str, codes: Iterable[str]) -> bool:
    return any(code in field for code in codes)


def export(
    json_path: Path | str,
    output: Path | str,
    gene3d_output: Path | str | None = None,
    fai_lengths: dict[str, int] | None = None,
    max_length: int = 1022,
) -> dict:
    """Stream the dump once, write the TSV(s), and return a summary of counts.

    Output is written to ``<path>.tmp`` and renamed on success, so an interrupted
    run never leaves a truncated TSV that looks complete.
    """
    output = Path(output)
    tmp_go = output.with_name(output.name + ".tmp")
    tmp_g3d = Path(gene3d_output).with_name(Path(gene3d_output).name + ".tmp") if gene3d_output else None

    n_entries = 0
    n_malformed = 0
    rows_per_aspect: Counter[str] = Counter()
    rows_per_evidence: Counter[str] = Counter()
    proteins_any_go: set[str] = set()
    core6_mf_by_code: set[str] = set()
    core6_mf_by_contains: set[str] = set()
    code_vs_contains_disagree = 0
    n_gene3d = 0
    accessions_seen: set[str] = set()
    duplicate_accessions = 0

    started = time.monotonic()
    go_handle = open(tmp_go, "w", encoding="utf-8")
    g3d_handle = open(tmp_g3d, "w", encoding="utf-8") if tmp_g3d else None
    try:
        go_handle.write("\t".join(GO_TSV_HEADER) + "\n")
        if g3d_handle:
            g3d_handle.write("\t".join(GENE3D_TSV_HEADER) + "\n")
        for entry in iter_sprot_entries(json_path):
            n_entries += 1
            accession = entry["primaryAccession"]
            if accession in accessions_seen:
                duplicate_accessions += 1
            accessions_seen.add(accession)

            rows, raw_evidence, malformed = go_annotations(entry)
            n_malformed += malformed
            for (pid, go_id, aspect, code), field in zip(rows, raw_evidence, strict=True):
                go_handle.write(f"{pid}\t{go_id}\t{aspect}\t{code}\n")
                rows_per_aspect[aspect] += 1
                rows_per_evidence[code] += 1
                proteins_any_go.add(pid)
                if aspect != "F":
                    continue
                by_code = code in CAFA_CORE6
                by_contains = _contains_any(field, CAFA_CORE6)
                if by_code:
                    core6_mf_by_code.add(pid)
                if by_contains:
                    core6_mf_by_contains.add(pid)
                if by_code != by_contains:
                    code_vs_contains_disagree += 1

            if g3d_handle:
                g3d = gene3d_ids(entry)
                if g3d:
                    g3d_handle.write(f"{accession}\t{';'.join(g3d)}\n")
                    n_gene3d += 1

            if n_entries % 100_000 == 0:
                logger.info("%d entries (%.0f s)", n_entries, time.monotonic() - started)
    except BaseException:
        go_handle.close()
        tmp_go.unlink(missing_ok=True)
        if g3d_handle:
            g3d_handle.close()
            tmp_g3d.unlink(missing_ok=True)
        raise
    go_handle.close()
    tmp_go.replace(output)
    if g3d_handle:
        g3d_handle.close()
        tmp_g3d.replace(Path(gene3d_output))

    summary: dict = {
        "source": str(json_path),
        "output": str(output),
        "gene3d_output": str(gene3d_output) if gene3d_output else None,
        "n_entries": n_entries,
        "duplicate_accessions": duplicate_accessions,
        "n_rows": sum(rows_per_aspect.values()),
        "rows_per_aspect": dict(sorted(rows_per_aspect.items())),
        "rows_per_evidence": dict(rows_per_evidence.most_common()),
        "malformed_go_xrefs_skipped": n_malformed,
        "proteins_with_any_go": len(proteins_any_go),
        "core6_codes": sorted(CAFA_CORE6),
        "proteins_core6_mf": len(core6_mf_by_code),
        "proteins_core6_mf_by_contains": len(core6_mf_by_contains),
        "mf_rows_code_vs_contains_disagree": code_vs_contains_disagree,
        "proteins_with_gene3d": n_gene3d if gene3d_output else None,
        "seconds": round(time.monotonic() - started, 1),
    }
    if fai_lengths is not None:
        in_fai = [p for p in core6_mf_by_code if p in fai_lengths]
        summary["max_length"] = max_length
        summary["proteins_core6_mf_in_fai"] = len(in_fai)
        summary["proteins_core6_mf_le_max_length"] = sum(
            1 for p in in_fai if fai_lengths[p] <= max_length
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export GO annotations with evidence codes (and optionally Gene3D ids) "
            "from the UniProtKB/Swiss-Prot JSON dump."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sprot_json", type=Path, required=True, help="UniProt REST JSON dump.")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="4-column TSV: protein_id, GO_term, aspect (F/P/C), evidence (code).",
    )
    parser.add_argument(
        "--gene3d_output", type=Path, default=None,
        help="Optional Entry/Gene3D TSV (';'-joined, sorted CATH superfamily ids).",
    )
    parser.add_argument(
        "--fai", type=Path, default=None,
        help="Optional FASTA index; adds the <= --max_length count of core-6 MF proteins "
        "to the summary.",
    )
    parser.add_argument("--max_length", type=int, default=1022)
    parser.add_argument(
        "--summary", type=Path, default=None,
        help="Summary JSON path (default: <output stem>.summary.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.sprot_json.exists():
        logger.error("Swiss-Prot JSON not found: %s", args.sprot_json)
        return 1
    fai_lengths = read_fai_lengths(args.fai) if args.fai else None
    summary = export(
        args.sprot_json, args.output, args.gene3d_output, fai_lengths, args.max_length
    )
    summary_path = args.summary or args.output.with_name(args.output.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("wrote %s (%d rows) and %s", args.output, summary["n_rows"], summary_path)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
