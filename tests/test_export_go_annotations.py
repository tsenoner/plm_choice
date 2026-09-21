"""The GO evidence code must survive the trip out of the 7.4 GB Swiss-Prot dump.

The GO arm answers reviewer R2.1 only because its labels are experimental, and
the only place the evidence code of a Swiss-Prot GO annotation exists is the
``GoEvidenceType`` property of the JSON cross-reference. Everything downstream
(the cohort, the transfer scores) reads the exporter's TSV, so a parsing slip
here does not crash anything — it silently re-admits IEA and turns the arm back
into the homology proxy the reviewers objected to. These tests pin the parse,
the streaming (the dump cannot be ``json.load``ed), and the hand-off to
``load_annotations_tsv``.
"""

from __future__ import annotations

import json

import pytest

from data_preparation.export_go_annotations import (
    export,
    gene3d_ids,
    go_annotations,
    iter_sprot_entries,
    read_fai_lengths,
)
from data_preparation.go_semantic_similarity import CAFA_CORE6, GOTerm, load_annotations_tsv


def _xref(database: str, xref_id: str, **props: str) -> dict:
    return {
        "database": database,
        "id": xref_id,
        "properties": [{"key": k, "value": v} for k, v in props.items()],
    }


def _go(xref_id: str, term: str, evidence: str) -> dict:
    """A GO cross-reference in the dump's own layout: ``F:<name>`` / ``IDA:UniProtKB``."""
    return _xref("GO", xref_id, GoTerm=term, GoEvidenceType=evidence)


ENTRIES = [
    {
        "primaryAccession": "P00001",
        "uniProtKBCrossReferences": [
            _go("GO:0003953", "F:NAD+ nucleosidase activity", "IDA:UniProtKB"),
            _go("GO:0061809", "F:NAD+ nucleotidase", "IEA:UniProtKB-EC"),
            _go("GO:0005576", "C:extracellular region", "IEA:UniProtKB-SubCell"),
            _xref("Gene3D", "3.40.50.720"),
            _xref("Gene3D", "1.10.10.10"),
            _xref("Pfam", "PF00001"),
        ],
    },
    {
        "primaryAccession": "P00002",
        "uniProtKBCrossReferences": [
            _go("GO:0004984", "F:olfactory receptor activity", "IMP:UniProtKB"),
            _go("GO:0006952", "P:defense response", "IDA:UniProtKB"),
            _xref("Gene3D", "1.10.10.10"),
        ],
    },
    {
        # Only electronic MF evidence, and no Gene3D: excluded from the core-6 count
        # and absent from the Gene3D TSV.
        "primaryAccession": "P00003",
        "uniProtKBCrossReferences": [
            _go("GO:0005515", "F:protein binding", "IEA:InterPro"),
        ],
    },
    {
        # High-throughput direct assay: experimental, but NOT one of the CAFA core 6.
        "primaryAccession": "P00004",
        "uniProtKBCrossReferences": [
            _go("GO:0004672", "F:protein kinase activity", "HDA:UniProtKB"),
        ],
    },
]


def _write_dump(tmp_path, entries=ENTRIES):
    """The dump's exact framing: a ``{"results": [`` wrapper, one entry per line."""
    lines = ['{"results": ['] + [json.dumps(e) + "," for e in entries]
    lines[-1] = lines[-1].rstrip(",")
    path = tmp_path / "sprot.json"
    path.write_text("\n".join(lines + ["]}"]) + "\n")
    return path


# ── streaming ────────────────────────────────────────────────────────────────


def test_entries_stream_one_at_a_time(tmp_path):
    entries = list(iter_sprot_entries(_write_dump(tmp_path)))
    assert [e["primaryAccession"] for e in entries] == ["P00001", "P00002", "P00003", "P00004"]


def test_a_reformatted_dump_fails_loudly(tmp_path):
    """Pretty-printed JSON must raise, not yield nothing.

    ``json.load`` is not an option at 7.4 GB, so the reader assumes one entry per
    line. If that assumption ever breaks, an empty export would look like a clean
    run with zero annotations.
    """
    path = tmp_path / "pretty.json"
    path.write_text(json.dumps({"results": ENTRIES}, indent=2))
    with pytest.raises(ValueError, match="one UniProtKB entry per line"):
        list(iter_sprot_entries(path))


# ── per-entry parsing ────────────────────────────────────────────────────────


def test_aspect_and_evidence_code_are_split_out_of_their_fields():
    rows, raw, malformed = go_annotations(ENTRIES[0])
    assert rows == [
        ("P00001", "GO:0003953", "F", "IDA"),
        ("P00001", "GO:0061809", "F", "IEA"),
        ("P00001", "GO:0005576", "C", "IEA"),
    ]
    assert raw == ["IDA:UniProtKB", "IEA:UniProtKB-EC", "IEA:UniProtKB-SubCell"]
    assert malformed == 0


def test_non_go_cross_references_are_ignored():
    rows, _, _ = go_annotations(ENTRIES[2])
    assert [r[1] for r in rows] == ["GO:0005515"]


def test_malformed_cross_references_are_counted_not_raised():
    """One odd record must not abort a 542k-entry export, nor vanish silently."""
    entry = {
        "primaryAccession": "P00009",
        "uniProtKBCrossReferences": [
            _go("GO:0003953", "F:fine", "IDA:UniProtKB"),
            _go("GO:0000001", "no aspect prefix", "IDA:UniProtKB"),  # unreadable aspect
            _go("GO:0000002", "F:fine", ""),  # unreadable evidence
            _go("0003953", "F:fine", "IDA:UniProtKB"),  # not a GO id
        ],
    }
    rows, _, malformed = go_annotations(entry)
    assert [r[1] for r in rows] == ["GO:0003953"]
    assert malformed == 3


def test_duplicate_cross_references_are_emitted_once():
    entry = {
        "primaryAccession": "P00010",
        "uniProtKBCrossReferences": [
            _go("GO:0003953", "F:x", "IDA:UniProtKB"),
            _go("GO:0003953", "F:x", "IDA:UniProtKB"),
            _go("GO:0003953", "F:x", "IMP:UniProtKB"),  # same term, other code: kept
        ],
    }
    rows, _, _ = go_annotations(entry)
    assert rows == [
        ("P00010", "GO:0003953", "F", "IDA"),
        ("P00010", "GO:0003953", "F", "IMP"),
    ]


def test_gene3d_ids_are_sorted_and_unique():
    assert gene3d_ids(ENTRIES[0]) == ["1.10.10.10", "3.40.50.720"]
    assert gene3d_ids(ENTRIES[2]) == []


def test_read_fai_lengths(tmp_path):
    fai = tmp_path / "s.fasta.fai"
    fai.write_text("P00001\t269\t12\t60\t61\nP00002\t1500\t298\t60\t61\n")
    assert read_fai_lengths(fai) == {"P00001": 269, "P00002": 1500}


# ── end-to-end export ────────────────────────────────────────────────────────


@pytest.fixture
def exported(tmp_path):
    fai = tmp_path / "s.fasta.fai"
    fai.write_text(
        "P00001\t269\t12\t60\t61\n"  # short
        "P00002\t1500\t298\t60\t61\n"  # too long
        "P00003\t100\t900\t60\t61\n"
        "P00004\t100\t1200\t60\t61\n"
    )
    out = tmp_path / "go.tsv"
    g3d = tmp_path / "gene3d.tsv"
    summary = export(
        _write_dump(tmp_path),
        out,
        gene3d_output=g3d,
        fai_lengths=read_fai_lengths(fai),
        max_length=1022,
    )
    return out, g3d, summary


def test_tsv_has_a_header_and_one_row_per_annotation(exported):
    out, _, summary = exported
    lines = out.read_text().splitlines()
    assert lines[0] == "protein_id\tGO_term\taspect\tevidence"
    assert lines[1] == "P00001\tGO:0003953\tF\tIDA"
    assert len(lines) - 1 == summary["n_rows"] == 7


def test_electronic_annotations_are_exported_not_filtered(exported):
    """Filtering is the loader's job, so the code set stays visible at the point of use."""
    out, _, summary = exported
    assert "P00003\tGO:0005515\tF\tIEA" in out.read_text()
    assert summary["rows_per_evidence"]["IEA"] == 3


def test_summary_counts_rows_per_aspect(exported):
    _, _, summary = exported
    assert summary["rows_per_aspect"] == {"C": 1, "F": 5, "P": 1}


def test_core6_excludes_high_throughput_and_electronic_only_proteins(exported):
    """P00003 is IEA-only and P00004 is HDA — experimental, but outside the core 6."""
    _, _, summary = exported
    assert summary["proteins_core6_mf"] == 2
    assert sorted(CAFA_CORE6) == summary["core6_codes"]


def test_code_and_contains_definitions_are_cross_checked(exported):
    """The cohort matches evidence by CONTAINS; the export proves that equals the code."""
    _, _, summary = exported
    assert summary["proteins_core6_mf_by_contains"] == summary["proteins_core6_mf"]
    assert summary["mf_rows_code_vs_contains_disagree"] == 0


def test_length_cut_is_reported_from_the_fai(exported):
    """P00001 (269 aa) counts, P00002 (1500 aa) does not — the >1022 aa proteins
    were never embedded, so they can never be a neighbour."""
    _, _, summary = exported
    assert summary["proteins_core6_mf_in_fai"] == 2
    assert summary["proteins_core6_mf_le_max_length"] == 1


def test_gene3d_tsv_is_the_uniprot_export_layout(exported):
    _, g3d, summary = exported
    assert g3d.read_text().splitlines() == [
        "Entry\tGene3D",
        "P00001\t1.10.10.10;3.40.50.720",
        "P00002\t1.10.10.10",
    ]
    assert summary["proteins_with_gene3d"] == 2


def test_an_interrupted_export_leaves_no_truncated_tsv(tmp_path):
    """A half-written TSV that looks complete would silently shrink the cohort."""
    bad = tmp_path / "sprot.json"
    bad.write_text('{"results": [\n' + json.dumps(ENTRIES[0]) + ",\nnot json\n]}\n")
    out = tmp_path / "go.tsv"
    with pytest.raises(ValueError):
        export(bad, out)
    assert not out.exists()
    assert not out.with_name(out.name + ".tmp").exists()


# ── hand-off to the similarity loader ────────────────────────────────────────

GO_TERMS = {
    "GO:0003953": GOTerm(id="GO:0003953", name="NAD+ nucleosidase", namespace="molecular_function"),
    "GO:0061809": GOTerm(id="GO:0061809", name="NAD+ nucleotidase", namespace="molecular_function"),
    "GO:0004984": GOTerm(id="GO:0004984", name="olfactory receptor", namespace="molecular_function"),
    "GO:0005515": GOTerm(id="GO:0005515", name="protein binding", namespace="molecular_function"),
    "GO:0004672": GOTerm(id="GO:0004672", name="protein kinase", namespace="molecular_function"),
    "GO:0006952": GOTerm(id="GO:0006952", name="defense response", namespace="biological_process"),
    "GO:0005576": GOTerm(id="GO:0005576", name="extracellular", namespace="cellular_component"),
}


def test_the_exported_tsv_loads_with_core6_filtering(exported):
    """The round trip the cohort depends on: 4 columns in, IEA out, header not an
    annotation. The header row previously passed format detection and was counted
    as an evidence code named ``EVIDENCE``."""
    out, _, _ = exported
    annotations = load_annotations_tsv(out, GO_TERMS, evidence_codes=set(CAFA_CORE6))

    assert annotations["P00001"]["MFO"] == {"GO:0003953"}  # the IEA term is gone
    assert annotations["P00002"]["MFO"] == {"GO:0004984"}
    assert annotations["P00002"]["BPO"] == {"GO:0006952"}
    assert "P00003" not in annotations  # IEA-only
    assert "P00004" not in annotations  # HDA is experimental but not core-6
    assert "protein_id" not in annotations
