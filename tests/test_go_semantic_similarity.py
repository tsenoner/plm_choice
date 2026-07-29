"""The GO functional axis must exclude homology-transferred annotations.

This arm exists to answer reviewer R2.1: HFSP is defined from percent identity,
so it cannot serve as an independent functional axis. GO can — but only if the
labels were assigned by experiment. ``IEA`` ("Inferred from Electronic
Annotation") is homology transfer by definition and is the majority of GO, so
loading it would rebuild the same circularity inside the replacement axis.

The evidence-code column was originally ignored: the GAF parser read columns
1, 4 and 8 and never looked at column 6, so every IEA annotation was loaded
silently. These tests pin the filter shut.
"""

from __future__ import annotations

import pytest

from data_preparation.go_semantic_similarity import (
    AUTHOR_EVIDENCE,
    ELECTRONIC_EVIDENCE,
    EXPERIMENTAL_EVIDENCE,
    GOTerm,
    load_annotations_tsv,
    parse_obo,
)

# A minimal ontology; only membership matters for the loader.
GO_TERMS = {
    "GO:0003674": GOTerm(id="GO:0003674", name="molecular_function", namespace="molecular_function"),
    "GO:0004672": GOTerm(id="GO:0004672", name="protein kinase activity", namespace="molecular_function"),
    "GO:0005515": GOTerm(id="GO:0005515", name="protein binding", namespace="molecular_function"),
}


def _gaf_line(protein: str, go_id: str, evidence: str, aspect: str = "F") -> str:
    """A 17-column GAF 2.2 row; only columns 1, 4, 6 and 8 are read.

    The trailing columns carry placeholders rather than empty strings: the
    loader does ``line.strip()`` before splitting, which would eat trailing tabs
    and drop the row below the 15-column GAF threshold.
    """
    cols = ["-"] * 17
    cols[0] = "UniProtKB"
    cols[1] = protein
    cols[2] = protein
    cols[4] = go_id
    cols[5] = "PMID:00000"
    cols[6] = evidence
    cols[8] = aspect
    cols[11] = "protein"
    cols[12] = "taxon:9606"
    cols[13] = "20240101"
    cols[14] = "UniProt"
    return "\t".join(cols)


@pytest.fixture
def gaf_file(tmp_path):
    """One experimental and one electronic annotation per protein."""
    path = tmp_path / "annotations.gaf"
    path.write_text(
        "!gaf-version: 2.2\n"
        + "\n".join(
            [
                _gaf_line("P00001", "GO:0004672", "IDA"),  # experimental
                _gaf_line("P00001", "GO:0005515", "IEA"),  # electronic
                _gaf_line("P00002", "GO:0004672", "IEA"),  # electronic
                _gaf_line("P00002", "GO:0005515", "IMP"),  # experimental
                _gaf_line("P00003", "GO:0004672", "TAS"),  # author statement
            ]
        )
        + "\n"
    )
    return path


def test_iea_is_excluded_by_default(gaf_file):
    annotations = load_annotations_tsv(
        gaf_file, GO_TERMS, evidence_codes=set(EXPERIMENTAL_EVIDENCE)
    )
    assert annotations["P00001"]["MFO"] == {"GO:0004672"}, "IEA leaked into the labels"
    assert annotations["P00002"]["MFO"] == {"GO:0005515"}


def test_author_statements_are_not_experimental_by_default(gaf_file):
    annotations = load_annotations_tsv(
        gaf_file, GO_TERMS, evidence_codes=set(EXPERIMENTAL_EVIDENCE)
    )
    assert "P00003" not in annotations or not annotations["P00003"]["MFO"]


def test_author_statements_can_be_opted_in(gaf_file):
    annotations = load_annotations_tsv(
        gaf_file, GO_TERMS, evidence_codes=set(EXPERIMENTAL_EVIDENCE | AUTHOR_EVIDENCE)
    )
    assert annotations["P00003"]["MFO"] == {"GO:0004672"}


def test_no_filter_loads_everything(gaf_file):
    """`--evidence_codes ALL` maps to None; it must be an explicit choice."""
    annotations = load_annotations_tsv(gaf_file, GO_TERMS, evidence_codes=None)
    assert annotations["P00001"]["MFO"] == {"GO:0004672", "GO:0005515"}


def test_iea_is_never_in_the_experimental_set():
    assert ELECTRONIC_EVIDENCE.isdisjoint(EXPERIMENTAL_EVIDENCE)
    assert "IEA" not in EXPERIMENTAL_EVIDENCE
    assert "IEA" not in AUTHOR_EVIDENCE


def test_high_throughput_codes_count_as_experimental():
    """HTP/HDA/HMP/HGI/HEP were added to GO in 2017 and are experimental."""
    for code in ("HTP", "HDA", "HMP", "HGI", "HEP"):
        assert code in EXPERIMENTAL_EVIDENCE


def test_four_column_tsv_supports_filtering(tmp_path):
    path = tmp_path / "annotations.tsv"
    path.write_text(
        "P00001\tGO:0004672\tF\tIDA\n"
        "P00001\tGO:0005515\tF\tIEA\n"
    )
    annotations = load_annotations_tsv(
        path, GO_TERMS, evidence_codes=set(EXPERIMENTAL_EVIDENCE)
    )
    assert annotations["P00001"]["MFO"] == {"GO:0004672"}


def test_three_column_tsv_warns_that_it_cannot_filter(tmp_path, caplog):
    """A file with no evidence column must not silently look filtered."""
    path = tmp_path / "annotations.tsv"
    path.write_text("P00001\tGO:0004672\tF\nP00001\tGO:0005515\tF\n")

    with caplog.at_level("WARNING"):
        annotations = load_annotations_tsv(
            path, GO_TERMS, evidence_codes=set(EXPERIMENTAL_EVIDENCE)
        )

    assert annotations["P00001"]["MFO"] == {"GO:0004672", "GO:0005515"}
    assert "NO EVIDENCE-CODE COLUMN" in caplog.text
    assert "R2.1" in caplog.text


# ── OBO parsing ───────────────────────────────────────────────────────────────
# The parser started a new GOTerm on every ``[Term]`` line without storing the
# previous one, so it kept only the last block before a ``[Typedef]``: a 47k-term
# go-basic.obo parsed down to ONE term. Every annotation then failed the
# "is this term in the ontology?" check, every pair scored NaN, and the GO arm
# would have reported a column of nothing while logging a cheerful summary.

OBO = """format-version: 1.2

[Term]
id: GO:0000001
name: alpha
namespace: molecular_function

[Term]
id: GO:0000002
name: beta
namespace: molecular_function
is_a: GO:0000001 ! alpha

[Term]
id: GO:0000003
name: gamma
namespace: biological_process
relationship: part_of GO:0000001 ! alpha

[Term]
id: GO:0000009
name: dead
namespace: molecular_function
is_obsolete: true

[Typedef]
id: part_of
name: part of
"""


@pytest.fixture
def obo_file(tmp_path):
    path = tmp_path / "mini.obo"
    path.write_text(OBO)
    return path


def test_every_term_block_is_kept(obo_file):
    terms = parse_obo(obo_file)
    assert set(terms) == {"GO:0000001", "GO:0000002", "GO:0000003"}, (
        "terms were dropped — a new [Term] block must flush the previous one"
    )


def test_obsolete_terms_are_dropped(obo_file):
    assert "GO:0000009" not in parse_obo(obo_file)


def test_parents_and_namespaces_survive(obo_file):
    terms = parse_obo(obo_file)
    assert terms["GO:0000002"].parents == [("GO:0000001", "is_a")]
    assert terms["GO:0000003"].parents == [("GO:0000001", "part_of")]
    assert terms["GO:0000001"].namespace == "molecular_function"
    assert terms["GO:0000003"].namespace == "biological_process"


def test_last_term_survives_without_a_trailing_typedef(tmp_path):
    """A file that ends on a [Term] block must still yield that term."""
    path = tmp_path / "no_typedef.obo"
    path.write_text(OBO.split("[Typedef]")[0])
    assert set(parse_obo(path)) == {"GO:0000001", "GO:0000002", "GO:0000003"}
