"""Tests for evaluation.label_adapters.

Ported from the SpeciesEmbedding reference (tools/eval/label_adapters.py) into the
upstream layout: import via ``from evaluation.label_adapters import ...``.

Covers the EC parser (ported verbatim) and the new CATH/Gene3D adapter (W2):
Gene3D -> Topology (``fold``) + Homologous-superfamily (``superfamily``), one
*set of domains* per protein. ``family`` is a placeholder (real CATH family
labels are an unmet people-track input) so the frame satisfies recall_fp's
hard ``protein_id/fold/superfamily/family`` column contract.
"""
from __future__ import annotations

import pandas as pd
import pytest

from evaluation.label_adapters import (
    load_cath_labels,
    parse_cath_from_gene3d,
    parse_ec_from_protein_names,
)


# ── EC parser (ported) ───────────────────────────────────────────────────────
def _ec_df(rows):
    return pd.DataFrame(rows, columns=["Entry", "Protein names"])


def test_parses_single_ec():
    df = _ec_df([("P1", "chitin synthase (EC 2.4.1.16)")])
    out = parse_ec_from_protein_names(df)
    assert list(out.columns) == ["protein_id", "ec_number"]
    assert out.to_records(index=False).tolist() == [("P1", "2.4.1.16")]


def test_skips_protein_without_ec():
    df = _ec_df([("P1", "hypothetical protein"), ("P2", "lipase (EC 3.1.1.3)")])
    out = parse_ec_from_protein_names(df)
    assert out["protein_id"].tolist() == ["P2"]


def test_partial_wildcard_ec_kept():
    df = _ec_df([("P1", "serine protease (EC 3.4.21.-)")])
    out = parse_ec_from_protein_names(df)
    assert out.iloc[0]["ec_number"] == "3.4.21.-"


def test_multiple_ec_takes_first():
    df = _ec_df([("P1", "bifunctional enzyme (EC 1.1.1.1) (EC 2.2.2.2)")])
    out = parse_ec_from_protein_names(df)
    assert out.iloc[0]["ec_number"] == "1.1.1.1"


def test_malformed_ec_is_skipped_not_raised():
    df = _ec_df([("P1", "weird (EC 9.9)"), ("P2", "real (EC 1.2.3.4)")])
    out = parse_ec_from_protein_names(df)
    assert out["protein_id"].tolist() == ["P2"]


# ── CATH / Gene3D adapter (W2) ───────────────────────────────────────────────
def _cath_df(rows):
    return pd.DataFrame(rows, columns=["Entry", "Gene3D"])


def test_cath_columns_match_recall_fp_contract():
    # recall_fp hard-requires protein_id + fold/superfamily/family.
    df = _cath_df([("P1", "3.90.550.10;")])
    out = parse_cath_from_gene3d(df)
    assert list(out.columns) == ["protein_id", "fold", "superfamily", "family"]


def test_single_domain_fold_is_three_field_topology():
    df = _cath_df([("P1", "3.90.550.10;")])
    out = parse_cath_from_gene3d(df)
    assert out.iloc[0]["fold"] == frozenset({"3.90.550"})


def test_single_domain_superfamily_is_four_field():
    df = _cath_df([("P1", "3.90.550.10;")])
    out = parse_cath_from_gene3d(df)
    assert out.iloc[0]["superfamily"] == frozenset({"3.90.550.10"})


def test_multi_domain_collected_as_set():
    df = _cath_df([("P1", "1.10.287.70;3.40.50.2300;3.40.190.10;")])
    out = parse_cath_from_gene3d(df)
    assert out.iloc[0]["fold"] == frozenset({"1.10.287", "3.40.50", "3.40.190"})
    assert out.iloc[0]["superfamily"] == frozenset(
        {"1.10.287.70", "3.40.50.2300", "3.40.190.10"}
    )


def test_duplicate_topology_collapses_in_fold_set():
    # Two domains that share a Topology but differ at the H level: fold set
    # collapses to one Topology, superfamily set keeps both.
    df = _cath_df([("P1", "3.40.50.2300;3.40.50.1000;")])
    out = parse_cath_from_gene3d(df)
    assert out.iloc[0]["fold"] == frozenset({"3.40.50"})
    assert out.iloc[0]["superfamily"] == frozenset({"3.40.50.2300", "3.40.50.1000"})


def test_family_is_none_placeholder():
    df = _cath_df([("P1", "3.90.550.10;")])
    out = parse_cath_from_gene3d(df)
    assert out.iloc[0]["family"] is None


def test_protein_without_gene3d_is_omitted():
    df = _cath_df([("P1", ""), ("P2", "3.90.550.10;")])
    out = parse_cath_from_gene3d(df)
    assert out["protein_id"].tolist() == ["P2"]


def test_nan_gene3d_is_omitted_not_raised():
    df = pd.DataFrame(
        [("P1", float("nan")), ("P2", "3.90.550.10;")],
        columns=["Entry", "Gene3D"],
    )
    out = parse_cath_from_gene3d(df)
    assert out["protein_id"].tolist() == ["P2"]


def test_malformed_code_skipped_protein_kept_if_any_valid():
    # A junk token is dropped; the protein survives on its valid domain.
    df = _cath_df([("P1", "garbage;3.90.550.10;")])
    out = parse_cath_from_gene3d(df)
    assert out.iloc[0]["superfamily"] == frozenset({"3.90.550.10"})


def test_protein_with_only_malformed_codes_is_omitted():
    df = _cath_df([("P1", "garbage;1.2.3;"), ("P2", "3.90.550.10;")])
    out = parse_cath_from_gene3d(df)
    assert out["protein_id"].tolist() == ["P2"]


def test_coverage_is_caller_computable_via_len():
    df = _cath_df([("P1", ""), ("P2", "3.90.550.10;"), ("P3", "1.10.10.10;")])
    out = parse_cath_from_gene3d(df)
    assert len(out) / len(df) == 2 / 3


def test_whitespace_around_codes_tolerated():
    df = _cath_df([("P1", " 3.90.550.10 ; 1.10.10.10 ;")])
    out = parse_cath_from_gene3d(df)
    assert out.iloc[0]["superfamily"] == frozenset({"3.90.550.10", "1.10.10.10"})


def test_non_string_gene3d_cell_omitted():
    df = _cath_df([("P1", 123), ("P2", "3.90.550.10;")])
    out = parse_cath_from_gene3d(df)
    assert out["protein_id"].tolist() == ["P2"]


def test_nan_protein_id_is_omitted():
    # A blank Entry parses to NaN under dtype=str; it must not become a
    # protein_id=nan row (would silently fail to join downstream).
    df = pd.DataFrame(
        [(float("nan"), "3.90.550.10;"), ("P2", "1.10.10.10;")],
        columns=["Entry", "Gene3D"],
    )
    out = parse_cath_from_gene3d(df)
    assert out["protein_id"].tolist() == ["P2"]


def test_blank_protein_id_is_omitted():
    df = _cath_df([("", "3.90.550.10;"), ("P2", "1.10.10.10;")])
    out = parse_cath_from_gene3d(df)
    assert out["protein_id"].tolist() == ["P2"]


def test_missing_gene3d_column_fails_closed():
    # A wholesale-missing column is a structural error, not a per-row skip:
    # fail loudly rather than silently report 0% coverage.
    df = pd.DataFrame([("P1",)], columns=["Entry"])
    with pytest.raises(KeyError, match="Gene3D"):
        parse_cath_from_gene3d(df)


def test_missing_entry_column_fails_closed():
    df = pd.DataFrame([("3.90.550.10;",)], columns=["Gene3D"])
    with pytest.raises(KeyError, match="Entry"):
        parse_cath_from_gene3d(df)


# ── load_cath_labels (file loader) ───────────────────────────────────────────
def _write_tsv(path, rows):
    lines = ["Entry\tOrganism\tProtein names\tGene3D\tPfam\tSUPFAM\tInterPro"]
    for entry, gene3d in rows:
        lines.append(f"{entry}\tOrg\tname\t{gene3d}\tPF;\tSSF;\tIPR;")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_cath_labels_reads_tsv(tmp_path):
    p = _write_tsv(tmp_path / "cath.tsv", [("P1", "3.90.550.10;"), ("P2", "")])
    out = load_cath_labels(p)
    assert out["protein_id"].tolist() == ["P1"]
    assert out.iloc[0]["fold"] == frozenset({"3.90.550"})


def test_load_cath_labels_returns_recall_fp_contract(tmp_path):
    p = _write_tsv(tmp_path / "cath.tsv", [("P1", "3.90.550.10;")])
    out = load_cath_labels(p)
    assert list(out.columns) == ["protein_id", "fold", "superfamily", "family"]


def test_load_cath_labels_multidomain_from_disk(tmp_path):
    # The ;-split must survive the file-read path, not just synthetic frames.
    p = _write_tsv(tmp_path / "cath.tsv", [("P1", "1.10.287.70;3.40.50.2300;")])
    out = load_cath_labels(p)
    assert out.iloc[0]["superfamily"] == frozenset({"1.10.287.70", "3.40.50.2300"})
