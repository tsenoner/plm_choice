"""Unit 1 — orphan cohort freeze (mirror of ec_freeze, headline = (p1,p2,sibling))."""
import pandas as pd
import pytest

from evaluation.orphan_freeze import (
    derive_orphan_freeze,
    headline_content_hash,
    verify_orphan_freeze,
)


def _pairs():
    # deliberately unsorted by (p1, p2) so the sort-normalisation is exercised
    return pd.DataFrame(
        {
            "p1": ["B", "A", "A"],
            "p2": ["C", "C", "B"],
            "tm": [0.50, 0.10, 0.90],
            "snn": [0.40, 0.20, 0.80],
            "sibling": [True, False, True],
        }
    )


def test_derive_ids_are_sorted_union_and_counts():
    m = derive_orphan_freeze(_pairs(), derived_from="bromberg_2024", source_tsv="x.tsv.gz")
    assert m["set_name"] == "orphan_bromberg"
    assert m["ids"] == ["A", "B", "C"]  # sorted union of endpoints
    assert m["n_proteins"] == 3
    assert m["n_pairs"] == 3
    assert m["n_siblings"] == 2
    assert m["schema_version"] == 1
    assert isinstance(m["content_sha256"], str) and len(m["content_sha256"]) == 64
    # provenance fields present + distinct from the headline
    assert len(m["snn_sha256"]) == 64 and len(m["tm_sha256"]) == 64


def test_headline_hash_is_row_order_invariant():
    a = _pairs()
    b = a.iloc[::-1].reset_index(drop=True)
    ha = derive_orphan_freeze(a, derived_from="x", source_tsv="x")["content_sha256"]
    hb = derive_orphan_freeze(b, derived_from="x", source_tsv="x")["content_sha256"]
    assert ha == hb


def test_known_answer_headline_hash():
    # LITERAL digest, computed independently for payload '[["A","B",true]]' (NOT via
    # headline_content_hash on both sides — that would pass through a canonicalisation
    # change). A refactor of the record shape / separators trips this wire.
    KNOWN = "ab4d8e16e88fbe616263637e01bb5de1cd79528f18a878ad38124c6fa2cb7faa"
    assert headline_content_hash([("A", "B", True)]) == KNOWN
    one = pd.DataFrame(
        {"p1": ["A"], "p2": ["B"], "tm": [0.5], "snn": [0.5], "sibling": [True]}
    )
    assert derive_orphan_freeze(one, derived_from="x", source_tsv="x")["content_sha256"] == KNOWN


def test_verify_passes_on_matching_pairs():
    m = derive_orphan_freeze(_pairs(), derived_from="x", source_tsv="x")
    assert verify_orphan_freeze(m, _pairs()) is True


def test_verify_fails_when_sibling_mutates():
    m = derive_orphan_freeze(_pairs(), derived_from="x", source_tsv="x")
    drifted = _pairs()
    drifted.loc[0, "sibling"] = False  # flip a sibling label -> cohort identity changes
    with pytest.raises(ValueError, match="drift"):
        verify_orphan_freeze(m, drifted)


def test_tm_only_change_keeps_headline_stable_but_changes_tm_provenance():
    base = derive_orphan_freeze(_pairs(), derived_from="x", source_tsv="x")
    mutated = _pairs()
    mutated.loc[0, "tm"] = 0.123  # re-score one TM cell only
    after = derive_orphan_freeze(mutated, derived_from="x", source_tsv="x")
    # headline (cohort identity) MUST be stable
    assert after["content_sha256"] == base["content_sha256"]
    # but the TM provenance hash MUST move
    assert after["tm_sha256"] != base["tm_sha256"]
    assert after["snn_sha256"] == base["snn_sha256"]
    # and verify still passes (TM is non-gating)
    assert verify_orphan_freeze(base, mutated) is True
