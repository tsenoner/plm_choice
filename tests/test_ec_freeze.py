import pandas as pd
import pytest

from evaluation.ec_freeze import (
    content_hash,
    derive_ec_freeze,
    verify_ec_freeze,
)


def _labels():
    # protein_id + ec_set frame (the parse_ec output shape).
    return pd.DataFrame(
        {
            "protein_id": ["P3", "P1", "P2"],  # deliberately unsorted
            "ec_set": [
                frozenset({"1.1.1.1"}),
                frozenset({"2.7.11.1", "2.7.11.2"}),
                frozenset({"3.4.21.62"}),
            ],
        }
    )


def test_derive_produces_sorted_ids_and_count():
    m = derive_ec_freeze(_labels(), derived_from="canonical_set_319",
                         source_tsv="freeze/x.tsv", wildcard_policy="exclude")
    assert m["set_name"] == "ec_positive_subset"
    assert m["ids"] == ["P1", "P2", "P3"]      # sorted
    assert m["n_proteins"] == 3
    assert m["schema_version"] == 1
    assert isinstance(m["content_sha256"], str) and len(m["content_sha256"]) == 64


def test_hash_is_order_invariant():
    # Row reorder must not change the hash (normalization invariance).
    a = _labels()
    b = a.iloc[::-1].reset_index(drop=True)
    ha = derive_ec_freeze(a, derived_from="x", source_tsv="x")["content_sha256"]
    hb = derive_ec_freeze(b, derived_from="x", source_tsv="x")["content_sha256"]
    assert ha == hb


def test_hash_changes_when_one_ec_mutates():
    a = _labels()
    m1 = derive_ec_freeze(a, derived_from="x", source_tsv="x")["content_sha256"]
    b = a.copy()
    b.loc[0, "ec_set"] = frozenset({"9.9.9.9"})  # mutate P3's EC
    m2 = derive_ec_freeze(b, derived_from="x", source_tsv="x")["content_sha256"]
    assert m1 != m2


def test_known_answer_hash():
    # Pin a LITERAL, independently-computed digest (not content_hash() on both sides —
    # that would pass through any canonicalisation change). This digest was computed
    # offline for payload '[["A",["1.1.1.1"]]]'; a refactor of the canonicalisation
    # (separators, sort, record shape) changes the real hash and trips this wire.
    one = pd.DataFrame({"protein_id": ["A"], "ec_set": [frozenset({"1.1.1.1"})]})
    KNOWN = "2fd8fec84e248acf32a0323ec1804263abaae9a6f39285bd220b2c5b6a6b88eb"
    assert content_hash([("A", ["1.1.1.1"])]) == KNOWN  # the kernel produces the literal
    assert derive_ec_freeze(one, derived_from="x", source_tsv="x")["content_sha256"] == KNOWN


def test_verify_passes_on_matching_labels_and_fails_on_drift():
    m = derive_ec_freeze(_labels(), derived_from="x", source_tsv="x")
    # independent re-derivation agrees
    assert verify_ec_freeze(m, _labels()) is True
    # mutate the labels -> verify must fail (hash drift)
    drifted = _labels()
    drifted.loc[0, "ec_set"] = frozenset({"9.9.9.9"})
    with pytest.raises(ValueError, match="drift"):
        verify_ec_freeze(m, drifted)
