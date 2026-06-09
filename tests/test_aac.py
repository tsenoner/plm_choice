"""Tests for the amino-acid composition (AAC) baseline.

Ported from the SpeciesEmbedding reference (tools/embeddings/aac.py) into the
upstream layout: import via `from data_preparation.aac import ...`.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_preparation.aac import STANDARD_AA, extract_aac


def test_aac_returns_20d_without_other_bucket():
    fa = {"p1": "ACDEFGHIKLMNPQRSTVWY"}
    out = extract_aac(fa, include_other=False)
    assert out["p1"].shape == (20,)
    np.testing.assert_allclose(out["p1"], np.full(20, 1.0 / 20.0), atol=1e-6)


def test_aac_21d_with_other_bucket():
    fa = {"p1": "ACDEFGHIKLMNPQRSTVWY"}
    out = extract_aac(fa, include_other=True)
    assert out["p1"].shape == (21,)
    assert out["p1"][20] == 0.0


def test_aac_sums_to_one_when_normalized_and_other_included():
    fa = {"p1": "ACDXXBJOUZ"}
    out = extract_aac(fa, normalize=True, include_other=True)
    assert pytest.approx(out["p1"].sum(), abs=1e-6) == 1.0


def test_aac_sums_le_one_when_other_dropped():
    fa = {"p1": "ACDXXBJOUZ"}
    out = extract_aac(fa, normalize=True, include_other=False)
    assert out["p1"].sum() <= 1.0 + 1e-6
    assert pytest.approx(out["p1"].sum(), abs=1e-6) == 0.3


def test_aac_raw_counts():
    fa = {"p1": "AAACC"}
    out = extract_aac(fa, normalize=False, include_other=False)
    assert out["p1"][0] == 3.0
    assert out["p1"][1] == 2.0
    assert out["p1"].sum() == 5.0


def test_aac_per_residue_onehot():
    fa = {"p1": "AC"}
    out = extract_aac(fa, reduce=False, include_other=False)
    assert out["p1"].shape == (2, 20)
    assert out["p1"][0, 0] == 1.0
    assert out["p1"][0, 1:].sum() == 0.0
    assert out["p1"][1, 1] == 1.0


def test_aac_handles_empty_sequence():
    fa = {"empty": ""}
    out = extract_aac(fa)
    assert out["empty"].shape == (21,)
    np.testing.assert_array_equal(out["empty"], np.zeros(21, dtype=np.float32))


def test_aac_dtype_is_float32():
    out = extract_aac({"p1": "ACDE"})
    assert out["p1"].dtype == np.float32


def test_aac_alphabet_consistency():
    assert STANDARD_AA == "ACDEFGHIKLMNPQRSTVWY"
    assert len(STANDARD_AA) == 20
