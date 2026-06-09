"""Tests for evaluation.ec_hierarchy.

Ported from the SpeciesEmbedding reference (tools/eval/ec_hierarchy.py) into the
upstream layout: import via `from evaluation.ec_hierarchy import ...`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.ec_hierarchy import (
    correlate_embedding_distance_with_ec,
    ec_distance,
    ec_distance_matrix,
)


def test_ec_distance_identical():
    assert ec_distance("1.1.1.1", "1.1.1.1") == 0


def test_ec_distance_match_first_three():
    assert ec_distance("1.1.1.1", "1.1.1.2") == 1


def test_ec_distance_match_first_two():
    assert ec_distance("1.1.1.1", "1.1.2.5") == 2


def test_ec_distance_match_first_one():
    assert ec_distance("1.1.1.1", "1.2.3.4") == 3


def test_ec_distance_no_overlap():
    assert ec_distance("1.1.1.1", "2.3.4.5") == 4


def test_ec_distance_wildcard_trailing_dash():
    assert ec_distance("1.1.1.-", "1.1.1.5") == 0
    assert ec_distance("1.1.-.-", "1.1.7.9") == 0


def test_ec_distance_missing_trailing_fields():
    assert ec_distance("1.1.1", "1.1.1.5") == 0
    assert ec_distance("1.1", "1.1.7.9") == 0
    assert ec_distance("1", "2.3.4.5") == 4


def test_ec_distance_non_string():
    with pytest.raises(TypeError):
        ec_distance(1.1, "1.1.1.1")  # type: ignore[arg-type]


def test_ec_distance_too_many_fields():
    with pytest.raises(ValueError, match="more than 4"):
        ec_distance("1.1.1.1.1", "1.1.1.1")


def test_ec_distance_matrix_shape():
    labels = pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3"],
            "ec_number": ["1.1.1.1", "1.1.1.2", "2.3.4.5"],
        }
    )
    long = ec_distance_matrix(labels)
    assert len(long) == 3  # 3-choose-2
    assert set(long.columns) == {"a", "b", "ec_dist"}
    assert (long["a"] < long["b"]).all()
    by_pair = long.set_index(["a", "b"])["ec_dist"]
    assert by_pair.loc[("P1", "P2")] == 1
    assert by_pair.loc[("P1", "P3")] == 4
    assert by_pair.loc[("P2", "P3")] == 4


def test_correlate_embedding_distance_with_ec_positive_rho():
    ec_long = pd.DataFrame(
        [
            ("P1", "P2", 1),
            ("P1", "P3", 4),
            ("P1", "P4", 2),
            ("P2", "P3", 4),
            ("P2", "P4", 2),
            ("P3", "P4", 3),
        ],
        columns=["a", "b", "ec_dist"],
    )
    emb_long = ec_long.copy()
    emb_long["dist"] = ec_long["ec_dist"] * 0.5 + np.linspace(0, 0.01, 6)
    emb_long = emb_long[["a", "b", "dist"]]
    result = correlate_embedding_distance_with_ec(emb_long, ec_long)
    assert result["n_pairs"] == 6
    assert result["spearman_rho"] > 0.9


def test_correlate_embedding_distance_with_ec_empty_join():
    ec_long = pd.DataFrame([("P1", "P2", 1)], columns=["a", "b", "ec_dist"])
    emb_long = pd.DataFrame([("P3", "P4", 0.5)], columns=["a", "b", "dist"])
    result = correlate_embedding_distance_with_ec(emb_long, ec_long)
    assert result["n_pairs"] == 0
    assert np.isnan(result["spearman_rho"])
