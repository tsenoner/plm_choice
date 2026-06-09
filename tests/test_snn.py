"""Tests for evaluation.snn (shared-nearest-neighbour Jaccard agreement).

Ported from the SpeciesEmbedding reference (tools/eval/snn.py) into the upstream
layout: import via `from evaluation.snn import ...`, BCa CI via evaluation.stats.
Adds a reproducibility test for the rng plumbing through the public functions
(the seed gate the completion plan moved to Phase 0).
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.snn import knn_jaccard_between_plms, knn_jaccard_matrix


@pytest.fixture
def toy_embeddings_a():
    rng = np.random.default_rng(0)
    return {f"P{i}": rng.normal(size=8).astype(np.float32) for i in range(20)}


def test_identity_jaccard_is_one(toy_embeddings_a):
    result = knn_jaccard_between_plms(
        toy_embeddings_a, toy_embeddings_a, k=5, distance="cosine"
    )
    assert result["mean_jaccard"] == pytest.approx(1.0)
    assert all(result["per_query"]["jaccard"] == 1.0)
    assert result["k"] == 5
    assert result["distance"] == "cosine"


def test_identity_jaccard_with_different_metric(toy_embeddings_a):
    result = knn_jaccard_between_plms(
        toy_embeddings_a, toy_embeddings_a, k=10, distance="euclidean"
    )
    assert result["mean_jaccard"] == pytest.approx(1.0)


def test_jaccard_random_pair_below_one():
    rng = np.random.default_rng(0)
    a = {f"P{i}": rng.normal(size=8).astype(np.float32) for i in range(30)}
    b = {f"P{i}": rng.normal(size=8).astype(np.float32) for i in range(30)}
    result = knn_jaccard_between_plms(a, b, k=5, distance="cosine")
    assert 0.0 <= result["mean_jaccard"] < 1.0
    low, high = result["ci"]
    assert low <= result["mean_jaccard"] <= high


def test_jaccard_ci_reproducible_with_seed():
    """Same seed -> identical bootstrap CI (the Phase-0 reproducibility gate)."""
    rng = np.random.default_rng(1)
    a = {f"P{i}": rng.normal(size=8).astype(np.float32) for i in range(40)}
    b = {f"P{i}": rng.normal(size=8).astype(np.float32) for i in range(40)}
    out1 = knn_jaccard_between_plms(a, b, k=5, distance="cosine", rng=42)
    out2 = knn_jaccard_between_plms(a, b, k=5, distance="cosine", rng=42)
    assert out1["ci"] == out2["ci"]


def test_jaccard_intersection_is_query_set(toy_embeddings_a):
    smaller = {k: v for k, v in list(toy_embeddings_a.items())[:10]}
    result = knn_jaccard_between_plms(
        smaller, toy_embeddings_a, k=3, distance="cosine"
    )
    assert len(result["per_query"]) == 10
    assert result["mean_jaccard"] < 1.0


def test_jaccard_matrix_shape_and_diagonal(toy_embeddings_a):
    embeddings_per_plm = {
        "plm_x": toy_embeddings_a,
        "plm_y": {k: -v for k, v in toy_embeddings_a.items()},
        "plm_z": {k: v[:4] for k, v in toy_embeddings_a.items()},
    }
    matrix = knn_jaccard_matrix(embeddings_per_plm, k=4, distance="cosine")
    assert matrix.shape == (3, 3)
    assert list(matrix.index) == ["plm_x", "plm_y", "plm_z"]
    assert list(matrix.columns) == ["plm_x", "plm_y", "plm_z"]
    assert np.allclose(np.diag(matrix.to_numpy()), 1.0)
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)


def test_jaccard_invalid_distance(toy_embeddings_a):
    with pytest.raises(ValueError, match="distance"):
        knn_jaccard_between_plms(
            toy_embeddings_a, toy_embeddings_a, k=5, distance="hamming"  # type: ignore[arg-type]
        )


def test_jaccard_too_few_common_ids():
    a = {"P1": np.array([1.0, 0.0])}
    b = {"P2": np.array([0.0, 1.0])}
    with pytest.raises(ValueError, match=">=2"):
        knn_jaccard_between_plms(a, b, k=1, distance="cosine")
