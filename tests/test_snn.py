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


def test_capped_partner_scores_against_common_cohort(toy_embeddings_a):
    # When one pLM covers fewer proteins (an architecture cap, e.g. esm1b), BOTH sides
    # must rank against the SAME common cohort. Here B is a strict subset of A with
    # IDENTICAL vectors on the overlap -> perfect agreement. The pre-fix code built A's
    # k-NN database from all of A (so ids absent from B entered the union as guaranteed
    # misses) and deflated the Jaccard well below 1.0.
    full = toy_embeddings_a
    capped = {k: full[k] for k in list(full)[:12]}  # 12 of 20, identical vectors
    res = knn_jaccard_between_plms(full, capped, k=4, distance="cosine")
    assert res["mean_jaccard"] == pytest.approx(1.0)
    assert (res["per_query"]["k_a"] == res["per_query"]["k_b"]).all()  # shared cohort -> equal k


def test_compute_ci_false_skips_bootstrap_keeps_per_query(toy_embeddings_a):
    # The analysis-DAG bridge recomputes its own seeded CI from per_query, so it asks
    # knn_jaccard_between_plms to skip the otherwise-discarded B=1000 bootstrap. The
    # mean/per_query must be identical; only the (unused) ci becomes (nan, nan).
    rng = np.random.default_rng(1)
    b = {f"P{i}": rng.normal(size=8).astype(np.float32) for i in range(20)}
    full = knn_jaccard_between_plms(toy_embeddings_a, b, k=5, distance="cosine", rng=0)
    fast = knn_jaccard_between_plms(toy_embeddings_a, b, k=5, distance="cosine",
                                    compute_ci=False)
    assert fast["mean_jaccard"] == full["mean_jaccard"]
    assert list(fast["per_query"]["jaccard"]) == list(full["per_query"]["jaccard"])
    assert np.isnan(fast["ci"][0]) and np.isnan(fast["ci"][1])
    assert np.isfinite(full["ci"][0])  # default path still computes a real CI


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
    # The query set is the intersection (the 10 ids in `smaller`).
    assert len(result["per_query"]) == 10
    # `smaller` carries IDENTICAL vectors to the overlap of the full set, so once both
    # sides rank against the same common cohort (the bug fix), agreement is perfect.
    # (Pre-fix, the full side's k-NN database spanned all 20 ids — including 10 absent
    # from `smaller` — which deflated this below 1.0; that asymmetry was the bug.)
    assert result["mean_jaccard"] == pytest.approx(1.0)


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
