"""Tests for the cosine/manhattan extension of distance_computation.

The upstream module computed euclidean distance only. The revision needs
cosine and manhattan as well (for the distance-based retrieval comparisons).
We extend it with a pure ``pairwise_distance(a, b, metric)`` function and thread
a ``metric`` selector through ``EmbeddingDistanceComputer`` and the CLI, keeping
euclidean as the default so existing outputs (``dist_<name>`` columns) are
unchanged.
"""

from __future__ import annotations

import h5py
import numpy as np
import polars as pl
import pytest

from data_preparation.distance_computation import (
    EmbeddingDistanceComputer,
    pairwise_distance,
)


# ---------------------------------------------------------------------------
# pairwise_distance — the pure metric kernel
# ---------------------------------------------------------------------------


def test_euclidean_matches_norm():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert pairwise_distance(a, b, "euclidean") == pytest.approx(5.0)


def test_manhattan_matches_sum_of_abs():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 0.0, 3.0])
    # |1-4| + |2-0| + |3-3| = 3 + 2 + 0 = 5
    assert pairwise_distance(a, b, "manhattan") == pytest.approx(5.0)


def test_cosine_identical_is_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert pairwise_distance(a, a.copy(), "cosine") == pytest.approx(0.0, abs=1e-12)


def test_cosine_orthogonal_is_one():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert pairwise_distance(a, b, "cosine") == pytest.approx(1.0)


def test_cosine_opposite_is_two():
    a = np.array([1.0, 1.0])
    b = np.array([-1.0, -1.0])
    assert pairwise_distance(a, b, "cosine") == pytest.approx(2.0)


def test_cosine_zero_vector_is_nan():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 3.0])
    assert np.isnan(pairwise_distance(a, b, "cosine"))


def test_cosine_nonfinite_is_nan():
    """A NaN/inf embedding component must make cosine undefined (nan), not a
    silent 0.0 -- which would rank a corrupt embedding as a perfect match."""
    good = np.array([1.0, 1.0, 1.0])
    assert np.isnan(pairwise_distance(np.array([np.nan, 1.0, 1.0]), good, "cosine"))
    assert np.isnan(pairwise_distance(np.array([np.inf, 1.0, 1.0]), good, "cosine"))
    assert np.isnan(pairwise_distance(good, np.array([np.nan, 1.0, 1.0]), "cosine"))


def test_unknown_metric_raises():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    with pytest.raises(ValueError):
        pairwise_distance(a, b, "mahalanobis")


# ---------------------------------------------------------------------------
# EmbeddingDistanceComputer — metric wired end-to-end
# ---------------------------------------------------------------------------


def _write_h5(path, vectors: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as f:
        for pid, vec in vectors.items():
            f.create_dataset(pid, data=vec)


def _make_computer(tmp_path, metric: str) -> EmbeddingDistanceComputer:
    emb_dir = tmp_path / "embs"
    emb_dir.mkdir()
    _write_h5(
        emb_dir / "modelX.h5",
        {
            "P1": np.array([1.0, 0.0, 0.0]),
            "P2": np.array([0.0, 1.0, 0.0]),  # orthogonal to P1
            "P3": np.array([1.0, 0.0, 0.0]),  # identical to P1
        },
    )
    return EmbeddingDistanceComputer(embeddings_dir=emb_dir, metric=metric)


def test_default_metric_is_euclidean_with_legacy_column_name(tmp_path):
    computer = _make_computer(tmp_path, metric="euclidean")
    df = pl.DataFrame({"query": ["P1"], "target": ["P2"]})
    series = computer.compute_distances_for_embedding(df, "modelX")
    # Backward-compatible column name and value (sqrt(2) for the orthogonal unit pair).
    assert series.name == "dist_modelX"
    assert series.to_list()[0] == pytest.approx(np.sqrt(2.0), abs=1e-4)


def test_cosine_metric_uses_metric_prefixed_column(tmp_path):
    computer = _make_computer(tmp_path, metric="cosine")
    df = pl.DataFrame({"query": ["P1", "P1"], "target": ["P2", "P3"]})
    series = computer.compute_distances_for_embedding(df, "modelX")
    assert series.name == "dist_cosine_modelX"
    vals = series.to_list()
    assert vals[0] == pytest.approx(1.0, abs=1e-4)  # orthogonal
    assert vals[1] == pytest.approx(0.0, abs=1e-4)  # identical


def test_manhattan_metric_uses_metric_prefixed_column(tmp_path):
    computer = _make_computer(tmp_path, metric="manhattan")
    df = pl.DataFrame({"query": ["P1"], "target": ["P2"]})
    series = computer.compute_distances_for_embedding(df, "modelX")
    assert series.name == "dist_manhattan_modelX"
    # |1-0| + |0-1| + |0-0| = 2
    assert series.to_list()[0] == pytest.approx(2.0, abs=1e-4)


def test_missing_protein_yields_nan(tmp_path):
    computer = _make_computer(tmp_path, metric="cosine")
    df = pl.DataFrame({"query": ["P1"], "target": ["MISSING"]})
    series = computer.compute_distances_for_embedding(df, "modelX")
    assert np.isnan(series.to_list()[0])


def test_unknown_metric_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError):
        _make_computer(tmp_path, metric="mahalanobis")
