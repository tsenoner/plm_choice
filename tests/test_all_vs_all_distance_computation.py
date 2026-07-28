"""Tests for the exact N x N all-vs-all distance table.

This module is the canonical all-vs-all implementation (F12): it produces the
multi-model ``all_vs_all_distances.parquet`` plus the visualization caches that
``docs/ALL_VS_ALL_ANALYSIS.md`` documents.

The assertions below deliberately check distance *values*, not just shapes. The
previous (untracked) test only checked row counts and column names, so it would
have passed even if the distance kernel returned nonsense.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import polars as pl
import pytest

from data_preparation.all_vs_all_distance_computation import AllVsAllEmbeddingAnalyzer

# Five proteins positioned so that several pairwise Euclidean distances are
# exact small integers, hand-checkable without running anything.
KNOWN = {
    "P00001": [0.0, 0.0, 0.0],
    "P00002": [3.0, 0.0, 0.0],
    "P00003": [0.0, 4.0, 0.0],
    "P00004": [0.0, 0.0, 12.0],
    "P00005": [6.0, 0.0, 0.0],
}

#: Hand-verified exact values (3-4-5 triangle plus two axis distances).
EXPECTED_EXACT = {
    ("P00001", "P00002"): 3.0,  # along x
    ("P00001", "P00003"): 4.0,  # along y
    ("P00002", "P00003"): 5.0,  # 3-4-5
    ("P00001", "P00004"): 12.0,  # along z
    ("P00001", "P00005"): 6.0,  # along x
    ("P00002", "P00005"): 3.0,  # 6 - 3 along x
}


def _write_h5(path, vectors):
    with h5py.File(path, "w") as f:
        for pid, vec in vectors.items():
            f.create_dataset(pid, data=np.asarray(vec, dtype=np.float32))


@pytest.fixture
def embeddings_dir(tmp_path):
    """Two embedding sets over the same proteins: one exact, one scaled 2x."""
    d = tmp_path / "embeddings"
    d.mkdir()
    _write_h5(d / "modela.h5", KNOWN)
    _write_h5(d / "modelb.h5", {k: [2 * c for c in v] for k, v in KNOWN.items()})
    return d


@pytest.fixture
def analyzer(embeddings_dir, tmp_path):
    return AllVsAllEmbeddingAnalyzer(
        embeddings_dir=embeddings_dir,
        output_dir=tmp_path / "out",
        chunk_size=2,  # force the chunked path, not a single shot
        precision=6,
    )


def test_discovers_both_embedding_sets_and_the_shared_protein_universe(analyzer):
    assert set(analyzer.embedding_info) == {"modela", "modelb"}
    assert sorted(analyzer.protein_universe) == sorted(KNOWN)


def test_table_is_the_full_square_including_the_diagonal(analyzer):
    df = analyzer.compute_all_vs_all_distances()
    n = len(KNOWN)
    assert len(df) == n * n
    assert {"query", "target", "dist_modela", "dist_modelb"} <= set(df.columns)


def test_distance_values_match_hand_checked_constants(analyzer):
    """The load-bearing assertion: real numbers, not just a well-shaped table."""
    df = analyzer.compute_all_vs_all_distances()
    lookup = {
        (row["query"], row["target"]): row["dist_modela"]
        for row in df.iter_rows(named=True)
    }
    for (a, b), expected in EXPECTED_EXACT.items():
        assert lookup[(a, b)] == pytest.approx(expected, abs=1e-4), f"{a}-{b}"


def test_every_distance_matches_an_independent_recomputation(analyzer):
    """Cross-check all 25 cells against numpy, i.e. a different code path."""
    df = analyzer.compute_all_vs_all_distances()
    vectors = {k: np.asarray(v, dtype=np.float64) for k, v in KNOWN.items()}
    for row in df.iter_rows(named=True):
        expected = float(
            np.linalg.norm(vectors[row["query"]] - vectors[row["target"]])
        )
        assert row["dist_modela"] == pytest.approx(expected, abs=1e-4), (
            f"{row['query']}-{row['target']}"
        )


def test_self_distance_is_zero(analyzer):
    df = analyzer.compute_all_vs_all_distances()
    diagonal = df.filter(pl.col("query") == pl.col("target"))
    assert len(diagonal) == len(KNOWN)
    assert diagonal["dist_modela"].abs().max() == pytest.approx(0.0, abs=1e-6)


def test_distances_are_symmetric(analyzer):
    df = analyzer.compute_all_vs_all_distances()
    lookup = {
        (row["query"], row["target"]): row["dist_modela"]
        for row in df.iter_rows(named=True)
    }
    for (a, b), value in lookup.items():
        assert value == pytest.approx(lookup[(b, a)], abs=1e-6), f"{a}-{b} asymmetric"


def test_scaling_an_embedding_scales_its_distances(analyzer):
    """modelb is modela x2, so every distance must double — this catches a
    silent mix-up of the per-embedding columns."""
    df = analyzer.compute_all_vs_all_distances()
    for row in df.iter_rows(named=True):
        assert row["dist_modelb"] == pytest.approx(2 * row["dist_modela"], abs=1e-4)


def test_max_proteins_truncates_deterministically(embeddings_dir, tmp_path):
    first = AllVsAllEmbeddingAnalyzer(
        embeddings_dir=embeddings_dir, output_dir=tmp_path / "a", max_proteins=3
    )
    second = AllVsAllEmbeddingAnalyzer(
        embeddings_dir=embeddings_dir, output_dir=tmp_path / "b", max_proteins=3
    )
    assert len(first.protein_universe) == 3
    assert first.protein_universe == second.protein_universe
    assert len(first.compute_all_vs_all_distances()) == 9


def test_complete_analysis_writes_parquet_and_every_cache(analyzer):
    outputs = analyzer.run_complete_analysis()
    parquet = analyzer.output_dir / "all_vs_all_distances.parquet"
    assert parquet.is_file()
    assert len(pl.read_parquet(parquet)) == len(KNOWN) ** 2

    expected_caches = {
        "hexbin_data.json",
        "correlation_data.json",
        "wasserstein_data.json",
        "distribution_data.json",
        "distribution_normalized_data.json",
    }
    written = {p.name for p in analyzer.cache_dir.glob("*.json")}
    assert expected_caches <= written, f"missing: {expected_caches - written}"
    assert outputs


def test_normalized_distribution_cache_uses_a_500_point_grid(analyzer):
    """Pins the density grid size.

    The published ridge figure was rendered from a 200-point grid that no
    surviving code path emits, which is why that figure is not reproducible.
    Pinning the value here stops the grid drifting again unnoticed.
    """
    analyzer.run_complete_analysis()
    cache = analyzer.cache_dir / "distribution_normalized_data.json"
    data = json.loads(cache.read_text())

    assert data["metadata"]["normalized"] is True
    distributions = data["distributions"]
    assert set(distributions) == {"dist_modela", "dist_modelb"}
    for name, entry in distributions.items():
        assert len(entry["x_range"]) == 500, name
        assert len(entry["density"]) == 500, name
