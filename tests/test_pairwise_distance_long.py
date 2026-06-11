import numpy as np
import pytest

from evaluation.analysis_io import pairwise_distance_long


def _toy():
    # 3 proteins, 2-D vectors; ids deliberately out of lexicographic order.
    return {
        "P2": np.array([0.0, 0.0], dtype=np.float32),
        "P1": np.array([3.0, 4.0], dtype=np.float32),  # euclid 5 from P2
        "P3": np.array([0.0, 1.0], dtype=np.float32),  # euclid 1 from P2
    }


def test_euclidean_pairs_are_lexicographic_unordered_and_correct():
    df = pairwise_distance_long(_toy(), distance="euclidean")
    assert list(df.columns) == ["a", "b", "dist"]
    assert len(df) == 3
    assert (df["a"] < df["b"]).all()
    row = df[(df["a"] == "P1") & (df["b"] == "P2")].iloc[0]
    assert row["dist"] == pytest.approx(5.0)
    row = df[(df["a"] == "P2") & (df["b"] == "P3")].iloc[0]
    assert row["dist"] == pytest.approx(1.0)


def test_cosine_matches_one_minus_cossim():
    emb = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}  # orthogonal
    df = pairwise_distance_long(emb, distance="cosine")
    assert df.iloc[0]["dist"] == pytest.approx(1.0)  # 1 - 0


def test_unknown_distance_raises():
    with pytest.raises(ValueError, match="distance"):
        pairwise_distance_long(_toy(), distance="mahalanobis")


def test_fewer_than_two_proteins_raises():
    with pytest.raises(ValueError, match=">=2"):
        pairwise_distance_long({"only": np.array([1.0])}, distance="euclidean")
