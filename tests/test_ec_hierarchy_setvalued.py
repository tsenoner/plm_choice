import pytest

from evaluation.ec_hierarchy import ec_distance, ec_distance_set


def test_min_agg_is_share_any_function():
    # A shares 1.1.1.1 with B's first EC -> min distance 0, even though the
    # second members are class-distant.
    A = frozenset({"1.1.1.1", "2.7.11.1"})
    B = frozenset({"1.1.1.1"})
    assert ec_distance_set(A, B, agg="min") == 0


def test_mean_agg_averages_cross_product():
    A = frozenset({"1.1.1.1"})
    B = frozenset({"1.1.1.1", "2.7.11.1"})
    # cross-product distances: d(1.1.1.1,1.1.1.1)=0, d(1.1.1.1,2.7.11.1)=4 -> mean 2.0
    assert ec_distance_set(A, B, agg="mean") == pytest.approx(2.0)


def test_hausdorff_agg_is_max_of_directed_mins():
    A = frozenset({"1.1.1.1", "2.7.11.1"})
    B = frozenset({"1.1.1.2"})
    # directed A->B: min over B for each a = {d(1.1.1.1,1.1.1.2)=1, d(2.7.11.1,1.1.1.2)=4} -> max 4
    # directed B->A: min over A for b=1.1.1.2 = min(1,4)=1
    # hausdorff = max(4, 1) = 4
    assert ec_distance_set(A, B, agg="hausdorff") == 4


def test_scalar_ec_distance_unchanged():
    # back-compat: the scalar path still works.
    assert ec_distance("1.1.1.1", "1.1.1.1") == 0
    assert ec_distance("1.1.1.1", "2.1.1.1") == 4


def test_unknown_agg_raises():
    with pytest.raises(ValueError, match="agg"):
        ec_distance_set(frozenset({"1.1.1.1"}), frozenset({"1.1.1.1"}), agg="median")


def test_empty_set_raises():
    with pytest.raises(ValueError, match="empty"):
        ec_distance_set(frozenset(), frozenset({"1.1.1.1"}), agg="min")


import pandas as pd

from evaluation.ec_hierarchy import (
    correlate_embedding_distance_with_ec,
    ec_distance_matrix_set,
)


def test_ec_distance_matrix_set_is_long_lexicographic():
    labels = pd.DataFrame(
        {
            "protein_id": ["P2", "P1"],
            "ec_set": [frozenset({"1.1.1.1"}), frozenset({"2.7.11.1"})],
        }
    )
    df = ec_distance_matrix_set(labels, agg="min")
    assert list(df.columns) == ["a", "b", "ec_dist"]
    assert len(df) == 1
    row = df.iloc[0]
    assert (row["a"], row["b"]) == ("P1", "P2")  # lexicographic
    assert row["ec_dist"] == 4.0  # 1.x vs 2.x -> class differs


def test_correlate_guards_zero_variance_ec():
    # All EC distances identical -> Spearman is undefined; must not return a bogus rho.
    emb = pd.DataFrame({"a": ["P1", "P1", "P2"], "b": ["P2", "P3", "P3"],
                        "dist": [0.1, 0.2, 0.3]})
    ec = pd.DataFrame({"a": ["P1", "P1", "P2"], "b": ["P2", "P3", "P3"],
                       "ec_dist": [2.0, 2.0, 2.0]})
    out = correlate_embedding_distance_with_ec(emb, ec)
    assert out["spearman_rho"] != out["spearman_rho"]  # NaN
    assert out["n_pairs"] == 3
