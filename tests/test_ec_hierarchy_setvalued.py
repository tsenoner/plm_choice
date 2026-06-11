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
