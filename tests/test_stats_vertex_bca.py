import numpy as np
import pytest
from scipy.stats import kendalltau

from evaluation.stats import kendall_tau_b


def test_matches_scipy_variant_b():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = x + rng.normal(size=50) * 0.5
    assert kendall_tau_b(x, y) == pytest.approx(kendalltau(x, y, variant="b").statistic)


def test_perfect_monotone_is_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert kendall_tau_b(x, y) == pytest.approx(1.0)


def test_constant_margin_returns_nan():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(kendall_tau_b(x, y))


from evaluation.stats import _induced_pair_values


def test_induced_pairs_full_index_is_all_upper_triangle():
    dist = np.array([[0.0, 1.0, 2.0],
                     [1.0, 0.0, 3.0],
                     [2.0, 3.0, 0.0]])
    ec = np.array([[0.0, 4.0, 4.0],
                   [4.0, 0.0, 1.0],
                   [4.0, 1.0, 0.0]])
    d, e = _induced_pair_values(dist, ec, np.array([0, 1, 2]))
    assert sorted(d.tolist()) == [1.0, 2.0, 3.0]
    assert sorted(e.tolist()) == [1.0, 4.0, 4.0]


def test_induced_pairs_drops_self_pairs_and_keeps_multiplicity():
    dist = np.array([[0.0, 1.0], [1.0, 0.0]])
    ec = np.array([[0.0, 4.0], [4.0, 0.0]])
    # resample picks index 0 twice and index 1 once: positions (0,1),(0,2),(1,2)
    # idx = [0, 0, 1]: pair(pos0,pos1)=idx(0,0) self -> dropped;
    #   pair(pos0,pos2)=idx(0,1) kept; pair(pos1,pos2)=idx(0,1) kept -> multiplicity 2
    d, e = _induced_pair_values(dist, ec, np.array([0, 0, 1]))
    assert d.tolist() == [1.0, 1.0]
    assert e.tolist() == [4.0, 4.0]
