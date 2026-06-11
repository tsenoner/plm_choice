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
