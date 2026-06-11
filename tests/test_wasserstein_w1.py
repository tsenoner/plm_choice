"""Unit 1 — the single W₁ (Wasserstein-1) owner in ``evaluation.stats``.

``stats.wasserstein_w1`` is the one earth-mover-distance implementation, wrapping
``scipy.stats.wasserstein_distance``. The ``pdb_tm_bias`` arm's inlined
``wasserstein_distance`` call is swapped to it — differential-tested byte-identical
on fixtures (the swap must not move pdb-TM's numbers, B4-gated though it is).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from evaluation.stats import wasserstein_w1


# ── known-answer / hand distributions ─────────────────────────────────────────
def test_known_answer_shifted_point_mass():
    # All mass of y is shifted +3 from x -> the earth-mover cost is exactly 3.
    x = np.array([0.0, 0.0, 0.0, 0.0])
    y = np.array([3.0, 3.0, 3.0, 3.0])
    assert wasserstein_w1(x, y) == pytest.approx(3.0)


def test_known_answer_uniform_shift():
    # Two equal-size uniform samples, y = x + 2 -> W1 = 2.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x + 2.0
    assert wasserstein_w1(x, y) == pytest.approx(2.0)


def test_matches_scipy_on_random():
    rng = np.random.default_rng(0)
    x = rng.normal(size=37)
    y = rng.normal(loc=0.7, scale=1.4, size=51)  # different size, allowed for W1
    assert wasserstein_w1(x, y) == pytest.approx(
        wasserstein_distance(x, y), abs=1e-12
    )


# ── symmetry ──────────────────────────────────────────────────────────────────
def test_symmetric():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20)
    y = rng.normal(loc=1.0, size=25)
    assert wasserstein_w1(x, y) == pytest.approx(wasserstein_w1(y, x), abs=1e-12)


# ── degenerate inputs -> 0.0 (explicit contract) ──────────────────────────────
def test_identical_distributions_zero():
    x = np.array([1.0, 2.0, 3.0])
    assert wasserstein_w1(x, x.copy()) == 0.0


def test_all_equal_both_sides_zero():
    x = np.array([5.0, 5.0, 5.0])
    y = np.array([5.0, 5.0])
    assert wasserstein_w1(x, y) == 0.0


def test_empty_both_sides_zero():
    assert wasserstein_w1(np.array([]), np.array([])) == 0.0


def test_one_side_empty_returns_nan():
    # An empty side vs a non-empty side has no defined transport -> NaN (documented).
    assert np.isnan(wasserstein_w1(np.array([]), np.array([1.0, 2.0])))
    assert np.isnan(wasserstein_w1(np.array([1.0, 2.0]), np.array([])))
