"""Tests for evaluation.stats.bounded_mean_bca_ci — the shared bounded-mean BCa CI.

The bounded-mean BCa CI logic (degenerate-honest, clipped to a bound, reproducible
from a seed) was first written privately as ``recall_fp_report._recall_ci``. It is the
common CI primitive for every absolute mean-of-per-row metric in the analysis DAG
(recall@first-FP, SNN Jaccard, AAC floor recall), so it lives in ``stats`` as one
source of truth. These tests pin the contract the arms rely on.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.stats import bounded_mean_bca_ci


def test_brackets_mean_and_is_reproducible():
    data = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])  # mean 0.5, varies
    lo1, hi1, deg1 = bounded_mean_bca_ci(data, n_boot=500, alpha=0.05, rng=np.random.default_rng(42))
    lo2, hi2, deg2 = bounded_mean_bca_ci(data, n_boot=500, alpha=0.05, rng=np.random.default_rng(42))
    assert (lo1, hi1) == (lo2, hi2)        # same seed -> byte-identical interval
    assert not deg1 and not deg2
    assert 0.0 <= lo1 < hi1 <= 1.0
    assert lo1 <= 0.5 <= hi1


def test_constant_is_zero_width_degenerate_not_nan():
    lo, hi, deg = bounded_mean_bca_ci(np.array([1.0, 1.0, 1.0, 1.0]), n_boot=500, alpha=0.05,
                                      rng=np.random.default_rng(0))
    assert (lo, hi) == (1.0, 1.0) and deg is True


def test_too_few_values_is_nan_degenerate():
    lo, hi, deg = bounded_mean_bca_ci(np.array([0.5]), n_boot=500, alpha=0.05,
                                      rng=np.random.default_rng(0))
    assert math.isnan(lo) and math.isnan(hi) and deg is True


def test_below_min_n_is_degenerate_not_a_minmax_interval():
    # At n=2/3 scipy's BCa degenerates to the data's (min, max) — a coverage-free
    # interval whose width is invariant to alpha. The floor flags it instead of
    # reporting a fake 95% CI. (n=4 is the first non-degenerate size.)
    for arr in ([0.1, 0.9], [0.0, 0.5, 1.0]):
        lo, hi, deg = bounded_mean_bca_ci(np.array(arr), n_boot=500, alpha=0.05,
                                          rng=np.random.default_rng(0))
        assert math.isnan(lo) and math.isnan(hi) and deg is True
    # n=4 with genuine spread is a real interval (the recall/SNN arms rely on this).
    lo, hi, deg = bounded_mean_bca_ci(np.array([0.0, 1.0, 1.0, 0.0]), n_boot=800,
                                      alpha=0.05, rng=np.random.default_rng(0))
    assert not deg and 0.0 <= lo <= hi <= 1.0


def test_near_constant_data_flagged_degenerate_point():
    # Near-constant (not bit-identical) data dodges an exact ptp==0 guard and makes
    # scipy's BCa acceleration jackknife singular -> coverage-free garbage interval. The
    # relative-tolerance guard treats it as a degenerate POINT at the mean instead.
    data = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5000001])
    lo, hi, deg = bounded_mean_bca_ci(data, n_boot=1000, alpha=0.05,
                                      rng=np.random.default_rng(0))
    assert deg is True
    assert lo == hi == pytest.approx(float(data.mean()))


def test_transposed_clip_bounds_raise():
    # A transposed clip (lo > hi) would silently collapse every result to a single
    # value with degenerate=False — reject it loudly instead.
    with pytest.raises(ValueError, match="clip"):
        bounded_mean_bca_ci(np.array([0.1, 0.4, 0.6, 0.9]), n_boot=200, alpha=0.05,
                            rng=np.random.default_rng(0), clip=(1.0, 0.0))


def test_alpha_widens_interval():
    data = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    lo95, hi95, _ = bounded_mean_bca_ci(data, n_boot=1500, alpha=0.05, rng=np.random.default_rng(3))
    lo80, hi80, _ = bounded_mean_bca_ci(data, n_boot=1500, alpha=0.20, rng=np.random.default_rng(3))
    assert (hi95 - lo95) > (hi80 - lo80)


def test_default_clip_keeps_interval_in_unit_range():
    # Heavily skewed near the 0 boundary: BCa can spill below 0; the default clip=(0,1)
    # must pull it back into the bounded statistic's range.
    data = np.array([0.0] * 18 + [1.0, 1.0], dtype=float)  # mostly 0, mean 0.1
    lo, hi, deg = bounded_mean_bca_ci(data, n_boot=2000, alpha=0.05, rng=np.random.default_rng(5))
    assert not deg
    assert lo >= 0.0 and hi <= 1.0


def test_clip_none_allows_out_of_unit_interval():
    # A metric on a different range (e.g. EC Spearman in [-1, 1], or a mean of values
    # >1) must be expressible: clip=None disables the unit-range clamp.
    data = np.array([10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 7.0, 12.0])
    lo, hi, deg = bounded_mean_bca_ci(data, n_boot=1000, alpha=0.05, rng=np.random.default_rng(7),
                                      clip=None)
    assert not deg
    assert hi > 1.0  # not clamped to the unit range
    assert lo <= float(np.mean(data)) <= hi


def test_custom_clip_bounds_respected():
    data = np.array([-0.9, 0.8, -0.7, 0.95, -0.85, 0.9, -0.6, 0.99])  # spread within [-1, 1]
    lo, hi, _ = bounded_mean_bca_ci(data, n_boot=1500, alpha=0.05, rng=np.random.default_rng(9),
                                    clip=(-1.0, 1.0))
    assert lo >= -1.0 and hi <= 1.0


def test_realistic_n_is_finite_and_bounded():
    # Production cells are ~283-319 queries; confirm BCa yields a finite, ordered,
    # in-bounds interval at realistic n (the small-n tests don't prove the
    # jackknife/accelerator is well-behaved at scale). This is the SHARED primitive
    # every arm leans on, so the production-scale guard belongs here.
    data = (np.random.default_rng(0).random(283) < 0.5).astype(float)
    lo, hi, deg = bounded_mean_bca_ci(data, n_boot=2000, alpha=0.05, rng=np.random.default_rng(1))
    assert not deg
    assert 0.0 <= lo < hi <= 1.0


def test_recall_ci_still_delegates_and_matches():
    # recall_fp_report._recall_ci must remain a thin wrapper over the shared helper
    # (clip=(0,1)) so the recall arm and the helper can never drift apart.
    from evaluation.recall_fp_report import _recall_ci

    data = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    a = _recall_ci(data, n_boot=600, alpha=0.05, rng=np.random.default_rng(11))
    b = bounded_mean_bca_ci(data, n_boot=600, alpha=0.05, rng=np.random.default_rng(11), clip=(0.0, 1.0))
    assert a == b
