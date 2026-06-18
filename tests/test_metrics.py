"""Tests for src/evaluation/metrics.py bootstrap CIs.

Focus: the Pearson R² confidence interval and bootstrap reproducibility.

Two issues being fixed here:

1. **R² CI via the signed r.** The previous code squared the correlation
   *inside* each bootstrap resample (``_square_transform`` as ``value_transform``)
   and then took percentiles of those r² values. When the true correlation is
   near zero, the resampled r straddles 0 and the r² values pile against the 0
   boundary, so the percentile lower bound is a positive artifact rather than 0.
   The fix bootstraps the signed r and maps the r-CI to an R²-CI with a
   zero-crossing-aware square.
2. **Reproducibility.** The bootstrap RNG was unseeded, so SE/CI changed run to
   run. ``calculate_regression_metrics`` now accepts ``seed=`` and both the
   parallel and sequential paths derive their resamples from it.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from evaluation.metrics import _bootstrap_stat, calculate_regression_metrics


def _linear(n: int, noise: float, seed: int):
    """targets ~ U(0,1); predictions = targets + N(0, noise) -> strong r."""
    rng = np.random.default_rng(seed)
    targets = rng.random(n)
    predictions = targets + rng.normal(0.0, noise, size=n)
    return targets, predictions


def test_bootstrap_reproducible_with_seed():
    """Same seed -> identical R² and Spearman CIs (parallel path, n>=100)."""
    targets, predictions = _linear(200, noise=0.05, seed=7)
    m1 = calculate_regression_metrics(targets, predictions, n_bootstrap=150, seed=42)
    m2 = calculate_regression_metrics(targets, predictions, n_bootstrap=150, seed=42)
    assert m1["Pearson_r2_95_CI_lower"] == m2["Pearson_r2_95_CI_lower"]
    assert m1["Pearson_r2_95_CI_upper"] == m2["Pearson_r2_95_CI_upper"]
    assert m1["Spearman_95_CI_lower"] == m2["Spearman_95_CI_lower"]
    assert m1["Spearman_95_CI_upper"] == m2["Spearman_95_CI_upper"]


def test_r2_ci_strong_correlation_valid_and_brackets_point():
    targets, predictions = _linear(300, noise=0.02, seed=11)
    m = calculate_regression_metrics(targets, predictions, n_bootstrap=200, seed=1)
    r2 = m["Pearson_r2"]
    lo = m["Pearson_r2_95_CI_lower"]
    hi = m["Pearson_r2_95_CI_upper"]
    assert r2 > 0.95
    assert 0.0 <= lo <= r2 <= hi <= 1.0
    assert lo > 0.0  # r-CI does not bracket 0 for a strong correlation


def test_bootstrap_parallel_and_sequential_agree_at_same_seed():
    """Parallel and sequential resampling must produce identical CIs for a fixed
    seed. Otherwise the published CI silently depends on the execution path (and
    on the parallel->sequential fallback that triggers on a transient error)."""
    targets, predictions = _linear(120, noise=0.05, seed=3)
    common = dict(
        n_bootstrap=150,
        confidence_level=0.95,
        stat_func=pearsonr,
        stat_name="Pearson_r2",
        square_after_ci=True,
        seed=42,
    )
    par = _bootstrap_stat(targets, predictions, use_parallel=True, **common)
    seq = _bootstrap_stat(targets, predictions, use_parallel=False, **common)
    assert par["Pearson_r2_95_CI_lower"] == seq["Pearson_r2_95_CI_lower"]
    assert par["Pearson_r2_95_CI_upper"] == seq["Pearson_r2_95_CI_upper"]


def test_r2_ci_near_zero_lower_bound_is_zero():
    """Independent targets/predictions -> r ~ 0. The signed-r map puts the R²
    lower bound at exactly 0 (the boundary the old square-inside code missed)."""
    rng = np.random.default_rng(99)
    targets = rng.normal(size=500)
    predictions = rng.normal(size=500)  # independent of targets
    m = calculate_regression_metrics(targets, predictions, n_bootstrap=300, seed=5)
    assert m["Pearson_r2"] < 0.05
    assert m["Pearson_r2_95_CI_lower"] == 0.0
    assert 0.0 < m["Pearson_r2_95_CI_upper"] <= 1.0
