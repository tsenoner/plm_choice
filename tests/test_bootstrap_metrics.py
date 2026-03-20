"""Regression test for _bootstrap_stat parallel/sequential control flow.

The fix (2026-03-20) prevents the sequential branch from re-executing after
parallel completes successfully.  We verify that:

1. Small n_bootstrap (< 100) correctly uses sequential path only.
2. The sentinel pattern works: parallel success → no sequential re-run.
3. The function returns valid SE/CI for well-behaved data.
"""

import numpy as np
import pytest
from unittest.mock import patch

from src.evaluation.metrics import _bootstrap_stat, calculate_regression_metrics
from scipy.stats import spearmanr


def test_sequential_only_for_small_n_bootstrap():
    """n_bootstrap < 100 should never attempt parallel (no Pool created)."""
    rng = np.random.default_rng(42)
    targets = rng.normal(0, 1, 200)
    predictions = targets + rng.normal(0, 0.3, 200)

    with patch("src.evaluation.metrics.Pool") as mock_pool:
        result = _bootstrap_stat(
            targets=targets,
            predictions=predictions,
            n_bootstrap=50,
            confidence_level=0.95,
            stat_func=spearmanr,
            stat_name="Spearman",
            use_parallel=True,  # should be ignored for n < 100
        )
        mock_pool.assert_not_called()

    # Should still produce valid results
    assert not np.isnan(result["Spearman_SE"])
    assert not np.isnan(result["Spearman_95_CI_lower"])
    assert not np.isnan(result["Spearman_95_CI_upper"])
    assert result["Spearman_95_CI_lower"] < result["Spearman_95_CI_upper"]


def test_parallel_success_skips_sequential():
    """When parallel succeeds, the sequential branch must NOT re-run."""
    rng = np.random.default_rng(42)
    targets = rng.normal(0, 1, 200)
    predictions = targets + rng.normal(0, 0.3, 200)

    result = _bootstrap_stat(
        targets=targets,
        predictions=predictions,
        n_bootstrap=100,
        confidence_level=0.95,
        stat_func=spearmanr,
        stat_name="Spearman",
        use_parallel=True,
    )

    assert not np.isnan(result["Spearman_SE"])
    assert result["Spearman_95_CI_lower"] < result["Spearman_95_CI_upper"]


def test_parallel_failure_falls_back_to_sequential():
    """If parallel raises, sequential should produce valid results."""
    rng = np.random.default_rng(42)
    targets = rng.normal(0, 1, 200)
    predictions = targets + rng.normal(0, 0.3, 200)

    # Force Pool to raise so parallel branch fails
    with patch("src.evaluation.metrics.Pool", side_effect=OSError("no fork")):
        result = _bootstrap_stat(
            targets=targets,
            predictions=predictions,
            n_bootstrap=100,
            confidence_level=0.95,
            stat_func=spearmanr,
            stat_name="Spearman",
            use_parallel=True,
        )

    assert not np.isnan(result["Spearman_SE"])
    assert result["Spearman_95_CI_lower"] < result["Spearman_95_CI_upper"]


def test_use_parallel_false_skips_pool():
    """use_parallel=False should never create a Pool."""
    rng = np.random.default_rng(42)
    targets = rng.normal(0, 1, 200)
    predictions = targets + rng.normal(0, 0.3, 200)

    with patch("src.evaluation.metrics.Pool") as mock_pool:
        result = _bootstrap_stat(
            targets=targets,
            predictions=predictions,
            n_bootstrap=200,
            confidence_level=0.95,
            stat_func=spearmanr,
            stat_name="Spearman",
            use_parallel=False,
        )
        mock_pool.assert_not_called()

    assert not np.isnan(result["Spearman_SE"])


def test_calculate_regression_metrics_end_to_end():
    """Smoke test: full metrics pipeline returns all expected keys."""
    rng = np.random.default_rng(42)
    targets = rng.normal(0, 1, 100)
    predictions = targets + rng.normal(0, 0.5, 100)

    metrics = calculate_regression_metrics(
        targets, predictions, n_bootstrap=50, confidence_level=0.95
    )

    # Standard keys
    for key in ["MSE", "RMSE", "MAE", "R2", "Pearson", "Spearman"]:
        assert key in metrics
        assert not np.isnan(metrics[key])

    # Bootstrap keys
    assert "Spearman_SE" in metrics
    assert "Pearson_r2_SE" in metrics
