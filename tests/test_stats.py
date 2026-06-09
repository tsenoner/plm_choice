"""Tests for evaluation.stats — the statistical toolkit for the paper revision.

Covers:
* BCa bootstrap CI vs scipy reference.
* Paired bootstrap of the ratio-of-means (the "retention" statistic).
* Holm-Bonferroni against a textbook example.
* BH-FDR rejection rate on uniform p-values.
* Paired Wilcoxon + Cliff's delta on synthetic data with a known effect.
* grid_test end-to-end on a tiny pLM x task grid (full pairwise, two-sided).
* r2_ci_via_r — bootstrap CI for R^2 via the SIGNED r (the B1 fix).

Ported from the SpeciesEmbedding reference (tools/stats/bootstrap.py) into the
upstream conventions: no statsmodels (self-contained Holm/BH), import via
`from evaluation.stats import ...`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from evaluation.stats import (
    bca_bootstrap,
    bh_fdr,
    grid_test,
    holm_bonferroni,
    paired_bootstrap_ratio,
    paired_wilcoxon,
    r2_ci_via_r,
)


# ---------------------------------------------------------------------------
# BCa
# ---------------------------------------------------------------------------


def test_bca_matches_scipy_on_normal_sample():
    rng = np.random.default_rng(42)
    sample = rng.normal(loc=3.0, scale=1.5, size=200)

    point, lo, hi = bca_bootstrap(
        sample,
        statistic=lambda x: float(np.mean(x)),
        B=2000,
        alpha=0.05,
        rng=np.random.default_rng(42),
    )

    ref = scipy_stats.bootstrap(
        (sample,),
        lambda x, axis=-1: np.mean(x, axis=axis),
        n_resamples=2000,
        confidence_level=0.95,
        method="BCa",
        random_state=np.random.default_rng(42),
        vectorized=True,
    )
    assert point == pytest.approx(float(np.mean(sample)))
    assert lo == pytest.approx(ref.confidence_interval.low, abs=0.05)
    assert hi == pytest.approx(ref.confidence_interval.high, abs=0.05)
    assert lo < point < hi


def test_bca_ci_covers_true_mean_at_nominal_rate():
    rng = np.random.default_rng(0)
    covered = 0
    n_trials = 60
    for _ in range(n_trials):
        sample = rng.normal(loc=0.0, scale=1.0, size=80)
        _, lo, hi = bca_bootstrap(
            sample,
            statistic=lambda x: float(np.mean(x)),
            B=500,
            alpha=0.05,
            rng=rng,
        )
        if lo <= 0.0 <= hi:
            covered += 1
    assert covered / n_trials >= 0.80


# ---------------------------------------------------------------------------
# Paired bootstrap ratio
# ---------------------------------------------------------------------------


def test_paired_ratio_of_identical_arrays_is_one():
    rng = np.random.default_rng(7)
    x = rng.uniform(0.5, 1.5, size=120)
    ratio, lo, hi = paired_bootstrap_ratio(x, x, B=1000, rng=rng)
    assert ratio == pytest.approx(1.0, abs=1e-9)
    assert lo == pytest.approx(1.0, abs=1e-9)
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_paired_ratio_of_noisy_pair_close_to_one():
    rng = np.random.default_rng(11)
    base = rng.uniform(0.5, 1.5, size=200)
    noise = rng.normal(0, 0.01, size=200)
    ratio, lo, hi = paired_bootstrap_ratio(base + noise, base, B=1000, rng=rng)
    assert ratio == pytest.approx(1.0, abs=0.05)
    assert lo <= 1.0 <= hi


def test_paired_ratio_recovers_known_ratio():
    rng = np.random.default_rng(19)
    b = rng.uniform(1.0, 2.0, size=300)
    a = 0.8 * b + rng.normal(0, 1e-4, size=b.size)
    ratio, lo, hi = paired_bootstrap_ratio(a, b, B=1000, rng=rng)
    assert ratio == pytest.approx(0.8, abs=1e-3)
    assert lo <= 0.8 <= hi


def test_paired_ratio_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        paired_bootstrap_ratio(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_paired_ratio_rejects_zero_denominator():
    with pytest.raises(ValueError):
        paired_bootstrap_ratio(np.array([1.0, 1.0]), np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------


def test_holm_bonferroni_textbook_example():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    rejected, adj = holm_bonferroni(p, alpha=0.05)
    expected_sorted = np.array([0.020, 0.030, 0.060, 0.060])
    order = np.argsort(p)
    np.testing.assert_allclose(adj[order], expected_sorted, atol=1e-9)
    assert rejected[order].tolist() == [True, True, False, False]


def test_holm_bonferroni_preserves_input_order_and_monotonicity():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, size=50)
    rejected, adj = holm_bonferroni(p)
    assert adj.shape == p.shape
    assert rejected.shape == p.shape
    assert np.all(adj + 1e-12 >= p)
    sorted_adj = adj[np.argsort(p)]
    assert np.all(np.diff(sorted_adj) >= -1e-12)


def test_holm_bonferroni_rejects_bad_inputs():
    with pytest.raises(ValueError):
        holm_bonferroni(np.array([0.1, np.nan]))
    with pytest.raises(ValueError):
        holm_bonferroni(np.array([-0.1, 0.5]))
    with pytest.raises(ValueError):
        holm_bonferroni(np.array([[0.1, 0.2]]))


# ---------------------------------------------------------------------------
# BH-FDR
# ---------------------------------------------------------------------------


def test_bh_fdr_on_uniform_pvalues_rejects_near_zero():
    rng = np.random.default_rng(123)
    n = 2000
    p = rng.uniform(0, 1, size=n)
    rejected, adj = bh_fdr(p, alpha=0.05)
    rejection_rate = rejected.mean()
    assert rejection_rate < 0.10
    assert adj.shape == p.shape


def test_bh_fdr_rejects_strong_signal():
    rng = np.random.default_rng(456)
    p_null = rng.uniform(0, 1, size=900)
    p_signal = rng.beta(0.1, 1.0, size=100)
    p = np.concatenate([p_signal, p_null])
    rejected, _ = bh_fdr(p, alpha=0.05)
    signal_reject_rate = rejected[:100].mean()
    null_reject_rate = rejected[100:].mean()
    assert signal_reject_rate > 0.50
    assert null_reject_rate < 0.10


def test_bh_fdr_monotone():
    rng = np.random.default_rng(7)
    p = rng.uniform(0, 1, size=200)
    _, adj = bh_fdr(p)
    sorted_adj = adj[np.argsort(p)]
    assert np.all(np.diff(sorted_adj) >= -1e-12)


# ---------------------------------------------------------------------------
# Wilcoxon + Cliff's delta
# ---------------------------------------------------------------------------


def test_paired_wilcoxon_known_effect():
    rng = np.random.default_rng(99)
    n = 50
    b = rng.uniform(0, 1, size=n)
    a = b + 0.1
    result = paired_wilcoxon(a, b)
    assert result["p_value"] < 0.001
    assert result["cliffs_delta"] == pytest.approx(1.0)


def test_paired_wilcoxon_no_effect():
    rng = np.random.default_rng(15)
    a = rng.normal(0, 1, size=500)
    b = rng.normal(0, 1, size=500)
    result = paired_wilcoxon(a, b)
    assert result["p_value"] > 0.1
    assert abs(result["cliffs_delta"]) < 0.1


def test_paired_wilcoxon_one_sided_greater():
    # B2/S2: rinit/floor comparisons are one-sided 'greater'. A uniformly
    # larger than B -> one-sided p is half the two-sided p and still tiny.
    rng = np.random.default_rng(77)
    b = rng.uniform(0, 1, size=60)
    a = b + 0.05
    two_sided = paired_wilcoxon(a, b, alternative="two-sided")["p_value"]
    greater = paired_wilcoxon(a, b, alternative="greater")["p_value"]
    assert greater < 0.001
    assert greater == pytest.approx(two_sided / 2.0, rel=1e-6)


def test_paired_wilcoxon_identical_inputs():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = paired_wilcoxon(a, a.copy())
    assert result["p_value"] == 1.0
    assert result["cliffs_delta"] == 0.0
    assert result["statistic"] == 0.0


def test_paired_wilcoxon_mismatched_shapes():
    with pytest.raises(ValueError):
        paired_wilcoxon(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# grid_test
# ---------------------------------------------------------------------------


def _make_grid(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    means = {"good": 0.80, "medium": 0.65, "bad": 0.40}
    for task in ("ss3", "hfsp"):
        for fold in range(30):
            for plm, mu in means.items():
                rows.append(
                    {
                        "pLM": plm,
                        "task": task,
                        "fold": fold,
                        "metric_value": mu + rng.normal(0, 0.01),
                    }
                )
    return pd.DataFrame(rows)


def test_grid_test_holm():
    rng = np.random.default_rng(31415)
    df = _make_grid(rng)
    out = grid_test(df, correction="holm")

    assert len(out) == 12
    assert out["significant"].all()
    good_rows = out[out["pLM_a"] == "good"]
    assert (good_rows["cliffs_delta"] > 0.9).all()
    assert (out["p_adj"] >= out["p_raw"] - 1e-12).all()


def test_grid_test_bh():
    rng = np.random.default_rng(2718)
    df = _make_grid(rng)
    out = grid_test(df, correction="bh")
    assert len(out) == 12
    assert "p_adj" in out.columns
    assert (out["p_adj"] >= out["p_raw"] - 1e-12).all()


def test_grid_test_rejects_unknown_correction():
    rng = np.random.default_rng(0)
    df = _make_grid(rng)
    with pytest.raises(ValueError):
        grid_test(df, correction="bonferroni-naive")


def test_grid_test_missing_column():
    df = pd.DataFrame({"pLM": ["a"], "task": ["t"], "fold": [0]})
    with pytest.raises(ValueError):
        grid_test(df)


# ---------------------------------------------------------------------------
# r2_ci_via_r — bootstrap CI for R^2 via the SIGNED r (B1 fix).
# ---------------------------------------------------------------------------


def _linear_pairs(n, noise, seed):
    """t ~ U(0,1); d = (1 - t) + N(0, noise) -> strong NEGATIVE r, high R^2."""
    rng = np.random.default_rng(seed)
    t = rng.random(n)
    d = (1.0 - t) + rng.normal(0.0, noise, size=n)
    return d, t


def test_r2_ci_via_r_reproducible_with_seed():
    d, t = _linear_pairs(300, noise=0.03, seed=7)
    out1 = r2_ci_via_r(d, t, B=2000, rng=42)
    out2 = r2_ci_via_r(d, t, B=2000, rng=42)
    assert out1["r2_ci"] == out2["r2_ci"]
    assert out1["r2"] == out2["r2"]


def test_r2_ci_via_r_strong_correlation_near_one():
    d, t = _linear_pairs(400, noise=0.01, seed=11)
    out = r2_ci_via_r(d, t, B=2000, rng=1)
    assert out["n_pairs"] == 400
    assert out["r2"] > 0.95
    lo, hi = out["r2_ci"]
    assert 0.0 <= lo <= out["r2"] <= hi <= 1.0


def test_r2_ci_via_r_near_zero_lower_bound_is_zero_not_degenerate():
    rng = np.random.default_rng(99)
    d = rng.normal(size=500)
    t = rng.normal(size=500)
    out = r2_ci_via_r(d, t, B=3000, rng=5)
    assert out["r2"] < 0.05
    lo, hi = out["r2_ci"]
    assert lo == 0.0
    assert 0.0 < hi <= 1.0


def test_r2_ci_via_r_degenerate_inputs_return_nan():
    out = r2_ci_via_r(np.arange(10.0), np.ones(10), B=500, rng=0)
    assert np.isnan(out["r2"])
    out2 = r2_ci_via_r(np.array([1.0]), np.array([2.0]), B=500, rng=0)
    assert out2["n_pairs"] == 1
    assert np.isnan(out2["r2"])
