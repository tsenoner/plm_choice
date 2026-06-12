"""Unit 4 — the cross-pLM per-cell permutation null (rho / R² only).

Mirrors stats.correlation_permutation_null's symmetric row+column protein-label
permutation, pluggable over the cross-pLM CORRELATION metrics (rho / r2-signed-r):
permute ONE pLM's protein labels (so its matrix stays a valid distance matrix over
relabelled proteins), recompute the metric against the other pLM's fixed matrix -> null ->
two-sided p (1 + #{|null| >= |obs|}) / (n_perm + 1). Lives in cross_plm.py (not stats.py)
per spec §5/§7 option (b).

W₁ has NO permutation null (FIX C1): a symmetric label permutation preserves each matrix's
marginal distance distribution, so every permuted W₁ == observed -> the null is degenerate
(w1_raw p≡1.0, w1_z float-noise). The function RAISES ValueError for the W₁ metrics; W₁ is
a descriptive distance reported with its BCa CI only. Downstream Holm families are built
ONLY over {rho, R²} x {euclidean, cosine, manhattan}, never W₁.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.cross_plm import cross_plm_permutation_null

# the metrics that DO carry a permutation p (W₁ is excluded by design — see FIX C1)
CORR_METRICS = ("rho", "r2")
W1_METRICS = ("w1_raw", "w1_z")


def _scored_matrix(score, noise_scale, rng):
    base = np.abs(score[:, None] - score[None, :]).astype(float)
    noise = rng.normal(scale=noise_scale, size=base.shape)
    m = base + (noise + noise.T) / 2.0
    np.fill_diagonal(m, 0.0)
    return m


def _correlated_pair(n=30, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    a = _scored_matrix(score, noise, rng)
    b = _scored_matrix(score, noise, rng)
    return a, b


def _independent_pair(n=30, seed=0):
    rng = np.random.default_rng(seed)
    a = _scored_matrix(rng.normal(size=n), 0.3, rng)
    b = _scored_matrix(rng.normal(size=n), 0.3, rng)
    return a, b


# ── noise pair -> p high, null centers appropriately ────────────────────────────
def test_rho_noise_pair_p_high_null_centers_near_zero():
    a, b = _independent_pair(n=30, seed=1)
    null, p = cross_plm_permutation_null(a, b, metric="rho", n_perm=200, seed=2)
    assert len(null) == 200
    assert abs(float(np.nanmean(null))) < 0.15  # rho null centers near 0
    assert p > 0.05


def test_r2_noise_pair_p_high():
    a, b = _independent_pair(n=30, seed=3)
    _, p = cross_plm_permutation_null(a, b, metric="r2", n_perm=200, seed=4)
    assert p > 0.05


# ── strong agreement -> p small (rho AND r2) ────────────────────────────────────
def test_rho_strong_agreement_p_small():
    a, b = _correlated_pair(n=30, noise=0.05, seed=5)
    _, p = cross_plm_permutation_null(a, b, metric="rho", n_perm=200, seed=6)
    assert p < 0.05


def test_r2_strong_agreement_p_small():
    a, b = _correlated_pair(n=30, noise=0.05, seed=5)
    _, p = cross_plm_permutation_null(a, b, metric="r2", n_perm=200, seed=6)
    assert p < 0.05


# ── FIX C1: W₁ has NO permutation null — it RAISES (was a vacuous p≡1.0 / float-noise) ──
@pytest.mark.parametrize("metric", W1_METRICS)
def test_w1_permutation_null_raises(metric):
    # A symmetric label permutation preserves each matrix's marginal distance distribution,
    # so every permuted W₁ == observed -> the null is degenerate and the p meaningless.
    # The function must REFUSE to compute it (FIX C1), not bless a vacuous p.
    rng = np.random.default_rng(7)
    n = 30
    score = rng.normal(size=n)
    a = _scored_matrix(score, 0.05, rng)
    b = a * 6.0
    np.fill_diagonal(b, 0.0)
    with pytest.raises(ValueError, match="W₁"):
        cross_plm_permutation_null(a, b, metric=metric, n_perm=200, seed=8)


@pytest.mark.parametrize("metric", W1_METRICS)
def test_w1_permutation_null_raises_even_when_identical(metric):
    # identical pLMs is exactly the degenerate case the old softened test blessed (p≡1.0);
    # it must now raise rather than return a meaningless number.
    a, _ = _correlated_pair(n=30, seed=11)
    b = a.copy()
    with pytest.raises(ValueError):
        cross_plm_permutation_null(a, b, metric=metric, n_perm=200, seed=12)


# ── identical-pLM behavior for rho/r2 (documented) ──────────────────────────────
def test_identical_plm_rho_p_is_floor():
    # identical pLMs -> obs rho = 1 (the max). Permutation breaks agreement so |null| < 1
    # almost always -> #{|null|>=1} ~ 0 -> p ~ floor 1/(n_perm+1). Documented: perfect
    # agreement is maximally significant under the two-sided null.
    a, _ = _correlated_pair(n=30, seed=9)
    b = a.copy()
    _, p = cross_plm_permutation_null(a, b, metric="rho", n_perm=200, seed=10)
    assert p == pytest.approx(1.0 / 201.0, abs=1e-6)


# ── FIX C1: the rho/r2 null is NON-DEGENERATE (it actually moves, spread > 0) ────
@pytest.mark.parametrize("metric", CORR_METRICS)
def test_rho_r2_null_is_non_degenerate(metric):
    # The valid (rho/r2) permutation null must be a real distribution that MOVES — unlike
    # the vacuous W₁ null where every draw == observed. Assert finite spread > 0 and that
    # not all draws collapse onto a single value.
    a, b = _correlated_pair(n=30, noise=0.3, seed=37)
    null, _ = cross_plm_permutation_null(a, b, metric=metric, n_perm=200, seed=39)
    finite = null[np.isfinite(null)]
    assert finite.size > 0
    assert float(np.ptp(finite)) > 1e-6
    assert np.unique(np.round(finite, 9)).size > 1


# ── reproducibility + dispatch ──────────────────────────────────────────────────
@pytest.mark.parametrize("metric", CORR_METRICS)
def test_metric_dispatch_runs(metric):
    a, b = _correlated_pair(n=24, seed=13)
    null, p = cross_plm_permutation_null(a, b, metric=metric, n_perm=100, seed=14)
    assert len(null) == 100
    assert math.isnan(p) or 0.0 <= p <= 1.0


def test_reproducible_under_fixed_seed():
    a, b = _correlated_pair(n=24, seed=15)
    n1, p1 = cross_plm_permutation_null(a, b, metric="rho", n_perm=150, seed=16)
    n2, p2 = cross_plm_permutation_null(a, b, metric="rho", n_perm=150, seed=16)
    assert p1 == p2
    np.testing.assert_array_equal(n1, n2)


def test_unknown_metric_raises():
    a, b = _correlated_pair(n=20, seed=17)
    with pytest.raises(ValueError):
        cross_plm_permutation_null(a, b, metric="nope", n_perm=50, seed=1)
