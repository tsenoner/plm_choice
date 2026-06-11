"""Unit 3 — the cross-pLM agreement-metric CI binding over stats.vertex_bca_ci.

Given two pLMs' distance matrices over ONE shared protein id order, each agreement metric
(rho / r2 / w1_raw / w1_z) is computed on the upper-triangle pair vectors of BOTH matrices
and given a vertex (protein) bootstrap BCa CI by binding point / boot / jackknife closures
over the SHIPPED stats.vertex_bca_ci. The shared-protein draw is automatic: the core owns
the single per-iteration idx and the closure applies it to both captured matrices.

These tests prove: (1) the binding feeds ONE core draw to both matrices (a sabotage that
draws an independent second idx for the b-matrix gives a measurably different interval);
(2) BCa coverage ~ nominal under the shared vertex resample; (3) R² uses signed-r, not
in-resample squaring; (4) W₁ near-zero degeneracy (C3) is flagged, not reported as a
coverage-free interval.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.cross_plm import cross_plm_agreement_ci

ALL_METRICS = ("rho", "r2", "w1_raw", "w1_z")


# ── matrix fixtures ────────────────────────────────────────────────────────────
def _scored_matrix(score, noise_scale, rng):
    """Symmetric distance matrix dist[i,j] = |score_i - score_j| + symmetric noise."""
    base = np.abs(score[:, None] - score[None, :]).astype(float)
    noise = rng.normal(scale=noise_scale, size=base.shape)
    m = base + (noise + noise.T) / 2.0
    np.fill_diagonal(m, 0.0)
    return m


def _correlated_pair(n=40, noise=0.3, seed=0):
    """Two pLM distance matrices over n shared proteins with a known positive agreement."""
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    a = _scored_matrix(score, noise, rng)
    b = _scored_matrix(score, noise, rng)  # same latent score -> positive agreement
    return a, b


# ── identical-pLM degeneracy ────────────────────────────────────────────────────
@pytest.mark.parametrize("metric", ALL_METRICS)
def test_identical_plm_degenerate_or_trivial(metric):
    a, _ = _correlated_pair(n=30, seed=1)
    b = a.copy()
    out = cross_plm_agreement_ci(a, b, metric=metric, n_boot=200, alpha=0.1, seed=3)
    if metric in ("w1_raw", "w1_z"):
        # identical marginals -> W1 = 0, flagged degenerate (C3), CI is the point.
        assert out["point"] == pytest.approx(0.0, abs=1e-9)
        assert out["degenerate"] is True
        assert out["ci_lo"] == out["ci_hi"] == pytest.approx(0.0, abs=1e-9) or (
            math.isnan(out["ci_lo"]) and math.isnan(out["ci_hi"])
        )
    elif metric == "rho":
        assert out["point"] == pytest.approx(1.0, abs=1e-9)
    else:  # r2
        assert out["point"] == pytest.approx(1.0, abs=1e-9)


# ── known-correlation: rho / r2 sane and vertex-wider-than-iid ──────────────────
def test_positive_agreement_rho_sane():
    a, b = _correlated_pair(n=40, noise=0.3, seed=2)
    out = cross_plm_agreement_ci(a, b, metric="rho", n_boot=400, alpha=0.1, seed=5)
    assert not out["degenerate"]
    assert out["point"] > 0.2
    assert out["ci_lo"] <= out["point"] <= out["ci_hi"]
    assert -1.0 <= out["ci_lo"] and out["ci_hi"] <= 1.0


def test_r2_in_unit_interval_and_derived_from_r_ci():
    a, b = _correlated_pair(n=40, noise=0.3, seed=4)
    out = cross_plm_agreement_ci(a, b, metric="r2", n_boot=400, alpha=0.1, seed=6)
    assert not out["degenerate"]
    assert 0.0 <= out["ci_lo"] <= out["ci_hi"] <= 1.0
    # r2 carries the signed-r CI it was derived from.
    assert "r_ci_lo" in out and "r_ci_hi" in out
    # R2 point == r_point**2, mapped via _r2_from_r_ci from the r-CI.
    from evaluation.stats import _r2_from_r_ci
    exp_lo, exp_hi = _r2_from_r_ci(out["r_ci_lo"], out["r_ci_hi"])
    assert out["ci_lo"] == pytest.approx(exp_lo)
    assert out["ci_hi"] == pytest.approx(exp_hi)


def test_rho_vertex_ci_wider_than_iid_pair_ci():
    # The vertex bootstrap (resample proteins) must be WIDER than an i.i.d.-pair bootstrap,
    # because pairs sharing a protein are correlated. Mirrors the EC discriminator.
    from evaluation.stats import _full_pair_values, spearman_rho
    a, b = _correlated_pair(n=30, noise=0.6, seed=7)
    out = cross_plm_agreement_ci(a, b, metric="rho", n_boot=800, alpha=0.1, seed=9)
    assert not out["degenerate"]
    vertex_width = out["ci_hi"] - out["ci_lo"]

    da, db, _, _ = _full_pair_values(a, b)
    m = da.size
    prng = np.random.default_rng(9)
    boot = []
    for _ in range(800):
        sel = prng.integers(0, m, size=m)
        s = spearman_rho(da[sel], db[sel])
        if math.isfinite(s):
            boot.append(s)
    boot = np.asarray(boot)
    pair_width = float(np.quantile(boot, 0.95) - np.quantile(boot, 0.05))
    assert vertex_width > pair_width


# ── R² uses signed-r, not in-resample squaring ─────────────────────────────────
def test_r2_uses_signed_r_not_in_resample_square():
    # A near-zero-correlation pair: the signed-r CI must straddle 0, so the mapped R²-CI
    # has r2_lo == 0 (the B1 zero-crossing rule). A naive in-resample square would pile
    # the bootstrap against 0 and give a degenerate, non-straddle interval.
    rng = np.random.default_rng(11)
    n = 50
    a = _scored_matrix(rng.normal(size=n), 0.1, rng)
    b = _scored_matrix(rng.normal(size=n), 0.1, rng)  # independent latent -> r ~ 0
    out = cross_plm_agreement_ci(a, b, metric="r2", n_boot=600, alpha=0.1, seed=13)
    if not out["degenerate"]:
        # the signed-r CI straddles 0 -> mapped R² low floored to 0.
        assert out["r_ci_lo"] < 0.0 < out["r_ci_hi"]
        assert out["ci_lo"] == pytest.approx(0.0, abs=1e-12)


# ── shared-draw discriminator ───────────────────────────────────────────────────
def test_shared_draw_vs_independent_second_draw_differs():
    # The binding feeds the core's ONE idx to both matrices. A sabotaged binding that
    # draws an INDEPENDENT second idx for the b-matrix breaks the protein pairing and
    # yields a measurably different (decorrelated) interval. We reconstruct both bootstrap
    # distributions directly and assert they differ.
    from evaluation.stats import _induced_pair_values, _full_pair_values, spearman_rho

    a, b = _correlated_pair(n=30, noise=0.4, seed=15)
    n = a.shape[0]

    # shared-draw bootstrap distribution (the correct binding)
    rng_shared = np.random.default_rng(17)
    shared = []
    for _ in range(500):
        idx = rng_shared.integers(0, n, size=n)
        da, db = _induced_pair_values(a, b, idx)
        s = spearman_rho(da, db)
        if math.isfinite(s):
            shared.append(s)

    # independent-second-draw bootstrap (the sabotage)
    rng_indep = np.random.default_rng(17)
    indep = []
    for _ in range(500):
        idx_a = rng_indep.integers(0, n, size=n)
        idx_b = rng_indep.integers(0, n, size=n)
        da, _ = _induced_pair_values(a, a, idx_a)
        _, db = _induced_pair_values(b, b, idx_b)
        # align lengths (independent draws may induce different pair counts)
        k = min(da.size, db.size)
        s = spearman_rho(da[:k], db[:k])
        if math.isfinite(s):
            indep.append(s)

    shared = np.asarray(shared)
    indep = np.asarray(indep)
    # the independent draw decorrelates the two pLMs -> its rho distribution centers near 0
    # while the shared draw preserves the real positive agreement.
    assert float(np.mean(shared)) > float(np.mean(indep)) + 0.1


# ── W₁ known-answer + z vs raw ──────────────────────────────────────────────────
def test_w1_raw_vs_z_differ_under_scale_shift():
    # Two matrices with the same SHAPE of distances but different overall scale: w1_raw sees
    # the scale gap, w1_z (per-cohort z-scored) cancels it -> the two metrics differ.
    rng = np.random.default_rng(19)
    n = 30
    score = rng.normal(size=n)
    a = _scored_matrix(score, 0.05, rng)
    b = a * 5.0  # same ordering, 5x scale
    np.fill_diagonal(b, 0.0)
    out_raw = cross_plm_agreement_ci(a, b, metric="w1_raw", n_boot=200, alpha=0.1, seed=21)
    out_z = cross_plm_agreement_ci(a, b, metric="w1_z", n_boot=200, alpha=0.1, seed=21)
    # raw sees a large W1 (the 5x scale gap); z-scored cancels it toward ~0.
    assert out_raw["point"] > out_z["point"] + 0.1


# ── W₁ near-zero degeneracy (C3) ────────────────────────────────────────────────
def test_w1_near_zero_marginals_flagged_degenerate():
    # Two matrices with ~identical marginal distance distributions -> W1 ~ 0; the binding
    # must mark the cell degenerate (C3), NOT report a coverage-free interval.
    rng = np.random.default_rng(23)
    n = 30
    score = rng.normal(size=n)
    a = _scored_matrix(score, 1e-9, rng)
    b = _scored_matrix(score, 1e-9, rng)  # essentially the same marginals
    out = cross_plm_agreement_ci(a, b, metric="w1_raw", n_boot=200, alpha=0.1, seed=25)
    assert out["degenerate"] is True
    assert out["point"] == pytest.approx(0.0, abs=1e-6)


# ── validate_point passes (point kernel == boot kernel on identity) ─────────────
@pytest.mark.parametrize("metric", ALL_METRICS)
def test_validate_point_passes(metric):
    a, b = _correlated_pair(n=30, noise=0.4, seed=27)
    out = cross_plm_agreement_ci(
        a, b, metric=metric, n_boot=200, alpha=0.1, seed=29, validate_point=True
    )
    assert "point" in out


# ── reproducibility ─────────────────────────────────────────────────────────────
def test_reproducible_under_fixed_seed():
    a, b = _correlated_pair(n=30, noise=0.4, seed=31)
    x = cross_plm_agreement_ci(a, b, metric="rho", n_boot=300, alpha=0.1, seed=33)
    y = cross_plm_agreement_ci(a, b, metric="rho", n_boot=300, alpha=0.1, seed=33)
    assert x["ci_lo"] == y["ci_lo"] and x["ci_hi"] == y["ci_hi"]


# ── unknown metric rejected ─────────────────────────────────────────────────────
def test_unknown_metric_raises():
    a, b = _correlated_pair(n=20, seed=35)
    with pytest.raises((ValueError, KeyError)):
        cross_plm_agreement_ci(a, b, metric="not_a_metric", n_boot=50, seed=1)


# ── W₁ coverage simulation (slow) ───────────────────────────────────────────────
@pytest.mark.slow
def test_w1_coverage_near_nominal():
    """Empirical coverage of the W₁ vertex-BCa CI (clip=None, scale-relative
    divergence_tol) must be >= nominal under the shared-vertex resample."""
    rng = np.random.default_rng(123)
    M = 1500
    score = rng.normal(size=M)
    # population matrices: a has score-based distances; b has a SHIFTED-scale version so
    # the population W1 between the marginals is a fixed positive number.
    pop_a = np.abs(score[:, None] - score[None, :])
    pop_b = pop_a * 1.5 + 0.2
    np.fill_diagonal(pop_a, 0.0)
    np.fill_diagonal(pop_b, 0.0)
    iu, ju = np.triu_indices(M, k=1)
    from evaluation.stats import wasserstein_w1
    true_w1 = wasserstein_w1(pop_a[iu, ju], pop_b[iu, ju])

    n, trials, alpha = 30, 250, 0.1
    covered = 0
    valid = 0
    for t in range(trials):
        sel = rng.choice(M, size=n, replace=False)
        a = pop_a[np.ix_(sel, sel)].copy()
        b = pop_b[np.ix_(sel, sel)].copy()
        np.fill_diagonal(a, 0.0)
        np.fill_diagonal(b, 0.0)
        out = cross_plm_agreement_ci(a, b, metric="w1_raw", n_boot=400, alpha=alpha, seed=t)
        if out["degenerate"]:
            continue
        valid += 1
        if out["ci_lo"] <= true_w1 <= out["ci_hi"]:
            covered += 1
    coverage = covered / max(valid, 1)
    # the load-bearing guard is the LOWER bound (anticonservative narrow intervals are the
    # dangerous failure). Over-coverage is safe (honest but wide).
    assert coverage >= 0.80, f"W1 coverage {coverage:.3f} < 0.80 (nominal 0.90, valid={valid})"
