"""Branch-coverage tests for the GENERIC `vertex_bca_ci` core via synthetic closures.

The differential battery (`test_vertex_bca_differential.py`) locks the EC correlation
wrapper byte-identical, but — as the foundation-refactor adversarial review found — the
correlation cohorts never drive the core's *guard* branches: the clip clamp, the
valid-fraction abort, the distinct-vertex floor, the cohort-floor boundary, the
`divergence_tol` flag, and `validate_point`. Those are exactly the branches the orphan
(``clip=(0,1)``) and cross-pLM (``clip=None``, scale-relative ``divergence_tol``) arms
will lean on. These tests exercise each one directly with controllable closures, and
double as executable documentation of the generic contract.

The synthetic statistic is the simplest valid vertex U-statistic: the mean of a
per-vertex value over the resampled / jackknifed vertices. That isolates the BCa
machinery from any pair-induction detail.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.stats import vertex_bca_ci


def _mean_closures(w: np.ndarray):
    """point / boot / jackknife closures for the vertex-mean of per-vertex values ``w``."""
    w = np.asarray(w, dtype=float)
    n = w.size
    point = float(np.mean(w))

    def boot(idx):
        return float(np.mean(w[idx]))

    def jack(k):
        keep = np.arange(n) != k
        return float(np.mean(w[keep]))

    return point, boot, jack


# ---- clip clamp (orphan uses (0,1); cross-pLM W₁ uses None) --------------------
def test_clip_upper_bound_clamps():
    # Every resample mean lies in [1.05, 1.30] (> 1); with clip=(0,1) BOTH endpoints
    # must clamp to exactly 1.0 — the branch a (0,1)-AUROC caller relies on.
    rng = np.random.default_rng(0)
    w = rng.uniform(1.05, 1.30, size=40)
    point, boot, jack = _mean_closures(w)
    lo, hi, pt, degenerate, _ = vertex_bca_ci(
        40, point=point, boot_statistic=boot, jackknife_statistic=jack,
        n_boot=400, alpha=0.1, seed=1, clip=(0.0, 1.0))
    assert not degenerate
    assert lo == 1.0 and hi == 1.0


def test_clip_lower_bound_clamps():
    rng = np.random.default_rng(0)
    w = rng.uniform(-1.30, -1.05, size=40)  # all < -1
    point, boot, jack = _mean_closures(w)
    lo, hi, _, degenerate, _ = vertex_bca_ci(
        40, point=point, boot_statistic=boot, jackknife_statistic=jack,
        n_boot=400, alpha=0.1, seed=1, clip=(-1.0, 1.0))
    assert not degenerate
    assert lo == -1.0 and hi == -1.0


def test_clip_none_does_not_clamp():
    # The W₁ case: clip=None must leave an out-of-[-1,1] interval intact.
    rng = np.random.default_rng(0)
    w = rng.uniform(1.05, 1.30, size=40)
    point, boot, jack = _mean_closures(w)
    lo, hi, _, degenerate, _ = vertex_bca_ci(
        40, point=point, boot_statistic=boot, jackknife_statistic=jack,
        n_boot=400, alpha=0.1, seed=1, clip=None)
    assert not degenerate
    assert hi > 1.0 and lo > 1.0  # unclamped


# ---- valid-fraction abort -----------------------------------------------------
def test_valid_fraction_abort_when_all_resamples_nan():
    point, _, jack = _mean_closures(np.arange(40, dtype=float))
    lo, hi, pt, degenerate, div = vertex_bca_ci(
        40, point=point, boot_statistic=lambda idx: float("nan"),
        jackknife_statistic=jack, n_boot=200, alpha=0.1, seed=1)
    assert degenerate and math.isnan(lo) and math.isnan(hi)
    assert pt == point and div is False


def test_valid_fraction_boundary_exact():
    # A counter makes EXACTLY `k` resamples finite; with n large the distinct-floor never
    # skips, so the closure is called once per n_boot. valid == int(0.5*n_boot) must PASS
    # (gate is `valid < thresh`); valid == thresh-1 must FAIL. Pins `<` vs `<=`.
    n_boot = 200
    thresh = int(0.5 * n_boot)  # 100
    point, _, jack = _mean_closures(np.arange(40, dtype=float))

    def make_boot(n_finite):
        state = {"i": 0}

        def boot(idx):
            i = state["i"]
            state["i"] += 1
            return point if i < n_finite else float("nan")
        return boot

    _, _, _, deg_at_thresh, _ = vertex_bca_ci(
        40, point=point, boot_statistic=make_boot(thresh), jackknife_statistic=jack,
        n_boot=n_boot, alpha=0.1, seed=1)
    _, _, _, deg_below, _ = vertex_bca_ci(
        40, point=point, boot_statistic=make_boot(thresh - 1), jackknife_statistic=jack,
        n_boot=n_boot, alpha=0.1, seed=1)
    assert deg_at_thresh is False
    assert deg_below is True


# ---- distinct-vertex floor ----------------------------------------------------
def test_distinct_floor_skips_every_resample_when_floor_exceeds_n():
    # min_distinct > n => no resample can reach the floor => every iteration `continue`s
    # => valid 0 => degenerate. Exercises the distinct-floor branch decisively.
    point, boot, jack = _mean_closures(np.arange(30, dtype=float))
    lo, hi, _, degenerate, _ = vertex_bca_ci(
        30, point=point, boot_statistic=boot, jackknife_statistic=jack,
        n_boot=200, alpha=0.1, seed=1, min_distinct=31)
    assert degenerate and math.isnan(lo) and math.isnan(hi)


# ---- cohort-floor boundary ----------------------------------------------------
def test_cohort_floor_boundary_default():
    # n == MIN_VERTEX_N (12) is NOT degenerate by the floor; n == 11 IS. Pins `<`.
    for n, expect_degen in ((11, True), (12, False)):
        point, boot, jack = _mean_closures(np.linspace(0.0, 1.0, n))
        _, _, _, degenerate, _ = vertex_bca_ci(
            n, point=point, boot_statistic=boot, jackknife_statistic=jack,
            n_boot=300, alpha=0.1, seed=2)
        assert degenerate is expect_degen, f"n={n}"


def test_cohort_floor_custom_min_vertices():
    point, boot, jack = _mean_closures(np.linspace(0.0, 1.0, 19))
    _, _, _, degenerate, _ = vertex_bca_ci(
        19, point=point, boot_statistic=boot, jackknife_statistic=jack,
        n_boot=200, alpha=0.1, seed=2, min_vertices=20)
    assert degenerate is True


# ---- divergence_tol -----------------------------------------------------------
def test_divergence_tol_controls_flag():
    # A right-skewed value vector gives a genuine BCa-vs-percentile gap; a tiny tol flags
    # it diverged, a huge tol does not. (No denom collapse here, so the flag is governed
    # purely by the gap-vs-tol comparison.)
    rng = np.random.default_rng(3)
    w = rng.exponential(scale=1.0, size=40)
    point, boot, jack = _mean_closures(w)
    common = dict(point=point, boot_statistic=boot, jackknife_statistic=jack,
                  n_boot=600, alpha=0.1, seed=5, clip=None)
    _, _, _, _, div_tight = vertex_bca_ci(40, divergence_tol=1e-9, **common)
    _, _, _, _, div_loose = vertex_bca_ci(40, divergence_tol=1e9, **common)
    assert div_tight is True
    assert div_loose is False


# ---- validate_point -----------------------------------------------------------
def test_validate_point_raises_on_kernel_mismatch():
    w = np.linspace(0.0, 1.0, 30)
    _, boot, jack = _mean_closures(w)
    wrong_point = float(np.mean(w)) + 0.25  # caller computed it with a different kernel
    with pytest.raises(ValueError, match="validate_point"):
        vertex_bca_ci(30, point=wrong_point, boot_statistic=boot,
                      jackknife_statistic=jack, n_boot=50, seed=1, validate_point=True)


def test_validate_point_passes_when_consistent():
    w = np.linspace(0.0, 1.0, 30)
    point, boot, jack = _mean_closures(w)
    lo, hi, _, degenerate, _ = vertex_bca_ci(
        30, point=point, boot_statistic=boot, jackknife_statistic=jack,
        n_boot=200, alpha=0.1, seed=1, validate_point=True, clip=None)
    assert not degenerate and lo <= point <= hi


# ---- denom-collapse fallback --------------------------------------------------
def test_denom_collapse_falls_back_to_percentile(monkeypatch):
    # The collapse condition |1 - a*(z0+zq)| < _BCA_DENOM_EPS is a measure-zero knife-edge
    # in (a, z0) space — which is exactly why no natural cohort ever hits it. To exercise
    # the FALLBACK BRANCH deterministically we widen _BCA_DENOM_EPS so an ordinary cohort's
    # denominator trips it; the branch logic under test (fall back to the plain percentile
    # quantile, raise percentile_diverged) is identical regardless of the threshold value.
    import evaluation.stats as stats_mod
    monkeypatch.setattr(stats_mod, "_BCA_DENOM_EPS", 2.0)  # > any |1 - a*(z0+zq)| here

    rng = np.random.default_rng(3)
    w = rng.exponential(scale=1.0, size=30)  # skewed -> nonzero a, real z0
    point, boot, jack = _mean_closures(w)
    # divergence_tol=1e9 makes the gap term unreachable, so `diverged` is True ONLY via
    # denom_collapsed -> this asserts the collapse branch specifically.
    lo, hi, _, degenerate, diverged = vertex_bca_ci(
        30, point=point, boot_statistic=boot, jackknife_statistic=jack,
        n_boot=600, alpha=0.1, seed=11, clip=None, divergence_tol=1e9)
    assert not degenerate
    assert diverged is True  # reachable only because the BCa denominator collapsed
    # Fallback returns the plain percentile interval for the collapsed tails.
    assert lo <= hi
