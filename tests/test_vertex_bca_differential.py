"""Differential test for the pluggable `vertex_bca_ci` foundation refactor.

Locks that the live `evaluation.stats.correlation_vertex_bca_ci` /
`correlation_permutation_null` — after being re-expressed as thin wrappers over the
generic `vertex_bca_ci` core — return results BYTE-IDENTICAL to the frozen
pre-refactor snapshot in `tests/_vertex_bca_head.py`, for the EC arm that consumes
them. "Byte-identical" = exact float equality on finite results (NOT approx), both-NaN
on degenerate results, exact bool equality on the degenerate/diverged flags.

The battery spans both statistics, several cohort shapes (high signal, noisy,
negative, degenerate-by-size, constant-margin), multiple seeds, n_boot and alpha
values — i.e. every branch of the BCa machinery (z0 sign, jackknife accel, denom
collapse, percentile-divergence flag, valid-fraction abort).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import evaluation.stats as live
import tests._vertex_bca_head as head


# ---- cohort fixtures (independent of either implementation) -------------------
def _monotone_matrices(n=40, seed=0):
    rng = np.random.default_rng(seed)
    ec_level = rng.integers(0, 5, size=n).astype(float)
    dist = np.abs(ec_level[:, None] - ec_level[None, :]).astype(float)
    ec = dist.copy()
    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(ec, 0.0)
    return dist, ec


def _noisy_matrices(n=30, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    ec = np.abs(score[:, None] - score[None, :])
    dist = ec + rng.normal(scale=1.0, size=(n, n))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(ec, 0.0)
    return dist, ec


def _negative_matrices(n=24, seed=2):
    dist, ec = _monotone_matrices(n=n, seed=seed)
    dist = dist.max() - dist
    np.fill_diagonal(dist, 0.0)
    return dist, ec


def _constant_margin_matrices(n=40, seed=0):
    dist, ec = _monotone_matrices(n=n, seed=seed)
    ec[:] = 1.0
    np.fill_diagonal(ec, 0.0)
    return dist, ec


def _too_few():
    return np.zeros((3, 3)), np.zeros((3, 3))


# (label, (dist, ec)) — covers every degeneracy + sign branch.
_COHORTS = [
    ("monotone_n40", _monotone_matrices(n=40, seed=0)),
    ("monotone_n13", _monotone_matrices(n=13, seed=4)),
    ("noisy_n30", _noisy_matrices(n=30, seed=3)),
    ("noisy_n25", _noisy_matrices(n=25, seed=9)),
    ("negative_n24", _negative_matrices(n=24, seed=2)),
    ("constant_margin", _constant_margin_matrices(n=40, seed=0)),
    ("too_few_n3", _too_few()),
]


def _assert_ci_identical(label, live_out, head_out):
    lo_l, hi_l, pt_l, deg_l, div_l = live_out
    lo_h, hi_h, pt_h, deg_h, div_h = head_out
    assert deg_l == deg_h, f"[{label}] degenerate flag differs"
    assert div_l == div_h, f"[{label}] diverged flag differs"
    # point estimate: exact equality (or both NaN).
    if math.isnan(pt_h):
        assert math.isnan(pt_l), f"[{label}] point NaN-ness differs"
    else:
        assert pt_l == pt_h, f"[{label}] point differs: {pt_l!r} != {pt_h!r}"
    for name, a, b in (("lo", lo_l, lo_h), ("hi", hi_l, hi_h)):
        if math.isnan(b):
            assert math.isnan(a), f"[{label}] {name} NaN-ness differs"
        else:
            assert a == b, f"[{label}] {name} differs: {a!r} != {b!r}"


@pytest.mark.parametrize("label,cohort", _COHORTS, ids=[c[0] for c in _COHORTS])
@pytest.mark.parametrize("statistic", ["tau_b", "spearman"])
@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("n_boot,alpha", [(200, 0.1), (500, 0.05), (800, 0.1)])
def test_vertex_ci_byte_identical(label, cohort, statistic, seed, n_boot, alpha):
    dist, ec = cohort
    kw = dict(statistic=statistic, n_boot=n_boot, alpha=alpha, seed=seed)
    _assert_ci_identical(
        label,
        live.correlation_vertex_bca_ci(dist, ec, **kw),
        head.correlation_vertex_bca_ci(dist, ec, **kw),
    )


@pytest.mark.parametrize("label,cohort", _COHORTS, ids=[c[0] for c in _COHORTS])
@pytest.mark.parametrize("statistic", ["tau_b", "spearman"])
@pytest.mark.parametrize("seed", [1, 11])
@pytest.mark.parametrize("n_perm", [100, 200])
def test_permutation_null_byte_identical(label, cohort, statistic, seed, n_perm):
    dist, ec = cohort
    kw = dict(statistic=statistic, n_perm=n_perm, seed=seed)
    null_l, p_l = live.correlation_permutation_null(dist, ec, **kw)
    null_h, p_h = head.correlation_permutation_null(dist, ec, **kw)
    # null array: exact element-wise equality with matched NaN positions.
    assert null_l.shape == null_h.shape, f"[{label}] null shape differs"
    np.testing.assert_array_equal(
        np.isnan(null_l), np.isnan(null_h),
        err_msg=f"[{label}] null NaN mask differs",
    )
    finite = ~np.isnan(null_h)
    assert np.array_equal(null_l[finite], null_h[finite]), f"[{label}] null values differ"
    if math.isnan(p_h):
        assert math.isnan(p_l), f"[{label}] p NaN-ness differs"
    else:
        assert p_l == p_h, f"[{label}] p differs: {p_l!r} != {p_h!r}"
