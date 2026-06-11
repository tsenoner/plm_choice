"""FROZEN HEAD snapshot of the vertex-bootstrap BCa core BEFORE the pluggable
`vertex_bca_ci` extraction (foundation refactor for the orphan + cross-pLM arms).

Verbatim copy of `evaluation.stats.correlation_vertex_bca_ci` /
`correlation_permutation_null` and their private dependencies, self-contained
(imports nothing from `evaluation.stats`). The differential test
(`test_vertex_bca_differential.py`) asserts the refactored live functions return
BYTE-IDENTICAL results to this oracle across a fixture battery, so the extraction
is provably behavior-preserving for the EC arm that consumes it.

DO NOT EDIT to track future changes — its whole value is being frozen at the
pre-refactor commit. If the live behavior is ever *intentionally* changed, delete
this file and its differential test in the same commit.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as _scipy_stats
from scipy.stats import norm as _norm


def _as_rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(_scipy_stats.kendalltau(x, y, variant="b").statistic)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(_scipy_stats.spearmanr(x, y).statistic)


_CORRELATION_KERNELS = {"tau_b": kendall_tau_b, "spearman": spearman_rho}

MIN_BOOTSTRAP_N = 4


def _induced_pair_values(dist_matrix, ec_matrix, idx):
    idx = np.asarray(idx)
    pi, qi = np.triu_indices(idx.size, k=1)
    a = idx[pi]
    b = idx[qi]
    keep = a != b
    a, b = a[keep], b[keep]
    return dist_matrix[a, b], ec_matrix[a, b]


MIN_VERTEX_N = 12
_MIN_VALID_BOOT_FRAC = 0.5
_BCA_DENOM_EPS = 1e-6


def _full_pair_values(dist_matrix, ec_matrix):
    iu, ju = np.triu_indices(dist_matrix.shape[0], k=1)
    return dist_matrix[iu, ju], ec_matrix[iu, ju], iu, ju


def correlation_vertex_bca_ci(
    dist_matrix,
    ec_matrix,
    *,
    statistic: str = "tau_b",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed=42,
):
    stat_fn = _CORRELATION_KERNELS[statistic]
    rng = _as_rng(seed)
    n = int(dist_matrix.shape[0])

    d_all, e_all, _, _ = _full_pair_values(dist_matrix, ec_matrix)
    point = stat_fn(d_all, e_all)
    if n < MIN_VERTEX_N or not math.isfinite(point):
        return float("nan"), float("nan"), float(point), True, False

    boot = np.empty(n_boot, dtype=float)
    valid = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.unique(idx).size < MIN_BOOTSTRAP_N:
            continue
        d, e = _induced_pair_values(dist_matrix, ec_matrix, idx)
        s = stat_fn(d, e)
        if math.isfinite(s):
            boot[valid] = s
            valid += 1
    if valid < int(_MIN_VALID_BOOT_FRAC * n_boot):
        return float("nan"), float("nan"), float(point), True, False
    boot = boot[:valid]

    less = np.count_nonzero(boot < point)
    equal = np.count_nonzero(boot == point)
    prop = (less + 0.5 * equal) / valid
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = _norm.ppf(prop)

    jack = np.empty(n, dtype=float)
    for k in range(n):
        keep = np.arange(n) != k
        sub_d = dist_matrix[np.ix_(keep, keep)]
        sub_e = ec_matrix[np.ix_(keep, keep)]
        du, eu, _, _ = _full_pair_values(sub_d, sub_e)
        jack[k] = stat_fn(du, eu)
    jack = jack[np.isfinite(jack)]
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    denom_collapsed = False

    def _bca_q(q):
        nonlocal denom_collapsed
        zq = _norm.ppf(q)
        denom = 1 - a * (z0 + zq)
        if abs(denom) < _BCA_DENOM_EPS:
            denom_collapsed = True
            return q
        return _norm.cdf(z0 + (z0 + zq) / denom)

    lo_q = _bca_q(alpha / 2)
    hi_q = _bca_q(1 - alpha / 2)
    lo = float(np.quantile(boot, lo_q))
    hi = float(np.quantile(boot, hi_q))
    plo = float(np.quantile(boot, alpha / 2))
    phi = float(np.quantile(boot, 1 - alpha / 2))
    diverged = denom_collapsed or abs(lo - plo) > 0.05 or abs(hi - phi) > 0.05

    if not (math.isfinite(lo) and math.isfinite(hi)):
        return float("nan"), float("nan"), float(point), True, False
    lo = min(max(lo, -1.0), 1.0)
    hi = min(max(hi, -1.0), 1.0)
    return lo, hi, float(point), False, diverged


def correlation_permutation_null(
    dist_matrix,
    ec_matrix,
    *,
    statistic: str = "tau_b",
    n_perm: int = 1000,
    seed=42,
):
    stat_fn = _CORRELATION_KERNELS[statistic]
    rng = _as_rng(seed)
    n = int(dist_matrix.shape[0])
    d_all, e_all, _, _ = _full_pair_values(dist_matrix, ec_matrix)
    obs = stat_fn(d_all, e_all)

    null = np.empty(n_perm, dtype=float)
    iu, ju = np.triu_indices(n, k=1)
    d_fixed = dist_matrix[iu, ju]
    for m in range(n_perm):
        perm = rng.permutation(n)
        e_perm = ec_matrix[np.ix_(perm, perm)]
        null[m] = stat_fn(d_fixed, e_perm[iu, ju])
    finite = null[np.isfinite(null)]
    if not math.isfinite(obs) or finite.size == 0:
        return null, float("nan")
    extreme = np.count_nonzero(np.abs(finite) >= abs(obs))
    p = (1 + extreme) / (finite.size + 1)
    return null, float(p)
