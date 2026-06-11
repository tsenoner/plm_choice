"""FROZEN HEAD snapshot of ``stats.r2_ci_via_r`` BEFORE the ``_r2_from_r_ci``
extraction (U2 of the cross-pLM arm).

Verbatim copy of ``evaluation.stats.r2_ci_via_r`` and its private dependency
``bca_bootstrap``-call path at the pre-refactor commit, self-contained except for the
shared ``bca_bootstrap`` (which is NOT touched by U2, so importing the live one is safe —
the only thing under test is the inline r-CI -> R²-CI mapping that U2 lifts into
``_r2_from_r_ci``). The differential test asserts the refactored live ``r2_ci_via_r``
returns BYTE-IDENTICAL results to this oracle across a fixture battery, so the extraction
is provably behavior-preserving.

DO NOT EDIT to track future changes — its value is being frozen at the pre-refactor
behavior. If the live mapping is ever *intentionally* changed, delete this file and its
differential test in the same commit.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _scipy_stats

from evaluation.stats import bca_bootstrap


def r2_ci_via_r(
    d: np.ndarray,
    t: np.ndarray,
    B: int = 10_000,
    alpha: float = 0.05,
    rng=None,
) -> dict:
    d_arr = np.asarray(d, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    if d_arr.shape != t_arr.shape:
        raise ValueError("d and t must have the same shape")
    n = int(d_arr.shape[0])
    if n < 2 or np.std(d_arr) == 0 or np.std(t_arr) == 0:
        return {
            "r": float("nan"),
            "r2": float("nan"),
            "r_ci": (float("nan"), float("nan")),
            "r2_ci": (float("nan"), float("nan")),
            "n_pairs": n,
        }

    def _pearson_r(pair: np.ndarray) -> float:
        di = pair[:, 0]
        ti = pair[:, 1]
        if len(di) < 2 or np.std(di) == 0 or np.std(ti) == 0:
            return 0.0
        return float(_scipy_stats.pearsonr(di, ti).statistic)

    r_point, r_lo, r_hi = bca_bootstrap(
        d_arr, statistic=_pearson_r, B=B, alpha=alpha, paired=t_arr, rng=rng
    )

    # ── the FROZEN inline mapping U2 extracts into stats._r2_from_r_ci ──
    r2_hi = max(r_lo * r_lo, r_hi * r_hi)
    if r_lo <= 0.0 <= r_hi:
        r2_lo = 0.0
    else:
        r2_lo = min(r_lo * r_lo, r_hi * r_hi)
    r2_lo = max(0.0, min(1.0, r2_lo))
    r2_hi = max(0.0, min(1.0, r2_hi))

    return {
        "r": float(r_point),
        "r2": float(r_point * r_point),
        "r_ci": (float(r_lo), float(r_hi)),
        "r2_ci": (float(r2_lo), float(r2_hi)),
        "n_pairs": n,
    }
