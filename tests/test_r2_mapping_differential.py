"""Differential test for the U2 ``_r2_from_r_ci`` extraction.

U2 lifts the zero-crossing-aware ``r_ci -> r2_ci`` mapping that was INLINE in
``stats.r2_ci_via_r`` (stats.py:253-260) into a private ``stats._r2_from_r_ci(r_lo, r_hi)``
and has ``r2_ci_via_r`` call it. This is the seam the cross-pLM R² path reuses so there is
exactly ONE B1 zero-crossing rule. The extraction must be behavior-preserving:

* ``_r2_from_r_ci`` reproduces the frozen inline expression on an r-CI fixture battery
  (r-CI straddling 0, all-positive, all-negative, boundary spill);
* the live ``r2_ci_via_r`` is byte-identical to the frozen-HEAD oracle end-to-end.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import evaluation.stats as live
import tests._r2_ci_head as head


# ── the extracted mapping reproduces the frozen inline expression directly ─────
def _frozen_inline_mapping(r_lo, r_hi):
    """The exact pre-extraction inline block (stats.py:253-260), verbatim."""
    r2_hi = max(r_lo * r_lo, r_hi * r_hi)
    if r_lo <= 0.0 <= r_hi:
        r2_lo = 0.0
    else:
        r2_lo = min(r_lo * r_lo, r_hi * r_hi)
    r2_lo = max(0.0, min(1.0, r2_lo))
    r2_hi = max(0.0, min(1.0, r2_hi))
    return r2_lo, r2_hi


# r-CI fixtures spanning every branch: straddling 0, all-positive, all-negative,
# boundary spill past ±1, a zero-width interval, and exact-0 endpoints.
_R_CI_FIXTURES = [
    (-0.3, 0.4),     # straddles 0 -> r2_lo must be 0
    (-0.05, 0.9),    # straddles 0
    (0.2, 0.8),      # all-positive
    (0.55, 0.55),    # zero-width positive
    (-0.9, -0.2),    # all-negative -> r2_lo = min(0.81, 0.04) = 0.04
    (-0.99, -0.1),   # all-negative
    (-1.2, 0.3),     # spill below -1, straddles 0
    (0.4, 1.3),      # spill above 1, all-positive
    (0.0, 0.6),      # left endpoint exactly 0 (the <= boundary)
    (-0.6, 0.0),     # right endpoint exactly 0 (the <= boundary)
    (0.0, 0.0),      # degenerate at 0
]


@pytest.mark.parametrize("r_lo,r_hi", _R_CI_FIXTURES)
def test_r2_from_r_ci_matches_frozen_inline(r_lo, r_hi):
    got = live._r2_from_r_ci(r_lo, r_hi)
    expected = _frozen_inline_mapping(r_lo, r_hi)
    assert got == expected, f"({r_lo},{r_hi}): {got!r} != {expected!r}"


def test_r2_from_r_ci_straddle_zero_floors_low_to_zero():
    lo, hi = live._r2_from_r_ci(-0.3, 0.4)
    assert lo == 0.0


def test_r2_from_r_ci_all_negative_uses_min_square():
    lo, hi = live._r2_from_r_ci(-0.9, -0.2)
    assert lo == pytest.approx(0.04)
    assert hi == pytest.approx(0.81)


def test_r2_from_r_ci_clips_to_unit_interval():
    lo, hi = live._r2_from_r_ci(-1.2, 0.3)
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


# ── end-to-end: live r2_ci_via_r byte-identical to frozen HEAD ─────────────────
def _xy(seed, n, kind):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    if kind == "positive":
        y = x + rng.normal(scale=0.5, size=n)
    elif kind == "negative":
        y = -x + rng.normal(scale=0.5, size=n)
    elif kind == "near_zero":
        y = rng.normal(size=n)  # independent -> r ~ 0, the straddle case
    elif kind == "strong":
        y = 2.0 * x + rng.normal(scale=0.05, size=n)
    else:
        raise ValueError(kind)
    return x, y


_E2E = [
    (1, 40, "positive"),
    (2, 30, "negative"),
    (3, 50, "near_zero"),
    (4, 60, "strong"),
    (5, 25, "near_zero"),
]


def _assert_r2_result_identical(label, a, b):
    for key in ("r", "r2", "n_pairs"):
        av, bv = a[key], b[key]
        if isinstance(bv, float) and math.isnan(bv):
            assert math.isnan(av), f"[{label}] {key} NaN-ness differs"
        else:
            assert av == bv, f"[{label}] {key} differs: {av!r} != {bv!r}"
    for key in ("r_ci", "r2_ci"):
        for i in range(2):
            av, bv = a[key][i], b[key][i]
            if math.isnan(bv):
                assert math.isnan(av), f"[{label}] {key}[{i}] NaN-ness differs"
            else:
                assert av == bv, f"[{label}] {key}[{i}] differs: {av!r} != {bv!r}"


@pytest.mark.parametrize("seed,n,kind", _E2E)
@pytest.mark.parametrize("B,alpha", [(500, 0.1), (1000, 0.05)])
def test_r2_ci_via_r_byte_identical_to_head(seed, n, kind, B, alpha):
    x, y = _xy(seed, n, kind)
    label = f"{kind}_s{seed}_n{n}_B{B}"
    live_out = live.r2_ci_via_r(x, y, B=B, alpha=alpha, rng=seed)
    head_out = head.r2_ci_via_r(x, y, B=B, alpha=alpha, rng=seed)
    _assert_r2_result_identical(label, live_out, head_out)


def test_r2_ci_via_r_degenerate_constant_column_identical():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    const = np.array([5.0, 5.0, 5.0, 5.0])
    live_out = live.r2_ci_via_r(x, const, B=200, rng=0)
    head_out = head.r2_ci_via_r(x, const, B=200, rng=0)
    _assert_r2_result_identical("constant", live_out, head_out)
