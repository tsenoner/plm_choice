"""Regression test for the HFSP score formula (Mahlich et al. 2018, Bioinformatics).

A prior version of ``_compute_hfsp_scores`` computed the length term as
``L ** (-0.33 * exp(1 + L/1000))`` instead of Mahlich Eq. 4's
``L ** (-0.33 * (1 + exp(-L/1000)))``. The bug produced a ~28-point discontinuity
at the L=450 branch boundary: the code's own ``L>450`` branch subtracts 28.4, which
matches the *correct* formula's length term at L=450 (~28.36) — but not the buggy
value (~0.14). This test pins the correct formula against an independent Python
implementation and asserts branch-boundary continuity, so a regression to the buggy
exponent fails loudly.
"""

from __future__ import annotations

import math

import polars as pl

from data_preparation.merge_datasets import ProteinAnalysisPipeline


def _mahlich_hfsp(fident: float, ungapped_len: int) -> float:
    """Reference HFSP per Mahlich et al. 2018 Eq. 4 (computed independently here)."""
    length = ungapped_len
    if length <= 11:
        return fident * 100 - 100
    if length <= 450:
        return fident * 100 - 770 * length ** (-0.33 * (1 + math.exp(-length / 1000)))
    return fident * 100 - 28.4


def _compute_hfsp(df: pl.DataFrame) -> pl.DataFrame:
    """Call the production HFSP path.

    ``_compute_hfsp_scores`` does not use ``self``; bypass ``__init__`` (which needs
    the on-disk data-directory layout) and invoke the bound method directly.
    """
    pipe = object.__new__(ProteinAnalysisPipeline)
    return pipe._compute_hfsp_scores(df)


def test_hfsp_matches_mahlich_across_all_three_branches():
    # (fident, nident, mismatch) -> ungapped_len = nident + mismatch
    rows = [
        (0.90, 5, 3),      # L=8   -> short branch (<= 11)
        (0.50, 200, 50),   # L=250 -> main branch
        (0.30, 300, 40),   # L=340 -> main branch
        (0.40, 400, 100),  # L=500 -> long branch (> 450)
    ]
    df = pl.DataFrame(
        {
            "fident": [r[0] for r in rows],
            "nident": [r[1] for r in rows],
            "mismatch": [r[2] for r in rows],
        }
    )
    out = _compute_hfsp(df)
    for fident, length, got in zip(
        out["fident"], out["ungapped_len"], out["hfsp"]
    ):
        expected = _mahlich_hfsp(fident, length)
        assert math.isclose(got, expected, rel_tol=0, abs_tol=1e-4), (
            f"L={length}: got {got}, expected {expected}"
        )


def test_hfsp_continuous_at_L450_boundary():
    # The correct length term at L=450 is ~28.36, matching the otherwise-branch
    # constant 28.4; the buggy exponent gives ~0.14, a ~28-point jump. A tight
    # continuity bound therefore rejects the bug.
    df = pl.DataFrame(
        {
            "fident": [0.6, 0.6],
            "nident": [350, 351],
            "mismatch": [100, 100],  # L = 450 and 451
        }
    )
    out = _compute_hfsp(df).sort("ungapped_len")
    h450, h451 = out["hfsp"].to_list()
    assert abs(h450 - h451) < 0.2, (
        f"discontinuity at L=450: {h450} vs {h451} — "
        "did the exponent regress to exp(1 + L/1000)?"
    )
