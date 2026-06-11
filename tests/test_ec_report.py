import numpy as np
import pandas as pd
import pytest

from evaluation.ec_report import (
    ec_dist_histogram,
    stratify_by_class,
)


def _pairs():
    return pd.DataFrame(
        {
            "a": ["P1", "P1", "P2"],
            "b": ["P2", "P3", "P3"],
            "dist": [0.1, 0.9, 0.5],
            "ec_dist": [0.0, 4.0, 4.0],
        }
    )


def _ec_class():
    # first EC field per protein
    return {"P1": "1", "P2": "1", "P3": "2"}


def test_histogram_counts_integer_bins():
    h = ec_dist_histogram(_pairs())
    assert h[0] == 1 and h[4] == 2


def test_stratify_within_vs_across_class():
    out = stratify_by_class(_pairs(), _ec_class())
    # within-class pairs: P1-P2 (both class 1). across: P1-P3, P2-P3.
    assert out["n_within"] == 1
    assert out["n_across"] == 2


def test_stratify_returns_real_stratum_tau_b():
    # A stratum with enough pairs to actually compute a correlation (the count-only
    # tests above never exercise the stratified statistic itself). Within class "1":
    # four proteins, embedding distance monotone in ec_dist -> within-class tau_b > 0.
    pairs = pd.DataFrame({
        "a": ["Q1", "Q1", "Q1", "Q2", "Q2", "Q3"],
        "b": ["Q2", "Q3", "Q4", "Q3", "Q4", "Q4"],
        "dist": [1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
        "ec_dist": [1.0, 2.0, 3.0, 1.0, 2.0, 1.0],  # == dist -> perfect monotone
    })
    ec_class = {"Q1": "1", "Q2": "1", "Q3": "1", "Q4": "1"}  # all same class
    out = stratify_by_class(pairs, ec_class)
    assert out["n_within"] == 6 and out["n_across"] == 0
    assert out["tau_b_within"] == pytest.approx(1.0)   # the real stratified statistic
    assert np.isnan(out["tau_b_across"])               # empty stratum -> NaN, not 0


from evaluation.ec_report import stratify_by_superfamily


def test_superfamily_within_across_and_nonhomologous_restriction():
    pairs = pd.DataFrame({
        "a": ["P1", "P1", "P2"],
        "b": ["P2", "P3", "P3"],
        "dist": [0.1, 0.9, 0.5],
        "ec_dist": [0.0, 4.0, 4.0],
    })
    # superfamily sets per protein (CATH multi-domain frozensets)
    sfam = {
        "P1": frozenset({"3.40.50.300"}),
        "P2": frozenset({"3.40.50.300"}),   # shares with P1 -> homologous
        "P3": frozenset({"1.10.10.10"}),
    }
    out = stratify_by_superfamily(pairs, sfam)
    assert out["n_within_superfamily"] == 1   # P1-P2
    assert out["n_across_superfamily"] == 2    # P1-P3, P2-P3
    # non-homologous restriction == across-superfamily subset
    assert out["n_nonhomologous"] == 2
