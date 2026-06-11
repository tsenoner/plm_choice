"""Differential test for the U1 ``pdb_tm_bias`` W₁ call-site swap.

``pdb_tm_bias.paired_tm_delta`` used to compute its ``wasserstein_w1`` field via an
INLINE ``scipy.stats.wasserstein_distance(pred, exp)`` (pdb_tm_bias.py:90). U1 promotes
W₁ to the single owner ``stats.wasserstein_w1`` and swaps the call-site to it. pdb-TM is
B4-gated, so this swap must NOT change its numbers — we assert the produced
``wasserstein_w1`` field equals the FROZEN inline expression byte-identically on a
fixture battery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as _frozen_inline_w1

from evaluation.pdb_tm_bias import paired_tm_delta


def _pair_table(a_ids, b_ids, vals):
    return pd.DataFrame({"a": a_ids, "b": b_ids, "tm_score": vals})


def _fixture(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(6, 40)
    a = [f"P{i}" for i in range(n)]
    b = [f"Q{i}" for i in range(n)]
    pred = rng.uniform(0.2, 0.95, size=n)
    exp = pred + rng.normal(scale=0.1, size=n)
    return (
        _pair_table(a, b, pred),
        _pair_table(a, b, exp),
        pred,
        exp,
    )


def test_pdb_tm_bias_w1_byte_identical_to_frozen_inline():
    for seed in range(12):
        pred_tbl, exp_tbl, pred, exp = _fixture(seed)
        report = paired_tm_delta(pred_tbl, exp_tbl, rng=0)
        # the FROZEN inline expression the swap replaced, on the SAME joined arrays
        # (the fixture pairs are 1:1 on (a, b) so the inner join preserves order).
        frozen = float(_frozen_inline_w1(pred, exp))
        assert report["wasserstein_w1"] == frozen, (
            f"[seed={seed}] swap changed pdb-TM W1: "
            f"{report['wasserstein_w1']!r} != {frozen!r}"
        )
