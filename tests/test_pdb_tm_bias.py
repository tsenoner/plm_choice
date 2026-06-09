"""Tests for evaluation.pdb_tm_bias.

Ported from the SpeciesEmbedding reference (tools/eval/pdb_tm_bias.py) into the
upstream layout: import via `from evaluation.pdb_tm_bias import ...`.

Synthetic example: predicted = experimental + Gaussian(0, σ). Expectations:
  - delta median ≈ 0 (centred noise)
  - Wasserstein W₁ on the marginals scales with σ
  - Pearson r between predicted and experimental is high (≥ 0.9 for the
    chosen σ relative to the TM-score scale of the synthetic data)
  - R² of distance vs TM is close under both predicted and experimental
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.pdb_tm_bias import (
    paired_tm_delta,
    pdb_bias_report,
    r2_pLM_distance_vs_tm,
)


def _make_pairs(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = [f"P{i:04d}" for i in range(n)]
    b = [f"P{(i + 1) % n:04d}" for i in range(n)]
    return pd.DataFrame({"a": a, "b": b, "_rng": rng.random(n)})


def _make_predicted_experimental(n: int = 400, sigma: float = 0.05, seed: int = 1):
    """Predicted = clip(experimental + N(0, σ), 0, 1)."""
    rng = np.random.default_rng(seed)
    base = _make_pairs(n, seed=seed)
    experimental_tm = base["_rng"].to_numpy()  # uniform in [0, 1]
    noise = rng.normal(0.0, sigma, size=n)
    predicted_tm = np.clip(experimental_tm + noise, 0.0, 1.0)

    exp_df = pd.DataFrame({"a": base["a"], "b": base["b"], "tm_score": experimental_tm})
    pred_df = pd.DataFrame({"a": base["a"], "b": base["b"], "tm_score": predicted_tm})
    return pred_df, exp_df


def test_paired_tm_delta_median_near_zero():
    pred, exp = _make_predicted_experimental(n=400, sigma=0.05, seed=42)
    report = paired_tm_delta(pred, exp)
    assert report["n_pairs"] == 400
    # Centred noise => median delta within ~0.01 of zero for n=400 σ=0.05.
    assert abs(report["median"]) < 0.015
    # CI should bracket zero.
    lo, hi = report["median_ci"]
    assert lo < 0.02 and hi > -0.02
    # Predicted and experimental are strongly correlated (Pearson ≥ 0.95
    # for σ=0.05 over a uniform [0, 1] base — most of the variance is the
    # base, not the added noise).
    assert report["pearson_r_predicted_vs_experimental"] >= 0.95


def test_paired_tm_delta_wasserstein_scales_with_noise():
    pred_low, exp_low = _make_predicted_experimental(n=400, sigma=0.02, seed=1)
    pred_high, exp_high = _make_predicted_experimental(n=400, sigma=0.10, seed=2)
    w1_low = paired_tm_delta(pred_low, exp_low)["wasserstein_w1"]
    w1_high = paired_tm_delta(pred_high, exp_high)["wasserstein_w1"]
    # W₁ on the marginals should scale monotonically with σ.
    assert w1_high > w1_low
    # And the magnitude is in the right ballpark — for centred Gaussian
    # noise of stddev σ added to a uniform base, the marginal W₁ is
    # dominated by σ scaled by a constant factor < 1 (since clipping
    # reduces tail mass). Loose bounds:
    assert 0.0 <= w1_low < 0.05
    assert 0.0 < w1_high < 0.15


def test_paired_tm_delta_empty_join():
    pred = pd.DataFrame({"a": ["X"], "b": ["Y"], "tm_score": [0.8]})
    exp = pd.DataFrame({"a": ["U"], "b": ["V"], "tm_score": [0.7]})
    report = paired_tm_delta(pred, exp)
    assert report["n_pairs"] == 0
    assert np.isnan(report["median"])


def test_r2_distance_vs_tm_strong_correlation():
    """When distance = 1 - tm_score + tiny noise, R² should be very close to 1."""
    pred, exp = _make_predicted_experimental(n=300, sigma=0.03, seed=7)
    rng = np.random.default_rng(7)
    distance = pd.DataFrame(
        {
            "a": exp["a"],
            "b": exp["b"],
            "embedding_dist": (1.0 - exp["tm_score"].to_numpy())
            + rng.normal(0.0, 0.01, len(exp)),
        }
    )
    out = r2_pLM_distance_vs_tm(distance, exp)
    assert out["n_pairs"] == 300
    assert out["r2"] > 0.95
    lo, hi = out["r2_ci"]
    # CI should contain the point estimate and be in [0, 1].
    assert 0.0 <= lo <= out["r2"] <= hi <= 1.0


def test_r2_distance_vs_tm_paired_consistency():
    """R² under predicted and experimental should both be high and close
    when σ is small — that is the PDB-bias headline."""
    pred, exp = _make_predicted_experimental(n=300, sigma=0.02, seed=11)
    rng = np.random.default_rng(11)
    distance = pd.DataFrame(
        {
            "a": exp["a"],
            "b": exp["b"],
            "embedding_dist": (1.0 - exp["tm_score"].to_numpy())
            + rng.normal(0.0, 0.01, len(exp)),
        }
    )
    r2_pred = r2_pLM_distance_vs_tm(distance, pred)["r2"]
    r2_exp = r2_pLM_distance_vs_tm(distance, exp)["r2"]
    assert abs(r2_pred - r2_exp) < 0.05


def test_r2_distance_vs_tm_reproducible_with_seed():
    """Seeded r2 CI is byte-identical across runs (Phase-0 reproducibility gate)."""
    pred, exp = _make_predicted_experimental(n=300, sigma=0.03, seed=7)
    rng = np.random.default_rng(7)
    distance = pd.DataFrame(
        {
            "a": exp["a"],
            "b": exp["b"],
            "embedding_dist": (1.0 - exp["tm_score"].to_numpy())
            + rng.normal(0.0, 0.01, len(exp)),
        }
    )
    out1 = r2_pLM_distance_vs_tm(distance, exp, B=2000, rng=42)
    out2 = r2_pLM_distance_vs_tm(distance, exp, B=2000, rng=42)
    assert out1["r2_ci"] == out2["r2_ci"]
    assert out1["r2"] == out2["r2"]


def test_paired_tm_delta_reproducible_with_seed():
    pred, exp = _make_predicted_experimental(n=300, sigma=0.05, seed=7)
    out1 = paired_tm_delta(pred, exp, rng=42)
    out2 = paired_tm_delta(pred, exp, rng=42)
    assert out1["median_ci"] == out2["median_ci"]


def test_pdb_bias_report_reproducible_with_seed():
    pred, exp = _make_predicted_experimental(n=200, sigma=0.04, seed=3)
    rng = np.random.default_rng(3)
    distance = pd.DataFrame(
        {
            "a": exp["a"],
            "b": exp["b"],
            "embedding_dist": (1.0 - exp["tm_score"].to_numpy())
            + rng.normal(0.0, 0.02, len(exp)),
        }
    )
    r1 = pdb_bias_report(distance, pred, exp, rng=42)
    r2 = pdb_bias_report(distance, pred, exp, rng=42)
    assert r1["r2_predicted"]["r2_ci"] == r2["r2_predicted"]["r2_ci"]
    assert r1["delta_median_ci"] == r2["delta_median_ci"]


def test_pdb_bias_report_payload_shape():
    pred, exp = _make_predicted_experimental(n=200, sigma=0.04, seed=3)
    rng = np.random.default_rng(3)
    distance = pd.DataFrame(
        {
            "a": exp["a"],
            "b": exp["b"],
            "embedding_dist": (1.0 - exp["tm_score"].to_numpy())
            + rng.normal(0.0, 0.02, len(exp)),
        }
    )
    report = pdb_bias_report(distance, pred, exp)
    # Manifest-shape sanity check.
    for key in (
        "n_pairs",
        "delta_median",
        "delta_median_ci",
        "delta_histogram",
        "wasserstein_w1",
        "pearson_r_predicted_vs_experimental",
        "r2_predicted",
        "r2_experimental",
    ):
        assert key in report, f"missing {key}"
    assert report["n_pairs"] == 200
    assert len(report["delta_histogram"]["counts"]) == 40
    assert len(report["delta_histogram"]["bin_edges"]) == 41
