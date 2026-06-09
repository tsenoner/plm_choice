"""PDB TM-score bias: predicted (ColabFold / ESMFold) vs experimental.

Reviewer concern: using AlphaFold-/ColabFold-predicted structures for
TM-score may overstate or understate true structural similarity. The PDB
experimental subset is the gold standard. This module quantifies the bias and
recomputes the pLM-distance-vs-TM-score relationship under both, with PDB as
the calibration set.

**B4-gated.** This analysis cannot be run for the paper until B4 (experimental-
PDB TM sign-off) is decided. The code + tests are ported now; the analysis
stays parked behind B4.

Three headline numbers
----------------------
1. ``paired_tm_delta`` — per-pair ΔTM = TM(predicted) − TM(experimental),
   95% BCa CI on the median, plus Wasserstein W₁ between the two TM
   distributions.
2. ``r2_pLM_distance_vs_tm`` — Pearson R² of embedding distance vs TM-score
   with paired BCa B=1000 bootstrap CI.
3. ``pdb_bias_report`` — single dict for direct insertion into the
   ``analysis/manifest.json`` payload.

Dependencies
------------
- ``pandas``, ``numpy``, ``scipy`` (Wasserstein W₁, Pearson R)
- ``evaluation.stats.bca_bootstrap`` / ``evaluation.stats.r2_ci_via_r``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wasserstein_distance

from evaluation.stats import bca_bootstrap, r2_ci_via_r


def _inner_join_pairs(left: pd.DataFrame, right: pd.DataFrame, value_cols: tuple[str, str]) -> pd.DataFrame:
    """Inner-join two pair tables on (a, b) preserving the value columns.

    Pairs are treated as ordered tuples — the caller is responsible for
    canonicalising direction (e.g. sorting (a, b) lexicographically) if
    pairs are symmetric. We do not symmetrise here because the upstream
    distance-computation step already emits canonical pairs.
    """
    l = left.rename(columns={value_cols[0]: "_lv"})[["a", "b", "_lv"]]
    r = right.rename(columns={value_cols[1]: "_rv"})[["a", "b", "_rv"]]
    return l.merge(r, on=["a", "b"], how="inner")


def paired_tm_delta(
    predicted_tm: pd.DataFrame,
    experimental_tm: pd.DataFrame,
    rng: int | np.random.Generator | None = None,
) -> dict:
    """Per-pair ΔTM, median + 95% BCa CI, and W₁ between the distributions.

    Parameters
    ----------
    predicted_tm, experimental_tm
        Long-form pair tables with columns ``[a, b, tm_score]``.

    Returns
    -------
    dict
        ``n_pairs``, ``delta`` (pandas.Series, predicted − experimental),
        ``median``, ``median_ci`` (low, high) from BCa B=10000,
        ``wasserstein_w1`` between the two marginal TM-score distributions
        (on the intersected pairs), and
        ``pearson_r_predicted_vs_experimental`` (sanity check — should be
        positive and substantial if the predicted structures are
        not pathological).
    """
    joined = _inner_join_pairs(predicted_tm, experimental_tm, ("tm_score", "tm_score"))
    if joined.empty:
        return {
            "n_pairs": 0,
            "delta": pd.Series(dtype=float),
            "median": float("nan"),
            "median_ci": (float("nan"), float("nan")),
            "wasserstein_w1": float("nan"),
            "pearson_r_predicted_vs_experimental": float("nan"),
        }

    pred = joined["_lv"].to_numpy()
    exp = joined["_rv"].to_numpy()
    delta = pred - exp

    median = float(np.median(delta))
    _, ci_low, ci_high = bca_bootstrap(delta, statistic=np.median, B=10_000, rng=rng)
    w1 = float(wasserstein_distance(pred, exp))
    r = float(pearsonr(pred, exp).statistic) if len(pred) >= 2 else float("nan")

    return {
        "n_pairs": int(len(delta)),
        "delta": pd.Series(delta, index=joined.set_index(["a", "b"]).index),
        "median": median,
        "median_ci": (float(ci_low), float(ci_high)),
        "wasserstein_w1": w1,
        "pearson_r_predicted_vs_experimental": r,
    }


def r2_pLM_distance_vs_tm(
    distance: pd.DataFrame,
    tm_score: pd.DataFrame,
    B: int = 10_000,
    rng: int | np.random.Generator | None = None,
) -> dict:
    """Pearson R² of embedding distance vs TM-score with a reproducible BCa CI.

    The CI is computed via :func:`evaluation.stats.r2_ci_via_r`, which
    bootstraps the *signed* r and maps the r-CI to an R²-CI with a
    zero-crossing-aware square (B1 fix — the previous in-resample square was
    boundary-degenerate near r=0). R² is an absolute metric, so B defaults to
    10000 per the spec (B3 fix), and ``rng`` makes the CI reproducible (NEW-2).

    Parameters
    ----------
    distance
        ``[a, b, embedding_dist]`` — pLM-distance pair table.
    tm_score
        ``[a, b, tm_score]`` — TM-score pair table (predicted OR
        experimental; the caller decides which subset).
    B
        Bootstrap resamples (default 10000, absolute-metric rule).
    rng
        Seed or ``numpy.random.Generator`` for reproducible CIs.

    Returns
    -------
    dict
        ``r2``, ``r2_ci`` (low, high), ``r`` (signed point estimate),
        ``r_ci`` (signed-r CI), and ``n_pairs``.
    """
    joined = _inner_join_pairs(distance, tm_score, ("embedding_dist", "tm_score"))
    n = len(joined)
    if n < 2:
        return {
            "r2": float("nan"),
            "r2_ci": (float("nan"), float("nan")),
            "r": float("nan"),
            "r_ci": (float("nan"), float("nan")),
            "n_pairs": int(n),
        }

    d = joined["_lv"].to_numpy()
    t = joined["_rv"].to_numpy()
    out = r2_ci_via_r(d, t, B=B, rng=rng)
    return {
        "r2": out["r2"],
        "r2_ci": out["r2_ci"],
        "r": out["r"],
        "r_ci": out["r_ci"],
        "n_pairs": int(n),
    }


def pdb_bias_report(
    distance: pd.DataFrame,
    predicted_tm: pd.DataFrame,
    experimental_tm: pd.DataFrame,
    B: int = 10_000,
    rng: int | np.random.Generator | None = None,
) -> dict:
    """Full headline payload for a single pLM × PDB-subset bias comparison.

    Returns a dict suitable for direct insertion into
    ``analysis/manifest.json``: ΔTM summary, R² under both TM sources, and
    the Wasserstein W₁ between the two TM distributions. ``rng`` makes every
    CI reproducible (NEW-2).
    """
    delta_report = paired_tm_delta(predicted_tm, experimental_tm, rng=rng)
    r2_predicted = r2_pLM_distance_vs_tm(distance, predicted_tm, B=B, rng=rng)
    r2_experimental = r2_pLM_distance_vs_tm(distance, experimental_tm, B=B, rng=rng)

    # The pandas.Series of per-pair deltas is heavy; we summarise it to a
    # histogram for the manifest. Callers that want the raw deltas should
    # use ``paired_tm_delta`` directly.
    if delta_report["n_pairs"] > 0:
        delta_arr = delta_report["delta"].to_numpy()
        hist_counts, hist_edges = np.histogram(delta_arr, bins=40)
        delta_hist = {
            "counts": hist_counts.tolist(),
            "bin_edges": hist_edges.tolist(),
        }
    else:
        delta_hist = {"counts": [], "bin_edges": []}

    return {
        "n_pairs": delta_report["n_pairs"],
        "delta_median": delta_report["median"],
        "delta_median_ci": delta_report["median_ci"],
        "delta_histogram": delta_hist,
        "wasserstein_w1": delta_report["wasserstein_w1"],
        "pearson_r_predicted_vs_experimental": delta_report[
            "pearson_r_predicted_vs_experimental"
        ],
        "r2_predicted": r2_predicted,
        "r2_experimental": r2_experimental,
    }
