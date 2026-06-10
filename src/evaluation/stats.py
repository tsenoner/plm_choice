"""Statistical toolkit for the pLM-choice paper revision.

Provides:

* BCa (bias-corrected and accelerated) bootstrap CIs — DiCiccio & Efron 1996.
* Paired bootstrap of the ratio-of-means (the "retention" statistic).
* ``r2_ci_via_r`` — bootstrap CI for R² via the *signed* r (the B1 fix; see
  ``evaluation.metrics`` for the same correction applied to the regression-metric
  table). Squaring r inside each resample piles the R² distribution against the 0
  boundary when the true r is near zero, degenerating the BCa accelerator.
* Holm-Bonferroni and Benjamini-Hochberg (BH-FDR) multiple-testing correction.
* Paired Wilcoxon signed-rank with Cliff's delta effect size.
* ``grid_test`` — full pairwise (two-sided) paired-Wilcoxon driver over the
  pLM × task grid; emits a tidy long-format DataFrame for figure scripts.

Design rules:

* scipy provides BCa; Holm/BH are self-contained (no statsmodels dependency).
* All randomness routes through a single ``numpy.random.Generator`` so result
  files are reproducible from a manifest seed.
* Public functions take and return plain numpy / pandas objects — nothing here
  imports from elsewhere in the repo, so it is reusable in notebooks, figure
  scripts, and the analysis DAG without a heavy import.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


# ---------------------------------------------------------------------------
# BCa bootstrap
# ---------------------------------------------------------------------------


def _as_rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def bca_bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    B: int = 10_000,
    alpha: float = 0.05,
    paired: np.ndarray | None = None,
    rng: int | np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """BCa bootstrap CI for an arbitrary statistic (DiCiccio & Efron 1996).

    Parameters
    ----------
    data:
        1-D array of observations, or 2-D where rows are observations (proteins,
        pairs, variants, ...). When ``paired`` is provided, ``data`` is treated
        as the first array of the pair.
    statistic:
        Callable ``f(x) -> float``. When ``paired`` is supplied the callable
        receives a 2-D array of shape ``(n, 2)`` whose columns are
        ``(data, paired)`` in that order.
    B:
        Number of bootstrap resamples (default 10 000).
    alpha:
        Two-sided coverage error; the CI is at ``1 - alpha``.
    paired:
        Optional second array. When given, row indices are resampled once and
        applied to both arrays so the pairing is preserved.
    rng:
        Either a numpy ``Generator`` or a seed; ``None`` uses the default RNG.

    Returns
    -------
    tuple[float, float, float]
        ``(point_estimate, lower, upper)``.
    """
    generator = _as_rng(rng)
    data_arr = np.asarray(data)
    if paired is not None:
        paired_arr = np.asarray(paired)
        if paired_arr.shape[0] != data_arr.shape[0]:
            raise ValueError(
                "paired array must have the same length as data along axis 0"
            )
        sample = np.column_stack([data_arr, paired_arr])
        # scipy accepts a tuple of paired samples and a `paired=True` flag that
        # resamples the row index once — exactly the DiCiccio & Efron pairing.
        result = _scipy_stats.bootstrap(
            (data_arr, paired_arr),
            lambda a, b, axis=-1: statistic(np.stack([a, b], axis=-1)),
            n_resamples=B,
            confidence_level=1 - alpha,
            method="BCa",
            paired=True,
            random_state=generator,
            vectorized=False,
        )
        point = float(statistic(sample))
        ci = result.confidence_interval
        return point, float(ci.low), float(ci.high)

    # Unpaired path.
    result = _scipy_stats.bootstrap(
        (data_arr,),
        lambda x, axis=-1: statistic(x),
        n_resamples=B,
        confidence_level=1 - alpha,
        method="BCa",
        random_state=generator,
        vectorized=False,
    )
    point = float(statistic(data_arr))
    ci = result.confidence_interval
    return point, float(ci.low), float(ci.high)


# Below this many observations a BCa CI is not defensible: the jackknife acceleration
# is chronically singular and scipy returns a coverage-free interval that is just the
# data's (min, max) — reported, misleadingly, as a 95% CI. Flag those degenerate. The
# real analysis cells are full cohorts (n ~ 260-319), so this floor never fires there;
# it guards sparse/stratified cells a future arm might produce.
MIN_BOOTSTRAP_N = 4

# Data whose spread is this small relative to its scale is effectively constant: the BCa
# acceleration jackknife denominator (sum of cubed deviations)^1.5 underflows toward 0 and
# the interval is garbage. Treat as a degenerate point. For the real per-row metrics
# (recall/Jaccard are discrete rationals) per-query values are either equal or differ by
# >= 1/(2k), so this only ever fires on exactly-constant data there; it additionally guards
# continuous-valued future metrics. Subsumes the exact-constant (spread == 0) case.
_NEAR_CONSTANT_RTOL = 1e-6


def bounded_mean_bca_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: int | np.random.Generator | None = None,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> tuple[float, float, bool]:
    """Degenerate-honest BCa CI for the *mean* of a bounded per-row metric.

    The shared CI primitive for every absolute mean-of-per-row statistic in the
    analysis DAG — recall@first-FP, SNN per-query Jaccard, AAC-floor recall — so the
    "degenerate is a point not an interval" and "clip to the statistic's range" rules
    live in exactly one place (this was first written privately as
    ``recall_fp_report._recall_ci``).

    Returns ``(lo, hi, degenerate)``. ``degenerate`` is True whenever the returned
    pair is NOT a genuine ``1 - alpha`` bootstrap coverage statement:

    * fewer than ``MIN_BOOTSTRAP_N`` (=4) values → ``(nan, nan, True)`` — at n=2/3 BCa
      degenerates to the data range, a coverage-free interval, so no CI is reported;
    * exactly- or near-constant values (e.g. perfect retrieval, every query 1.0; spread
      negligible relative to scale) → ``(mean, mean, True)`` — every resample is ~the
      constant, so the bootstrap is inapplicable and this is a point (scipy's BCa
      acceleration jackknife is singular and would return NaN or coverage-free garbage);
    * BCa fails to form a finite interval → ``(nan, nan, True)`` rather than passing
      scipy's NaN through as if it were real.

    Otherwise the BCa interval is clipped to ``clip`` (default ``(0.0, 1.0)`` for a
    unit-range metric; pass ``clip=None`` to disable, or a custom ``(lo, hi)`` for a
    different range such as a correlation in ``[-1, 1]``) — BCa can spill past a
    boundary on skewed data (cf. ``r2_ci_via_r``).

    Raises ``ValueError`` if ``clip`` is given with ``clip[0] > clip[1]`` (a transposed
    bound would silently collapse every result to a single value).
    """
    if clip is not None and clip[0] > clip[1]:
        raise ValueError(f"clip lower bound {clip[0]} > upper bound {clip[1]}")
    arr = np.asarray(values, dtype=float)
    if arr.size < MIN_BOOTSTRAP_N:
        return float("nan"), float("nan"), True
    # Exactly- or near-constant data -> degenerate point (the bootstrap is inapplicable;
    # its acceleration jackknife is singular). Return the mean as the point estimate.
    spread = float(np.ptp(arr))
    scale = max(1.0, float(np.max(np.abs(arr))))
    if spread <= _NEAR_CONSTANT_RTOL * scale:
        c = float(np.mean(arr))
        return c, c, True
    _, lo, hi = bca_bootstrap(arr, np.mean, B=n_boot, alpha=alpha, rng=rng)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return float("nan"), float("nan"), True
    if clip is not None:
        lo_b, hi_b = clip
        lo = min(max(lo, lo_b), hi_b)
        hi = min(max(hi, lo_b), hi_b)
    return float(lo), float(hi), False


def r2_ci_via_r(
    d: np.ndarray,
    t: np.ndarray,
    B: int = 10_000,
    alpha: float = 0.05,
    rng: int | np.random.Generator | None = None,
) -> dict:
    """Bootstrap CI for R² (squared Pearson r) of ``d`` vs ``t`` via the signed r.

    The naive pattern — squaring r *inside* each bootstrap resample, then
    bootstrapping the r² values — makes the R² bootstrap distribution pile
    against the 0 boundary whenever the true r is near zero (the resampled r
    straddles 0), which degenerates the BCa bias-correction/accelerator. We
    instead BCa-bootstrap the *signed* r (well-behaved away from |r|=1) and map
    the r-CI to an R²-CI with a zero-crossing-aware square:

    * ``r2_hi = max(r_lo², r_hi²)``
    * ``r2_lo = 0`` if the r-CI brackets 0, else ``min(r_lo², r_hi²)``

    This is the standalone counterpart to the in-resample fix in
    ``evaluation.metrics`` (B1). Use for the predicted-TM R² and the cross-pLM
    R²-vs-target table.

    Returns ``{"r", "r2", "r_ci", "r2_ci", "n_pairs"}``. Degenerate inputs
    (n < 2 or a constant column) return NaN point estimates. Default B=10000
    for an absolute metric.
    """
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

    r2_hi = max(r_lo * r_lo, r_hi * r_hi)
    if r_lo <= 0.0 <= r_hi:
        r2_lo = 0.0
    else:
        r2_lo = min(r_lo * r_lo, r_hi * r_hi)
    # Guard against floating-point spill outside [0, 1].
    r2_lo = max(0.0, min(1.0, r2_lo))
    r2_hi = max(0.0, min(1.0, r2_hi))

    return {
        "r": float(r_point),
        "r2": float(r_point * r_point),
        "r_ci": (float(r_lo), float(r_hi)),
        "r2_ci": (float(r2_lo), float(r2_hi)),
        "n_pairs": n,
    }


def paired_bootstrap_ratio(
    a: np.ndarray,
    b: np.ndarray,
    B: int = 10_000,
    alpha: float = 0.05,
    rng: int | np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Paired BCa bootstrap of the ratio of means ``mean(a) / mean(b)``.

    The retention statistic: "method A achieves X% of method B's score on the
    same proteins". Sampling is paired by protein index, so the numerator and
    denominator move together.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have the same shape")
    if a_arr.size == 0:
        raise ValueError("inputs are empty")
    if np.mean(b_arr) == 0:
        raise ValueError("denominator mean is zero; ratio is undefined")

    def _ratio(pair: np.ndarray) -> float:
        return float(np.mean(pair[:, 0]) / np.mean(pair[:, 1]))

    point = float(np.mean(a_arr) / np.mean(b_arr))
    # Degenerate case: if the per-protein ratio is constant, the bootstrap
    # distribution collapses to a point and the BCa accelerator is undefined.
    # Return a zero-width CI so downstream code doesn't have to special-case NaNs.
    per_protein_ratio = a_arr / np.where(b_arr == 0, np.nan, b_arr)
    if np.all(np.isfinite(per_protein_ratio)) and np.allclose(
        per_protein_ratio, per_protein_ratio[0]
    ):
        return point, point, point

    return bca_bootstrap(a_arr, _ratio, B=B, alpha=alpha, paired=b_arr, rng=rng)


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------


def _validate_pvalues(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be 1-D")
    if np.any(np.isnan(p)):
        raise ValueError("p_values contains NaNs")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must lie in [0, 1]")
    return p


def holm_bonferroni(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction (Holm 1979).

    Returns ``(rejected_mask, adjusted_p_values)`` in the original input order.
    Adjusted p-values are clipped to ``[0, 1]`` and enforced monotone over the
    sorted sequence.
    """
    p = _validate_pvalues(p_values)

    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    # Multipliers run from n down to 1 in sort order.
    raw_adj = p[order] * (n - np.arange(n))
    # Enforce monotone non-decreasing along sorted order and clip to 1.
    monotone = np.maximum.accumulate(np.minimum(raw_adj, 1.0))
    adjusted = monotone[ranks]
    # ``<=`` matches the canonical Holm 1979 definition (reject when the
    # adjusted p-value is at most alpha).
    rejected = adjusted <= alpha
    return rejected, adjusted


def bh_fdr(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR (Benjamini & Hochberg 1995).

    Returns ``(rejected_mask, adjusted_p_values)`` in the original input order.
    """
    p = _validate_pvalues(p_values)

    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    sorted_p = p[order]
    raw_adj = sorted_p * n / (np.arange(n) + 1)
    # Reverse cumulative min — monotone non-decreasing adjusted p-values over
    # the sorted sequence (Benjamini-Hochberg "step-up").
    monotone = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adjusted = np.minimum(monotone, 1.0)[ranks]
    rejected = adjusted <= alpha
    return rejected, adjusted


# ---------------------------------------------------------------------------
# Wilcoxon + Cliff's delta
# ---------------------------------------------------------------------------


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size for paired data: ``(#{a>b} - #{a<b}) / n``.

    We follow the paired convention used in the paper revision (one comparison
    per protein), which is what the methodology spec calls for.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    diffs = a - b
    n = diffs.size
    if n == 0:
        return float("nan")
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / n)


def paired_wilcoxon(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test with Cliff's delta effect size.

    Returns a dict with keys ``statistic``, ``p_value``, ``cliffs_delta``. When
    every pair is identical (zero variance), the p-value is set to 1.0 and the
    statistic to 0.0 — scipy raises in that case and we'd rather emit a neutral
    result. ``alternative`` is passed through to scipy (use ``"greater"`` for
    the one-sided rinit/floor comparisons).
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have the same shape")
    delta = _cliffs_delta(a_arr, b_arr)

    if np.all(a_arr == b_arr):
        return {"statistic": 0.0, "p_value": 1.0, "cliffs_delta": delta}

    try:
        result = _scipy_stats.wilcoxon(a_arr, b_arr, alternative=alternative)
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "cliffs_delta": delta,
        }
    except ValueError:
        # scipy raises if all differences are zero after zero-handling.
        return {"statistic": 0.0, "p_value": 1.0, "cliffs_delta": delta}


# ---------------------------------------------------------------------------
# Grid driver
# ---------------------------------------------------------------------------


def grid_test(
    metric_table: pd.DataFrame,
    plm_axis: str = "pLM",
    task_axis: str = "task",
    fold_axis: str = "fold",
    value_axis: str = "metric_value",
    correction: str = "holm",
    alpha: float = 0.05,
    correction_scope: str = "per_task",
) -> pd.DataFrame:
    """Full pairwise (two-sided) paired-Wilcoxon over the pLM × task grid.

    Parameters
    ----------
    metric_table:
        Long-format DataFrame with columns ``[pLM, task, fold, metric_value]``
        (axis names overridable via the keyword arguments). Within each
        (pLM, task) cell the values must be aligned along the ``fold`` axis so
        the pairing across pLMs is by fold index.
    plm_axis, task_axis, fold_axis, value_axis:
        Column names in ``metric_table``.
    correction:
        Either ``"holm"`` (Holm-Bonferroni; default) or ``"bh"``
        (Benjamini-Hochberg FDR).
    alpha:
        Family-wise α used for the ``significant`` flag.
    correction_scope:
        Either ``"per_task"`` (default — one correction family per task) or
        ``"global"`` (pool all tasks into one correction family).

    Returns
    -------
    pandas.DataFrame
        One row per ``(pLM_a, pLM_b, task)`` ordered pair with columns
        ``[pLM_a, pLM_b, task, n_pairs, statistic, p_raw, p_adj, cliffs_delta,
        significant]``. Self-comparisons are dropped; (a, b) and (b, a) both
        appear so callers can read off the matrix row-wise. The correction
        family is the set of *unique unordered pairs* per scope — directional
        pairs share the same ``p_adj`` because the two-sided Wilcoxon gives
        identical p-values (B2: full pairwise grid, two-sided).
    """
    if correction not in {"holm", "bh"}:
        raise ValueError("correction must be 'holm' or 'bh'")
    if correction_scope not in {"per_task", "global"}:
        raise ValueError("correction_scope must be 'per_task' or 'global'")
    required = {plm_axis, task_axis, fold_axis, value_axis}
    missing = required - set(metric_table.columns)
    if missing:
        raise ValueError(f"metric_table is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for task, task_df in metric_table.groupby(task_axis, sort=True):
        # Wide: rows = fold, cols = pLM.
        wide = task_df.pivot_table(
            index=fold_axis, columns=plm_axis, values=value_axis, aggfunc="mean"
        )
        plms = list(wide.columns)
        for i, plm_a in enumerate(plms):
            for j, plm_b in enumerate(plms):
                if i == j:
                    continue
                paired = wide[[plm_a, plm_b]].dropna()
                if paired.empty:
                    continue
                a = paired[plm_a].to_numpy()
                b = paired[plm_b].to_numpy()
                test = paired_wilcoxon(a, b)
                # Canonical (unordered) pair key — used to deduplicate the
                # correction family so the two directions don't double-count.
                pair_key = tuple(sorted([plm_a, plm_b]))
                rows.append(
                    {
                        "pLM_a": plm_a,
                        "pLM_b": plm_b,
                        "task": task,
                        "n_pairs": int(paired.shape[0]),
                        "statistic": test["statistic"],
                        "p_raw": test["p_value"],
                        "cliffs_delta": test["cliffs_delta"],
                        "_pair_key": pair_key,
                    }
                )

    out_columns = [
        "pLM_a",
        "pLM_b",
        "task",
        "n_pairs",
        "statistic",
        "p_raw",
        "p_adj",
        "cliffs_delta",
        "significant",
    ]
    if not rows:
        return pd.DataFrame(columns=out_columns)

    out = pd.DataFrame(rows)
    correct = holm_bonferroni if correction == "holm" else bh_fdr

    # Apply correction within each scope (per task by default), using only the
    # unique unordered pairs as the family, then propagate p_adj to both
    # directional rows.
    out["p_adj"] = np.nan
    out["significant"] = False

    scope_groups = (
        out.groupby("task", sort=False)
        if correction_scope == "per_task"
        else [("__all__", out)]
    )

    for _, group_rows in scope_groups:
        group_df = (
            group_rows.copy()
            if isinstance(group_rows, pd.DataFrame)
            else out.loc[group_rows.index]
        )
        # Pick one representative per unordered pair — the (a, b) where a <= b.
        canonical_mask = group_df.apply(
            lambda r: r["pLM_a"] == r["_pair_key"][0]
            and r["pLM_b"] == r["_pair_key"][1],
            axis=1,
        )
        canonical = group_df[canonical_mask]
        if canonical.empty:
            continue
        rejected, p_adj = correct(canonical["p_raw"].to_numpy(), alpha=alpha)
        pair_to_padj = dict(zip(canonical["_pair_key"], p_adj))
        pair_to_rej = dict(zip(canonical["_pair_key"], rejected))
        for idx, key in zip(group_df.index, group_df["_pair_key"]):
            out.at[idx, "p_adj"] = pair_to_padj[key]
            out.at[idx, "significant"] = bool(pair_to_rej[key])

    return out[out_columns]
