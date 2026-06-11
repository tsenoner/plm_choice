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
from scipy.stats import norm as _norm


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


# ---------------------------------------------------------------------------
# Rank-correlation kernels (the single τ-b / ρ owners) + vertex bootstrap
# ---------------------------------------------------------------------------


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall τ-b (tie-corrected) — the one τ-b implementation in the codebase.

    Used by the EC point estimate, every bootstrap resample, and the permutation
    null, so there is exactly one tie-handling convention. Returns NaN on a constant
    margin (scipy returns NaN there too; we surface it as a float NaN).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(_scipy_stats.kendalltau(x, y, variant="b").statistic)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ — the one ρ implementation (NaN on a constant margin)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(_scipy_stats.spearmanr(x, y).statistic)


_CORRELATION_KERNELS = {"tau_b": kendall_tau_b, "spearman": spearman_rho}


def _induced_pair_values(
    dist_matrix: np.ndarray, ec_matrix: np.ndarray, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectors over the induced unordered pairs of a resampled protein-index array.

    For all resample positions ``p < q`` take the matrix entry between the *original*
    proteins ``idx[p], idx[q]``, dropping self-pairs (``idx[p] == idx[q]``). Repeated
    distinct indices reproduce the correct multiplicity (the vertex-bootstrap induced
    multiset). Returns ``(dist_vals, ec_vals)`` aligned 1-D arrays.
    """
    idx = np.asarray(idx)
    pi, qi = np.triu_indices(idx.size, k=1)
    a = idx[pi]
    b = idx[qi]
    keep = a != b
    a, b = a[keep], b[keep]
    return dist_matrix[a, b], ec_matrix[a, b]


# A vertex-bootstrap CI of a U-statistic needs more than the absolute n>=4 floor that
# guards a mean (a 4-protein cohort yields only C(4,2)=6 pairs -> a coverage-free
# correlation CI). MIN_VERTEX_N is the protein floor below which the CI is declared
# degenerate; it guards the stratified / non-homologous sub-cells the design must serve.
MIN_VERTEX_N = 12
# Fraction of bootstrap resamples that must survive the diversity/finiteness filter for
# the interval to be reported (else the surviving set is too conditioned to trust).
_MIN_VALID_BOOT_FRAC = 0.5
# Guard on the BCa quantile denominator 1 - a*(z0+zq): when it approaches 0 the adjusted
# quantile blows up and the interval silently collapses to a clipped boundary.
_BCA_DENOM_EPS = 1e-6


def _full_pair_values(dist_matrix, ec_matrix):
    iu, ju = np.triu_indices(dist_matrix.shape[0], k=1)
    return dist_matrix[iu, ju], ec_matrix[iu, ju], iu, ju


def correlation_vertex_bca_ci(
    dist_matrix: np.ndarray,
    ec_matrix: np.ndarray,
    *,
    statistic: str = "tau_b",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | np.random.Generator | None = 42,
):
    """Vertex-bootstrap BCa CI for a rank correlation of two NxN distance matrices.

    Resamples PROTEINS (matrix indices) with replacement — the correct unit, because
    the C(N,2) pairs are not independent (pairs sharing a protein are correlated). BCa
    acceleration via leave-one-protein-out jackknife. ``statistic`` is ``"tau_b"`` or
    ``"spearman"`` (the kernels in this module). Returns
    ``(lo, hi, point, degenerate, percentile_diverged)``:

    * ``degenerate`` True (-> lo/hi NaN) when N < MIN_VERTEX_N (12), a constant margin,
      or BCa fails to form a finite interval;
    * ``percentile_diverged`` True when the BCa interval and the plain percentile
      interval disagree by more than 0.05 (a sensitivity flag the report records).

    The interval is clipped to ``[-1, 1]`` after the BCa z0/a adjustment.
    """
    stat_fn = _CORRELATION_KERNELS[statistic]
    rng = _as_rng(seed)
    n = int(dist_matrix.shape[0])

    d_all, e_all, _, _ = _full_pair_values(dist_matrix, ec_matrix)
    point = stat_fn(d_all, e_all)
    if n < MIN_VERTEX_N or not math.isfinite(point):
        return float("nan"), float("nan"), float(point), True, False

    # Bootstrap distribution over resampled proteins.
    boot = np.empty(n_boot, dtype=float)
    valid = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Per-resample diversity floor is the SMALL pair-computability floor (>=4 distinct
        # proteins -> >=6 pairs), NOT the cohort floor MIN_VERTEX_N: a resample of an
        # n=MIN_VERTEX_N cohort rarely draws MIN_VERTEX_N distinct proteins, so reusing the
        # cohort floor here would wrongly mark every small-but-valid cohort degenerate.
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

    # z0: tie-stabilized bias correction.
    less = np.count_nonzero(boot < point)
    equal = np.count_nonzero(boot == point)
    prop = (less + 0.5 * equal) / valid
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = _norm.ppf(prop)

    # Leave-one-protein-out jackknife acceleration.
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
            return q  # fall back to the plain percentile quantile for this tail
        return _norm.cdf(z0 + (z0 + zq) / denom)

    lo_q = _bca_q(alpha / 2)
    hi_q = _bca_q(1 - alpha / 2)
    lo = float(np.quantile(boot, lo_q))
    hi = float(np.quantile(boot, hi_q))
    # Percentile interval for the divergence flag.
    plo = float(np.quantile(boot, alpha / 2))
    phi = float(np.quantile(boot, 1 - alpha / 2))
    diverged = denom_collapsed or abs(lo - plo) > 0.05 or abs(hi - phi) > 0.05

    if not (math.isfinite(lo) and math.isfinite(hi)):
        return float("nan"), float("nan"), float(point), True, False
    lo = min(max(lo, -1.0), 1.0)
    hi = min(max(hi, -1.0), 1.0)
    return lo, hi, float(point), False, diverged


def _pair_bootstrap_ci_width(
    dist_matrix, ec_matrix, *, statistic="tau_b", n_boot=2000, alpha=0.05, seed=42
) -> float:
    """Test-only: naive i.i.d.-pair percentile-CI width (the vertex CI must exceed this)."""
    stat_fn = _CORRELATION_KERNELS[statistic]
    rng = _as_rng(seed)
    d_all, e_all, _, _ = _full_pair_values(dist_matrix, ec_matrix)
    m = d_all.size
    boot = []
    for _ in range(n_boot):
        sel = rng.integers(0, m, size=m)
        s = stat_fn(d_all[sel], e_all[sel])
        if math.isfinite(s):
            boot.append(s)
    boot = np.asarray(boot)
    return float(np.quantile(boot, 1 - alpha / 2) - np.quantile(boot, alpha / 2))


def correlation_permutation_null(
    dist_matrix: np.ndarray,
    ec_matrix: np.ndarray,
    *,
    statistic: str = "tau_b",
    n_perm: int = 1000,
    seed: int | np.random.Generator | None = 42,
):
    """M-permutation null for the matrix rank correlation + a two-sided p-value.

    Permutes the EC-matrix protein labels (a symmetric row+column permutation, so the
    EC matrix stays a valid distance matrix over relabelled proteins) ``n_perm`` times,
    recomputing the statistic against the fixed embedding-distance matrix → the null
    distribution. The two-sided permutation p-value is
    ``(1 + #{|null| >= |obs|}) / (n_perm + 1)`` (the add-one keeps it strictly positive).
    Returns ``(null_values, p_value)``.
    """
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
