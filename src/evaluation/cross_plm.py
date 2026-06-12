"""Cross-pLM AGREEMENT-MATRIX arm — the per-cell agreement-metric CI binding.

Descriptive (Supplementary) arm: how similarly do two pLMs order / scale / linearly
associate the SAME frozen protein pairs? For each unordered pLM-pair and distance it
reports three symmetric agreement metrics — Spearman ρ, R² (signed-r vertex bootstrap),
and Wasserstein-1 (raw + per-cohort z-scored) — each as a point + a per-cell vertex BCa CI.
ρ and R² additionally carry a per-cell permutation p-value (U4). This is NOT a pLM ranking
(Ivan's call) — the "which pLM" answer lives in the ground-truth arms
(recall-FP / SNN / EC / pdb-TM).

**W₁ has NO permutation p-value, by design.** A symmetric row+column label permutation
only REORDERS the upper-triangle multiset of a distance matrix; it leaves each matrix's
marginal distance multiset unchanged. W₁ is a function of those two marginals ONLY, so
every permuted W₁ equals the observed W₁ → the null is degenerate (``w1_raw`` p≡1.0,
``w1_z`` float-noise). The permutation null is the wrong null for a distributional
distance. W₁ is therefore reported as a DESCRIPTIVE distance with its BCa CI only.
``cross_plm_permutation_null`` RAISES ``ValueError`` for the W₁ metrics.

**Downstream contract (U7 / Holm):** the multiple-comparison families are built ONLY over
ρ and R² — six families ``{ρ, R²} × {euclidean, cosine, manhattan}``. W₁ carries point + CI
ONLY and MUST NOT enter any Holm family (it has no p-value).

The CI binds three closures over the SHIPPED ``stats.vertex_bca_ci`` (no new stats.py CI
fn). Because the two columns of every agreement cell are BOTH pLM-dependent vectors induced
from the SAME proteins, the resample must draw the same proteins for both — which the core
guarantees: it owns the single per-iteration ``idx`` and each closure applies that one draw
to BOTH captured matrices via ``stats._induced_pair_values``.

Metric knobs (per design §3):

* ``rho``    — ``stats.spearman_rho``;  ``clip=(-1, 1)``, default ``divergence_tol``.
* ``r2``     — vertex-bootstrap the SIGNED Pearson r (``clip=(-1, 1)``), then map the r-CI
  to an R²-CI via ``stats._r2_from_r_ci`` (the shared B1 zero-crossing rule). NO
  in-resample squaring. The record carries both the R²-CI and the r-CI it derived from.
* ``w1_raw`` / ``w1_z`` — ``stats.wasserstein_w1`` on the two induced pair vectors. W₁ is
  UNBOUNDED, so the binding passes ``clip=None`` and a SCALE-RELATIVE ``divergence_tol``
  (a small multiple of the W₁ point magnitude — the default 0.05 is meaningless off
  ``[-1, 1]``). ``w1_z`` z-scores EACH pLM's induced vector by its OWN per-resample
  mean/std before W₁ (so a pure scale gap between the two pLMs cancels); ``w1_raw`` does
  not. **Near-zero-W₁ degeneracy (C3):** when the two marginals are ~identical the point
  W₁ ≈ 0 and the BCa is coverage-free — the binding marks the cell degenerate (mirroring
  the constant-margin path) rather than report a spurious interval.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as _scipy_stats

from evaluation.stats import (
    _as_rng,
    _full_pair_values,
    _induced_pair_values,
    _r2_from_r_ci,
    spearman_rho,
    vertex_bca_ci,
    wasserstein_w1,
)

# The W₁ scale-relative divergence_tol: tol = max(_W1_DIVERGENCE_ABS_FLOOR,
# _W1_DIVERGENCE_REL * |point|). A fixed 0.05 (the bounded-statistic default) is
# meaningless for an unbounded W₁ whose scale is the data's; the rel-multiple tracks the
# cell's own magnitude. The absolute floor keeps a small-but-nonzero W₁ from getting a
# vanishing tol that trips the divergence flag on float noise alone.
_W1_DIVERGENCE_REL = 0.25
_W1_DIVERGENCE_ABS_FLOOR = 1e-9

# Near-zero-W₁ (C3): if the point W₁ is within this *relative* tolerance of 0 (scaled by the
# typical distance magnitude of the two cohorts), the two marginals are effectively
# identical and the BCa is degenerate — report a point, not a coverage-free interval.
_W1_NEAR_ZERO_RTOL = 1e-6


def _zscore(v: np.ndarray) -> np.ndarray:
    """Z-score a 1-D vector by its OWN mean/std; constant vector -> all-zeros.

    Used per-resample by the ``w1_z`` kernel: each pLM's induced pair-distance vector is
    standardised by its own (resample-local) mean/std before W₁, so a pure overall-scale
    difference between the two pLMs cancels and only distribution *shape* differences
    survive.

    Constant guard: a numerically-constant vector has ``np.std`` ≈ 1e-16 (NOT exactly 0),
    so an ``sd == 0.0`` test misses it and divides ~1e-16/~1e-16 -> spurious unit values
    that silently corrupt ``w1_z``. Mirror the kernels' relative ``np.ptp(v) == 0`` guard
    (``spearman_rho`` / the signed-r kernel both use it) and add a relative-std floor so a
    near-constant column also maps to all-zeros rather than amplified float noise.
    """
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v
    mean = float(np.mean(v))
    sd = float(np.std(v))
    if np.ptp(v) == 0 or sd <= 1e-12 * max(1.0, abs(mean)):
        return np.zeros_like(v)
    return (v - mean) / sd


def _w1_raw_kernel(da: np.ndarray, db: np.ndarray) -> float:
    return wasserstein_w1(da, db)


def _w1_z_kernel(da: np.ndarray, db: np.ndarray) -> float:
    return wasserstein_w1(_zscore(da), _zscore(db))


def _signed_r_kernel(da: np.ndarray, db: np.ndarray) -> float:
    """Signed Pearson r; NaN on a constant margin (mirrors spearman_rho's guard)."""
    da = np.asarray(da, dtype=float)
    db = np.asarray(db, dtype=float)
    if da.size < 2 or np.ptp(da) == 0 or np.ptp(db) == 0:
        return float("nan")
    return float(_scipy_stats.pearsonr(da, db).statistic)


# metric name -> (pairwise kernel m(da, db) -> float). r2 dispatches to the signed-r kernel
# and post-maps the r-CI; the others map straight through vertex_bca_ci.
_METRIC_KERNELS = {
    "rho": spearman_rho,
    "r2": _signed_r_kernel,  # CI is over signed r; mapped to R² after
    "w1_raw": _w1_raw_kernel,
    "w1_z": _w1_z_kernel,
}


def _make_closures(dist_a: np.ndarray, dist_b: np.ndarray, kernel):
    """Build (point, boot_statistic, jackknife_statistic) over BOTH matrices.

    ``boot_statistic(idx)`` applies the SINGLE core-drawn ``idx`` to both captured
    matrices via ``_induced_pair_values`` (shared-protein draw is automatic).
    """
    n = int(dist_a.shape[0])

    da_all, db_all, _, _ = _full_pair_values(dist_a, dist_b)
    point = kernel(da_all, db_all)

    def _boot(idx: np.ndarray) -> float:
        da, db = _induced_pair_values(dist_a, dist_b, idx)
        return kernel(da, db)

    def _jack(k: int) -> float:
        keep = np.arange(n) != k
        sub_a = dist_a[np.ix_(keep, keep)]
        sub_b = dist_b[np.ix_(keep, keep)]
        da, db, _, _ = _full_pair_values(sub_a, sub_b)
        return kernel(da, db)

    return n, point, _boot, _jack


def cross_plm_agreement_ci(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    *,
    metric: str,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | np.random.Generator | None = 42,
    validate_point: bool = False,
) -> dict:
    """Vertex-bootstrap BCa CI for one cross-pLM agreement metric over two distance matrices.

    ``dist_a`` / ``dist_b`` are square symmetric distance matrices over ONE shared protein
    id order (length ``n``). ``metric`` is one of ``rho`` / ``r2`` / ``w1_raw`` / ``w1_z``.

    Returns ``{metric, point, ci_lo, ci_hi, degenerate, diverged}``. For ``r2`` it
    additionally carries ``r_point``, ``r_ci_lo``, ``r_ci_hi`` (the signed-r CI the R²-CI
    was mapped from). ``degenerate`` True means the returned pair is a point, not a
    ``1 - alpha`` coverage statement.
    """
    if metric not in _METRIC_KERNELS:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {sorted(_METRIC_KERNELS)}"
        )
    dist_a = np.asarray(dist_a, dtype=float)
    dist_b = np.asarray(dist_b, dtype=float)
    if dist_a.shape != dist_b.shape or dist_a.ndim != 2 or dist_a.shape[0] != dist_a.shape[1]:
        raise ValueError("dist_a and dist_b must be square matrices of the same shape")

    kernel = _METRIC_KERNELS[metric]
    n, point, boot_fn, jack_fn = _make_closures(dist_a, dist_b, kernel)

    if metric in ("w1_raw", "w1_z"):
        return _w1_ci(metric, n, point, boot_fn, jack_fn, n_boot, alpha, seed, validate_point)
    if metric == "r2":
        return _r2_ci(n, point, boot_fn, jack_fn, n_boot, alpha, seed, validate_point)

    # rho — straight bounded correlation through the core.
    lo, hi, pt, degenerate, diverged = vertex_bca_ci(
        n,
        point=point,
        boot_statistic=boot_fn,
        jackknife_statistic=jack_fn,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        clip=(-1.0, 1.0),
        validate_point=validate_point,
    )
    return {
        "metric": metric,
        "point": float(pt),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "degenerate": bool(degenerate),
        "diverged": bool(diverged),
    }


def _r2_ci(n, r_point, boot_fn, jack_fn, n_boot, alpha, seed, validate_point) -> dict:
    """R² via the signed-r vertex bootstrap, mapped through ``_r2_from_r_ci`` (B1)."""
    r_lo, r_hi, r_pt, degenerate, diverged = vertex_bca_ci(
        n,
        point=r_point,
        boot_statistic=boot_fn,
        jackknife_statistic=jack_fn,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        clip=(-1.0, 1.0),
        validate_point=validate_point,
    )
    if degenerate or not math.isfinite(r_lo) or not math.isfinite(r_hi):
        r2_point = float(r_pt) ** 2 if math.isfinite(r_pt) else float("nan")
        return {
            "metric": "r2",
            "point": r2_point,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "degenerate": True,
            "diverged": bool(diverged),
            "r_point": float(r_pt),
            "r_ci_lo": float(r_lo),
            "r_ci_hi": float(r_hi),
        }
    r2_lo, r2_hi = _r2_from_r_ci(r_lo, r_hi)
    return {
        "metric": "r2",
        "point": float(r_pt) ** 2,
        "ci_lo": float(r2_lo),
        "ci_hi": float(r2_hi),
        "degenerate": False,
        "diverged": bool(diverged),
        "r_point": float(r_pt),
        "r_ci_lo": float(r_lo),
        "r_ci_hi": float(r_hi),
    }


def _w1_ci(metric, n, point, boot_fn, jack_fn, n_boot, alpha, seed, validate_point) -> dict:
    """W₁ vertex BCa CI: clip=None, scale-relative divergence_tol, near-zero guard (C3)."""
    # Near-zero-W₁ (C3): if both marginals are ~identical the point W₁ is ~0 and the BCa is
    # coverage-free. Scale the near-zero tolerance by the typical magnitude of the W₁'s own
    # operands so it is unit-agnostic; flag degenerate and return the point.
    if not math.isfinite(point):
        return _w1_record(metric, point, float("nan"), float("nan"), True, False)
    scale = _w1_point_scale(point)
    if point <= _W1_NEAR_ZERO_RTOL * scale:
        return _w1_record(metric, point, point, point, True, False)

    divergence_tol = max(_W1_DIVERGENCE_ABS_FLOOR, _W1_DIVERGENCE_REL * abs(point))
    lo, hi, pt, degenerate, diverged = vertex_bca_ci(
        n,
        point=point,
        boot_statistic=boot_fn,
        jackknife_statistic=jack_fn,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        clip=None,
        divergence_tol=divergence_tol,
        validate_point=validate_point,
    )
    return _w1_record(metric, pt, lo, hi, degenerate, diverged)


def _w1_point_scale(point: float) -> float:
    """Magnitude scale for the W₁ near-zero test — at least 1.0 so the rtol never collapses
    to 0 on a genuinely tiny-but-meaningful cohort; otherwise the point's own size."""
    return max(1.0, abs(float(point)))


def _w1_record(metric, point, lo, hi, degenerate, diverged) -> dict:
    return {
        "metric": metric,
        "point": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "degenerate": bool(degenerate),
        "diverged": bool(diverged),
    }


# ---------------------------------------------------------------------------
# Per-cell permutation null (U4) — cross_plm-local, metric-pluggable
# ---------------------------------------------------------------------------


def cross_plm_permutation_null(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    *,
    metric: str,
    n_perm: int = 1000,
    seed: int | np.random.Generator | None = 42,
):
    """Per-cell permutation null + two-sided p for the ρ / R² cross-pLM agreement metrics.

    Mirrors ``stats.correlation_permutation_null``: permutes pLM-B's protein labels with a
    SYMMETRIC row+column permutation (so ``dist_b`` stays a valid distance matrix over
    relabelled proteins), recomputes ``metric(da_fixed, db_perm)`` on the upper-triangle
    pairs ``n_perm`` times -> the null distribution. The two-sided permutation p-value is
    ``(1 + #{|null| >= |obs|}) / (n_perm + 1)`` (the add-one keeps it strictly positive).
    Returns ``(null_values, p_value)``. Kept in ``cross_plm.py`` (not ``stats.py``) per
    spec §5/§7 option (b) — metric-pluggable for a single consumer.

    **W₁ is REJECTED here (``metric in {w1_raw, w1_z}`` -> ``ValueError``).** A symmetric
    row+column protein-label permutation only REORDERS the upper-triangle multiset of a
    distance matrix; it leaves each matrix's marginal distance multiset UNCHANGED. W₁
    depends only on those two marginals, so every permuted W₁ equals the observed W₁ — the
    null is degenerate (``w1_raw`` p≡1.0, ``w1_z`` float-noise around it). A permutation
    null is the wrong null for a distributional distance. W₁ is reported as a descriptive
    distance with its BCa CI only (see ``cross_plm_agreement_ci``); it has no permutation p
    by design, and MUST NOT enter any downstream Holm family.

    Identical-pLM behaviour for ρ / R² (documented decision, two-sided ``|null| >= |obs|``):
    perfect agreement is the statistic MAXIMUM (1.0). Label permutation breaks it, so
    ``|null| < |obs|`` almost surely -> p -> the floor ``1/(n_perm+1)``. Perfect agreement
    is maximally significant — the principled call.
    """
    if metric not in _METRIC_KERNELS:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {sorted(_METRIC_KERNELS)}"
        )
    if metric in ("w1_raw", "w1_z"):
        raise ValueError(
            "permutation null is undefined for W₁ — a symmetric label permutation "
            "preserves each matrix's marginal distance distribution, so the null is "
            "degenerate (every permuted W₁ == observed); report W₁ as a descriptive "
            "distance with its BCa CI only (cross_plm_agreement_ci), not a permutation p."
        )
    kernel = _METRIC_KERNELS[metric]
    dist_a = np.asarray(dist_a, dtype=float)
    dist_b = np.asarray(dist_b, dtype=float)
    rng = _as_rng(seed)
    n = int(dist_a.shape[0])

    iu, ju = np.triu_indices(n, k=1)
    da_fixed = dist_a[iu, ju]
    obs = kernel(da_fixed, dist_b[iu, ju])

    null = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        perm = rng.permutation(n)
        db_perm = dist_b[np.ix_(perm, perm)]
        null[p] = kernel(da_fixed, db_perm[iu, ju])

    finite = null[np.isfinite(null)]
    if not math.isfinite(obs) or finite.size == 0:
        return null, float("nan")
    extreme = np.count_nonzero(np.abs(finite) >= abs(obs))
    p_value = (1 + extreme) / (finite.size + 1)
    return null, float(p_value)
