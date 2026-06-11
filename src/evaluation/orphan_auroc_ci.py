"""Vertex-bootstrap BCa CI for the orphan dyadic AUROC, bound over ``stats.vertex_bca_ci``.

Each Bromberg pair ``(p1, p2)`` has BOTH endpoints in the orphan set, so pairs that share
an orphan are correlated — the dependence is *dyadic*, exactly the EC arm's vertex
U-statistic. The correct CI therefore resamples ORPHANS (vertices) and recomputes AUROC
over the induced pair set, reusing the shipped pluggable core ``stats.vertex_bca_ci``
(design §11-R0). This module does NOT add a new function to ``stats.py``; it binds three
closures over the core:

* ``point`` = AUROC over ALL kept Bromberg pairs;
* ``boot_statistic(idx)`` = AUROC over the **sparse** induced pair set — the Bromberg
  pairs whose BOTH endpoints are in the resampled orphan index ``idx`` (the vertex
  bootstrap induces a *multiset*: pair ``(u, v)`` gets weight ``count(u)·count(v)``).
  Returns NaN when the induced set has zero sibling OR zero non-sibling pairs (one class
  absent) — the core skips NaN draws and declares the CI degenerate if too many die;
* ``jackknife_statistic(k)`` = AUROC with orphan ``k`` removed, computed INCREMENTALLY
  from precomputed per-pair concordance contributions (n ≈ 11,444 orphans × O(309k pairs)
  rebuilt-from-scratch would be a wall-clock cliff).

AUROC is the Mann-Whitney / concordance-U form, ``P(cos_sibling > cos_nonsibling)`` with
ties at 0.5, which equals ``sklearn.metrics.roc_auc_score`` but is cheaper over the
induced multiset and has a single degeneracy condition (one class empty).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.stats import vertex_bca_ci


# ── the weighted concordance-U kernel (== roc_auc_score for unit weights) ─────────────
def _sorted_kernel_contrib(
    cos_self: np.ndarray,
    cos_other: np.ndarray,
    w_other: np.ndarray,
) -> np.ndarray:
    """For each self row, ``Σ_j w_other[j] · (1[cos_self>cos_other_j] + 0.5·1[==])``.

    O((m+k) log k) via sorting the *other* class once + searchsorted (the weighted
    Mann-Whitney rank contribution). Returns a per-self-row array aligned to ``cos_self``.
    """
    if cos_other.size == 0:
        return np.zeros(cos_self.size, dtype=np.float64)
    order = np.argsort(cos_other, kind="mergesort")
    o_sorted = cos_other[order]
    w_sorted = w_other[order]
    cum = np.concatenate([[0.0], np.cumsum(w_sorted)])  # cum[i] = weight of first i others
    # weight strictly below: others with cos < cos_self  -> searchsorted 'left'
    lo = np.searchsorted(o_sorted, cos_self, side="left")
    below = cum[lo]
    # weight equal: others with cos == cos_self
    hi = np.searchsorted(o_sorted, cos_self, side="right")
    equal = cum[hi] - cum[lo]
    return below + 0.5 * equal


def weighted_concordance_auc(
    cos: np.ndarray, sibling: np.ndarray, weights: np.ndarray
) -> float:
    """Weighted concordance-U AUROC = ``P(cos_sib > cos_nonsib)`` (ties 0.5).

    Equals ``roc_auc_score(sibling, cos)`` for unit weights; a weight of ``w`` on a row
    is equivalent to duplicating that row ``w`` times. Returns NaN when either class has
    zero total weight (one class empty — AUROC undefined).
    """
    cos = np.asarray(cos, dtype=np.float64)
    sibling = np.asarray(sibling, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    pos = sibling
    neg = ~sibling
    w_pos = weights[pos]
    w_neg = weights[neg]
    sw_pos = float(w_pos.sum())
    sw_neg = float(w_neg.sum())
    if sw_pos <= 0.0 or sw_neg <= 0.0:
        return float("nan")
    contrib = _sorted_kernel_contrib(cos[pos], cos[neg], w_neg)  # per-pos vs all neg
    numer = float(np.dot(w_pos, contrib))
    return numer / (sw_pos * sw_neg)


# ── the vertex-bootstrap binding ──────────────────────────────────────────────────────
def orphan_auroc_vertex_bca_ci(
    per_pair_df: pd.DataFrame,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | np.random.Generator | None = 42,
    validate_point: bool = False,
) -> dict:
    """Vertex-bootstrap BCa CI for the orphan sibling AUROC.

    ``per_pair_df`` is the :func:`orphan_score.score_orphan_pairs` output (columns
    ``p1, p2, cos, ..., sibling``). Maps orphan string-ids -> contiguous vertex indices,
    binds the point/boot/jackknife closures over :func:`stats.vertex_bca_ci`, and returns
    ``{point, ci_lo, ci_hi, degenerate, diverged, n_boot_undefined}``.

    ``n_boot_undefined`` counts bootstrap draws whose induced pair set lost a class
    (zero sibling OR zero non-sibling) — those are skipped by the core. Set
    ``validate_point=True`` (tests) to assert the point kernel == the boot kernel on the
    identity resample.
    """
    cos, sibling, n, u, v = _prepare_vertices(per_pair_df)
    point = weighted_concordance_auc(cos, sibling, np.ones(u.size))

    boot_fn, undefined_counter = _build_boot(cos, sibling, n, u, v)
    _, jack_fn, _, _, _ = _build_jackknife(per_pair_df, _prepared=(cos, sibling, n, u, v))

    lo, hi, point_out, degenerate, diverged = vertex_bca_ci(
        n,
        point=point,
        boot_statistic=boot_fn,
        jackknife_statistic=jack_fn,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        clip=(0.0, 1.0),
        validate_point=validate_point,
    )
    return {
        "point": float(point_out),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "degenerate": bool(degenerate),
        "diverged": bool(diverged),
        "n_boot_undefined": int(undefined_counter[0]),
    }


def _prepare_vertices(per_pair_df: pd.DataFrame):
    """``per_pair_df`` -> ``(cos, sibling, n_orphans, u, v)`` with contiguous vertex ids.

    ``u``/``v`` are the per-pair endpoint vertex indices into the sorted orphan-id list.
    """
    cos = per_pair_df["cos"].to_numpy(dtype=np.float64)
    sibling = per_pair_df["sibling"].to_numpy().astype(bool)
    p1 = per_pair_df["p1"].astype(str).to_numpy()
    p2 = per_pair_df["p2"].astype(str).to_numpy()
    ids = sorted(set(p1) | set(p2))
    vid = {pid: i for i, pid in enumerate(ids)}
    n = len(ids)
    u = np.fromiter((vid[a] for a in p1), dtype=np.int64, count=p1.size)
    v = np.fromiter((vid[b] for b in p2), dtype=np.int64, count=p2.size)
    return cos, sibling, n, u, v


def _build_boot(cos, sibling, n, u, v):
    """Return ``(boot_statistic, undefined_counter)`` for :func:`stats.vertex_bca_ci`.

    The vertex bootstrap of pair ``(u, v)``, ``u != v``, has multiplicity
    ``count(u)·count(v)`` (mirrors ``stats._induced_pair_values``' triu-over-positions
    semantics; all Bromberg pairs have ``u != v``). ``undefined_counter`` is a 1-element
    list incremented on each draw that lost a class (so the caller can read the final
    count after the bootstrap).
    """
    undefined = [0]

    def _boot(idx: np.ndarray) -> float:
        counts = np.bincount(idx, minlength=n)
        w = counts[u].astype(np.float64) * counts[v].astype(np.float64)
        active = w > 0
        if not active.any():
            undefined[0] += 1
            return float("nan")
        val = weighted_concordance_auc(cos[active], sibling[active], w[active])
        if not np.isfinite(val):
            undefined[0] += 1
            return float("nan")
        return val

    return _boot, undefined


def _build_jackknife(per_pair_df: pd.DataFrame, *, _prepared=None):
    """Build the INCREMENTAL leave-one-orphan-out AUROC closure.

    Returns ``(n, jack_fn, ids, u, v)`` (the extra returns are a test seam). Precomputes,
    ONCE: each pair's concordance contribution against the FULL opposite class
    (``_sorted_kernel_contrib`` / ``_kernel_above``), the full numerator ``C_full``, and
    the orphan->incident-pair index. ``jack_fn(k)`` then removes only the pairs incident
    to ``k`` and updates the U counts by inclusion-exclusion — O(deg(k) + deg_pos(k)·
    deg_neg(k)) per call, NOT O(pairs) rebuilt from scratch.
    """
    if _prepared is None:
        cos, sibling, n, u, v = _prepare_vertices(per_pair_df)
    else:
        cos, sibling, n, u, v = _prepared
    m = u.size

    pos_mask = sibling
    neg_mask = ~sibling
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    # per-row concordance contributions vs the FULL opposite class.
    against_neg_pos = _sorted_kernel_contrib(cos[pos_mask], cos[neg_mask], np.ones(n_neg))
    # a neg row's contribution to C is the count of positives ABOVE it (+0.5 ties).
    against_pos_neg = _kernel_above(cos[neg_mask], cos[pos_mask], np.ones(n_pos))
    C_full = float(against_neg_pos.sum())  # == against_pos_neg.sum()

    pos_rows = np.where(pos_mask)[0]
    neg_rows = np.where(neg_mask)[0]
    row_to_posidx = {int(r): i for i, r in enumerate(pos_rows)}
    row_to_negidx = {int(r): j for j, r in enumerate(neg_rows)}

    incident: list[list[int]] = [[] for _ in range(n)]
    for r in range(m):
        incident[int(u[r])].append(r)
        incident[int(v[r])].append(r)

    def _jack(k: int) -> float:
        rows = incident[k]
        if not rows:
            if n_pos == 0 or n_neg == 0:
                return float("nan")
            return C_full / (n_pos * n_neg)
        rem_pos = [r for r in rows if pos_mask[r]]
        rem_neg = [r for r in rows if neg_mask[r]]
        new_npos = n_pos - len(rem_pos)
        new_nneg = n_neg - len(rem_neg)
        if new_npos <= 0 or new_nneg <= 0:
            return float("nan")
        # ΔC = (removed pos vs ALL neg) + (removed neg vs ALL pos)
        #      − (removed pos vs removed neg)  [inclusion-exclusion: counted twice]
        delta = 0.0
        for r in rem_pos:
            delta += against_neg_pos[row_to_posidx[r]]
        for r in rem_neg:
            delta += against_pos_neg[row_to_negidx[r]]
        if rem_pos and rem_neg:
            cp = cos[rem_pos]
            cn = cos[rem_neg]
            gt = (cp[:, None] > cn[None, :]).sum()
            eq = (cp[:, None] == cn[None, :]).sum()
            delta -= gt + 0.5 * eq
        return (C_full - delta) / (new_npos * new_nneg)

    ids = sorted(set(per_pair_df["p1"].astype(str)) | set(per_pair_df["p2"].astype(str)))
    return n, _jack, ids, u, v


def _kernel_above(
    cos_self: np.ndarray, cos_other: np.ndarray, w_other: np.ndarray
) -> np.ndarray:
    """For each self row, ``Σ_j w_other[j] · (1[cos_other_j>cos_self] + 0.5·1[==])``.

    The role-swapped companion of :func:`_sorted_kernel_contrib`: counts the weight of the
    *other* class lying ABOVE this row (used for a neg row's concordance contribution,
    which is the count of positives above it). O((m+k) log k).
    """
    if cos_other.size == 0:
        return np.zeros(cos_self.size, dtype=np.float64)
    order = np.argsort(cos_other, kind="mergesort")
    o_sorted = cos_other[order]
    w_sorted = w_other[order]
    cum = np.concatenate([[0.0], np.cumsum(w_sorted)])
    total = cum[-1]
    lo = np.searchsorted(o_sorted, cos_self, side="left")
    hi = np.searchsorted(o_sorted, cos_self, side="right")
    # strictly above = total − weight(<= self) = total − cum[hi]
    above = total - cum[hi]
    equal = cum[hi] - cum[lo]
    return above + 0.5 * equal
