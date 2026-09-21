"""Component (cluster) bootstrap BCa CI for the orphan sibling AUROC.

The shipped :mod:`orphan_auroc_ci` resamples ORPHANS (vertices) -- the right unit for a
dyadic U-statistic. This module is the stricter sensitivity companion: it resamples
CONNECTED COMPONENTS of the orphan pair graph.

Why that is well posed here: every Bromberg pair lies entirely inside one connected
component of the pair graph (that is what a connected component is), so the components
partition the 309,549 pairs into disjoint blocks. Resampling components with replacement
is therefore a textbook cluster bootstrap with the clusters as the exchangeable unit,
and a pair's bootstrap weight is simply the number of times its component was drawn
(linear), not ``count(u) * count(v)``.

It is deliberately the CONSERVATIVE reading: the orphan pair graph has one giant
component, so the effective number of independent clusters is far smaller than the
component count and the interval widens accordingly. Report it beside the vertex
interval, not instead of it.

Reuses the shipped generic core ``stats.vertex_bca_ci`` (n = number of components) and
the shipped weighted concordance kernel ``orphan_auroc_ci.weighted_concordance_auc``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.orphan_auroc_ci import weighted_concordance_auc
from evaluation.stats import vertex_bca_ci


def pair_components(per_pair_df: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    """Label each pair with its connected component id in the orphan pair graph.

    Returns ``(component_of_pair, n_components, n_vertices)``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    p1 = per_pair_df["p1"].astype(str).to_numpy()
    p2 = per_pair_df["p2"].astype(str).to_numpy()
    ids = sorted(set(p1) | set(p2))
    vid = {p: i for i, p in enumerate(ids)}
    n = len(ids)
    u = np.fromiter((vid[a] for a in p1), dtype=np.int64, count=p1.size)
    v = np.fromiter((vid[b] for b in p2), dtype=np.int64, count=p2.size)
    g = coo_matrix((np.ones(u.size), (u, v)), shape=(n, n))
    n_comp, label = connected_components(g, directed=False)
    return label[u], int(n_comp), n


def orphan_auroc_component_bca_ci(
    per_pair_df: pd.DataFrame,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | np.random.Generator | None = 42,
) -> dict:
    """Cluster-bootstrap BCa CI for the sibling AUROC, resampling connected components."""
    comp_of_pair, n_comp, n_vert = pair_components(per_pair_df)
    cos = per_pair_df["cos"].to_numpy(dtype=np.float64)
    sibling = per_pair_df["sibling"].to_numpy().astype(bool)
    ones = np.ones(cos.size)
    point = weighted_concordance_auc(cos, sibling, ones)

    undefined = [0]

    def _boot(idx: np.ndarray) -> float:
        counts = np.bincount(idx, minlength=n_comp).astype(np.float64)
        w = counts[comp_of_pair]
        active = w > 0
        if not active.any():
            undefined[0] += 1
            return float("nan")
        val = weighted_concordance_auc(cos[active], sibling[active], w[active])
        if not np.isfinite(val):
            undefined[0] += 1
            return float("nan")
        return val

    def _jack(k: int) -> float:
        keep = comp_of_pair != k
        if not keep.any():
            return float("nan")
        return weighted_concordance_auc(cos[keep], sibling[keep], ones[keep])

    lo, hi, point_out, degenerate, diverged = vertex_bca_ci(
        n_comp,
        point=point,
        boot_statistic=_boot,
        jackknife_statistic=_jack,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        clip=(0.0, 1.0),
    )
    return {
        "point": float(point_out),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "degenerate": bool(degenerate),
        "diverged": bool(diverged),
        "n_boot_undefined": int(undefined[0]),
        "n_components": int(n_comp),
        "n_vertices": int(n_vert),
    }
