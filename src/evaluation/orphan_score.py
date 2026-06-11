"""Per-pair cosine scoring kernel for the orphan arm (lifted from cmd_orphan).

The orphan metric, made a pure tested function over ``(embeddings, pairs)``. It is the
single source of truth for *how a pair is scored* — the report (Unit 4) and the
vertex-AUROC CI (Unit 3) both consume its per-pair frame, so the cosine convention lives
in exactly one place.

The math, read from ``run_pipeline.cmd_orphan`` (the source of truth):

1. L2-normalise each embedding vector ONCE.
2. For each pair present in BOTH the embeddings and the pairs file, ``cos = dot(â, b̂)``.
3. Scalars: ``siblings_AUROC`` (sklearn ``roc_auc_score`` over the pairs' own ``sibling``
   column), ``spearman_cos_vs_SNN``, ``spearman_cos_vs_TM`` (scipy ``spearmanr``), plus
   the bookkeeping counts.

Pairs whose endpoints are not both in the embeddings are dropped and counted
(``n_pairs_dropped``) — the legacy path dropped them silently.

The embeddings dict is expected per-protein reduced (one ``(D,)`` vector per id) — the
orphan H5 is 128-d reduced (design Q6/R3). ``analysis_io.load_embeddings_h5`` mean-pools
a 2-D ``(L, D)`` dataset to ``(D,)``, so a per-residue H5 would also be accepted, but for
the orphan H5 that pool is a no-op (note: do NOT feed an already-pooled-then-stacked
matrix here — pass the per-id dict).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PER_PAIR_COLUMNS: tuple[str, ...] = ("p1", "p2", "cos", "snn", "tm", "sibling")


def _safe_auroc(sibling: np.ndarray, cos: np.ndarray) -> float:
    """``roc_auc_score`` that returns NaN on the one-class degeneracy instead of raising."""
    if sibling.size == 0 or sibling.all() or not sibling.any():
        return float("nan")  # empty, all-sibling, or no-sibling -> AUROC undefined
    try:
        return float(roc_auc_score(sibling, cos))
    except ValueError:
        return float("nan")


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def score_orphan_pairs(
    embeddings: dict[str, np.ndarray], pairs: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Score the orphan pairs for one pLM. Pure; no I/O.

    Parameters
    ----------
    embeddings:
        ``{protein_id: 1-D np.ndarray}`` (per-protein reduced).
    pairs:
        Frame with columns ``[p1, p2, tm, snn, sibling]`` (the
        :func:`orphan_io.load_orphan_pairs` schema).

    Returns
    -------
    ``(per_pair_df, scalars)`` where ``per_pair_df`` has columns
    :data:`PER_PAIR_COLUMNS` (one row per kept pair, in the input order) and ``scalars``
    is ``{siblings_AUROC, spearman_cos_vs_SNN, spearman_cos_vs_TM, n_pairs,
    n_pairs_dropped, n_siblings, n_proteins}``.
    """
    for col in ("p1", "p2", "tm", "snn", "sibling"):
        if col not in pairs.columns:
            raise KeyError(f"pairs frame missing column {col!r}")

    ids = list(embeddings)
    n_proteins = len(ids)
    pos = {pid: i for i, pid in enumerate(ids)}

    if n_proteins == 0:
        empty = pd.DataFrame(columns=list(PER_PAIR_COLUMNS))
        return empty, {
            "siblings_AUROC": float("nan"),
            "spearman_cos_vs_SNN": float("nan"),
            "spearman_cos_vs_TM": float("nan"),
            "n_pairs": 0,
            "n_pairs_dropped": int(len(pairs)),
            "n_siblings": 0,
            "n_proteins": 0,
        }

    # L2-normalise each vector once (cosine similarity becomes a dot product).
    mat = np.stack([np.asarray(embeddings[k], dtype=np.float32) for k in ids])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

    p1 = pairs["p1"].astype(str).to_numpy()
    p2 = pairs["p2"].astype(str).to_numpy()
    keep = np.fromiter(
        ((a in pos and b in pos) for a, b in zip(p1, p2)),
        dtype=bool,
        count=len(pairs),
    )
    idx = np.where(keep)[0]
    n_dropped = int((~keep).sum())

    ia = np.fromiter((pos[p1[i]] for i in idx), dtype=np.int64, count=idx.size)
    ib = np.fromiter((pos[p2[i]] for i in idx), dtype=np.int64, count=idx.size)
    cos = np.sum(mat[ia] * mat[ib], axis=1).astype(np.float64)

    sub = pairs.iloc[idx]
    sibling = sub["sibling"].to_numpy().astype(bool)
    snn = sub["snn"].to_numpy().astype(np.float64)
    tm = sub["tm"].to_numpy().astype(np.float64)

    per_pair = pd.DataFrame(
        {
            "p1": p1[idx],
            "p2": p2[idx],
            "cos": cos,
            "snn": snn,
            "tm": tm,
            "sibling": sibling,
        }
    )[list(PER_PAIR_COLUMNS)]

    scalars = {
        "siblings_AUROC": _safe_auroc(sibling, cos),
        "spearman_cos_vs_SNN": _safe_spearman(cos, snn),
        "spearman_cos_vs_TM": _safe_spearman(cos, tm),
        "n_pairs": int(idx.size),
        "n_pairs_dropped": n_dropped,
        "n_siblings": int(sibling.sum()),
        "n_proteins": n_proteins,
    }
    return per_pair, scalars
