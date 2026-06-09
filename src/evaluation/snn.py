"""Shared-Nearest-Neighbour (SNN) Jaccard agreement between pLMs.

For each protein ``q`` present in two embedding sets ``A`` and ``B``, build its
k-nearest-neighbour set under each pLM (excluding ``q`` itself), then report the
Jaccard index ``|A_k ∩ B_k| / |A_k ∪ B_k|``. The falsification target is "if
pLMs encode the same biology, k-NN agreement on orphans should be >= X" — we
report the raw per-query Jaccard plus mean + 95% BCa-bootstrap CI.

``knn_jaccard_matrix`` returns the N x N grid for the cross-pLM comparison
figure. The public functions take an ``rng`` so the bootstrap CI is reproducible
from a manifest seed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from evaluation.stats import bca_bootstrap

DistanceName = Literal["cosine", "euclidean", "manhattan"]

_SKLEARN_METRIC_MAP = {
    "cosine": "cosine",
    "euclidean": "euclidean",
    "manhattan": "manhattan",
}


def _safe_bootstrap_ci(
    values: np.ndarray,
    B: int = 1000,
    rng: int | np.random.Generator | None = None,
) -> tuple[float, float]:
    """95% BCa CI of the mean via evaluation.stats.bca_bootstrap.

    Returns a zero-width CI when the input is constant — scipy's BCa accelerator
    is undefined there and we don't want NaN leaking into figure pipelines.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    # Constant input — BCa undefined; return zero-width CI at the point estimate.
    if np.allclose(arr, arr[0]):
        return float(arr[0]), float(arr[0])

    _, lo, hi = bca_bootstrap(arr, statistic=np.mean, B=B, rng=rng)
    return float(lo), float(hi)


def _knn_sets(
    embeddings: dict[str, np.ndarray],
    query_ids: list[str],
    k: int,
    distance: DistanceName,
) -> dict[str, set[str]]:
    """Compute the k-NN id-set for each query, excluding the query itself."""
    if distance not in _SKLEARN_METRIC_MAP:
        raise ValueError(
            f"distance={distance!r} not in {list(_SKLEARN_METRIC_MAP)}"
        )
    ids = list(embeddings.keys())
    matrix = np.stack(
        [np.asarray(embeddings[pid], dtype=np.float32) for pid in ids]
    )
    # Request k+1 because the query itself is in the database.
    n_neighbors = min(k + 1, len(ids))
    nn = NearestNeighbors(
        n_neighbors=n_neighbors, metric=_SKLEARN_METRIC_MAP[distance]
    ).fit(matrix)

    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    out: dict[str, set[str]] = {}
    for qid in query_ids:
        if qid not in id_to_idx:
            continue
        qvec = matrix[id_to_idx[qid]].reshape(1, -1)
        _, idx = nn.kneighbors(qvec, n_neighbors=n_neighbors)
        neighbors = [ids[i] for i in idx[0] if ids[i] != qid][:k]
        out[qid] = set(neighbors)
    return out


def knn_jaccard_between_plms(
    embeddings_a: dict[str, np.ndarray],
    embeddings_b: dict[str, np.ndarray],
    k: int = 10,
    distance: DistanceName = "cosine",
    rng: int | np.random.Generator | None = None,
) -> dict:
    """Per-query Jaccard of k-NN sets between two pLMs.

    Args:
        embeddings_a, embeddings_b: ``{protein_id: 1-D np.ndarray}`` per pLM.
            The query set is the intersection of the two id sets.
        k: Number of neighbours per query (default 10).
        distance: Distance metric (same for both pLMs).
        rng: Seed or Generator for the bootstrap CI (reproducibility).

    Returns:
        ``{"mean_jaccard": float, "ci": (low, high), "k": int, "distance": str,
        "per_query": pd.DataFrame[query, jaccard, k_a, k_b]}``. ``ci`` is the
        95% BCa-bootstrap CI of the mean (B=1000).
    """
    common_ids = sorted(set(embeddings_a) & set(embeddings_b))
    if len(common_ids) < 2:
        raise ValueError(
            f"Need >=2 proteins in both embedding sets (got {len(common_ids)})"
        )

    nn_a = _knn_sets(embeddings_a, common_ids, k, distance)
    nn_b = _knn_sets(embeddings_b, common_ids, k, distance)

    records: list[tuple[str, float, int, int]] = []
    for qid in common_ids:
        sa, sb = nn_a[qid], nn_b[qid]
        union = sa | sb
        jacc = len(sa & sb) / len(union) if union else 0.0
        records.append((qid, jacc, len(sa), len(sb)))

    per_query_df = pd.DataFrame(
        records, columns=["query", "jaccard", "k_a", "k_b"]
    )
    jaccards = per_query_df["jaccard"].to_numpy()
    mean_jacc = float(jaccards.mean())

    ci_low, ci_high = _safe_bootstrap_ci(jaccards, rng=rng)

    return {
        "mean_jaccard": mean_jacc,
        "ci": (float(ci_low), float(ci_high)),
        "k": k,
        "distance": distance,
        "per_query": per_query_df,
    }


def knn_jaccard_matrix(
    embeddings_per_plm: dict[str, dict[str, np.ndarray]],
    k: int = 10,
    distance: DistanceName = "cosine",
    rng: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Symmetric N x N matrix of pairwise mean Jaccard across pLMs.

    Diagonal is 1.0 (a pLM agrees with itself). Lower and upper triangles match
    (symmetric statistic).
    """
    plm_names = list(embeddings_per_plm)
    n = len(plm_names)
    matrix = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            res = knn_jaccard_between_plms(
                embeddings_per_plm[plm_names[i]],
                embeddings_per_plm[plm_names[j]],
                k=k,
                distance=distance,
                rng=rng,
            )
            matrix[i, j] = matrix[j, i] = res["mean_jaccard"]
    return pd.DataFrame(matrix, index=plm_names, columns=plm_names)
