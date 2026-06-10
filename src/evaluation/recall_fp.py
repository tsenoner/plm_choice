"""Recall-at-first-FP per Lin et al. 2023 (Nat Biotech).

For each query protein q, rank every other protein in the embedding set by
distance ascending. Count the fraction of the query's total positives (same
CATH Fold/Superfamily/Family, or an arbitrary predicate) retrieved BEFORE the
first false positive. The reported headline is the mean of this fraction over
queries that have at least one positive in the lookup database.

Edge cases (locked here so the figure pipeline can rely on them)
----------------------------------------------------------------
- ``n_positives == 0``: query dropped from the mean; the count of skipped
  queries is reported as ``n_queries_skipped_no_positives``.
- Ties at a distance: adversarial strict walk (Lin et al. 2023). A positive at
  exactly the first-FP distance counts AFTER the FP, so only positives strictly
  closer than the nearest FP are credited. The number of positives tied at that
  distance is reported per query as ``n_ties_at_first_fp``. Order-independent
  (no argsort tie-break bias). Matters most for the discrete AAC 20-d floor.
- ``query_id`` is excluded from its own ranked list (self-match).
- Proteins in ``embeddings`` but not in ``labels`` are silently dropped from the
  lookup database (defensive; callers should keep them aligned).
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

DistanceName = Literal["cosine", "euclidean", "manhattan"]
LevelName = Literal["fold", "superfamily", "family"]

_DISTANCE_METRIC_MAP = {
    "cosine": "cosine",
    "euclidean": "euclidean",
    "manhattan": "cityblock",
}


def _is_null_label(v) -> bool:
    """True for None or a NaN float (both mean "no label")."""
    return v is None or (isinstance(v, float) and v != v)


def _stack_embeddings(
    embeddings: dict[str, np.ndarray], labels: pd.DataFrame, level: str
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Stack embeddings + matching label vector for proteins present in both."""
    if level not in labels.columns:
        raise KeyError(f"level={level!r} not in labels columns {list(labels.columns)}")

    label_lookup = dict(zip(labels["protein_id"], labels[level]))
    ids = [pid for pid in embeddings if pid in label_lookup]
    if len(ids) < 2:
        raise ValueError(
            f"Need >=2 proteins present in both embeddings and labels (got {len(ids)})"
        )

    matrix = np.stack([np.asarray(embeddings[pid], dtype=np.float32) for pid in ids])
    label_vec = np.array([label_lookup[pid] for pid in ids], dtype=object)
    return matrix, ids, label_vec


def _recall_one_query(
    distances_row: np.ndarray, is_positive_row: np.ndarray, query_idx: int
) -> tuple[float | None, int, int]:
    """Recall-at-first-FP for one query (adversarial ties).

    ``is_positive_row`` is a boolean array over all proteins (aligned with
    ``distances_row``); the query's own position is excluded via ``query_idx``.
    Returns ``(recall, n_positives, n_ties_at_first_fp)`` or ``(None, 0, 0)``
    if the query has no positives.
    """
    mask = np.ones_like(distances_row, dtype=bool)
    mask[query_idx] = False
    dd = distances_row[mask]
    is_positive = np.asarray(is_positive_row, dtype=bool)[mask]
    n_positives = int(is_positive.sum())
    if n_positives == 0:
        return None, 0, 0

    fp_dists = dd[~is_positive]
    pos_dists = dd[is_positive]
    if fp_dists.size == 0:
        # No false positives: every positive precedes the (non-existent) first FP.
        return 1.0, n_positives, 0

    d_fp = float(fp_dists.min())
    positives_before_first_fp = int(np.count_nonzero(pos_dists < d_fp))
    n_ties_at_first_fp = int(np.count_nonzero(pos_dists == d_fp))
    return positives_before_first_fp / n_positives, n_positives, n_ties_at_first_fp


def recall_at_first_fp(
    embeddings: dict[str, np.ndarray],
    labels: pd.DataFrame,
    distance: DistanceName = "cosine",
    level: LevelName = "fold",
    per_query: bool = True,
    is_positive_fn: "Callable[[str, str], bool] | None" = None,
) -> dict:
    """Per-query recall-at-first-FP at a single CATH level.

    Parameters
    ----------
    embeddings
        ``{protein_id: 1-D np.ndarray}`` of per-protein embeddings.
    labels
        DataFrame with ``protein_id`` and the CATH ``level`` column. The level
        column also defines the lookup population (proteins in both inputs).
    distance
        ``"cosine"``, ``"euclidean"``, or ``"manhattan"``.
    level
        CATH column used as the positive class (and population). Ignored for
        *positivity* when ``is_positive_fn`` is given, but must exist.
    per_query
        If True, include a per-query DataFrame in the return value.
    is_positive_fn
        Optional ``(query_id, target_id) -> bool`` predicate. When given, it —
        not scalar label equality — decides positivity, so multi-domain
        set-intersection positives (target shares ANY CATH domain with the
        query) can be expressed. When ``None``, positivity is scalar equality on
        ``labels[level]``.

    Returns
    -------
    dict
        ``mean_recall_1stFP``, ``n_queries_with_positives``,
        ``n_queries_skipped_no_positives``, ``level``, ``distance``, and (if
        requested) ``per_query`` = DataFrame[query_id, n_positives, recall,
        n_ties_at_first_fp].
    """
    if distance not in _DISTANCE_METRIC_MAP:
        raise ValueError(f"distance={distance!r} not in {list(_DISTANCE_METRIC_MAP)}")
    metric = _DISTANCE_METRIC_MAP[distance]
    matrix, ids, label_vec = _stack_embeddings(embeddings, labels, level)

    if is_positive_fn is None:
        if any(_is_null_label(v) for v in label_vec):
            raise ValueError(
                f"level={level!r} has null label(s) (None/NaN) and no is_positive_fn "
                f"was given; scalar-equality positivity would treat null == null as a "
                f"match (fabricating recall) or silently drop every query. Provide "
                f"is_positive_fn or score an available level (e.g. the family column is "
                f"a placeholder until real labels are joined)."
            )
        if any(isinstance(v, (set, frozenset)) for v in label_vec):
            raise ValueError(
                f"level={level!r} has set-valued labels and no is_positive_fn was "
                f"given; scalar equality on sets scores exact-set identity and "
                f"undercounts multi-domain positives. Pass is_positive_fn, e.g. "
                f"label_adapters.make_cath_is_positive_fn(labels, {level!r})."
            )

    dmat = cdist(matrix, matrix, metric=metric)

    records: list[tuple[str, int, float, int]] = []
    skipped = 0
    for i, pid in enumerate(ids):
        if is_positive_fn is None:
            is_pos_row = label_vec == label_vec[i]
        else:
            is_pos_row = np.fromiter(
                (is_positive_fn(pid, ids[j]) for j in range(len(ids))),
                dtype=bool,
                count=len(ids),
            )
        recall, n_pos, n_ties = _recall_one_query(dmat[i], is_pos_row, i)
        if recall is None:
            skipped += 1
            continue
        records.append((pid, n_pos, recall, n_ties))

    per_query_df = pd.DataFrame(
        records, columns=["query_id", "n_positives", "recall", "n_ties_at_first_fp"]
    )
    mean_recall = (
        float(per_query_df["recall"].mean()) if len(per_query_df) else float("nan")
    )

    out: dict = {
        "mean_recall_1stFP": mean_recall,
        "n_queries_with_positives": len(per_query_df),
        "n_queries_skipped_no_positives": skipped,
        "level": level,
        "distance": distance,
    }
    if per_query:
        out["per_query"] = per_query_df
    return out


def recall_at_first_fp_multi_level(
    embeddings: dict[str, np.ndarray],
    labels: pd.DataFrame,
    distance: DistanceName = "cosine",
    is_positive_fn_builder: "Callable[[pd.DataFrame, str], Callable[[str, str], bool]] | None" = None,
) -> dict[str, dict]:
    """Run :func:`recall_at_first_fp` at all three CATH levels.

    A level whose label column is entirely null (e.g. the ``family`` placeholder
    before real CATH family labels are joined in) is **not scored** — it would
    fabricate a perfect recall under scalar equality. Such a level appears in the
    result with a ``skipped_reason`` instead of metrics (visible, not silently
    dropped).

    Every value carries a ``scored`` bool so callers can branch on a guaranteed
    field rather than probing for ``mean_recall_1stFP`` vs ``skipped_reason``.

    Parameters
    ----------
    is_positive_fn_builder
        Optional ``(labels, level) -> predicate`` factory. When given, each
        scored level uses the built ``is_positive_fn`` (e.g. CATH set
        intersection from
        :func:`evaluation.label_adapters.make_cath_is_positive_fn`) instead of
        scalar label equality — required for correct multi-domain (set-valued)
        labels.
    """
    out: dict[str, dict] = {}
    for level in ("fold", "superfamily", "family"):
        if level not in labels.columns or labels[level].isna().all():
            out[level] = {
                "level": level,
                "scored": False,
                "skipped_reason": (
                    f"level {level!r} unavailable (column missing or all-null); "
                    f"not scored to avoid fabricated recall"
                ),
            }
            continue
        is_pos = (
            is_positive_fn_builder(labels, level) if is_positive_fn_builder else None
        )
        res = recall_at_first_fp(
            embeddings,
            labels,
            distance=distance,
            level=level,
            per_query=True,
            is_positive_fn=is_pos,
        )
        res["scored"] = True
        out[level] = res
    return out
