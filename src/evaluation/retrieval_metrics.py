# --- Ivan infrastructure (2026-03-19) ---
"""Retrieval-style evaluation metrics for pLM embedding quality.

In the pLM choice framework, we evaluate whether embeddings from different
protein language models capture biologically meaningful similarity. Given a
query protein and a set of candidates with known relationships (e.g., same
EC class, same fold, same family), we rank candidates by embedding distance
and measure how well the ranking separates true positives from negatives.

Two complementary metrics:

- **Recall at first false positive (recall@1FP):** Walk down the ranked list
  and count how many true positives appear before the first false positive.
  This measures the "safe retrieval zone" — how many correct hits a user
  would find before encountering the first mistake. Stringent but
  interpretable, especially for curators scanning ranked lists.

- **AUROC:** Area under the ROC curve over the full ranking. Measures global
  discriminative ability across all thresholds. More robust to individual
  outliers but less interpretable for top-of-list quality.

Both metrics are computed per-query and then aggregated across queries to
compare pLMs at different levels of biological hierarchy (fold, superfamily,
family, EC level, etc.).
"""

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def recall_at_first_fp(
    distances: np.ndarray,
    labels: np.ndarray,
    lower_is_similar: bool = True,
) -> Dict[str, float]:
    """Compute recall at the point of the first false positive.

    Sorts (distance, label) pairs and scans in order of increasing similarity.
    Counts consecutive true positives until the first false positive is
    encountered. Returns the fraction of all positives retrieved before that
    point.

    Parameters
    ----------
    distances : np.ndarray
        Pairwise distances (or scores) between a query and candidates.
    labels : np.ndarray
        Boolean array — True if the candidate is a true positive.
    lower_is_similar : bool
        If True (default), smaller distances mean higher similarity.
        If False, larger values mean higher similarity (scores).

    Returns
    -------
    dict
        Keys: ``recall_at_first_fp``, ``n_retrieved``, ``n_positives``.
    """
    distances = np.asarray(distances, dtype=float)
    labels = np.asarray(labels, dtype=bool)

    n_positives = int(labels.sum())

    if n_positives == 0:
        return {"recall_at_first_fp": 0.0, "n_retrieved": 0, "n_positives": 0}

    # Sort by distance (ascending = most similar first when lower_is_similar)
    order = np.argsort(distances) if lower_is_similar else np.argsort(-distances)
    sorted_labels = labels[order]

    # No negatives → everything is a true positive
    if n_positives == len(labels):
        return {
            "recall_at_first_fp": 1.0,
            "n_retrieved": n_positives,
            "n_positives": n_positives,
        }

    # Count consecutive TPs before first FP
    n_retrieved = 0
    for lab in sorted_labels:
        if lab:
            n_retrieved += 1
        else:
            break

    recall = n_retrieved / n_positives

    return {
        "recall_at_first_fp": recall,
        "n_retrieved": n_retrieved,
        "n_positives": n_positives,
    }


def auroc_at_level(
    distances: np.ndarray,
    labels: np.ndarray,
    lower_is_similar: bool = True,
) -> float:
    """Compute AUROC for a retrieval ranking at one hierarchy level.

    Parameters
    ----------
    distances : np.ndarray
        Pairwise distances (or scores) between a query and candidates.
    labels : np.ndarray
        Boolean array — True if the candidate is a true positive.
    lower_is_similar : bool
        If True (default), smaller distances mean higher similarity,
        so distances are negated before passing to ``roc_auc_score``.
        If False, values are used directly as similarity scores.

    Returns
    -------
    float
        AUROC value, or ``np.nan`` if only one class is present or all
        distances are NaN.
    """
    distances = np.asarray(distances, dtype=float)
    labels = np.asarray(labels, dtype=bool)

    # Filter NaN distances
    valid = ~np.isnan(distances)
    if valid.sum() < 2:
        return np.nan

    distances = distances[valid]
    labels = labels[valid]

    # Need both classes for AUROC
    if labels.all() or (~labels).all():
        return np.nan

    # sklearn expects higher score = more likely positive
    scores = -distances if lower_is_similar else distances

    return float(roc_auc_score(labels, scores))


def evaluate_retrieval(
    pairs_df,
    distance_col: str,
    label_col: str,
    lower_is_similar: bool = True,
) -> Dict[str, float]:
    """Convenience wrapper: extract columns from a polars DataFrame and compute
    both retrieval metrics.

    Parameters
    ----------
    pairs_df : polars.DataFrame
        DataFrame containing at least ``distance_col`` and ``label_col``.
    distance_col : str
        Name of the column with distance/score values.
    label_col : str
        Name of the column with boolean labels.
    lower_is_similar : bool
        Passed through to both metric functions.

    Returns
    -------
    dict
        Combined results from ``recall_at_first_fp`` and ``auroc_at_level``.
    """
    distances = pairs_df[distance_col].to_numpy().astype(float)
    labels = pairs_df[label_col].to_numpy().astype(bool)

    # Filter rows where distance is NaN
    valid = ~np.isnan(distances)
    distances = distances[valid]
    labels = labels[valid]

    results = recall_at_first_fp(distances, labels, lower_is_similar=lower_is_similar)
    results["auroc"] = auroc_at_level(
        distances, labels, lower_is_similar=lower_is_similar
    )

    return results
