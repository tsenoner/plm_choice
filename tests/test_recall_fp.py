"""Tests for src/evaluation/recall_fp.py (recall-at-first-FP, Lin et al. 2023)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.recall_fp import (
    recall_at_first_fp,
    recall_at_first_fp_multi_level,
)


@pytest.fixture
def tiny_db():
    """5 proteins, 2 folds, embeddings on a line.

    Fold A: P1@0, P2@1, P3@200; Fold B: P4@100, P5@101.
    Hand-computed mean recall at fold level = 0.6 (see assertions below).
    """
    embeddings = {
        "P1": np.array([0.0], dtype=np.float32),
        "P2": np.array([1.0], dtype=np.float32),
        "P3": np.array([200.0], dtype=np.float32),
        "P4": np.array([100.0], dtype=np.float32),
        "P5": np.array([101.0], dtype=np.float32),
    }
    labels = pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3", "P4", "P5"],
            "fold": ["A", "A", "A", "B", "B"],
            "superfamily": ["A1", "A1", "A2", "B1", "B1"],
            "family": ["A1a", "A1b", "A2a", "B1a", "B1b"],
        }
    )
    return embeddings, labels


def test_recall_at_first_fp_hand_computable(tiny_db):
    embeddings, labels = tiny_db
    result = recall_at_first_fp(embeddings, labels, distance="euclidean", level="fold")
    assert result["n_queries_with_positives"] == 5
    assert result["n_queries_skipped_no_positives"] == 0
    assert result["mean_recall_1stFP"] == pytest.approx(0.6, abs=1e-9)
    per_q = result["per_query"].set_index("query_id")
    assert per_q.loc["P1", "recall"] == pytest.approx(0.5)
    assert per_q.loc["P3", "recall"] == pytest.approx(0.0)
    assert per_q.loc["P4", "recall"] == pytest.approx(1.0)
    assert per_q.loc["P1", "n_positives"] == 2


def test_recall_skips_singleton_classes_at_family_level(tiny_db):
    embeddings, labels = tiny_db
    result = recall_at_first_fp(embeddings, labels, distance="euclidean", level="family")
    assert result["n_queries_with_positives"] == 0
    assert result["n_queries_skipped_no_positives"] == 5
    assert np.isnan(result["mean_recall_1stFP"])


def test_recall_superfamily_level(tiny_db):
    embeddings, labels = tiny_db
    result = recall_at_first_fp(
        embeddings, labels, distance="euclidean", level="superfamily"
    )
    assert result["n_queries_with_positives"] == 4
    assert result["n_queries_skipped_no_positives"] == 1
    assert result["mean_recall_1stFP"] == pytest.approx(1.0)


def test_recall_multi_level(tiny_db):
    embeddings, labels = tiny_db
    out = recall_at_first_fp_multi_level(embeddings, labels, distance="euclidean")
    assert set(out) == {"fold", "superfamily", "family"}
    assert out["fold"]["mean_recall_1stFP"] == pytest.approx(0.6)


def test_recall_invalid_distance(tiny_db):
    embeddings, labels = tiny_db
    with pytest.raises(ValueError, match="distance"):
        recall_at_first_fp(embeddings, labels, distance="hamming", level="fold")


def test_recall_missing_level(tiny_db):
    embeddings, labels = tiny_db
    with pytest.raises(KeyError):
        recall_at_first_fp(embeddings, labels, distance="euclidean", level="class")


def test_recall_drops_proteins_without_labels(tiny_db):
    embeddings, labels = tiny_db
    embeddings = dict(embeddings)
    embeddings["P_orphan"] = np.array([50.0], dtype=np.float32)
    result = recall_at_first_fp(embeddings, labels, distance="euclidean", level="fold")
    assert result["n_queries_with_positives"] == 5
    assert "P_orphan" not in result["per_query"]["query_id"].tolist()


def test_recall_adversarial_ties_excludes_tied_positive():
    """A positive tied (identical distance) with the first FP counts AFTER it
    (Lin et al. strict walk). Q@0; positives A@1, B@5 (fold X); FP C@-5 (fold Y)
    -> only A counts. recall = 1/2 = 0.5; n_ties_at_first_fp = 1."""
    embeddings = {
        "Q": np.array([0.0], dtype=np.float32),
        "A": np.array([1.0], dtype=np.float32),
        "B": np.array([5.0], dtype=np.float32),
        "C": np.array([-5.0], dtype=np.float32),
    }
    labels = pd.DataFrame(
        {
            "protein_id": ["Q", "A", "B", "C"],
            "fold": ["X", "X", "X", "Y"],
            "superfamily": ["X1", "X1", "X1", "Y1"],
            "family": ["Xa", "Xb", "Xc", "Ya"],
        }
    )
    result = recall_at_first_fp(embeddings, labels, distance="euclidean", level="fold")
    per_q = result["per_query"].set_index("query_id")
    assert per_q.loc["Q", "n_positives"] == 2
    assert per_q.loc["Q", "recall"] == pytest.approx(0.5)
    assert per_q.loc["Q", "n_ties_at_first_fp"] == 1


def test_recall_no_fp_retrieves_all_positives():
    embeddings = {
        "Q": np.array([0.0], dtype=np.float32),
        "A": np.array([1.0], dtype=np.float32),
        "B": np.array([2.0], dtype=np.float32),
    }
    labels = pd.DataFrame(
        {
            "protein_id": ["Q", "A", "B"],
            "fold": ["X", "X", "X"],
            "superfamily": ["X1", "X1", "X1"],
            "family": ["Xa", "Xb", "Xc"],
        }
    )
    result = recall_at_first_fp(embeddings, labels, distance="euclidean", level="fold")
    per_q = result["per_query"].set_index("query_id")
    assert per_q.loc["Q", "recall"] == pytest.approx(1.0)
    assert per_q.loc["Q", "n_ties_at_first_fp"] == 0


def test_recall_positive_predicate_multidomain_set_intersection():
    """An additive is_positive_fn overrides scalar-label equality, expressing
    multi-domain set-intersection positives. Domains: Q={d1,d2}, A={d2}, B={d3};
    Q@0, A@1, B@2. A is positive (shares d2), B is an FP -> recall(Q)=1.0."""
    embeddings = {
        "Q": np.array([0.0], dtype=np.float32),
        "A": np.array([1.0], dtype=np.float32),
        "B": np.array([2.0], dtype=np.float32),
    }
    labels = pd.DataFrame(
        {
            "protein_id": ["Q", "A", "B"],
            "fold": ["F1", "F2", "F3"],
            "superfamily": ["S1", "S2", "S3"],
            "family": ["fa", "fb", "fc"],
        }
    )
    domains = {"Q": {"d1", "d2"}, "A": {"d2"}, "B": {"d3"}}

    def shares_domain(q: str, t: str) -> bool:
        return len(domains[q] & domains[t]) > 0

    result = recall_at_first_fp(
        embeddings, labels, distance="euclidean", level="family",
        is_positive_fn=shares_domain,
    )
    per_q = result["per_query"].set_index("query_id")
    assert "Q" in per_q.index
    assert per_q.loc["Q", "n_positives"] == 1
    assert per_q.loc["Q", "recall"] == pytest.approx(1.0)
