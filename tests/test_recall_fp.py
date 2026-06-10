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
    # All five per-query recalls asserted so the 0.6 mean can't be hit by a
    # compensating swap of two unverified queries.
    assert per_q.loc["P1", "recall"] == pytest.approx(0.5)
    assert per_q.loc["P2", "recall"] == pytest.approx(0.5)
    assert per_q.loc["P3", "recall"] == pytest.approx(0.0)
    assert per_q.loc["P4", "recall"] == pytest.approx(1.0)
    assert per_q.loc["P5", "recall"] == pytest.approx(1.0)
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
    with pytest.raises(KeyError, match="class"):
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


# ── W1: fail-closed guard on null-label levels (the family=None footgun) ──────
def _none_family_db():
    embeddings = {
        "P1": np.array([0.0], dtype=np.float32),
        "P2": np.array([1.0], dtype=np.float32),
        "P3": np.array([100.0], dtype=np.float32),
    }
    labels = pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3"],
            "fold": ["A", "A", "B"],
            "superfamily": ["A1", "A1", "B1"],
            "family": [None, None, None],
        }
    )
    return embeddings, labels


def test_recall_fails_closed_on_all_none_level():
    # Without a predicate, scalar equality on all-None labels would treat
    # None == None as a match for every pair and fabricate recall = 1.0.
    embeddings, labels = _none_family_db()
    with pytest.raises(ValueError, match="family"):
        recall_at_first_fp(embeddings, labels, distance="euclidean", level="family")


def test_recall_fails_closed_on_all_nan_level():
    # NaN-float nulls (e.g. from a pandas merge with missing rows) must trip the
    # same guard as None — not silently drop every query (nan != nan).
    embeddings, labels = _none_family_db()
    labels = labels.copy()
    labels["family"] = [float("nan"), float("nan"), float("nan")]
    with pytest.raises(ValueError, match="family"):
        recall_at_first_fp(embeddings, labels, distance="euclidean", level="family")


def test_recall_fails_closed_on_partial_null_level():
    # A single stray null in an otherwise-real level is also fail-closed (the
    # caller must decide how to handle it rather than get a silent under/over-count).
    embeddings, labels = _none_family_db()
    labels = labels.copy()
    labels["superfamily"] = ["A1", None, "B1"]
    with pytest.raises(ValueError, match="superfamily"):
        recall_at_first_fp(embeddings, labels, distance="euclidean", level="superfamily")


def test_recall_fails_closed_on_set_valued_level_without_predicate():
    # Set-valued labels without a predicate would score exact-set identity and
    # silently undercount multi-domain positives -> fail closed, name the builder.
    embeddings = {
        "Q": np.array([0.0], dtype=np.float32),
        "A": np.array([1.0], dtype=np.float32),
    }
    labels = pd.DataFrame(
        {
            "protein_id": ["Q", "A"],
            "fold": [frozenset({"d1"}), frozenset({"d1"})],
            "superfamily": [frozenset({"s1"}), frozenset({"s1"})],
            "family": [None, None],
        }
    )
    with pytest.raises(ValueError, match="is_positive_fn"):
        recall_at_first_fp(embeddings, labels, distance="euclidean", level="fold")


def test_recall_fails_closed_on_mixed_str_and_set_level():
    # A column mixing scalar strings and frozensets (e.g. a partial parse or a
    # concat of two frames) is set-valued enough to be unscorable by scalar ==.
    embeddings = {
        "Q": np.array([0.0], dtype=np.float32),
        "A": np.array([1.0], dtype=np.float32),
    }
    labels = pd.DataFrame(
        {
            "protein_id": ["Q", "A"],
            "fold": ["A1", frozenset({"d1"})],
            "superfamily": ["S1", "S1"],
            "family": [None, None],
        }
    )
    with pytest.raises(ValueError, match="is_positive_fn"):
        recall_at_first_fp(embeddings, labels, distance="euclidean", level="fold")


def test_recall_null_level_ok_with_predicate():
    # A predicate overrides scalar equality, so a null label column is fine.
    embeddings, labels = _none_family_db()
    result = recall_at_first_fp(
        embeddings, labels, distance="euclidean", level="family",
        is_positive_fn=lambda q, t: labels.set_index("protein_id").loc[q, "fold"]
        == labels.set_index("protein_id").loc[t, "fold"],
    )
    assert result["n_queries_with_positives"] == 2  # P1, P2 share fold A


def test_multi_level_skips_unavailable_null_level():
    # family is all-None (unacquired) -> scored levels exclude it (no fabricated
    # 1.0); it is reported as skipped, not silently dropped.
    embeddings, labels = _none_family_db()
    out = recall_at_first_fp_multi_level(embeddings, labels, distance="euclidean")
    assert out["fold"]["scored"] is True
    assert out["superfamily"]["scored"] is True
    assert out["family"]["scored"] is False
    assert "mean_recall_1stFP" in out["fold"]
    assert "mean_recall_1stFP" not in out["family"]
    assert "skipped_reason" in out["family"]


def test_multi_level_uses_positive_predicate_builder():
    # With a builder, multi_level scores multi-domain frozenset labels by set
    # intersection instead of exact-set equality (which would undercount).
    embeddings = {
        "Q": np.array([0.0], dtype=np.float32),
        "A": np.array([1.0], dtype=np.float32),
        "B": np.array([2.0], dtype=np.float32),
    }
    labels = pd.DataFrame(
        {
            "protein_id": ["Q", "A", "B"],
            "fold": [frozenset({"d1", "d2"}), frozenset({"d2"}), frozenset({"d3"})],
            "superfamily": [frozenset({"s1"}), frozenset({"s2"}), frozenset({"s3"})],
            "family": [None, None, None],
        }
    )

    def builder(lab, level):
        lk = dict(zip(lab["protein_id"], lab[level]))
        return lambda q, t: len(lk[q] & lk[t]) > 0

    out = recall_at_first_fp_multi_level(
        embeddings, labels, distance="euclidean", is_positive_fn_builder=builder
    )
    per_q = out["fold"]["per_query"].set_index("query_id")
    # Q shares d2 with A (positive), B is disjoint (FP) -> recall(Q) = 1.0.
    assert per_q.loc["Q", "n_positives"] == 1
    assert per_q.loc["Q", "recall"] == pytest.approx(1.0)
    assert out["fold"]["scored"] is True
    assert out["superfamily"]["scored"] is True
    assert out["family"]["scored"] is False
    assert "skipped_reason" in out["family"]


def test_multi_level_partial_null_level_raises_without_builder():
    # Documents the all-or-nothing assumption: multi_level only *skips* a level
    # that is entirely null (isna().all()). A partially-null level is passed
    # through and trips recall_at_first_fp's per-row guard, aborting the sweep.
    # (Real partial-coverage handling is a W3 concern once family labels land.)
    embeddings, labels = _none_family_db()
    labels = labels.copy()
    labels["superfamily"] = ["A1", None, "B1"]
    with pytest.raises(ValueError, match="superfamily"):
        recall_at_first_fp_multi_level(embeddings, labels, distance="euclidean")
