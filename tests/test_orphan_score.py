"""Unit 2 — per-pair cosine scoring kernel (lifted from cmd_orphan)."""
import numpy as np
import pandas as pd
import pytest

from evaluation.orphan_score import score_orphan_pairs


def _pairs(rows):
    return pd.DataFrame(rows, columns=["p1", "p2", "tm", "snn", "sibling"])


def test_monotone_cos_sibling_gives_auroc_one():
    # Place 4 unit vectors at increasing angles off the x-axis. Sibling pairs are the
    # small-angle (high-cos) ones; non-siblings the large-angle (low/negative-cos) ones,
    # so the two classes are cleanly separable -> AUROC 1.0.
    emb = {
        "A": np.array([1.0, 0.0], dtype=np.float32),       # 0 deg
        "B": np.array([0.966, 0.259], dtype=np.float32),   # ~15 deg
        "C": np.array([0.0, 1.0], dtype=np.float32),       # 90 deg
        "D": np.array([-1.0, 0.0], dtype=np.float32),      # 180 deg
    }
    pairs = _pairs(
        [
            ("A", "B", 0.9, 0.8, True),    # cos ~0.966  (small angle) -> sibling
            ("B", "C", 0.8, 0.7, True),    # cos ~0.259  (still > the non-siblings) -> sibling
            ("A", "C", 0.1, 0.2, False),   # cos 0
            ("A", "D", 0.0, 0.0, False),   # cos -1
        ]
    )
    per_pair, scalars = score_orphan_pairs(emb, pairs)
    # sibling cos {0.966, 0.259} both strictly above non-sibling cos {0, -1}
    assert scalars["siblings_AUROC"] == pytest.approx(1.0)
    assert scalars["n_pairs"] == 4
    assert scalars["n_siblings"] == 2
    assert scalars["n_proteins"] == 4
    assert list(per_pair.columns) == ["p1", "p2", "cos", "snn", "tm", "sibling"]
    # cosine is dot of L2-normalised vectors -> in [-1, 1]
    assert per_pair["cos"].between(-1.0, 1.0).all()


def test_scrambled_labels_give_auroc_near_half():
    rng = np.random.default_rng(0)
    ids = [f"P{i}" for i in range(12)]
    emb = {i: rng.normal(size=8).astype(np.float32) for i in ids}
    rows = []
    for a in range(12):
        for b in range(a + 1, 12):
            rows.append((ids[a], ids[b], 0.0, 0.0, bool(rng.integers(0, 2))))
    per_pair, scalars = score_orphan_pairs(emb, _pairs(rows))
    assert 0.3 < scalars["siblings_AUROC"] < 0.7


def test_missing_id_pairs_dropped_and_counted():
    emb = {
        "A": np.array([1.0, 0.0], dtype=np.float32),
        "B": np.array([0.0, 1.0], dtype=np.float32),
    }
    pairs = _pairs(
        [
            ("A", "B", 0.5, 0.5, True),
            ("A", "Z", 0.5, 0.5, False),   # Z absent
            ("Y", "B", 0.5, 0.5, False),   # Y absent
        ]
    )
    per_pair, scalars = score_orphan_pairs(emb, pairs)
    assert scalars["n_pairs"] == 1            # only A-B kept
    assert scalars["n_pairs_dropped"] == 2
    assert per_pair.shape[0] == 1


def test_constant_cos_or_one_class_gives_auroc_nan():
    # All pairs are siblings -> roc_auc_score has one class -> NaN, not a crash.
    emb = {
        "A": np.array([1.0, 0.0], dtype=np.float32),
        "B": np.array([0.0, 1.0], dtype=np.float32),
        "C": np.array([1.0, 1.0], dtype=np.float32),
    }
    pairs = _pairs(
        [
            ("A", "B", 0.5, 0.5, True),
            ("A", "C", 0.6, 0.6, True),
            ("B", "C", 0.7, 0.7, True),
        ]
    )
    _, scalars = score_orphan_pairs(emb, pairs)
    assert np.isnan(scalars["siblings_AUROC"])


def test_cosine_matches_normalised_dot():
    emb = {
        "A": np.array([3.0, 4.0], dtype=np.float32),   # norm 5
        "B": np.array([4.0, 3.0], dtype=np.float32),   # norm 5
    }
    pairs = _pairs([("A", "B", 0.5, 0.5, True)])
    per_pair, _ = score_orphan_pairs(emb, pairs)
    # dot(â,b̂) = (3*4 + 4*3) / 25 = 24/25
    assert per_pair["cos"].iloc[0] == pytest.approx(24.0 / 25.0, abs=1e-6)


def test_spearman_scalars_present():
    rng = np.random.default_rng(1)
    ids = [f"P{i}" for i in range(8)]
    emb = {i: rng.normal(size=6).astype(np.float32) for i in ids}
    rows = []
    for a in range(8):
        for b in range(a + 1, 8):
            rows.append((ids[a], ids[b], rng.random(), rng.random(), bool(rng.integers(0, 2))))
    _, scalars = score_orphan_pairs(emb, _pairs(rows))
    assert "spearman_cos_vs_SNN" in scalars
    assert "spearman_cos_vs_TM" in scalars
    assert -1.0 <= scalars["spearman_cos_vs_SNN"] <= 1.0
