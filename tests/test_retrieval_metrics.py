import numpy as np
import pytest
from src.evaluation.retrieval_metrics import recall_at_first_fp, auroc_at_level

def test_recall_at_first_fp_perfect():
    distances = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    labels = np.array([True, True, True, False, False])
    result = recall_at_first_fp(distances, labels)
    assert result["recall_at_first_fp"] == 1.0
    assert result["n_retrieved"] == 3

def test_recall_at_first_fp_immediate_failure():
    distances = np.array([0.1, 0.2, 0.3])
    labels = np.array([False, True, True])
    result = recall_at_first_fp(distances, labels)
    assert result["recall_at_first_fp"] == 0.0
    assert result["n_retrieved"] == 0

def test_recall_at_first_fp_partial():
    distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    labels = np.array([True, True, False, True, False])
    result = recall_at_first_fp(distances, labels)
    assert result["recall_at_first_fp"] == pytest.approx(2/3)
    assert result["n_retrieved"] == 2

def test_recall_at_first_fp_no_positives():
    distances = np.array([0.1, 0.2, 0.3])
    labels = np.array([False, False, False])
    result = recall_at_first_fp(distances, labels)
    assert result["recall_at_first_fp"] == 0.0
    assert result["n_positives"] == 0

def test_auroc_at_level_perfect():
    distances = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([True, True, False, False])
    assert auroc_at_level(distances, labels) == pytest.approx(1.0)

def test_auroc_at_level_random():
    np.random.seed(42)
    distances = np.random.rand(1000)
    labels = np.random.choice([True, False], 1000)
    auc = auroc_at_level(distances, labels)
    assert 0.4 < auc < 0.6

def test_auroc_at_level_inverted():
    distances = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([True, True, False, False])
    auc = auroc_at_level(distances, labels, lower_is_similar=False)
    assert auc == pytest.approx(1.0)

def test_auroc_single_class():
    distances = np.array([0.1, 0.2, 0.3])
    labels = np.array([True, True, True])
    auc = auroc_at_level(distances, labels)
    assert np.isnan(auc)
