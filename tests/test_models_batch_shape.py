"""A predictor must return one prediction per pair, even when the batch holds one pair.

`.squeeze()` with no argument drops *every* size-1 dimension, so a trailing batch of one
collapses [1, 1] to a 0-dim scalar instead of [1]. No DataLoader in this repo sets
`drop_last`, so that batch occurs whenever a split's row count is 1 modulo the batch size.
Evaluation then dies in `evaluation.evaluate`, which concatenates the per-batch outputs.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from training.models import (
    FNNPredictor,
    LinearDistancePredictor,
    LinearRegressionPredictor,
)

EMB = 8


def _build(cls):
    torch.manual_seed(0)
    kwargs = {"embedding_size": EMB}
    if cls is FNNPredictor:
        kwargs["hidden_size"] = 4
    return cls(**kwargs).eval()


def _pair(rows):
    g = torch.Generator().manual_seed(rows)
    return torch.randn(rows, EMB, generator=g), torch.randn(rows, EMB, generator=g)


ALL = [FNNPredictor, LinearRegressionPredictor, LinearDistancePredictor]


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
@pytest.mark.parametrize("rows", [1, 2, 7])
def test_prediction_has_one_entry_per_pair(cls, rows):
    """The batch dimension survives, so predictions line up with targets one-to-one."""
    model = _build(cls)
    with torch.no_grad():
        out = model(*_pair(rows))
    assert out.shape == (rows,), f"{cls.__name__} at batch {rows} returned {tuple(out.shape)}"


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
def test_predictions_concatenate_across_a_one_row_final_batch(cls):
    """Mirrors evaluation.evaluate, which does torch.cat over the per-batch outputs."""
    model = _build(cls)
    preds = []
    with torch.no_grad():
        for rows in (4, 4, 1):  # a split of 9 rows at batch size 4
            preds.append(model(*_pair(rows)))
    assert torch.cat(preds).numel() == 9


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
def test_single_row_loss_does_not_broadcast(cls):
    """MSELoss must see matching shapes; broadcasting [] against [1] is a silent trap."""
    model = _build(cls)
    target = torch.rand(1)
    with torch.no_grad():
        pred = model(*_pair(1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        nn.MSELoss()(pred, target)
