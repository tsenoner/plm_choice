"""The recall@1FP variant must travel with the artifact it produced.

`evaluation/retrieval_metrics` and `evaluation/recall_fp` both compute
recall-at-first-false-positive and do NOT share tie-handling. `recall_fp` is
canonical for anything reaching the manuscript; `classification_eval` uses the
other one because it needs a flat-vector entry point that `recall_fp` does not
expose, and there is a live path from its output parquet to a figure
(`visualization/create_retrieval_plots.py`).

Until the two are reconciled, the only thing standing between that path and a
mislabelled number in the paper is the provenance column. A comment cannot be
enforced; this can. If someone drops the column, this test fails.
"""

import itertools

import numpy as np
import polars as pl

from evaluation.classification_eval import (
    RECALL_METRIC_SOURCE,
    evaluate_at_hierarchy_levels,
)


def _toy_inputs():
    """Ten proteins in two classes — clears the >=10 positive / >=10 negative gate.

    All 45 unordered pairs: 20 same-class, 25 different-class. Same-class pairs are
    given smaller distances so the metrics are well defined rather than degenerate.
    """
    proteins = [f"p{i}" for i in range(10)]
    classes = {p: ("1" if i < 5 else "2") for i, p in enumerate(proteins)}

    queries, targets, dists = [], [], []
    for q, t in itertools.combinations(proteins, 2):
        queries.append(q)
        targets.append(t)
        dists.append(0.1 if classes[q] == classes[t] else 0.9)

    pairs = pl.DataFrame({"query": queries, "target": targets, "dist_toy": dists})
    return pairs, {"fold_id": classes}


def test_results_carry_the_recall_metric_source():
    pairs, cmap = _toy_inputs()
    out = evaluate_at_hierarchy_levels(pairs, ["dist_toy"], cmap)

    assert not out.is_empty(), "fixture should clear the min-pair gates"
    assert "recall_metric_source" in out.columns, (
        "the provenance column was dropped — a recall@1FP number can now reach a "
        "figure without recording which tie-handling variant produced it"
    )
    assert out["recall_metric_source"].to_list() == [RECALL_METRIC_SOURCE] * len(out)


def test_source_string_names_the_non_canonical_implementation():
    """The stamp must be readable as 'not recall_fp' by someone holding only the parquet."""
    assert "retrieval_metrics" in RECALL_METRIC_SOURCE
    assert "recall_fp" in RECALL_METRIC_SOURCE, (
        "the stamp should say explicitly that this is NOT the canonical recall_fp"
    )


def test_provenance_survives_a_roundtrip_through_parquet(tmp_path):
    """The guard is worthless if it does not survive the write the pipeline performs."""
    pairs, cmap = _toy_inputs()
    out = evaluate_at_hierarchy_levels(pairs, ["dist_toy"], cmap)

    path = tmp_path / "classification_eval_results.parquet"
    out.write_parquet(path)
    back = pl.read_parquet(path)

    assert back["recall_metric_source"].to_list() == [RECALL_METRIC_SOURCE] * len(back)
    # and the metric it labels is actually present and finite
    assert np.isfinite(back["recall_at_first_fp"].to_numpy()).all()
