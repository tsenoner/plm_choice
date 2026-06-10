"""Tests for evaluation.recall_fp_report — the analysis-step bridge that turns
the in-memory recall-FP result into a barrier-checkable on-disk parquet.

The bridge owns the error-prone glue the per-function tests don't cover:
subset the pLM embeddings to the frozen canonical set, assert population BEFORE
scoring (S3), score each available CATH level with the set-intersection
predicate, and atomic-write the per-query parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.analysis_barrier import ArtifactSpec, check_artifact
from evaluation.population import PopulationError
from evaluation.recall_fp_report import recall_fp_report


def _db():
    # Two folds on a line: {P1,P2} near 0, {P3,P4} near 5 -> clean retrieval.
    embeddings = {
        "P1": np.array([0.0, 0.0], dtype=np.float32),
        "P2": np.array([0.1, 0.0], dtype=np.float32),
        "P3": np.array([5.0, 5.0], dtype=np.float32),
        "P4": np.array([5.1, 5.0], dtype=np.float32),
    }
    labels = pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3", "P4"],
            "fold": [
                frozenset({"a"}),
                frozenset({"a"}),
                frozenset({"b"}),
                frozenset({"b"}),
            ],
            "superfamily": [
                frozenset({"a1"}),
                frozenset({"a1"}),
                frozenset({"b1"}),
                frozenset({"b1"}),
            ],
            "family": [None, None, None, None],
        }
    )
    return embeddings, labels


def _parquets(out_dir):
    return sorted(p.name for p in out_dir.glob("*.parquet"))


def test_writes_one_parquet_per_level_and_returns_manifest(tmp_path):
    emb, labels = _db()
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    assert set(manifest["levels"]) == {"fold", "superfamily"}
    for level in ("fold", "superfamily"):
        info = manifest["levels"][level]
        assert info["n_queries_with_positives"] == 4
        assert info["mean_recall_1stFP"] == pytest.approx(1.0)
        assert info["path"].endswith(".parquet")
    # two parquet files actually landed on disk
    assert len(_parquets(tmp_path)) == 2


def test_default_levels_exclude_family(tmp_path):
    emb, labels = _db()
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    assert "family" not in manifest["levels"]
    # note: "superfamily" contains the substring "family" -> match the level suffix
    assert not any(n.endswith("_family.parquet") for n in _parquets(tmp_path))


def test_output_parquet_passes_the_barrier(tmp_path):
    emb, labels = _db()
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    path = manifest["levels"]["fold"]["path"]
    spec = ArtifactSpec(
        label="recall_fp:prott5:fold",
        path=path,
        expected_rows=4,
        required_columns=("query_id", "n_positives", "recall", "n_ties_at_first_fp"),
        finite_columns=("recall",),
        unique_columns=("query_id",),
        non_null_columns=("query_id",),
    )
    status = check_artifact(spec)
    assert status.ok, status.reasons


def test_population_asserted_before_scoring_writes_nothing_on_drift(tmp_path):
    # A pLM silently missing a frozen id (and not flagged capped) must raise
    # BEFORE any parquet is written.
    emb, labels = _db()
    with pytest.raises(PopulationError):
        recall_fp_report(
            emb, labels, tmp_path, pLM="broken",
            expected_ids=["P1", "P2", "P3", "P4", "P5_missing"],
            distance="euclidean",
        )
    assert _parquets(tmp_path) == []


def test_capped_plm_allowed_as_subset(tmp_path):
    emb, labels = _db()
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="esm1b",
        expected_ids=["P1", "P2", "P3", "P4", "P5_missing"],
        allow_capped=True, distance="euclidean",
    )
    assert manifest["population_n"] == 4
    assert manifest["levels"]["fold"]["n_queries_with_positives"] == 4


def test_superset_embeddings_subset_to_frozen_set(tmp_path):
    # prott5/esm3 carry a 1225-key pool; the bridge must subset to the frozen
    # set rather than score (or population-fail on) the extras.
    emb, labels = _db()
    emb = dict(emb)
    emb["PX_extra"] = np.array([99.0, 99.0], dtype=np.float32)
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    assert manifest["population_n"] == 4
    df = pd.read_parquet(manifest["levels"]["fold"]["path"])
    assert "PX_extra" not in df["query_id"].tolist()


def test_rerun_replaces_in_place_at_fixed_path(tmp_path):
    # The barrier validates a FIXED spec path, so a re-run must atomically
    # replace that path (default overwrite=True), not orphan the fresh result at
    # a timestamped sibling the barrier never checks.
    emb, labels = _db()
    first = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    second = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    assert first["levels"]["fold"]["path"] == second["levels"]["fold"]["path"]
    assert len(_parquets(tmp_path)) == 2  # fold + superfamily, replaced in place


def test_no_overwrite_keeps_prior_via_timestamped_sibling(tmp_path):
    # The explicit opt-out still works for ad-hoc never-clobber use.
    emb, labels = _db()
    first = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    second = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
        overwrite=False,
    )
    assert first["levels"]["fold"]["path"] != second["levels"]["fold"]["path"]
    assert len(_parquets(tmp_path)) == 4


def test_representation_axis_in_filename_prevents_collision(tmp_path):
    # raw and ffn recall-FP of the same pLM/level must land on distinct paths.
    emb, labels = _db()
    raw = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5", representation="raw",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    ffn = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5", representation="ffn",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    assert raw["levels"]["fold"]["path"] != ffn["levels"]["fold"]["path"]
    assert "prott5_raw_fold" in raw["levels"]["fold"]["path"]
    assert "prott5_ffn_fold" in ffn["levels"]["fold"]["path"]
    assert len(_parquets(tmp_path)) == 4  # 2 reps x 2 levels, no collision


def test_n_scored_distinguishes_cohort_from_labelled(tmp_path):
    # A canonical protein with no CATH label is in the cohort (population_n) but
    # not ranked (n_scored) -> the two denominators must differ, not be conflated.
    emb, labels = _db()
    emb = dict(emb)
    emb["P5_nolabel"] = np.array([0.2, 0.0], dtype=np.float32)
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4", "P5_nolabel"], distance="euclidean",
    )
    assert manifest["population_n"] == 5
    assert manifest["levels"]["fold"]["n_scored"] == 4  # P5_nolabel not in labels
