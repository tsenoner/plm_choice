"""Differential test: refactored builders == frozen HEAD on success; same
first-failing check on error (accepting the deliberate suffix unifications listed
in the plan's 'Accepted behavior changes')."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import evaluation.recall_fp_barrier_spec as live_recall
import evaluation.snn_barrier_spec as live_snn
import tests._barrier_spec_head as head_recall
import tests._snn_barrier_spec_head as head_snn
from evaluation.recall_fp_report import PER_QUERY_COLUMNS
from evaluation.snn_report import SNN_PER_QUERY_COLUMNS


# ---- recall-fp fixtures -------------------------------------------------------
def _recall_sidecar(d, plm, rep, *, levels=("fold", "superfamily"), n_pos=4,
                    population_n=4, per_query_columns=None, level_paths=None):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    block = {}
    for lvl in levels:
        info = {
            "path": str(d / f"recall_fp_{plm}_{rep}_{lvl}.parquet"),
            "n_queries_with_positives": n_pos, "n_queries_skipped_no_positives": 0,
            "n_scored": n_pos, "mean_recall_1stFP": 1.0,
        }
        if level_paths and lvl in level_paths:
            info["path"] = str(level_paths[lvl])
        block[lvl] = info
    m = {"pLM": plm, "representation": rep, "distance": "euclidean",
         "population_n": population_n, "levels": block,
         "per_query_columns": list(per_query_columns or PER_QUERY_COLUMNS)}
    (d / f"recall_fp_{plm}_{rep}.manifest.json").write_text(json.dumps(m, indent=2) + "\n")


def test_recall_success_identical(tmp_path):
    for plm in ("prott5", "esm2"):
        for rep in ("raw", "ffn"):
            _recall_sidecar(tmp_path, plm, rep, population_n=267 if plm == "esm2" else 319)
    kw = dict(plms=["prott5", "esm2"], representations=["raw", "ffn"],
              levels=["fold", "superfamily"])
    assert live_recall.build_recall_fp_barrier_spec(tmp_path, **kw) == \
           head_recall.build_recall_fp_barrier_spec(tmp_path, **kw)


def test_recall_missing_sidecar_reconstruct_identical(tmp_path):
    _recall_sidecar(tmp_path, "prott5", "raw")  # esm2 absent
    kw = dict(plms=["prott5", "esm2"], representations=["raw"])
    assert live_recall.build_recall_fp_barrier_spec(tmp_path, **kw) == \
           head_recall.build_recall_fp_barrier_spec(tmp_path, **kw)


def test_recall_written_bytes_identical(tmp_path):
    _recall_sidecar(tmp_path, "prott5", "raw")
    spec = live_recall.build_recall_fp_barrier_spec(tmp_path, plms=["prott5"],
                                                    representations=["raw"])
    out = tmp_path / "live.json"
    live_recall.write_barrier_spec(spec, out)
    assert out.read_text() == json.dumps(spec, indent=2) + "\n"


def test_recall_written_bytes_match_head(tmp_path):
    # Explicit live-vs-HEAD byte comparison (dict == is order-insensitive; this locks
    # JSON field order too). Optional hardening per the plan-review fan (#4).
    for plm in ("prott5", "esm2"):
        for rep in ("raw", "ffn"):
            _recall_sidecar(tmp_path, plm, rep)
    kw = dict(plms=["prott5", "esm2"], representations=["raw", "ffn"])
    live_out = tmp_path / "live.json"
    head_out = tmp_path / "head.json"
    live_recall.write_barrier_spec(
        live_recall.build_recall_fp_barrier_spec(tmp_path, **kw), live_out)
    head_recall.write_barrier_spec(
        head_recall.build_recall_fp_barrier_spec(tmp_path, **kw), head_out)
    assert live_out.read_bytes() == head_out.read_bytes()


@pytest.mark.parametrize("which", ["orphan", "drift", "grid"])
def test_recall_error_same_first_check(tmp_path, which):
    if which == "orphan":
        (tmp_path / "recall_fp_prott5_raw_fold.parquet").write_bytes(b"x")
        kw, disc = dict(plms=["prott5"], representations=["raw"]), "orphan"
    elif which == "drift":
        _recall_sidecar(tmp_path, "prott5", "raw", per_query_columns=["query_id"])
        kw, disc = dict(plms=["prott5"], representations=["raw"]), "per_query_columns"
    else:
        _recall_sidecar(tmp_path, "prott5", "raw")
        kw, disc = dict(plms=["prott5"], representations=["raw"], expected_n_plms=9), "expected"
    for mod in (live_recall, head_recall):
        with pytest.raises(mod.SpecBuildError, match=disc):
            mod.build_recall_fp_barrier_spec(tmp_path, **kw)


# ---- SNN fixtures -------------------------------------------------------------
def _snn_parquet(path, n):
    df = {c: (np.arange(n) if c != "query" else [f"Q{i}" for i in range(n)])
          for c in SNN_PER_QUERY_COLUMNS}
    import pandas as pd
    pd.DataFrame(df).to_parquet(path, index=False)


def _snn_cell(d, a, b, rep, dist, *, n_common=6, population=(6, 6),
              per_query_columns=None):
    d = Path(d)
    pq = d / f"snn_{a}__{b}_{rep}_{dist}.parquet"
    _snn_parquet(pq, n_common)
    (d / f"snn_{a}__{b}_{rep}_{dist}.manifest.json").write_text(json.dumps({
        "plm_a": a, "plm_b": b, "representation": rep, "distance": dist,
        "n_common": n_common, "population_n_a": population[0],
        "population_n_b": population[1],
        "per_query_columns": list(per_query_columns or SNN_PER_QUERY_COLUMNS),
        "path": str(pq),
    }))


def test_snn_success_identical(tmp_path):
    for a, b in [("prott5", "esm2"), ("ankh", "esm1b")]:
        for dist in ("cosine", "euclidean"):
            _snn_cell(tmp_path, a, b, "raw", dist,
                      population=(6, 5) if b == "esm1b" else (6, 6))
    kw = dict(pairs=[("prott5", "esm2"), ("ankh", "esm1b")],
              representations=["raw"], distances=["cosine", "euclidean"])
    assert live_snn.build_snn_barrier_spec(tmp_path, **kw) == \
           head_snn.build_snn_barrier_spec(tmp_path, **kw)


def test_snn_missing_and_population_identical(tmp_path):
    _snn_cell(tmp_path, "prott5", "esm1b", "raw", "cosine", population=(6, 5))  # euclidean absent
    kw = dict(pairs=[("prott5", "esm1b")], representations=["raw"],
              distances=["cosine", "euclidean"])
    assert live_snn.build_snn_barrier_spec(tmp_path, **kw) == \
           head_snn.build_snn_barrier_spec(tmp_path, **kw)


def test_snn_written_bytes_match_head(tmp_path):
    # Lock SNN's guard-dict field order too (optional hardening, plan-review #4).
    for dist in ("cosine", "euclidean"):
        _snn_cell(tmp_path, "prott5", "esm2", "raw", dist)
    kw = dict(pairs=[("prott5", "esm2")], representations=["raw"],
              distances=["cosine", "euclidean"])
    live_out = tmp_path / "live.json"
    head_out = tmp_path / "head.json"
    live_snn.write_barrier_spec(live_snn.build_snn_barrier_spec(tmp_path, **kw), live_out)
    head_snn.write_barrier_spec(head_snn.build_snn_barrier_spec(tmp_path, **kw), head_out)
    assert live_out.read_bytes() == head_out.read_bytes()


@pytest.mark.parametrize("which", ["orphan", "drift", "n_common", "expected"])
def test_snn_error_same_first_check(tmp_path, which):
    if which == "orphan":
        _snn_parquet(tmp_path / "snn_prott5__esm2_raw_cosine.parquet", 6)
        disc = "orphan"
    elif which == "drift":
        _snn_cell(tmp_path, "prott5", "esm2", "raw", "cosine",
                  per_query_columns=["query", "WRONG"])
        disc = "per_query_columns"
    elif which == "n_common":
        _snn_parquet(tmp_path / "snn_prott5__esm2_raw_cosine.parquet", 6)
        (tmp_path / "snn_prott5__esm2_raw_cosine.manifest.json").write_text(json.dumps({
            "plm_a": "prott5", "plm_b": "esm2", "n_common": "6",
            "population_n_a": 6, "population_n_b": 6,
            "per_query_columns": list(SNN_PER_QUERY_COLUMNS),
            "path": str(tmp_path / "snn_prott5__esm2_raw_cosine.parquet"),
        }))
        disc = "n_common"
    else:
        _snn_cell(tmp_path, "prott5", "esm2", "raw", "cosine")
        disc = "expected"
    kw = dict(pairs=[("prott5", "esm2")], representations=["raw"], distances=["cosine"])
    if which == "expected":
        kw["expected_n_pairs"] = 9
    for mod in (live_snn, head_snn):
        with pytest.raises(mod.SpecBuildError, match=disc):
            mod.build_snn_barrier_spec(tmp_path, **kw)
