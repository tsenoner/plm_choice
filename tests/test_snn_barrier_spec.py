"""Tests for evaluation.snn_barrier_spec — the fan-in barrier spec-builder for the
cross-pLM SNN grid (pLM-pair x representation x distance).

Mirrors the recall-fp barrier_spec contract: one artifact per cell even when a sidecar
is absent (so a dead fan-out cell is surfaced MISSING, not dropped); sidecar-path
authoritative; guards transcribed from the single SNN_PARQUET_GUARDS source of truth; an
orphan parquet (no sidecar) fails closed; an under/over-specified grid fails loud.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.analysis_barrier import run_barrier, _spec_from_dict
from evaluation.snn_barrier_spec import (
    SpecBuildError,
    build_snn_barrier_spec,
    main,
)
from evaluation.snn_report import SNN_PER_QUERY_COLUMNS


def _write_parquet(path: Path, n_rows: int, cols=SNN_PER_QUERY_COLUMNS):
    df = pd.DataFrame({c: (np.arange(n_rows) if c != "query"
                           else [f"Q{i}" for i in range(n_rows)]) for c in cols})
    df["jaccard"] = np.linspace(0.0, 1.0, n_rows) if "jaccard" in cols else None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_cell(d: Path, a, b, rep, dist, *, n_common=6, parquet=True,
                per_query_columns=list(SNN_PER_QUERY_COLUMNS), population=(6, 6)):
    """Write a (parquet + sidecar) cell the way the SNN CLI does."""
    parquet_path = d / f"snn_{a}__{b}_{rep}_{dist}.parquet"
    if parquet:
        _write_parquet(parquet_path, n_common)
    sidecar = d / f"snn_{a}__{b}_{rep}_{dist}.manifest.json"
    sidecar.write_text(json.dumps({
        "plm_a": a, "plm_b": b, "representation": rep, "distance": dist,
        "n_common": n_common, "population_n_a": population[0], "population_n_b": population[1],
        "per_query_columns": per_query_columns, "path": str(parquet_path),
    }))
    return parquet_path, sidecar


def test_full_grid_one_artifact_per_cell_with_guards(tmp_path):
    for dist in ("cosine", "euclidean"):
        _write_cell(tmp_path, "prott5", "esm2", "raw", dist, n_common=6)
    spec = build_snn_barrier_spec(
        tmp_path, pairs=[("prott5", "esm2")], representations=["raw"],
        distances=["cosine", "euclidean"],
    )
    arts = spec["artifacts"]
    assert len(arts) == 2
    a0 = arts[0]
    assert a0["label"] == "snn:prott5:esm2:raw:cosine"
    assert a0["expected_rows"] == 6
    assert tuple(a0["required_columns"]) == SNN_PER_QUERY_COLUMNS
    assert tuple(a0["unique_columns"]) == ("query",)
    assert tuple(a0["finite_columns"]) == ("jaccard",)
    assert a0["kind"] == "parquet"


def test_sidecar_path_is_authoritative(tmp_path):
    # The sidecar records a non-canonical (timestamped) parquet path; the spec must use it.
    parquet = tmp_path / "snn_prott5__esm2_raw_cosine.20260610_120000.parquet"
    _write_parquet(parquet, 6)
    sidecar = tmp_path / "snn_prott5__esm2_raw_cosine.manifest.json"
    sidecar.write_text(json.dumps({
        "plm_a": "prott5", "plm_b": "esm2", "n_common": 6,
        "population_n_a": 6, "population_n_b": 6,
        "per_query_columns": list(SNN_PER_QUERY_COLUMNS), "path": str(parquet),
    }))
    spec = build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                  representations=["raw"], distances=["cosine"])
    assert spec["artifacts"][0]["path"] == str(parquet)


def test_missing_sidecar_emitted_as_reconstructed_not_dropped(tmp_path):
    # No cell written at all -> the grid cell must still appear (barrier reports MISSING).
    spec = build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                  representations=["raw"], distances=["cosine"])
    assert len(spec["artifacts"]) == 1
    assert spec["artifacts"][0]["path"].endswith("snn_prott5__esm2_raw_cosine.parquet")
    assert spec["artifacts"][0]["expected_rows"] is None
    assert spec["_meta"]["n_cells_without_sidecar"] == 1
    assert spec["_meta"]["reconstructed_cells"] == ["snn:prott5:esm2:raw:cosine"]


def test_orphan_parquet_without_sidecar_fails_closed(tmp_path):
    # A canonical parquet present with NO sidecar is a stale/partial artifact.
    _write_parquet(tmp_path / "snn_prott5__esm2_raw_cosine.parquet", 6)
    with pytest.raises(SpecBuildError, match="orphan"):
        build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                               representations=["raw"], distances=["cosine"])


def test_empty_pairs_raises(tmp_path):
    with pytest.raises(SpecBuildError, match="empty"):
        build_snn_barrier_spec(tmp_path, pairs=[], representations=["raw"],
                               distances=["cosine"])


def test_expected_n_pairs_mismatch_raises(tmp_path):
    _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine")
    with pytest.raises(SpecBuildError, match="expected"):
        build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                               representations=["raw"], distances=["cosine"],
                               expected_n_pairs=105)  # C(15,2)


def test_malformed_sidecar_no_path_raises(tmp_path):
    sidecar = tmp_path / "snn_prott5__esm2_raw_cosine.manifest.json"
    sidecar.write_text(json.dumps({"plm_a": "prott5", "n_common": 6}))  # no 'path'
    with pytest.raises(SpecBuildError):
        build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                               representations=["raw"], distances=["cosine"])


def test_per_query_columns_drift_raises(tmp_path):
    parquet = tmp_path / "snn_prott5__esm2_raw_cosine.parquet"
    _write_parquet(parquet, 6)
    sidecar = tmp_path / "snn_prott5__esm2_raw_cosine.manifest.json"
    sidecar.write_text(json.dumps({
        "plm_a": "prott5", "plm_b": "esm2", "n_common": 6,
        "population_n_a": 6, "population_n_b": 6,
        "per_query_columns": ["query", "WRONG"], "path": str(parquet),
    }))
    with pytest.raises(SpecBuildError, match="per_query_columns"):
        build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                               representations=["raw"], distances=["cosine"])


def test_population_n_propagated_into_meta(tmp_path):
    _write_cell(tmp_path, "prott5", "esm1b", "raw", "cosine", n_common=5, population=(6, 5))
    spec = build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm1b")],
                                  representations=["raw"], distances=["cosine"])
    pop = spec["_meta"]["population_n"]["prott5__esm1b:raw:cosine"]
    assert pop == {"a": 6, "b": 5}  # capped esm1b cap survives the spec step


def test_malformed_sidecar_non_int_n_common_raises(tmp_path):
    # n_common flows into expected_rows; a string/float/bool must fail loud at build time
    # rather than surfacing as a confusing "row count 6 != expected '6'" at barrier time.
    parquet = tmp_path / "snn_prott5__esm2_raw_cosine.parquet"
    _write_parquet(parquet, 6)
    sidecar = tmp_path / "snn_prott5__esm2_raw_cosine.manifest.json"
    sidecar.write_text(json.dumps({
        "plm_a": "prott5", "plm_b": "esm2", "n_common": "6",  # string, not int
        "population_n_a": 6, "population_n_b": 6,
        "per_query_columns": list(SNN_PER_QUERY_COLUMNS), "path": str(parquet),
    }))
    with pytest.raises(SpecBuildError, match="n_common"):
        build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                               representations=["raw"], distances=["cosine"])


def test_artifact_order_is_deterministic_grid_order(tmp_path):
    # Cells must enumerate pairs x reps x distances in a stable nested order so a
    # reordering bug in the iterator (or a dict-iteration surprise) is caught.
    pairs = [("prott5", "esm2"), ("ankh", "esm3")]
    for a, b in pairs:
        for rep in ("raw", "ffn"):
            for dist in ("cosine", "euclidean"):
                _write_cell(tmp_path, a, b, rep, dist)
    spec = build_snn_barrier_spec(tmp_path, pairs=pairs,
                                  representations=["raw", "ffn"],
                                  distances=["cosine", "euclidean"])
    labels = [a["label"] for a in spec["artifacts"]]
    assert labels == [
        "snn:prott5:esm2:raw:cosine", "snn:prott5:esm2:raw:euclidean",
        "snn:prott5:esm2:ffn:cosine", "snn:prott5:esm2:ffn:euclidean",
        "snn:ankh:esm3:raw:cosine", "snn:ankh:esm3:raw:euclidean",
        "snn:ankh:esm3:ffn:cosine", "snn:ankh:esm3:ffn:euclidean",
    ]


def test_no_expected_rows_leaves_rows_none(tmp_path):
    _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine", n_common=6)
    spec = build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                  representations=["raw"], distances=["cosine"],
                                  use_expected_rows=False)
    assert spec["artifacts"][0]["expected_rows"] is None


def test_dedup_axes(tmp_path):
    _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine")
    spec = build_snn_barrier_spec(
        tmp_path, pairs=[("prott5", "esm2"), ("prott5", "esm2")],
        representations=["raw", "raw"], distances=["cosine", "cosine"],
    )
    assert len(spec["artifacts"]) == 1  # duplicates collapsed


# ── the built spec actually BITES through the REAL barrier ────────────────────
def test_built_spec_passes_real_barrier_on_good_cell(tmp_path):
    _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine", n_common=6)
    spec = build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                  representations=["raw"], distances=["cosine"])
    specs = [_spec_from_dict(a, i) for i, a in enumerate(spec["artifacts"])]
    assert run_barrier(specs).ok


def test_built_spec_catches_truncated_parquet(tmp_path):
    parquet, sidecar = _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine", n_common=6)
    spec = build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                  representations=["raw"], distances=["cosine"])
    # Truncate the parquet AFTER the spec recorded expected_rows=6 -> row-count guard bites.
    _write_parquet(parquet, 3)
    specs = [_spec_from_dict(a, i) for i, a in enumerate(spec["artifacts"])]
    report = run_barrier(specs)
    assert not report.ok
    assert any("row count" in r for s in report.failures for r in s.reasons)


def test_built_spec_catches_dropped_column(tmp_path):
    parquet, _ = _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine", n_common=6)
    spec = build_snn_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                  representations=["raw"], distances=["cosine"])
    # Drop the 'jaccard' column -> required-columns + finite guard bite.
    pd.read_parquet(parquet).drop(columns=["jaccard"]).to_parquet(parquet, index=False)
    specs = [_spec_from_dict(a, i) for i, a in enumerate(spec["artifacts"])]
    assert not run_barrier(specs).ok


# ── CLI ──────────────────────────────────────────────────────────────────────
def test_cli_writes_spec_and_returns_0(tmp_path):
    _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine")
    out = tmp_path / "snn_barrier_spec.json"
    rc = main(["--sidecar-dir", str(tmp_path), "--pairs", "prott5,esm2",
               "--representations", "raw", "--distances", "cosine", "--out", str(out)])
    assert rc == 0
    spec = json.loads(out.read_text())
    assert len(spec["artifacts"]) == 1


def test_cli_no_pairs_returns_2(tmp_path):
    out = tmp_path / "spec.json"
    rc = main(["--sidecar-dir", str(tmp_path), "--pairs",
               "--representations", "raw", "--distances", "cosine", "--out", str(out)])
    assert rc == 2


def test_cli_orphan_returns_2(tmp_path):
    _write_parquet(tmp_path / "snn_prott5__esm2_raw_cosine.parquet", 6)
    out = tmp_path / "spec.json"
    rc = main(["--sidecar-dir", str(tmp_path), "--pairs", "prott5,esm2",
               "--representations", "raw", "--distances", "cosine", "--out", str(out)])
    assert rc == 2
