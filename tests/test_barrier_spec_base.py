"""Unit tests for the shared barrier-spec base."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.barrier_spec_base import (
    SpecBuildError,
    check_per_query_columns_drift,
    dedup,
    emit_cell,
    make_artifact,
    read_sidecar_dict,
    require_grid_size,
    write_barrier_spec,
)


def test_dedup_preserves_order_strings_and_tuples():
    assert dedup(["a", "b", "a", "c"]) == ["a", "b", "c"]
    assert dedup([("p", "q"), ("p", "q"), ("r", "s")]) == [("p", "q"), ("r", "s")]


def test_make_artifact_shape_and_field_order():
    guards = {"required_columns": ("query",), "unique_columns": ("query",),
              "non_null_columns": ("query",), "finite_columns": ("jaccard",)}
    art = make_artifact("L", Path("/x/y.parquet"), 6, guards)
    assert list(art.keys()) == [
        "label", "path", "expected_rows", "kind",
        "required_columns", "unique_columns", "non_null_columns", "finite_columns",
    ]
    assert art["path"] == "/x/y.parquet" and art["kind"] == "parquet"
    assert art["required_columns"] == ["query"]  # tuples coerced to list


def test_make_artifact_rejects_unknown_guard_key():
    with pytest.raises(SpecBuildError, match="unknown guard"):
        make_artifact("L", "/x.parquet", 1, {"required_cols": ("a",)})  # typo


def test_read_sidecar_dict_structural_only(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"any": 1}))
    assert read_sidecar_dict(p) == {"any": 1}
    p.write_text("{ not json")
    with pytest.raises(SpecBuildError, match="not valid JSON"):
        read_sidecar_dict(p)
    p.write_text(json.dumps([1, 2]))  # valid JSON, not an object
    with pytest.raises(SpecBuildError, match="must be a JSON object"):
        read_sidecar_dict(p)
    d = tmp_path / "dir.json"
    d.mkdir()
    with pytest.raises(SpecBuildError, match="unreadable"):
        read_sidecar_dict(d)


def test_check_drift_skips_when_absent_or_none():
    contract = ("query", "jaccard")
    check_per_query_columns_drift({}, contract, "p")               # absent -> ok
    check_per_query_columns_drift({"per_query_columns": None}, contract, "p")  # None -> ok
    check_per_query_columns_drift({"per_query_columns": ["query", "jaccard"]}, contract, "p")


def test_check_drift_raises_on_mismatch_and_non_list():
    contract = ("query", "jaccard")
    with pytest.raises(SpecBuildError, match="per_query_columns"):
        check_per_query_columns_drift({"per_query_columns": ["query"]}, contract, "p")
    with pytest.raises(SpecBuildError, match="per_query_columns"):
        check_per_query_columns_drift({"per_query_columns": "query"}, contract, "p")


def test_require_grid_size_noun_and_echo():
    require_grid_size(["a"], None, singular="pLM", plural_key="plms")  # None -> no check
    require_grid_size(["a", "b"], 2, singular="pLM", plural_key="plms")
    with pytest.raises(SpecBuildError, match=r"unique pLM\(s\) but expected 15"):
        require_grid_size(["a"], 15, singular="pLM", plural_key="plms")
    with pytest.raises(SpecBuildError, match=r"pairs=\[\('p', 'q'\)\]"):
        require_grid_size([("p", "q")], 2, singular="pair", plural_key="pairs")


def test_emit_cell_covered_uses_authoritative(tmp_path):
    guards = {"required_columns": ("query",)}
    art, recon = emit_cell(
        "L", covered=True, get_path_rows=lambda: ("/stamped.parquet", 9),
        canonical_parquet=tmp_path / "canon.parquet", guards=guards,
    )
    assert recon is False
    assert art["path"] == "/stamped.parquet" and art["expected_rows"] == 9


def _must_not_be_called():
    raise AssertionError("get_path_rows must not be called when covered=False")


def test_emit_cell_uncovered_reconstructs(tmp_path):
    canon = tmp_path / "canon.parquet"  # does NOT exist
    art, recon = emit_cell(
        "L", covered=False, get_path_rows=_must_not_be_called,
        canonical_parquet=canon, guards={"required_columns": ("query",)},
    )
    assert recon is True
    assert art["path"] == str(canon) and art["expected_rows"] is None


def test_emit_cell_uncovered_orphan_fails_closed(tmp_path):
    canon = tmp_path / "canon.parquet"
    canon.write_bytes(b"stale")  # canonical parquet present, no sidecar
    with pytest.raises(SpecBuildError, match="orphan"):
        emit_cell("L", covered=False, get_path_rows=lambda: ("x", 1),
                  canonical_parquet=canon, guards={"required_columns": ("query",)})


def test_write_barrier_spec_replace_default(tmp_path):
    target = tmp_path / "spec.json"
    a = write_barrier_spec({"artifacts": []}, target)
    b = write_barrier_spec({"artifacts": []}, target)
    assert a == target == b
    assert len(list(tmp_path.glob("spec*.json"))) == 1
    assert target.read_text().endswith("\n")
