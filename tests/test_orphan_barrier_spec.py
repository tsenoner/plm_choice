"""Unit 5 — orphan fan-in barrier spec-builder (clone of test_ec_barrier_spec)."""
import json

import pytest

from evaluation.orphan_barrier_spec import build_orphan_barrier_spec
from evaluation.barrier_spec_base import SpecBuildError


def _sidecar(d, plm, distance="cosine", n_pairs=10):
    path = d / f"orphan_{plm}_raw_{distance}.manifest.json"
    parquet = d / f"orphan_{plm}_raw_{distance}.parquet"
    parquet.write_text("")  # presence only; the barrier reads it later
    path.write_text(json.dumps({
        "path": str(parquet), "n_pairs": n_pairs, "population_n": 30,
        "per_pair_columns": ["pair_key", "p1", "p2", "cos", "snn", "tm", "sibling"],
    }))


def test_full_grid_one_artifact_per_cell(tmp_path):
    for plm in ("prott5", "esm2", "prottucker"):
        _sidecar(tmp_path, plm)
    spec = build_orphan_barrier_spec(tmp_path, plms=["prott5", "esm2", "prottucker"])
    assert len(spec["artifacts"]) == 3       # cosine-only grid -> one cell per pLM
    assert spec["_meta"]["n_cells"] == 3
    assert spec["_meta"]["n_cells_without_sidecar"] == 0
    # population carried through
    assert spec["_meta"]["population_n"]["prott5:cosine"] == 30


def test_missing_sidecar_emits_reconstructed_cell(tmp_path):
    _sidecar(tmp_path, "prott5")
    spec = build_orphan_barrier_spec(tmp_path, plms=["prott5", "esm2"])
    assert spec["_meta"]["n_cells_without_sidecar"] == 1  # esm2 cell absent
    assert "orphan:esm2:cosine" in spec["_meta"]["reconstructed_cells"]


def test_orphan_parquet_without_sidecar_fails(tmp_path):
    (tmp_path / "orphan_prott5_raw_cosine.parquet").write_text("")  # parquet, no sidecar
    with pytest.raises(SpecBuildError, match="orphan"):
        build_orphan_barrier_spec(tmp_path, plms=["prott5"])


def test_drift_in_per_pair_columns_fails(tmp_path):
    path = tmp_path / "orphan_prott5_raw_cosine.manifest.json"
    (tmp_path / "orphan_prott5_raw_cosine.parquet").write_text("")
    path.write_text(json.dumps({"path": str(tmp_path / "orphan_prott5_raw_cosine.parquet"),
                                "n_pairs": 10, "per_pair_columns": ["WRONG"]}))
    with pytest.raises(SpecBuildError, match="drift"):
        build_orphan_barrier_spec(tmp_path, plms=["prott5"])


def test_grid_size_drift_raises(tmp_path):
    for plm in ("prott5", "esm2"):
        _sidecar(tmp_path, plm)
    with pytest.raises(SpecBuildError, match="expected"):
        build_orphan_barrier_spec(tmp_path, plms=["prott5", "esm2"], expected_n_plms=3)


def test_empty_plms_fails(tmp_path):
    with pytest.raises(SpecBuildError, match="empty"):
        build_orphan_barrier_spec(tmp_path, plms=[])


def test_guards_transcribed_from_orphan_report(tmp_path):
    # The single-source-of-truth check: the spec's artifact guards must equal the ones
    # orphan_report owns (imported, not re-typed).
    from evaluation.orphan_report import ORPHAN_PARQUET_GUARDS

    _sidecar(tmp_path, "prott5")
    spec = build_orphan_barrier_spec(tmp_path, plms=["prott5"])
    art = spec["artifacts"][0]
    assert art["required_columns"] == list(ORPHAN_PARQUET_GUARDS["required_columns"])
    assert art["unique_columns"] == list(ORPHAN_PARQUET_GUARDS["unique_columns"])
    assert art["finite_columns"] == list(ORPHAN_PARQUET_GUARDS["finite_columns"])
    assert art["non_null_columns"] == list(ORPHAN_PARQUET_GUARDS["non_null_columns"])
