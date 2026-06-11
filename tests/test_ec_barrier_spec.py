import json
import pytest

from evaluation.ec_barrier_spec import build_ec_barrier_spec
from evaluation.barrier_spec_base import SpecBuildError


def _sidecar(d, plm, distance, n_pairs=10):
    path = d / f"ec_{plm}_raw_{distance}.manifest.json"
    parquet = d / f"ec_{plm}_raw_{distance}.parquet"
    parquet.write_text("")  # presence only; the barrier reads it later
    path.write_text(json.dumps({
        "path": str(parquet), "n_pairs": n_pairs, "population_n": 30,
        "per_pair_columns": ["pair_key", "a", "b", "dist", "ec_dist"],
    }))


def test_full_grid_one_artifact_per_cell(tmp_path):
    for plm in ("prott5", "esm2"):
        for dist in ("euclidean", "cosine"):
            _sidecar(tmp_path, plm, dist)
    spec = build_ec_barrier_spec(
        tmp_path, plms=["prott5", "esm2"], distances=["euclidean", "cosine"])
    assert len(spec["artifacts"]) == 4
    assert spec["_meta"]["n_cells"] == 4
    assert spec["_meta"]["n_cells_without_sidecar"] == 0


def test_missing_sidecar_emits_reconstructed_cell(tmp_path):
    _sidecar(tmp_path, "prott5", "euclidean")
    spec = build_ec_barrier_spec(
        tmp_path, plms=["prott5"], distances=["euclidean", "cosine"])
    assert spec["_meta"]["n_cells_without_sidecar"] == 1  # the cosine cell absent


def test_orphan_parquet_without_sidecar_fails(tmp_path):
    (tmp_path / "ec_prott5_raw_cosine.parquet").write_text("")  # parquet, no sidecar
    with pytest.raises(SpecBuildError, match="orphan"):
        build_ec_barrier_spec(tmp_path, plms=["prott5"], distances=["cosine"])


def test_drift_in_per_pair_columns_fails(tmp_path):
    path = tmp_path / "ec_prott5_raw_euclidean.manifest.json"
    (tmp_path / "ec_prott5_raw_euclidean.parquet").write_text("")
    path.write_text(json.dumps({"path": str(tmp_path / "ec_prott5_raw_euclidean.parquet"),
                                "n_pairs": 10, "per_pair_columns": ["WRONG"]}))
    with pytest.raises(SpecBuildError, match="drift"):
        build_ec_barrier_spec(tmp_path, plms=["prott5"], distances=["euclidean"])
