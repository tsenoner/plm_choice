"""Tests for evaluation.cross_plm_barrier_spec — the fan-in barrier spec-builder for the
cross-pLM agreement grid (pLM-pair x representation x distance).

Structurally the SNN builder (pLM-pair grid, per-cell ``population_n {a, b}``) but with the
EC sidecar fields (``n_pairs`` -> expected_rows, ``per_pair_columns`` drift key). The
load-bearing manhattan guard: ``DEFAULT_DISTANCES`` MUST be the 3-element tuple — cloning
``ec_barrier_spec`` (2-element ``{euclidean, cosine}``) would silently drop the entire
manhattan column with no error, the single highest-risk manhattan spot.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.analysis_barrier import run_barrier, _spec_from_dict
from evaluation.cross_plm_barrier_spec import (
    DEFAULT_DISTANCES,
    SpecBuildError,
    build_cross_plm_barrier_spec,
    main,
)
from evaluation.cross_plm_report import CROSS_PLM_PER_PAIR_COLUMNS


def _write_parquet(path: Path, n_rows: int, cols=CROSS_PLM_PER_PAIR_COLUMNS):
    df = pd.DataFrame({
        "pair_key": [f"a{i}\tb{i}" for i in range(n_rows)],
        "a": [f"a{i}" for i in range(n_rows)],
        "b": [f"b{i}" for i in range(n_rows)],
        "dist_a": np.linspace(0.0, 1.0, n_rows),
        "dist_b": np.linspace(1.0, 2.0, n_rows),
    })[list(cols)]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_cell(d: Path, a, b, rep, dist, *, n_pairs=15, parquet=True,
                per_pair_columns=list(CROSS_PLM_PER_PAIR_COLUMNS), population=(6, 6)):
    """Write a (parquet + sidecar) cross-pLM cell the way the CLI does."""
    parquet_path = d / f"cross_plm_{a}__{b}_{rep}_{dist}.parquet"
    if parquet:
        _write_parquet(parquet_path, n_pairs)
    sidecar = d / f"cross_plm_{a}__{b}_{rep}_{dist}.manifest.json"
    sidecar.write_text(json.dumps({
        "plm_a": a, "plm_b": b, "representation": rep, "distance": dist,
        "n_pairs": n_pairs, "population_n_a": population[0], "population_n_b": population[1],
        "per_pair_columns": per_pair_columns, "path": str(parquet_path),
    }))
    return parquet_path, sidecar


# ── the manhattan anti-regression (the fan's top risk) ─────────────────────────
def test_default_distances_is_three_element_with_manhattan():
    assert DEFAULT_DISTANCES == ("cosine", "euclidean", "manhattan")


def test_build_over_default_distances_emits_manhattan_cell(tmp_path):
    for dist in DEFAULT_DISTANCES:
        _write_cell(tmp_path, "prott5", "esm2", "raw", dist)
    spec = build_cross_plm_barrier_spec(
        tmp_path, pairs=[("prott5", "esm2")], representations=["raw"],
    )  # distances defaulted -> must include manhattan
    labels = [a["label"] for a in spec["artifacts"]]
    assert "cross_plm:prott5:esm2:raw:manhattan" in labels
    assert len(spec["artifacts"]) == 3


# ── core grid behaviour ────────────────────────────────────────────────────────
def test_full_grid_one_artifact_per_cell_with_guards(tmp_path):
    for dist in ("cosine", "euclidean"):
        _write_cell(tmp_path, "prott5", "esm2", "raw", dist, n_pairs=15)
    spec = build_cross_plm_barrier_spec(
        tmp_path, pairs=[("prott5", "esm2")], representations=["raw"],
        distances=["cosine", "euclidean"],
    )
    arts = spec["artifacts"]
    assert len(arts) == 2
    a0 = arts[0]
    assert a0["label"] == "cross_plm:prott5:esm2:raw:cosine"
    assert a0["expected_rows"] == 15  # n_pairs, NOT n_common
    assert tuple(a0["required_columns"]) == CROSS_PLM_PER_PAIR_COLUMNS
    assert tuple(a0["unique_columns"]) == ("pair_key",)
    assert tuple(a0["finite_columns"]) == ("dist_a", "dist_b")
    assert a0["kind"] == "parquet"


def test_sidecar_path_is_authoritative(tmp_path):
    parquet = tmp_path / "cross_plm_prott5__esm2_raw_cosine.20260612_120000.parquet"
    _write_parquet(parquet, 15)
    sidecar = tmp_path / "cross_plm_prott5__esm2_raw_cosine.manifest.json"
    sidecar.write_text(json.dumps({
        "plm_a": "prott5", "plm_b": "esm2", "n_pairs": 15,
        "population_n_a": 6, "population_n_b": 6,
        "per_pair_columns": list(CROSS_PLM_PER_PAIR_COLUMNS), "path": str(parquet),
    }))
    spec = build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                        representations=["raw"], distances=["cosine"])
    assert spec["artifacts"][0]["path"] == str(parquet)


def test_missing_sidecar_emitted_as_reconstructed_not_dropped(tmp_path):
    spec = build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                        representations=["raw"], distances=["cosine"])
    assert len(spec["artifacts"]) == 1
    assert spec["artifacts"][0]["path"].endswith("cross_plm_prott5__esm2_raw_cosine.parquet")
    assert spec["artifacts"][0]["expected_rows"] is None
    assert spec["_meta"]["n_cells_without_sidecar"] == 1
    assert spec["_meta"]["reconstructed_cells"] == ["cross_plm:prott5:esm2:raw:cosine"]


def test_orphan_parquet_without_sidecar_fails_closed(tmp_path):
    _write_parquet(tmp_path / "cross_plm_prott5__esm2_raw_cosine.parquet", 15)
    with pytest.raises(SpecBuildError, match="orphan"):
        build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                     representations=["raw"], distances=["cosine"])


def test_empty_pairs_raises(tmp_path):
    with pytest.raises(SpecBuildError, match="empty"):
        build_cross_plm_barrier_spec(tmp_path, pairs=[], representations=["raw"],
                                     distances=["cosine"])


def test_expected_n_pairs_mismatch_raises(tmp_path):
    _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine")
    with pytest.raises(SpecBuildError, match="expected"):
        build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                     representations=["raw"], distances=["cosine"],
                                     expected_n_pairs=105)  # C(15,2)


def test_malformed_sidecar_no_path_raises(tmp_path):
    sidecar = tmp_path / "cross_plm_prott5__esm2_raw_cosine.manifest.json"
    sidecar.write_text(json.dumps({"plm_a": "prott5", "n_pairs": 15}))  # no 'path'
    with pytest.raises(SpecBuildError):
        build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                     representations=["raw"], distances=["cosine"])


def test_per_pair_columns_drift_raises(tmp_path):
    parquet = tmp_path / "cross_plm_prott5__esm2_raw_cosine.parquet"
    _write_parquet(parquet, 15)
    sidecar = tmp_path / "cross_plm_prott5__esm2_raw_cosine.manifest.json"
    sidecar.write_text(json.dumps({
        "plm_a": "prott5", "plm_b": "esm2", "n_pairs": 15,
        "population_n_a": 6, "population_n_b": 6,
        "per_pair_columns": ["pair_key", "WRONG"], "path": str(parquet),
    }))
    with pytest.raises(SpecBuildError, match="per_pair_columns"):
        build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                     representations=["raw"], distances=["cosine"])


def test_population_n_propagated_into_meta(tmp_path):
    _write_cell(tmp_path, "prott5", "esm1b", "raw", "cosine", n_pairs=10, population=(6, 5))
    spec = build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm1b")],
                                        representations=["raw"], distances=["cosine"])
    pop = spec["_meta"]["population_n"]["prott5__esm1b:raw:cosine"]
    assert pop == {"a": 6, "b": 5}  # capped esm1b cap survives the spec step


def test_malformed_sidecar_non_int_n_pairs_raises(tmp_path):
    parquet = tmp_path / "cross_plm_prott5__esm2_raw_cosine.parquet"
    _write_parquet(parquet, 15)
    sidecar = tmp_path / "cross_plm_prott5__esm2_raw_cosine.manifest.json"
    sidecar.write_text(json.dumps({
        "plm_a": "prott5", "plm_b": "esm2", "n_pairs": "15",  # string, not int
        "population_n_a": 6, "population_n_b": 6,
        "per_pair_columns": list(CROSS_PLM_PER_PAIR_COLUMNS), "path": str(parquet),
    }))
    with pytest.raises(SpecBuildError, match="n_pairs"):
        build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                     representations=["raw"], distances=["cosine"])


def test_population_n_non_int_rejected(tmp_path):
    parquet = tmp_path / "cross_plm_prott5__esm2_raw_cosine.parquet"
    _write_parquet(parquet, 15)
    sidecar = tmp_path / "cross_plm_prott5__esm2_raw_cosine.manifest.json"
    sidecar.write_text(json.dumps({
        "plm_a": "prott5", "plm_b": "esm2", "n_pairs": 15,
        "population_n_a": 6.0, "population_n_b": 6,  # float, not int
        "per_pair_columns": list(CROSS_PLM_PER_PAIR_COLUMNS), "path": str(parquet),
    }))
    with pytest.raises(SpecBuildError, match="population_n_a"):
        build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                     representations=["raw"], distances=["cosine"])


def test_artifact_order_is_deterministic_grid_order(tmp_path):
    pairs = [("prott5", "esm2"), ("ankh", "esm3")]
    for a, b in pairs:
        for dist in ("cosine", "euclidean", "manhattan"):
            _write_cell(tmp_path, a, b, "raw", dist)
    spec = build_cross_plm_barrier_spec(tmp_path, pairs=pairs, representations=["raw"],
                                        distances=["cosine", "euclidean", "manhattan"])
    labels = [a["label"] for a in spec["artifacts"]]
    assert labels == [
        "cross_plm:prott5:esm2:raw:cosine", "cross_plm:prott5:esm2:raw:euclidean",
        "cross_plm:prott5:esm2:raw:manhattan",
        "cross_plm:ankh:esm3:raw:cosine", "cross_plm:ankh:esm3:raw:euclidean",
        "cross_plm:ankh:esm3:raw:manhattan",
    ]


def test_no_expected_rows_leaves_rows_none(tmp_path):
    _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine", n_pairs=15)
    spec = build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                        representations=["raw"], distances=["cosine"],
                                        use_expected_rows=False)
    assert spec["artifacts"][0]["expected_rows"] is None


def test_dedup_axes(tmp_path):
    _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine")
    spec = build_cross_plm_barrier_spec(
        tmp_path, pairs=[("prott5", "esm2"), ("prott5", "esm2")],
        representations=["raw", "raw"], distances=["cosine", "cosine"],
    )
    assert len(spec["artifacts"]) == 1


# ── the built spec actually BITES through the REAL barrier ────────────────────
def test_built_spec_passes_real_barrier_on_good_cell(tmp_path):
    _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine", n_pairs=15)
    spec = build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                        representations=["raw"], distances=["cosine"])
    specs = [_spec_from_dict(a, i) for i, a in enumerate(spec["artifacts"])]
    assert run_barrier(specs).ok


def test_built_spec_catches_truncated_parquet(tmp_path):
    parquet, _ = _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine", n_pairs=15)
    spec = build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                        representations=["raw"], distances=["cosine"])
    _write_parquet(parquet, 8)  # truncate AFTER spec recorded expected_rows=15
    specs = [_spec_from_dict(a, i) for i, a in enumerate(spec["artifacts"])]
    report = run_barrier(specs)
    assert not report.ok
    assert any("row count" in r for s in report.failures for r in s.reasons)


def test_built_spec_catches_dropped_column(tmp_path):
    parquet, _ = _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine", n_pairs=15)
    spec = build_cross_plm_barrier_spec(tmp_path, pairs=[("prott5", "esm2")],
                                        representations=["raw"], distances=["cosine"])
    pd.read_parquet(parquet).drop(columns=["dist_a"]).to_parquet(parquet, index=False)
    specs = [_spec_from_dict(a, i) for i, a in enumerate(spec["artifacts"])]
    assert not run_barrier(specs).ok


# ── CLI ──────────────────────────────────────────────────────────────────────
def test_cli_writes_spec_and_returns_0(tmp_path):
    _write_cell(tmp_path, "prott5", "esm2", "raw", "cosine")
    out = tmp_path / "cross_plm_barrier_spec.json"
    rc = main(["--sidecar-dir", str(tmp_path), "--pairs", "prott5,esm2",
               "--representations", "raw", "--distances", "cosine", "--out", str(out)])
    assert rc == 0
    spec = json.loads(out.read_text())
    assert len(spec["artifacts"]) == 1


def test_cli_accepts_manhattan_distance(tmp_path):
    # argparse choices MUST include manhattan (a 2-element choices clone would reject it).
    _write_cell(tmp_path, "prott5", "esm2", "raw", "manhattan")
    out = tmp_path / "spec.json"
    rc = main(["--sidecar-dir", str(tmp_path), "--pairs", "prott5,esm2",
               "--representations", "raw", "--distances", "manhattan", "--out", str(out)])
    assert rc == 0
    spec = json.loads(out.read_text())
    assert spec["artifacts"][0]["label"] == "cross_plm:prott5:esm2:raw:manhattan"


def test_cli_no_pairs_returns_2(tmp_path):
    out = tmp_path / "spec.json"
    rc = main(["--sidecar-dir", str(tmp_path), "--pairs",
               "--representations", "raw", "--distances", "cosine", "--out", str(out)])
    assert rc == 2


def test_cli_orphan_returns_2(tmp_path):
    _write_parquet(tmp_path / "cross_plm_prott5__esm2_raw_cosine.parquet", 15)
    out = tmp_path / "spec.json"
    rc = main(["--sidecar-dir", str(tmp_path), "--pairs", "prott5,esm2",
               "--representations", "raw", "--distances", "cosine", "--out", str(out)])
    assert rc == 2


def test_cli_default_distances_cover_all_three(tmp_path):
    for dist in DEFAULT_DISTANCES:
        _write_cell(tmp_path, "prott5", "esm2", "raw", dist)
    out = tmp_path / "spec.json"
    rc = main(["--sidecar-dir", str(tmp_path), "--pairs", "prott5,esm2",
               "--representations", "raw", "--out", str(out)])  # distances defaulted
    assert rc == 0
    spec = json.loads(out.read_text())
    dists = {a["label"].rsplit(":", 1)[1] for a in spec["artifacts"]}
    assert dists == {"cosine", "euclidean", "manhattan"}
