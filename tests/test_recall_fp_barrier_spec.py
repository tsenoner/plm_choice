"""Tests for evaluation.recall_fp_barrier_spec — the fan-in barrier spec-builder.

The spec-builder is the *caller* the generic ``analysis_barrier`` defers to: it
walks the recall-fp grid (pLM × representation × CATH level), reads each per-(pLM,
representation) sidecar manifest that ``recall_fp_report``'s CLI wrote, and emits a
``barrier_spec.json`` = ``{"artifacts": [ {<ArtifactSpec fields>}, ... ]}`` that
``analysis_barrier`` validates. Key contracts under test:

* one artifact per (pLM, rep, level) cell — the full grid, so a missing fan-out
  cell is surfaced by the barrier, never silently dropped;
* the parquet path is **sidecar-authoritative** (use the path the producing run
  recorded, which survives ``--no-overwrite`` timestamping) and only falls back to
  the canonical reconstructed name when a sidecar is absent;
* the column/guard contract is transcribed from the single source of truth
  (``recall_fp_report.PARQUET_GUARDS``), and a sidecar that disagrees fails loud.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.analysis_barrier import _load_specs, run_barrier
from evaluation.recall_fp_barrier_spec import (
    SpecBuildError,
    build_recall_fp_barrier_spec,
    main,
    write_barrier_spec,
)
from evaluation.recall_fp_report import PARQUET_GUARDS, PER_QUERY_COLUMNS


_OMIT = object()  # sentinel: omit per_query_columns entirely (older-sidecar back-compat)


def _load_specs_dict(spec, tmp_path):
    """Write a built spec to JSON and parse it back via the REAL barrier loader."""
    spec_path = tmp_path / "barrier_spec_under_test.json"
    write_barrier_spec(spec, spec_path)
    return _load_specs(spec_path)


def _level_info(out_dir, plm, rep, level, n_pos):
    return {
        "path": str(out_dir / f"recall_fp_{plm}_{rep}_{level}.parquet"),
        "n_queries_with_positives": n_pos,
        "n_queries_skipped_no_positives": 0,
        "n_scored": n_pos,
        "mean_recall_1stFP": 1.0,
    }


def _make_real_recall_fp(tmp_path):
    """Produce REAL parquets + sidecar via the task-1 CLI; return the out dir."""
    import h5py

    from evaluation.recall_fp_report import main as recall_main

    emb = {
        "P1": np.array([0.0, 0.0], dtype=np.float32),
        "P2": np.array([0.1, 0.0], dtype=np.float32),
        "P3": np.array([5.0, 5.0], dtype=np.float32),
        "P4": np.array([5.1, 5.0], dtype=np.float32),
    }
    h5 = tmp_path / "prott5.h5"
    with h5py.File(h5, "w") as f:
        for pid, vec in emb.items():
            f.create_dataset(pid, data=vec)
    tsv = tmp_path / "cath.tsv"
    tsv.write_text(
        "Entry\tGene3D\n"
        "P1\t3.30.70.10\nP2\t3.30.70.10\nP3\t1.10.10.10\nP4\t1.10.10.10\n"
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"ids": ["P1", "P2", "P3", "P4"], "n_proteins": 4}))
    out = tmp_path / "out"
    rc = recall_main([
        "--plm", "prott5", "--emb-h5", str(h5), "--cath-tsv", str(tsv),
        "--freeze", str(freeze), "--out-dir", str(out),
        "--distance", "euclidean", "--representation", "raw",
    ])
    assert rc == 0
    return out


def _write_sidecar(
    out_dir,
    plm,
    rep,
    *,
    levels=("fold", "superfamily"),
    n_pos=4,
    population_n=4,
    per_query_columns=None,
    level_paths=None,
):
    """Write a recall_fp_<plm>_<rep>.manifest.json like the task-1 CLI does."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels_block = {}
    for lvl in levels:
        info = _level_info(out_dir, plm, rep, lvl, n_pos)
        if level_paths and lvl in level_paths:
            info["path"] = str(level_paths[lvl])
        levels_block[lvl] = info
    manifest = {
        "pLM": plm,
        "representation": rep,
        "distance": "euclidean",
        "population_n": population_n,
        "levels": levels_block,
    }
    if per_query_columns is not _OMIT:
        manifest["per_query_columns"] = list(
            per_query_columns if per_query_columns is not None else PER_QUERY_COLUMNS
        )
    path = out_dir / f"recall_fp_{plm}_{rep}.manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


# ── grid construction ────────────────────────────────────────────────────────
def test_builds_one_artifact_per_grid_cell(tmp_path):
    for plm in ("prott5", "esm2"):
        for rep in ("raw", "ffn"):
            _write_sidecar(tmp_path, plm, rep)
    spec = build_recall_fp_barrier_spec(
        tmp_path, plms=["prott5", "esm2"], representations=["raw", "ffn"],
        levels=["fold", "superfamily"],
    )
    arts = spec["artifacts"]
    assert len(arts) == 2 * 2 * 2  # pLM × rep × level
    labels = {a["label"] for a in arts}
    assert "recall_fp:prott5:raw:fold" in labels
    assert "recall_fp:esm2:ffn:superfamily" in labels


def test_family_excluded_by_default_levels(tmp_path):
    _write_sidecar(tmp_path, "prott5", "raw")
    spec = build_recall_fp_barrier_spec(tmp_path, plms=["prott5"], representations=["raw"])
    assert not any(a["label"].endswith(":family") for a in spec["artifacts"])
    assert len(spec["artifacts"]) == 2  # fold + superfamily only


def test_guards_transcribed_from_constant(tmp_path):
    _write_sidecar(tmp_path, "prott5", "raw")
    spec = build_recall_fp_barrier_spec(tmp_path, plms=["prott5"], representations=["raw"])
    art = spec["artifacts"][0]
    assert tuple(art["required_columns"]) == PARQUET_GUARDS["required_columns"]
    assert tuple(art["unique_columns"]) == PARQUET_GUARDS["unique_columns"]
    assert tuple(art["non_null_columns"]) == PARQUET_GUARDS["non_null_columns"]
    assert tuple(art["finite_columns"]) == PARQUET_GUARDS["finite_columns"]
    assert art["kind"] == "parquet"


def test_expected_rows_from_sidecar(tmp_path):
    _write_sidecar(tmp_path, "prott5", "raw", n_pos=7)
    spec = build_recall_fp_barrier_spec(tmp_path, plms=["prott5"], representations=["raw"])
    assert all(a["expected_rows"] == 7 for a in spec["artifacts"])


def test_expected_rows_omitted_when_disabled(tmp_path):
    _write_sidecar(tmp_path, "prott5", "raw", n_pos=7)
    spec = build_recall_fp_barrier_spec(
        tmp_path, plms=["prott5"], representations=["raw"], use_expected_rows=False
    )
    assert all(a["expected_rows"] is None for a in spec["artifacts"])


# ── sidecar-path authority ─────────────────────────────────────────────────────
def test_path_is_sidecar_authoritative_not_reconstructed(tmp_path):
    # A --no-overwrite producing run records a TIMESTAMPED parquet path; the spec
    # must point at THAT, not the canonical reconstructed name (else the barrier
    # validates a stale/absent file).
    stamped = tmp_path / "recall_fp_prott5_raw_fold.20260610_120000.parquet"
    _write_sidecar(tmp_path, "prott5", "raw", levels=["fold"], level_paths={"fold": stamped})
    spec = build_recall_fp_barrier_spec(
        tmp_path, plms=["prott5"], representations=["raw"], levels=["fold"]
    )
    assert spec["artifacts"][0]["path"] == str(stamped)


def test_missing_sidecar_emits_reconstructed_cell_flagged(tmp_path):
    # prott5/raw present; esm2/raw sidecar absent -> the esm2 cells must STILL be
    # emitted (canonical reconstructed path, expected_rows None) so the barrier
    # reports them missing, and the count of reconstructed cells is recorded.
    _write_sidecar(tmp_path, "prott5", "raw")
    spec = build_recall_fp_barrier_spec(
        tmp_path, plms=["prott5", "esm2"], representations=["raw"]
    )
    by_label = {a["label"]: a for a in spec["artifacts"]}
    assert len(spec["artifacts"]) == 2 * 2  # 2 pLM × 1 rep × 2 level
    esm2_fold = by_label["recall_fp:esm2:raw:fold"]
    assert esm2_fold["path"] == str(tmp_path / "recall_fp_esm2_raw_fold.parquet")
    assert esm2_fold["expected_rows"] is None
    assert spec["_meta"]["n_cells_without_sidecar"] == 2
    assert spec["_meta"]["n_cells"] == 4


# ── drift / fault handling ──────────────────────────────────────────────────────
def test_per_query_columns_drift_raises(tmp_path):
    # A sidecar whose schema disagrees with the single source of truth is a drift
    # signal (task-1/task-2 out of sync) -> fail loud, do not silently transcribe.
    _write_sidecar(
        tmp_path, "prott5", "raw",
        per_query_columns=["query_id", "recall"],  # wrong/short
    )
    with pytest.raises(SpecBuildError):
        build_recall_fp_barrier_spec(tmp_path, plms=["prott5"], representations=["raw"])


def test_malformed_sidecar_json_raises(tmp_path):
    bad = tmp_path / "recall_fp_prott5_raw.manifest.json"
    bad.write_text("{ not json")
    with pytest.raises(SpecBuildError):
        build_recall_fp_barrier_spec(tmp_path, plms=["prott5"], representations=["raw"])


def test_empty_plms_raises(tmp_path):
    with pytest.raises(SpecBuildError):
        build_recall_fp_barrier_spec(tmp_path, plms=[], representations=["raw"])


# ── round-trip through the real barrier ─────────────────────────────────────────
def test_spec_loads_via_barrier_and_passes_on_real_parquets(tmp_path):
    # End-to-end: task-1 CLI produces real parquets + sidecars -> build the spec ->
    # the real analysis_barrier._load_specs parses it and run_barrier validates the
    # actual artifacts as complete.
    from evaluation.recall_fp_report import main as recall_main

    emb = {
        "P1": np.array([0.0, 0.0], dtype=np.float32),
        "P2": np.array([0.1, 0.0], dtype=np.float32),
        "P3": np.array([5.0, 5.0], dtype=np.float32),
        "P4": np.array([5.1, 5.0], dtype=np.float32),
    }
    import h5py

    h5 = tmp_path / "prott5.h5"
    with h5py.File(h5, "w") as f:
        for pid, vec in emb.items():
            f.create_dataset(pid, data=vec)
    tsv = tmp_path / "cath.tsv"
    tsv.write_text(
        "Entry\tGene3D\n"
        "P1\t3.30.70.10\nP2\t3.30.70.10\nP3\t1.10.10.10\nP4\t1.10.10.10\n"
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"ids": ["P1", "P2", "P3", "P4"], "n_proteins": 4}))
    out = tmp_path / "out"
    rc = recall_main([
        "--plm", "prott5", "--emb-h5", str(h5), "--cath-tsv", str(tsv),
        "--freeze", str(freeze), "--out-dir", str(out),
        "--distance", "euclidean", "--representation", "raw",
    ])
    assert rc == 0

    spec = build_recall_fp_barrier_spec(out, plms=["prott5"], representations=["raw"])
    spec_path = out / "barrier_spec.json"
    write_barrier_spec(spec, spec_path)

    specs = _load_specs(spec_path)  # the real barrier loader, no SpecError
    assert len(specs) == 2
    report = run_barrier(specs)
    assert report.ok, report.format_report()


# ── writer ──────────────────────────────────────────────────────────────────────
def test_write_barrier_spec_replaces_in_place_by_default(tmp_path):
    _write_sidecar(tmp_path, "prott5", "raw")
    spec = build_recall_fp_barrier_spec(tmp_path, plms=["prott5"], representations=["raw"])
    target = tmp_path / "barrier_spec.json"
    first = write_barrier_spec(spec, target)
    second = write_barrier_spec(spec, target)
    assert first == target == second
    assert len(list(tmp_path.glob("barrier_spec*.json"))) == 1


# ── CLI ───────────────────────────────────────────────────────────────────────
def test_cli_writes_spec_returns_0(tmp_path):
    _write_sidecar(tmp_path, "prott5", "raw")
    _write_sidecar(tmp_path, "prott5", "ffn")
    out = tmp_path / "barrier_spec.json"
    rc = main([
        "--sidecar-dir", str(tmp_path), "--plms", "prott5",
        "--representations", "raw", "ffn", "--out", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert len(payload["artifacts"]) == 4  # 1 pLM × 2 rep × 2 level


def test_cli_drift_returns_2(tmp_path):
    _write_sidecar(tmp_path, "prott5", "raw", per_query_columns=["query_id"])
    out = tmp_path / "barrier_spec.json"
    rc = main([
        "--sidecar-dir", str(tmp_path), "--plms", "prott5",
        "--representations", "raw", "--out", str(out),
    ])
    assert rc == 2
    assert not out.exists()


# ── negative round-trips: prove the spec's guards/expected_rows actually BITE ────
def test_barrier_fails_when_a_referenced_parquet_is_deleted(tmp_path):
    out = _make_real_recall_fp(tmp_path)
    spec = build_recall_fp_barrier_spec(out, plms=["prott5"], representations=["raw"])
    Path(spec["artifacts"][0]["path"]).unlink()  # delete one real parquet
    report = run_barrier(_load_specs_dict(spec, tmp_path))
    assert not report.ok
    assert any("missing" in r for s in report.failures for r in s.reasons)


def test_barrier_fails_on_row_count_when_parquet_truncated(tmp_path):
    out = _make_real_recall_fp(tmp_path)
    spec = build_recall_fp_barrier_spec(out, plms=["prott5"], representations=["raw"])
    target = spec["artifacts"][0]["path"]
    assert spec["artifacts"][0]["expected_rows"] == 4
    df = pd.read_parquet(target)
    df.iloc[:2].to_parquet(target, index=False)  # truncate 4 -> 2 rows
    report = run_barrier(_load_specs_dict(spec, tmp_path))
    assert not report.ok
    assert any("row count" in r for s in report.failures for r in s.reasons)


def test_barrier_fails_when_guard_column_dropped(tmp_path):
    out = _make_real_recall_fp(tmp_path)
    spec = build_recall_fp_barrier_spec(out, plms=["prott5"], representations=["raw"])
    target = spec["artifacts"][0]["path"]
    df = pd.read_parquet(target).drop(columns=["query_id"])  # break the contract
    df.to_parquet(target, index=False)
    report = run_barrier(_load_specs_dict(spec, tmp_path))
    assert not report.ok  # missing required/unique/non-null column on query_id


# ── stale / orphan artifact (C2) ────────────────────────────────────────────────
def test_orphan_parquet_without_sidecar_raises(tmp_path):
    # A canonical parquet present with NO sidecar is a stale/partial artifact the
    # barrier would otherwise pass on shape alone -> the builder must fail closed.
    (tmp_path / "recall_fp_prott5_raw_fold.parquet").write_bytes(b"not-a-real-parquet")
    with pytest.raises(SpecBuildError):
        build_recall_fp_barrier_spec(tmp_path, plms=["prott5"], representations=["raw"])


# ── back-compat + provenance + grid guards ──────────────────────────────────────
def test_sidecar_without_per_query_columns_still_builds(tmp_path):
    # An older task-1 sidecar that predates per_query_columns skips the drift check
    # and still transcribes the canonical guards.
    _write_sidecar(tmp_path, "prott5", "raw", per_query_columns=_OMIT)
    spec = build_recall_fp_barrier_spec(tmp_path, plms=["prott5"], representations=["raw"])
    art = spec["artifacts"][0]
    assert tuple(art["required_columns"]) == PARQUET_GUARDS["required_columns"]


def test_population_n_propagated_to_meta(tmp_path):
    # Capped pLM (population_n=267) must be carried forward so a downstream step can
    # keep it out of a bare cross-pLM mean.
    _write_sidecar(tmp_path, "esm1b", "raw", population_n=267)
    spec = build_recall_fp_barrier_spec(tmp_path, plms=["esm1b"], representations=["raw"])
    assert spec["_meta"]["population_n"]["esm1b:raw"] == 267


def test_expected_n_plms_guard_catches_under_coverage(tmp_path):
    _write_sidecar(tmp_path, "prott5", "raw")
    # asking for 15 but only 1 pLM in the grid -> fail closed
    with pytest.raises(SpecBuildError):
        build_recall_fp_barrier_spec(
            tmp_path, plms=["prott5"], representations=["raw"], expected_n_plms=15
        )


def test_duplicate_plms_deduped_not_inflated(tmp_path):
    _write_sidecar(tmp_path, "prott5", "raw")
    spec = build_recall_fp_barrier_spec(
        tmp_path, plms=["prott5", "prott5"], representations=["raw"]
    )
    assert spec["_meta"]["n_cells"] == 2  # not 4


def test_artifact_order_is_deterministic_grid_order(tmp_path):
    for plm in ("prott5", "esm2"):
        for rep in ("raw", "ffn"):
            _write_sidecar(tmp_path, plm, rep)
    spec = build_recall_fp_barrier_spec(
        tmp_path, plms=["prott5", "esm2"], representations=["raw", "ffn"],
        levels=["fold", "superfamily"],
    )
    assert [a["label"] for a in spec["artifacts"]] == [
        "recall_fp:prott5:raw:fold",
        "recall_fp:prott5:raw:superfamily",
        "recall_fp:prott5:ffn:fold",
        "recall_fp:prott5:ffn:superfamily",
        "recall_fp:esm2:raw:fold",
        "recall_fp:esm2:raw:superfamily",
        "recall_fp:esm2:ffn:fold",
        "recall_fp:esm2:ffn:superfamily",
    ]


def test_every_artifact_carries_full_guard_contract(tmp_path):
    for rep in ("raw", "ffn"):
        _write_sidecar(tmp_path, "prott5", rep)
    spec = build_recall_fp_barrier_spec(
        tmp_path, plms=["prott5"], representations=["raw", "ffn"]
    )
    for art in spec["artifacts"]:  # not just artifacts[0]
        assert art["kind"] == "parquet"
        assert tuple(art["required_columns"]) == PARQUET_GUARDS["required_columns"]
        assert tuple(art["unique_columns"]) == PARQUET_GUARDS["unique_columns"]
        assert tuple(art["finite_columns"]) == PARQUET_GUARDS["finite_columns"]
        assert tuple(art["non_null_columns"]) == PARQUET_GUARDS["non_null_columns"]


# ── CLI flag plumbing + the "no silent caps" warning ─────────────────────────────
def test_cli_missing_sidecar_warns_on_stderr(tmp_path, capsys):
    _write_sidecar(tmp_path, "prott5", "raw")  # esm2 absent
    out = tmp_path / "barrier_spec.json"
    rc = main([
        "--sidecar-dir", str(tmp_path), "--plms", "prott5", "esm2",
        "--representations", "raw", "--out", str(out),
    ])
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" in err and "no sidecar" in err
    assert "recall_fp:esm2:raw:fold" in err


def test_cli_no_expected_rows_flag(tmp_path):
    _write_sidecar(tmp_path, "prott5", "raw", n_pos=4)
    out = tmp_path / "barrier_spec.json"
    rc = main([
        "--sidecar-dir", str(tmp_path), "--plms", "prott5",
        "--representations", "raw", "--out", str(out), "--no-expected-rows",
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert all(a["expected_rows"] is None for a in payload["artifacts"])


def test_cli_expected_n_plms_mismatch_returns_2(tmp_path):
    _write_sidecar(tmp_path, "prott5", "raw")
    out = tmp_path / "barrier_spec.json"
    rc = main([
        "--sidecar-dir", str(tmp_path), "--plms", "prott5",
        "--representations", "raw", "--out", str(out), "--expected-n-plms", "15",
    ])
    assert rc == 2
    assert not out.exists()


# ── Phase-0 backfill: characterization tests pinning behavior the refactor must preserve ──

def test_unreadable_sidecar_raises_specbuilderror(tmp_path):
    # OSError branch of sidecar read: a DIRECTORY at the manifest path is unreadable as text.
    (tmp_path / "recall_fp_prott5_raw.manifest.json").mkdir()
    with pytest.raises(SpecBuildError):
        build_recall_fp_barrier_spec(tmp_path, plms=["prott5"], representations=["raw"])

def test_write_barrier_spec_emits_indent2_trailing_newline(tmp_path):
    # Byte-level pin so a refactor that drops indent=2 / trailing "\n" is caught.
    _write_sidecar(tmp_path, "prott5", "raw")
    spec = build_recall_fp_barrier_spec(tmp_path, plms=["prott5"], representations=["raw"])
    target = tmp_path / "barrier_spec.json"
    write_barrier_spec(spec, target)
    text = target.read_text()
    assert text.endswith("\n")
    assert "\n  " in text  # indent=2 produced indented lines

def test_specbuilderror_match_on_grid_size_orphan_drift(tmp_path):
    # Backfill match= on recall-side raises so message DRIFT is caught (recall used bare raises).
    _write_sidecar(tmp_path, "prott5", "raw")
    with pytest.raises(SpecBuildError, match="expected"):
        build_recall_fp_barrier_spec(
            tmp_path, plms=["prott5"], representations=["raw"], expected_n_plms=15
        )
    (tmp_path / "recall_fp_esm2_raw_fold.parquet").write_bytes(b"x")
    with pytest.raises(SpecBuildError, match="orphan"):
        build_recall_fp_barrier_spec(tmp_path, plms=["esm2"], representations=["raw"])
    _write_sidecar(tmp_path, "ankh", "raw", per_query_columns=["query_id"])
    with pytest.raises(SpecBuildError, match="per_query_columns"):
        build_recall_fp_barrier_spec(tmp_path, plms=["ankh"], representations=["raw"])
