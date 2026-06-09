"""Tests for evaluation.analysis_barrier — the fan-in barrier (revision plan v3, B6).

The barrier is the `afterok` parent of every analysis fan-in step. Before
cross-pLM / grid-stat / manifest run, it asserts that EVERY expected fan-out
artifact (per-pLM parquet pair tables, per-pLM H5 embeddings) exists AND is
COMPLETE — correct row count, required columns present, finite values, and
(for embeddings) non-zero norms. A killed job that left a valid-looking but
truncated file must fail the barrier, not be silently consumed (B7 hazard).

Generic by design: the barrier takes a list of ArtifactSpec and validates each.
The caller (run_pipeline / submit_analysis_dag.sh) supplies the 15-pLM grid —
the barrier itself imports no project modules so it stays trivially testable.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from evaluation.analysis_barrier import (
    ArtifactSpec,
    BarrierReport,
    check_artifact,
    main,
    run_barrier,
)


# ── fixtures ──────────────────────────────────────────────────────────────────
def _write_pairs(path: Path, n: int = 10, *, cols=("a", "b", "embedding_dist")) -> Path:
    rng = np.random.default_rng(0)
    data = {}
    for c in cols:
        if c in ("a", "b"):
            data[c] = [f"P{i:03d}" for i in range(n)]
        else:
            data[c] = rng.random(n)
    pd.DataFrame(data).to_parquet(path)
    return path


def _write_h5(
    path: Path, n: int = 10, dim: int = 8, *, zero_row=-1, nan_row=-1, ragged_row=-1
) -> Path:
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        for i in range(n):
            d = dim if i != ragged_row else dim // 2  # truncated last write
            vec = rng.standard_normal(d).astype(np.float32)
            if i == zero_row:
                vec[:] = 0.0
            if i == nan_row:
                vec[0] = np.nan
            f.create_dataset(f"P{i:03d}", data=vec)
    return path


# ── single-artifact checks ──────────────────────────────────────────────────
def test_good_parquet_passes(tmp_path):
    p = _write_pairs(tmp_path / "esm2_650m_distances.parquet", n=10)
    spec = ArtifactSpec(
        label="esm2_650m:distances",
        path=p,
        expected_rows=10,
        required_columns=("a", "b", "embedding_dist"),
        finite_columns=("embedding_dist",),
    )
    status = check_artifact(spec)
    assert status.ok, status.reasons
    assert status.n_rows == 10
    assert status.reasons == ()


def test_missing_file_fails(tmp_path):
    spec = ArtifactSpec(label="esm2_650m:distances", path=tmp_path / "nope.parquet")
    status = check_artifact(spec)
    assert not status.ok
    assert any("missing" in r.lower() for r in status.reasons)


def test_wrong_row_count_fails(tmp_path):
    p = _write_pairs(tmp_path / "x.parquet", n=7)
    status = check_artifact(ArtifactSpec(label="x", path=p, expected_rows=10))
    assert not status.ok
    assert status.n_rows == 7
    assert any("row" in r.lower() for r in status.reasons)


def test_missing_required_column_fails(tmp_path):
    p = _write_pairs(tmp_path / "x.parquet", n=5, cols=("a", "b"))
    status = check_artifact(
        ArtifactSpec(label="x", path=p, required_columns=("a", "b", "embedding_dist"))
    )
    assert not status.ok
    assert any("embedding_dist" in r for r in status.reasons)


def test_non_finite_column_fails(tmp_path):
    p = tmp_path / "x.parquet"
    pd.DataFrame(
        {"a": ["P0", "P1"], "b": ["P1", "P2"], "embedding_dist": [0.5, np.inf]}
    ).to_parquet(p)
    status = check_artifact(
        ArtifactSpec(label="x", path=p, finite_columns=("embedding_dist",))
    )
    assert not status.ok
    assert any("finite" in r.lower() for r in status.reasons)


def test_good_h5_passes(tmp_path):
    p = _write_h5(tmp_path / "esm2_650m.h5", n=319, dim=16)
    spec = ArtifactSpec(
        label="esm2_650m:emb", path=p, expected_rows=319, require_positive_norm=True
    )
    status = check_artifact(spec)
    assert status.ok, status.reasons
    assert status.n_rows == 319


def test_h5_truncated_dataset_count_fails(tmp_path):
    # A killed extraction left only 200 of 319 proteins — valid file, wrong count.
    p = _write_h5(tmp_path / "esm2_650m.h5", n=200, dim=16)
    status = check_artifact(
        ArtifactSpec(label="esm2_650m:emb", path=p, expected_rows=319)
    )
    assert not status.ok
    assert status.n_rows == 200
    assert any("row" in r.lower() or "count" in r.lower() for r in status.reasons)


def test_h5_zero_norm_vector_fails(tmp_path):
    p = _write_h5(tmp_path / "x.h5", n=10, dim=8, zero_row=3)
    status = check_artifact(
        ArtifactSpec(label="x", path=p, expected_rows=10, require_positive_norm=True)
    )
    assert not status.ok
    assert any("norm" in r.lower() for r in status.reasons)


def test_h5_nan_vector_fails(tmp_path):
    p = _write_h5(tmp_path / "x.h5", n=10, dim=8, nan_row=4)
    status = check_artifact(
        ArtifactSpec(label="x", path=p, expected_rows=10, require_positive_norm=True)
    )
    assert not status.ok
    assert any("finite" in r.lower() or "nan" in r.lower() for r in status.reasons)


def test_corrupt_file_fails_gracefully(tmp_path):
    p = tmp_path / "broken.parquet"
    p.write_bytes(b"not a parquet file")
    status = check_artifact(ArtifactSpec(label="x", path=p))
    assert not status.ok
    assert any("unreadable" in r.lower() for r in status.reasons)


# ── full barrier over a grid ──────────────────────────────────────────────────
def test_barrier_all_present_is_ok(tmp_path):
    specs = []
    for plm in ("esm2_650m", "prot_t5", "prost_t5"):
        p = _write_pairs(tmp_path / f"{plm}_distances.parquet", n=10)
        specs.append(
            ArtifactSpec(label=f"{plm}:distances", path=p, expected_rows=10)
        )
    report = run_barrier(specs)
    assert isinstance(report, BarrierReport)
    assert report.ok
    assert report.failures == ()


def test_barrier_one_missing_fails_and_names_it(tmp_path):
    specs = []
    for plm in ("esm2_650m", "prot_t5"):
        p = _write_pairs(tmp_path / f"{plm}_distances.parquet", n=10)
        specs.append(ArtifactSpec(label=f"{plm}:distances", path=p, expected_rows=10))
    # prost_t5 never written → missing.
    specs.append(
        ArtifactSpec(
            label="prost_t5:distances",
            path=tmp_path / "prost_t5_distances.parquet",
            expected_rows=10,
        )
    )
    report = run_barrier(specs)
    assert not report.ok
    failure_labels = {s.label for s in report.failures}
    assert failure_labels == {"prost_t5:distances"}
    text = report.format_report()
    assert "prost_t5:distances" in text
    assert "PASS" in text or "pass" in text  # report shows the passing ones too


# ── CLI ───────────────────────────────────────────────────────────────────────
def test_main_returns_zero_on_success(tmp_path):
    p = _write_pairs(tmp_path / "esm2_650m_distances.parquet", n=10)
    spec_file = tmp_path / "barrier_spec.json"
    spec_file.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "label": "esm2_650m:distances",
                        "path": str(p),
                        "expected_rows": 10,
                        "required_columns": ["a", "b", "embedding_dist"],
                    }
                ]
            }
        )
    )
    assert main(["--spec", str(spec_file)]) == 0


def test_main_returns_one_on_failure(tmp_path):
    spec_file = tmp_path / "barrier_spec.json"
    spec_file.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "label": "missing:distances",
                        "path": str(tmp_path / "absent.parquet"),
                        "expected_rows": 10,
                    }
                ]
            }
        )
    )
    assert main(["--spec", str(spec_file)]) == 1


# ── review fixes: H5 dimension / key identity ─────────────────────────────────
def test_h5_ragged_dimension_fails(tmp_path):
    # Right key count, but one vector was truncated to half-dim by a killed job.
    p = _write_h5(tmp_path / "x.h5", n=10, dim=16, ragged_row=5)
    status = check_artifact(
        ArtifactSpec(label="x", path=p, expected_rows=10, expected_dim=16)
    )
    assert not status.ok
    assert any("dim" in r.lower() for r in status.reasons)


def test_h5_expected_dim_passes(tmp_path):
    p = _write_h5(tmp_path / "x.h5", n=10, dim=16)
    status = check_artifact(
        ArtifactSpec(label="x", path=p, expected_rows=10, expected_dim=16)
    )
    assert status.ok, status.reasons


def test_h5_expected_keys_identity_mismatch_fails(tmp_path):
    # 10 keys present, but they are the WRONG proteins (count alone would pass).
    p = _write_h5(tmp_path / "x.h5", n=10, dim=8)
    wanted = tuple(f"Q{i:03d}" for i in range(10))  # disjoint id space
    status = check_artifact(
        ArtifactSpec(label="x", path=p, expected_rows=10, expected_keys=wanted)
    )
    assert not status.ok
    assert any("key" in r.lower() for r in status.reasons)


def test_h5_expected_keys_identity_match_passes(tmp_path):
    p = _write_h5(tmp_path / "x.h5", n=5, dim=8)
    wanted = tuple(f"P{i:03d}" for i in range(5))
    status = check_artifact(
        ArtifactSpec(label="x", path=p, expected_rows=5, expected_keys=wanted)
    )
    assert status.ok, status.reasons


def test_min_norm_flags_near_dead_vector(tmp_path):
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("P000", data=np.full(8, 1.0, dtype=np.float32))
        f.create_dataset("P001", data=np.full(8, 1e-9, dtype=np.float32))  # ~dead
    status = check_artifact(
        ArtifactSpec(
            label="x", path=p, expected_rows=2,
            require_positive_norm=True, min_norm=1e-3,
        )
    )
    assert not status.ok
    assert any("norm" in r.lower() for r in status.reasons)


# ── review fixes: no-crash failure paths ──────────────────────────────────────
def test_unknown_suffix_reported_not_crashed(tmp_path):
    p = tmp_path / "mystery.txt"
    p.write_text("hello")
    status = check_artifact(ArtifactSpec(label="x", path=p))  # kind=auto
    assert not status.ok
    assert any("kind" in r.lower() for r in status.reasons)
    # And it must not abort a whole barrier run.
    report = run_barrier([ArtifactSpec(label="x", path=p)])
    assert not report.ok


def test_finite_check_on_nonnumeric_column_fails_gracefully(tmp_path):
    p = _write_pairs(tmp_path / "x.parquet", n=4)
    # 'a' is a string id column — np.isfinite would raise; barrier must report.
    status = check_artifact(ArtifactSpec(label="x", path=p, finite_columns=("a",)))
    assert not status.ok
    assert any("numeric" in r.lower() or "finite" in r.lower() for r in status.reasons)


# ── review fixes: id integrity on pair tables ─────────────────────────────────
def test_duplicate_pairs_fail(tmp_path):
    p = tmp_path / "x.parquet"
    pd.DataFrame(
        {"a": ["P0", "P0", "P1"], "b": ["P1", "P1", "P2"], "embedding_dist": [0.1, 0.1, 0.2]}
    ).to_parquet(p)
    status = check_artifact(
        ArtifactSpec(label="x", path=p, unique_columns=("a", "b"))
    )
    assert not status.ok
    assert any("dupl" in r.lower() for r in status.reasons)


def test_null_id_fails(tmp_path):
    p = tmp_path / "x.parquet"
    pd.DataFrame(
        {"a": ["P0", None], "b": ["P1", "P2"], "embedding_dist": [0.1, 0.2]}
    ).to_parquet(p)
    status = check_artifact(
        ArtifactSpec(label="x", path=p, non_null_columns=("a", "b"))
    )
    assert not status.ok
    assert any("null" in r.lower() for r in status.reasons)


# ── review fixes: empty artifact is never valid ───────────────────────────────
def test_empty_parquet_fails_without_expected_rows(tmp_path):
    p = tmp_path / "x.parquet"
    pd.DataFrame({"a": [], "b": [], "embedding_dist": []}).to_parquet(p)
    status = check_artifact(ArtifactSpec(label="x", path=p))  # no expected_rows
    assert not status.ok
    assert any("empty" in r.lower() for r in status.reasons)


def test_empty_h5_fails_without_expected_rows(tmp_path):
    p = tmp_path / "x.h5"
    with h5py.File(p, "w"):
        pass
    status = check_artifact(ArtifactSpec(label="x", path=p))
    assert not status.ok
    assert any("empty" in r.lower() for r in status.reasons)


# ── review fixes: report hygiene + mixed grid ─────────────────────────────────
def test_report_is_pure_ascii_and_shows_reasons(tmp_path):
    p = _write_pairs(tmp_path / "x.parquet", n=7)
    report = run_barrier([ArtifactSpec(label="x:distances", path=p, expected_rows=10)])
    text = report.format_report()
    text.encode("ascii")  # must not raise (SLURM LANG=C safety)
    assert "row count 7" in text  # the failing reason is rendered


def test_barrier_mixed_parquet_and_h5(tmp_path):
    pq = _write_pairs(tmp_path / "esm2_650m_distances.parquet", n=10)
    h5 = _write_h5(tmp_path / "esm2_650m.h5", n=10, dim=8)
    report = run_barrier(
        [
            ArtifactSpec(label="esm2_650m:distances", path=pq, expected_rows=10),
            ArtifactSpec(label="esm2_650m:emb", path=h5, expected_rows=10,
                         require_positive_norm=True),
        ]
    )
    assert report.ok, report.format_report()


# ── review fixes: CLI config errors get a distinct exit code (2) ──────────────
def test_main_missing_spec_file_returns_2(tmp_path):
    assert main(["--spec", str(tmp_path / "nope.json")]) == 2


def test_main_malformed_json_returns_2(tmp_path):
    spec_file = tmp_path / "bad.json"
    spec_file.write_text("{ this is not json")
    assert main(["--spec", str(spec_file)]) == 2


def test_main_missing_artifacts_key_returns_2(tmp_path):
    spec_file = tmp_path / "bad.json"
    spec_file.write_text(json.dumps({"wrong_key": []}))
    assert main(["--spec", str(spec_file)]) == 2


def test_main_spec_entry_missing_path_returns_2(tmp_path):
    spec_file = tmp_path / "bad.json"
    spec_file.write_text(json.dumps({"artifacts": [{"label": "x"}]}))
    assert main(["--spec", str(spec_file)]) == 2
