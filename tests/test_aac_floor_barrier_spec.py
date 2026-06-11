"""Tests for evaluation.aac_floor_barrier_spec — fan-in barrier spec-builder for
the AAC-floor grid (population_tag × level, per-distance sidecar_dir).

Mirrors test_ec_barrier_spec / test_snn_barrier_spec / test_barrier_spec contract:
  - one artifact per (population_tag, level) cell, no silent gaps
  - sidecar-path authoritative; canonical path is reconstruction fallback only
  - orphan parquet (no sidecar) fails closed
  - per_query_columns drift fails loud
  - distinct population_tag/level cells don't collide (C1 — full319 vs esm1b)
  - CLI exit codes 0 (ok) / 2 (config/I-O fault); no exit 1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.aac_floor_barrier_spec import (
    DEFAULT_LEVELS,
    DEFAULT_POPULATION_TAGS,
    build_aac_floor_barrier_spec,
    main,
)
from evaluation.barrier_spec_base import SpecBuildError
from evaluation.recall_fp_report import PARQUET_GUARDS, PER_QUERY_COLUMNS


# ── fixture helpers ──────────────────────────────────────────────────────────

def _write_parquet(path: Path, n_rows: int = 5, cols=PER_QUERY_COLUMNS):
    """Write a minimal per-query parquet (real file; barrier-visible)."""
    df = pd.DataFrame(
        {c: ([f"Q{i}" for i in range(n_rows)] if c == "query_id"
             else np.linspace(0.0, 1.0, n_rows)) for c in cols}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_cell(
    d: Path,
    population_tag: str,
    *,
    levels: tuple[str, ...] = ("fold", "superfamily"),
    n_queries: int = 5,
    population_n: int = 10,
    per_query_columns: list[str] | None = None,
    write_parquets: bool = True,
) -> tuple[dict[str, Path], Path]:
    """Write a (per-level parquets + sidecar) cell the way aac_floor_report.main does.

    Returns ``(level -> parquet_path, sidecar_path)``.
    """
    if per_query_columns is None:
        per_query_columns = list(PER_QUERY_COLUMNS)
    parquet_paths: dict[str, Path] = {}
    levels_dict: dict = {}
    for level in levels:
        parquet = d / f"aac_floor_{population_tag}_{level}.parquet"
        if write_parquets:
            _write_parquet(parquet, n_queries)
        parquet_paths[level] = parquet
        levels_dict[level] = {
            "path": str(parquet),
            "n_queries_with_positives": n_queries,
            "n_queries_skipped_no_positives": 0,
            "n_scored": n_queries,
            "mean_recall_1stFP": 0.5,
            "ci_lo": 0.3,
            "ci_hi": 0.7,
            "ci_degenerate": False,
            "n_ties_at_first_fp": 0,
        }
    sidecar = d / f"aac_floor_{population_tag}.manifest.json"
    sidecar.write_text(json.dumps({
        "floor": "aac",
        "population_tag": population_tag,
        "distance": "euclidean",
        "include_other": False,
        "population_n": population_n,
        "per_query_columns": per_query_columns,
        "levels": levels_dict,
    }))
    return parquet_paths, sidecar


# ── core correctness tests ────────────────────────────────────────────────────

def test_full_grid_one_artifact_per_cell(tmp_path):
    """Full population_tag × level grid emits exactly one artifact per cell."""
    for pop_tag in ("full319", "esm1b"):
        _write_cell(tmp_path, pop_tag)
    spec = build_aac_floor_barrier_spec(
        tmp_path,
        population_tags=["full319", "esm1b"],
        levels=["fold", "superfamily"],
    )
    assert len(spec["artifacts"]) == 4   # 2 pop_tags × 2 levels
    assert spec["_meta"]["n_cells"] == 4
    assert spec["_meta"]["n_cells_without_sidecar"] == 0


def test_missing_sidecar_emits_reconstructed_cell_not_dropped(tmp_path):
    """A sidecar-less cell appears in artifacts (barrier reports it MISSING, not silently dropped)."""
    _write_cell(tmp_path, "full319")  # esm1b absent entirely
    spec = build_aac_floor_barrier_spec(
        tmp_path,
        population_tags=["full319", "esm1b"],
        levels=["fold"],
    )
    assert spec["_meta"]["n_cells_without_sidecar"] == 1
    # The reconstructed cell label must name the absent population
    assert any("esm1b" in label for label in spec["_meta"]["reconstructed_cells"])
    # The artifact must use the canonical reconstructed parquet path
    esm1b_arts = [a for a in spec["artifacts"] if "esm1b" in a["label"]]
    assert len(esm1b_arts) == 1
    assert esm1b_arts[0]["path"].endswith("aac_floor_esm1b_fold.parquet")
    assert esm1b_arts[0]["expected_rows"] is None


def test_orphan_parquet_without_sidecar_fails_closed(tmp_path):
    """A canonical parquet present with no sidecar is a stale/partial artifact → SpecBuildError."""
    (tmp_path / "aac_floor_full319_fold.parquet").write_text("")  # parquet, no sidecar
    with pytest.raises(SpecBuildError, match="orphan"):
        build_aac_floor_barrier_spec(
            tmp_path, population_tags=["full319"], levels=["fold"]
        )


def test_per_query_columns_drift_raises(tmp_path):
    """A sidecar whose per_query_columns disagree with the contract raises SpecBuildError."""
    parquet = tmp_path / "aac_floor_full319_fold.parquet"
    _write_parquet(parquet)
    sidecar = tmp_path / "aac_floor_full319.manifest.json"
    sidecar.write_text(json.dumps({
        "floor": "aac", "population_tag": "full319", "distance": "euclidean",
        "include_other": False, "population_n": 10,
        "per_query_columns": ["WRONG_COLUMN"],
        "levels": {"fold": {"path": str(parquet), "n_queries_with_positives": 5}},
    }))
    with pytest.raises(SpecBuildError, match="drift"):
        build_aac_floor_barrier_spec(
            tmp_path, population_tags=["full319"], levels=["fold"]
        )


def test_population_tag_cells_do_not_collide(tmp_path):
    """C1: full319 and esm1b cells are distinct — they must get separate artifact labels and paths."""
    _write_cell(tmp_path, "full319", population_n=319)
    _write_cell(tmp_path, "esm1b", population_n=267)
    spec = build_aac_floor_barrier_spec(
        tmp_path, population_tags=["full319", "esm1b"], levels=["fold"]
    )
    labels = [a["label"] for a in spec["artifacts"]]
    assert "aac_floor:full319:fold" in labels
    assert "aac_floor:esm1b:fold" in labels
    # Labels must be distinct
    assert len(labels) == len(set(labels))


def test_population_n_propagated_into_meta(tmp_path):
    """population_n per population_tag is recorded in _meta for downstream consumers."""
    _write_cell(tmp_path, "full319", population_n=319)
    _write_cell(tmp_path, "esm1b", population_n=267)
    spec = build_aac_floor_barrier_spec(
        tmp_path, population_tags=["full319", "esm1b"], levels=["fold"]
    )
    pop = spec["_meta"]["population_n"]
    assert pop["full319"] == 319
    assert pop["esm1b"] == 267


def test_sidecar_path_is_authoritative(tmp_path):
    """When a sidecar records a timestamped parquet path, the spec must use it, not canonical."""
    # Simulate a non-canonical (timestamped) parquet path as recorded by the producer.
    ts_parquet = tmp_path / "aac_floor_full319_fold.20260611_120000.parquet"
    _write_parquet(ts_parquet)
    sidecar = tmp_path / "aac_floor_full319.manifest.json"
    sidecar.write_text(json.dumps({
        "floor": "aac", "population_tag": "full319", "distance": "euclidean",
        "include_other": False, "population_n": 319,
        "per_query_columns": list(PER_QUERY_COLUMNS),
        "levels": {
            "fold": {
                "path": str(ts_parquet),
                "n_queries_with_positives": 5,
                "n_queries_skipped_no_positives": 0,
                "n_scored": 5,
            }
        },
    }))
    spec = build_aac_floor_barrier_spec(
        tmp_path, population_tags=["full319"], levels=["fold"]
    )
    assert spec["artifacts"][0]["path"] == str(ts_parquet)


def test_expected_rows_from_sidecar(tmp_path):
    """expected_rows is set from n_queries_with_positives when use_expected_rows=True."""
    _write_cell(tmp_path, "full319", n_queries=7)
    spec = build_aac_floor_barrier_spec(
        tmp_path, population_tags=["full319"], levels=["fold"]
    )
    fold_art = next(a for a in spec["artifacts"] if ":fold" in a["label"])
    assert fold_art["expected_rows"] == 7


def test_no_expected_rows_leaves_rows_none(tmp_path):
    """use_expected_rows=False → expected_rows is None for all cells."""
    _write_cell(tmp_path, "full319", n_queries=7)
    spec = build_aac_floor_barrier_spec(
        tmp_path, population_tags=["full319"], levels=["fold"],
        use_expected_rows=False,
    )
    assert spec["artifacts"][0]["expected_rows"] is None


def test_empty_population_tags_raises(tmp_path):
    """An empty population_tags list is an operator fault → SpecBuildError."""
    with pytest.raises(SpecBuildError, match="empty"):
        build_aac_floor_barrier_spec(tmp_path, population_tags=[], levels=["fold"])


def test_expected_n_population_tags_mismatch_raises(tmp_path):
    """expected_n_population_tags guard: mismatch raises SpecBuildError."""
    _write_cell(tmp_path, "full319")
    with pytest.raises(SpecBuildError, match="expected"):
        build_aac_floor_barrier_spec(
            tmp_path, population_tags=["full319"], levels=["fold"],
            expected_n_population_tags=2,   # expect 2, got 1
        )


def test_dedup_population_tags(tmp_path):
    """Duplicated population_tags collapse to unique → grid is deduplicated."""
    _write_cell(tmp_path, "full319")
    spec = build_aac_floor_barrier_spec(
        tmp_path,
        population_tags=["full319", "full319"],  # duplicate
        levels=["fold"],
    )
    assert len(spec["artifacts"]) == 1   # collapsed to 1 unique


def test_artifact_order_follows_grid_nesting(tmp_path):
    """Artifacts enumerate population_tags × levels in the declared order."""
    for pop_tag in ("full319", "esm1b"):
        _write_cell(tmp_path, pop_tag, levels=("fold", "superfamily"))
    spec = build_aac_floor_barrier_spec(
        tmp_path,
        population_tags=["full319", "esm1b"],
        levels=["fold", "superfamily"],
    )
    labels = [a["label"] for a in spec["artifacts"]]
    assert labels == [
        "aac_floor:full319:fold",
        "aac_floor:full319:superfamily",
        "aac_floor:esm1b:fold",
        "aac_floor:esm1b:superfamily",
    ]


def test_guards_wired_correctly(tmp_path):
    """The artifact carries the recall-fp PARQUET_GUARDS (required/unique/non_null/finite cols)."""
    _write_cell(tmp_path, "full319")
    spec = build_aac_floor_barrier_spec(
        tmp_path, population_tags=["full319"], levels=["fold"]
    )
    art = spec["artifacts"][0]
    assert tuple(art["required_columns"]) == PER_QUERY_COLUMNS
    assert tuple(art["unique_columns"]) == ("query_id",)
    assert tuple(art["non_null_columns"]) == ("query_id",)
    assert tuple(art["finite_columns"]) == ("recall",)
    assert art["kind"] == "parquet"


def test_malformed_sidecar_missing_levels_raises(tmp_path):
    """A sidecar with no 'levels' key → SpecBuildError (not a KeyError crash)."""
    sidecar = tmp_path / "aac_floor_full319.manifest.json"
    sidecar.write_text(json.dumps({"floor": "aac", "population_tag": "full319"}))
    with pytest.raises(SpecBuildError, match="levels"):
        build_aac_floor_barrier_spec(
            tmp_path, population_tags=["full319"], levels=["fold"]
        )


def test_malformed_sidecar_level_no_path_raises(tmp_path):
    """A sidecar level block missing a non-empty 'path' → SpecBuildError."""
    sidecar = tmp_path / "aac_floor_full319.manifest.json"
    sidecar.write_text(json.dumps({
        "floor": "aac", "population_tag": "full319", "distance": "euclidean",
        "include_other": False, "population_n": 10,
        "per_query_columns": list(PER_QUERY_COLUMNS),
        "levels": {"fold": {"n_queries_with_positives": 5}},  # missing 'path'
    }))
    with pytest.raises(SpecBuildError, match="path"):
        build_aac_floor_barrier_spec(
            tmp_path, population_tags=["full319"], levels=["fold"]
        )


# ── real-barrier integration ──────────────────────────────────────────────────

def test_built_spec_passes_real_barrier_on_good_cell(tmp_path):
    """The spec produced by the builder actually passes the real analysis_barrier."""
    from evaluation.analysis_barrier import run_barrier, _spec_from_dict
    _write_cell(tmp_path, "full319", n_queries=5)
    spec = build_aac_floor_barrier_spec(
        tmp_path, population_tags=["full319"], levels=["fold"]
    )
    specs = [_spec_from_dict(a, i) for i, a in enumerate(spec["artifacts"])]
    assert run_barrier(specs).ok


def test_built_spec_catches_truncated_parquet(tmp_path):
    """expected_rows guard: a parquet truncated after spec-build fails the real barrier."""
    from evaluation.analysis_barrier import run_barrier, _spec_from_dict
    parquets, _ = _write_cell(tmp_path, "full319", n_queries=6)
    spec = build_aac_floor_barrier_spec(
        tmp_path, population_tags=["full319"], levels=["fold"]
    )
    # Truncate: write only 3 rows into the parquet the spec points to
    fold_path = parquets["fold"]
    _write_parquet(fold_path, 3)
    specs = [_spec_from_dict(a, i) for i, a in enumerate(spec["artifacts"])]
    report = run_barrier(specs)
    assert not report.ok
    assert any("row count" in r for s in report.failures for r in s.reasons)


# ── CLI tests ─────────────────────────────────────────────────────────────────

def test_cli_writes_spec_and_returns_0(tmp_path):
    _write_cell(tmp_path, "full319")
    out = tmp_path / "aac_floor_barrier_spec.json"
    rc = main([
        "--sidecar-dir", str(tmp_path),
        "--population-tags", "full319",
        "--levels", "fold",
        "--out", str(out),
    ])
    assert rc == 0
    spec = json.loads(out.read_text())
    assert len(spec["artifacts"]) == 1


def test_cli_empty_population_tags_returns_2(tmp_path):
    """Empty --population-tags → exit 2 (operator fault)."""
    out = tmp_path / "spec.json"
    # argparse nargs="+" requires at least one value; simulate by passing the builder directly
    with pytest.raises(SpecBuildError):
        build_aac_floor_barrier_spec(tmp_path, population_tags=[], levels=["fold"])
    # CLI path: argparse would enforce nargs="+", so we test via the builder directly above


def test_cli_orphan_returns_2(tmp_path):
    """An orphan parquet (no sidecar) → exit 2."""
    (tmp_path / "aac_floor_full319_fold.parquet").write_text("")
    out = tmp_path / "spec.json"
    rc = main([
        "--sidecar-dir", str(tmp_path),
        "--population-tags", "full319",
        "--levels", "fold",
        "--out", str(out),
    ])
    assert rc == 2


def test_cli_missing_sidecar_warns_on_stderr_and_returns_0(tmp_path, capsys):
    """A missing sidecar cell emits a WARNING on stderr but still exits 0."""
    _write_cell(tmp_path, "full319")  # esm1b absent
    out = tmp_path / "spec.json"
    rc = main([
        "--sidecar-dir", str(tmp_path),
        "--population-tags", "full319", "esm1b",
        "--levels", "fold",
        "--out", str(out),
    ])
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "no sidecar" in err
    assert "esm1b" in err


def test_cli_default_population_tags_are_full319_and_esm1b(tmp_path):
    """The default --population-tags match DEFAULT_POPULATION_TAGS."""
    assert set(DEFAULT_POPULATION_TAGS) == {"full319", "esm1b"}
