"""Tests for shared.atomic_io — atomic-write + completeness-guarded skip (plan v3, B7).

B7 hazard: subcommands guard on ``out_path.exists()`` only, and ``save_h5`` writes
incrementally into ``h5py.File(path, "w")``, so a job killed by OOM/walltime leaves a
valid-looking but truncated file the next run *skips as done*. The fix has two parts:

1. ``atomic_write`` — write to a sibling tmp file, then ``os.replace`` (atomic on the
   same filesystem) so a killed job never leaves a partial *final* file. On
   ``mode="timestamp"`` an existing target is never clobbered: the new content lands at
   ``<stem>.<YYYYMMDD_HHMMSS><ext>`` (the CLAUDE.md / B7 overwrite-safety convention).
2. ``needs_rebuild`` — the skip decision validates *completeness* (reusing the fan-in
   barrier's predicate), not mere existence, so a truncated artifact is rebuilt.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.analysis_barrier import ArtifactSpec
from shared.atomic_io import atomic_write, needs_rebuild


# ── atomic_write ──────────────────────────────────────────────────────────────
def test_atomic_write_creates_file_and_returns_path(tmp_path):
    final = tmp_path / "out.txt"
    landed = atomic_write(final, lambda p: p.write_text("hello"))
    assert landed == final
    assert final.read_text() == "hello"
    # No tmp sibling left behind.
    assert list(tmp_path.glob("*.tmp*")) == []


def test_atomic_write_replace_overwrites_in_place(tmp_path):
    final = tmp_path / "out.txt"
    final.write_text("old")
    landed = atomic_write(final, lambda p: p.write_text("new"), mode="replace")
    assert landed == final
    assert final.read_text() == "new"


def test_default_mode_is_timestamp_safe_never_clobbers(tmp_path):
    # The B7/CLAUDE.md safety default: calling atomic_write the obvious way on an
    # EXISTING file must NOT clobber it in place — the new content lands timestamped.
    final = tmp_path / "out.txt"
    final.write_text("original")
    landed = atomic_write(final, lambda p: p.write_text("new"), timestamp="20260609_154500")
    assert landed == tmp_path / "out.20260609_154500.txt"
    assert final.read_text() == "original"
    assert landed.read_text() == "new"


def test_replace_mode_crash_cleans_tmp_and_preserves(tmp_path):
    final = tmp_path / "out.txt"
    final.write_text("original")

    def bad_writer(p: Path) -> None:
        p.write_text("partial...")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        atomic_write(final, bad_writer, mode="replace")
    assert final.read_text() == "original"
    assert list(tmp_path.glob("*.tmp*")) == []


def test_timestamp_disambiguates_when_sibling_exists(tmp_path):
    final = tmp_path / "dist.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(final)
    sibling = tmp_path / "dist.20260609_154500.parquet"
    pd.DataFrame({"a": [9]}).to_parquet(sibling)  # same-second rerun already landed
    landed = atomic_write(
        final,
        lambda p: pd.DataFrame({"a": [1, 2]}).to_parquet(p),
        mode="timestamp",
        timestamp="20260609_154500",
    )
    # Must not clobber the existing timestamped sibling.
    assert landed != sibling
    assert landed.exists()
    assert pd.read_parquet(sibling)["a"].tolist() == [9]


def test_unknown_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        atomic_write(tmp_path / "x.txt", lambda p: p.write_text("y"), mode="bogus")


def test_atomic_write_no_partial_and_preserves_existing_on_crash(tmp_path):
    final = tmp_path / "out.txt"
    final.write_text("original")

    def bad_writer(p: Path) -> None:
        p.write_text("partial...")
        raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError, match="killed mid-write"):
        atomic_write(final, bad_writer)

    # Final untouched (we wrote to tmp, never to final), tmp cleaned up.
    assert final.read_text() == "original"
    assert list(tmp_path.glob("*.tmp*")) == []


def test_atomic_write_timestamp_mode_never_clobbers(tmp_path):
    final = tmp_path / "dist.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(final)
    original_bytes = final.read_bytes()

    landed = atomic_write(
        final,
        lambda p: pd.DataFrame({"a": [1, 2, 3]}).to_parquet(p),
        mode="timestamp",
        timestamp="20260609_154500",
    )
    # New content lands at the timestamped sibling; original is untouched.
    assert landed == tmp_path / "dist.20260609_154500.parquet"
    assert landed.exists()
    assert final.read_bytes() == original_bytes


def test_atomic_write_timestamp_mode_new_file_lands_at_final(tmp_path):
    final = tmp_path / "dist.parquet"
    landed = atomic_write(
        final,
        lambda p: pd.DataFrame({"a": [1]}).to_parquet(p),
        mode="timestamp",
        timestamp="20260609_154500",
    )
    assert landed == final
    assert final.exists()


# ── needs_rebuild (completeness-guarded skip; reuses the barrier predicate) ────
def test_needs_rebuild_true_when_missing(tmp_path):
    spec = ArtifactSpec(label="x", path=tmp_path / "absent.parquet", expected_rows=10)
    assert needs_rebuild(spec) is True


def test_needs_rebuild_false_when_complete(tmp_path):
    p = tmp_path / "x.parquet"
    pd.DataFrame(
        {"a": ["P0", "P1"], "b": ["P1", "P2"], "embedding_dist": [0.1, 0.2]}
    ).to_parquet(p)
    spec = ArtifactSpec(
        label="x", path=p, expected_rows=2, required_columns=("a", "b", "embedding_dist")
    )
    assert needs_rebuild(spec) is False


def test_needs_rebuild_true_when_truncated(tmp_path):
    # Exists, but only 7 of an expected 10 rows — the partial-write case.
    p = tmp_path / "x.parquet"
    pd.DataFrame({"a": list(range(7))}).to_parquet(p)
    spec = ArtifactSpec(label="x", path=p, expected_rows=10)
    assert needs_rebuild(spec) is True


def test_needs_rebuild_true_for_zero_norm_h5(tmp_path):
    # Proves "completeness, not existence" for the embedding path B7 cares about:
    # a present H5 with a dead (zero-norm) vector must trigger a rebuild.
    import h5py

    p = tmp_path / "emb.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("P0", data=np.ones(8, dtype=np.float32))
        f.create_dataset("P1", data=np.zeros(8, dtype=np.float32))  # dead
    spec = ArtifactSpec(label="emb", path=p, expected_rows=2, require_positive_norm=True)
    assert needs_rebuild(spec) is True
