"""The common protein cohort: one evaluation set shared by every embedding arm.

The 15 arms do not cover the same proteins (see
``freeze/embedding_key_coverage.json``), and because
``src/shared/datasets.py`` drops a pair when *either* protein is missing from
that arm's HDF5, each arm silently gets a different -- and differently sized --
test set. A cross-pLM ranking built that way is not a ranking.

Restricting every arm to the intersection fixes it. This is done as a
**load-time filter over a committed id list**, never by deleting datasets from
the HDF5 files: those files are the md5-verified Zenodo deposit, deletion is
irreversible, and an arm's file would then no longer match its published
checksum.

The freeze stores the **excluded** ids rather than the included ones because
the exclusion is ~34x smaller (15,367 vs 526,871 ids).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.protein_cohort import (
    derive_exclusion,
    load_excluded_proteins,
    restrict_to_cohort,
    verify_exclusion,
    write_exclusion_freeze,
)


def _kept(keys, excluded):
    return restrict_to_cohort(keys, excluded)[0]


def _freeze(tmp_path, ids, **extra):
    p = tmp_path / "excluded.json"
    p.write_text(json.dumps({"excluded_ids": list(ids), **extra}))
    return p


def test_loads_the_excluded_ids(tmp_path):
    path = _freeze(tmp_path, ["P1", "P2"])
    assert load_excluded_proteins(path) == frozenset({"P1", "P2"})


def test_missing_freeze_excludes_nothing(tmp_path):
    """Absent freeze must be a no-op, so behaviour is unchanged until we opt in."""
    assert load_excluded_proteins(tmp_path / "does_not_exist.json") == frozenset()


def test_restrict_removes_excluded_keys(tmp_path):
    keys = {"P1", "P2", "P3"}
    assert _kept(keys, frozenset({"P2"})) == {"P1", "P3"}


def test_restrict_is_a_noop_for_an_empty_exclusion():
    keys = {"P1", "P2"}
    assert _kept(keys, frozenset()) == keys


def test_restrict_ignores_excluded_ids_absent_from_this_arm():
    """An arm may already lack an excluded id; that must not be an error."""
    assert _kept({"P1"}, frozenset({"P2", "P3"})) == {"P1"}


def test_exclusion_summary_reports_what_was_removed():
    _, summary = restrict_to_cohort({"P1", "P2", "P3"}, frozenset({"P2", "P9"}))
    assert summary.kept == 2
    assert summary.removed == 1
    # P9 is not in this arm at all -- reported separately, never counted as removed
    assert summary.not_present == 1


def test_freeze_may_carry_provenance_without_breaking_the_loader(tmp_path):
    path = _freeze(
        tmp_path, ["P1"], cohort="sprot_pre2024", reason="missing from >=1 arm"
    )
    assert load_excluded_proteins(path) == frozenset({"P1"})


# --------------------------------------------------------------------------- #
# Derivation + freeze writing
#
# Before this existed, DEFAULT_EXCLUSION_FREEZE named a file nothing produced, so
# the filter above was a permanent no-op. These pin the half that makes it real.
# --------------------------------------------------------------------------- #


def test_excluded_is_union_minus_intersection():
    """The cohort is what EVERY arm has; excluded is everything at least one lacks."""
    manifest = derive_exclusion(
        {"a": {"P1", "P2", "P3"}, "b": {"P1", "P2"}, "c": {"P1", "P2", "P4"}}
    )
    assert manifest["universe"] == 4          # P1..P4
    assert manifest["intersection_all"] == 2  # P1, P2
    assert manifest["excluded_ids"] == ["P3", "P4"]
    assert manifest["n_excluded"] == 2


def test_arms_that_all_agree_exclude_nothing():
    manifest = derive_exclusion({"a": {"P1"}, "b": {"P1"}})
    assert manifest["excluded_ids"] == []
    assert manifest["universe"] == manifest["intersection_all"] == 1


def test_derive_rejects_an_empty_arm_set():
    with pytest.raises(ValueError, match="nothing to derive"):
        derive_exclusion({})


def test_write_then_load_round_trips(tmp_path):
    manifest = derive_exclusion({"a": {"P1", "P2"}, "b": {"P1"}})
    out = write_exclusion_freeze(manifest, tmp_path / "excluded.json")
    assert load_excluded_proteins(out) == frozenset({"P2"})


def test_write_refuses_to_clobber_without_intent(tmp_path):
    """A stale freeze must never be silently left behind a 'regenerated' one."""
    manifest = derive_exclusion({"a": {"P1", "P2"}, "b": {"P1"}})
    path = tmp_path / "excluded.json"
    write_exclusion_freeze(manifest, path)
    with pytest.raises(FileExistsError, match="--overwrite"):
        write_exclusion_freeze(manifest, path)
    assert write_exclusion_freeze(manifest, path, overwrite=True) == path


def test_verify_catches_a_hand_edited_id_list():
    """Editing excluded_ids without re-deriving must fail, not filter a different cohort."""
    manifest = derive_exclusion({"a": {"P1", "P2", "P3"}, "b": {"P1"}})
    manifest["excluded_ids"] = ["P2"]  # dropped P3 by hand; hash/counts now stale
    with pytest.raises(ValueError, match="n_excluded"):
        verify_exclusion(manifest)


def test_verify_catches_content_drift_when_counts_still_line_up():
    manifest = derive_exclusion({"a": {"P1", "P2", "P3"}, "b": {"P1"}})
    manifest["excluded_ids"] = ["P2", "P9"]  # same length, different set
    with pytest.raises(ValueError, match="content drift"):
        verify_exclusion(manifest)


def test_verify_rejects_duplicate_ids():
    manifest = derive_exclusion({"a": {"P1", "P2"}, "b": {"P1"}})
    manifest["excluded_ids"] = ["P2", "P2"]
    with pytest.raises(ValueError, match="duplicate"):
        verify_exclusion(manifest)


def test_derivation_matches_the_committed_coverage_freeze():
    """Anti-tautology: the arm counts must reproduce the independently-measured freeze.

    freeze/embedding_key_coverage_cohort2k.json was measured on the cluster from the
    real .h5 files. Deriving from key sets reconstructed out of its own pattern
    bitmasks must land on the same universe/intersection, or the two committed
    artifacts disagree about what the cohort is.
    """
    blob = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "freeze"
            / "embedding_key_coverage_cohort2k.json"
        ).read_text()
    )
    models = list(blob["models"])
    # Rebuild synthetic key sets with the right membership structure from the patterns.
    keysets = {m: set() for m in models}
    pid = 0
    for mask_str, n in blob["patterns"].items():
        mask = int(mask_str)
        members = [m for i, m in enumerate(models) if mask & (1 << i)]
        for _ in range(n):
            for m in members:
                keysets[m].add(f"P{pid}")
            pid += 1

    manifest = derive_exclusion(keysets)
    assert manifest["counts"] == blob["counts"]
    assert manifest["universe"] == blob["universe"]
    assert manifest["intersection_all"] == blob["intersection_all"]
    # 540,881 - 526,871 = the 14,010 proteins clean/esm1b lack (ESM-1b's 1022 cap)
    assert manifest["n_excluded"] == blob["universe"] - blob["intersection_all"] == 14010
