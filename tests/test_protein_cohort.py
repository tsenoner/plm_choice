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

from shared.protein_cohort import (
    exclusion_summary,
    load_excluded_proteins,
    restrict_to_cohort,
)


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
    assert restrict_to_cohort(keys, frozenset({"P2"})) == {"P1", "P3"}


def test_restrict_is_a_noop_for_an_empty_exclusion():
    keys = {"P1", "P2"}
    assert restrict_to_cohort(keys, frozenset()) == keys


def test_restrict_ignores_excluded_ids_absent_from_this_arm():
    """An arm may already lack an excluded id; that must not be an error."""
    assert restrict_to_cohort({"P1"}, frozenset({"P2", "P3"})) == {"P1"}


def test_exclusion_summary_reports_what_was_removed():
    summary = exclusion_summary({"P1", "P2", "P3"}, frozenset({"P2", "P9"}))
    assert summary.kept == 2
    assert summary.removed == 1
    # P9 is not in this arm at all -- reported separately, never counted as removed
    assert summary.not_present == 1


def test_freeze_may_carry_provenance_without_breaking_the_loader(tmp_path):
    path = _freeze(
        tmp_path, ["P1"], cohort="sprot_pre2024", reason="missing from >=1 arm"
    )
    assert load_excluded_proteins(path) == frozenset({"P1"})
