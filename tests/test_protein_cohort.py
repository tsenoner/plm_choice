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
the exclusion is ~38x smaller (14,010 vs 526,871 ids, against the <=2000 cohort of
540,881; 15,367 was the pre-cut figure against the 542,238 deposit).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.protein_cohort import (
    cohort_size,
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


def _cohort2k_freeze() -> dict:
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "freeze"
            / "embedding_key_coverage_cohort2k.json"
        ).read_text()
    )


def test_the_committed_freeze_states_the_exclusion_size_it_implies():
    """Pure arithmetic on the freeze -- no ids materialised.

    540,881 - 526,871 = the 14,010 proteins clean/esm1b lack to ESM-1b's 1022-token cap.
    """
    blob = _cohort2k_freeze()
    assert blob["universe"] - blob["intersection_all"] == 14010


def test_derivation_reproduces_the_committed_coverage_freeze_structure():
    """Anti-tautology: derive_exclusion must agree with the independently-measured freeze.

    Scaled 1000x down. The value here is the pattern-bitmask -> membership -> union /
    intersection round trip, which is scale-free; materialising all 540,881 ids cost
    ~1 GB and ~3 s for an identity that holds at any size.
    """
    blob = _cohort2k_freeze()
    models = list(blob["models"])
    scale = 1000

    keysets: dict[str, set] = {m: set() for m in models}
    expected: dict[str, int] = {m: 0 for m in models}
    pid = 0
    for mask_str, n in blob["patterns"].items():
        mask = int(mask_str)
        members = [m for i, m in enumerate(models) if mask & (1 << i)]
        for _ in range(max(1, n // scale)):
            for m in members:
                keysets[m].add(f"P{pid}")
                expected[m] += 1
            pid += 1

    manifest = derive_exclusion(keysets)
    assert manifest["counts"] == expected
    assert manifest["universe"] == pid
    # the deficient arms must be exactly the ones the freeze reports short
    short_here = {m for m, n in manifest["counts"].items() if n < manifest["universe"]}
    short_there = {m for m, n in blob["counts"].items() if n < blob["universe"]}
    assert short_here == short_there == {"clean", "esm1b"}
    assert manifest["n_excluded"] == manifest["universe"] - manifest["intersection_all"]


def test_a_hand_edited_freeze_fails_on_LOAD_not_only_on_write(tmp_path):
    """The guard has to sit where drift can actually happen.

    At write time the manifest was just derived in-process and cannot have been
    edited; on disk between runs it can. Verifying only on write puts the check in
    the one place it cannot help.
    """
    manifest = derive_exclusion({"a": {"P1", "P2", "P3"}, "b": {"P1"}})
    path = write_exclusion_freeze(manifest, tmp_path / "ex.json")
    blob = json.loads(path.read_text())
    blob["excluded_ids"] = ["P2"]  # dropped P3 by hand; hash and counts now stale
    path.write_text(json.dumps(blob))
    load_excluded_proteins.cache_clear()
    with pytest.raises(ValueError):
        load_excluded_proteins(path)


def test_derive_accepts_frozensets():
    """set().union() takes a frozenset; the unbound set.intersection() does not."""
    manifest = derive_exclusion({"a": frozenset({"P1", "P2"}), "b": frozenset({"P1"})})
    assert manifest["excluded_ids"] == ["P2"]


def test_default_freeze_path_does_not_assume_a_source_checkout(tmp_path, monkeypatch):
    """The wheel ships src/* but NOT freeze/, so parents[2] resolves outside the tree."""
    from shared import protein_cohort

    # No pyproject.toml above a site-packages-like location -> cwd-relative fallback,
    # never a path built by counting parents off __file__.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(protein_cohort, "__file__", str(tmp_path / "sp" / "shared" / "protein_cohort.py"))
    assert protein_cohort._default_freeze_path() == Path("freeze") / protein_cohort.FREEZE_NAME


def _committed_exclusion() -> dict:
    return json.loads(
        (
            Path(__file__).resolve().parents[1] / "freeze" / "embedding_excluded_proteins.json"
        ).read_text()
    )


def test_the_committed_exclusion_freeze_is_the_one_the_paper_describes():
    """Pin the artifact, not just the arithmetic that produced it.

    ``test_the_committed_freeze_states_the_exclusion_size_it_implies`` above checks a
    subtraction inside the *coverage* freeze. It would still pass if the exclusion
    freeze were absent, truncated, or derived from a different directory -- which is
    the whole failure mode, because an absent freeze is a deliberate silent no-op
    (``load_excluded_proteins`` returns an empty set). So pin the four numbers that
    identify this specific cohort, and the hash that identifies its exact membership.

    Measured 2026-09-15 on the 15 physical ``.h5`` files in
    ``embeddings_cohort2k`` -- the first measurement of that directory, as opposed to
    the projection ``freeze/embedding_key_coverage_cohort2k.json`` recorded ("these
    counts are what that job will produce").
    """
    blob = _committed_exclusion()
    assert blob["universe"] == 540881
    assert blob["intersection_all"] == 526871
    assert blob["n_excluded"] == 14010
    assert (
        blob["content_sha256"]
        == "b9fba30708b2074129ff4f2c1b08e4d9f65c07de0ac3745e6e6351cbfbecf10e"
    )
    verify_exclusion(blob)  # re-derives hash and counts from excluded_ids alone


def test_the_exclusion_and_coverage_freezes_describe_the_same_cohort():
    """Two files, two derivations, one cohort -- so they must not drift apart.

    The coverage freeze counts keys per arm; the exclusion freeze materialises ids.
    Committing them separately means a re-measurement of one can silently outrun the
    other, and the pair is what Methods quotes.
    """
    excl, cov = _committed_exclusion(), _cohort2k_freeze()
    assert excl["universe"] == cov["universe"]
    assert excl["intersection_all"] == cov["intersection_all"]
    assert excl["counts"] == cov["counts"]
    assert sorted(excl["arms"]) == sorted(cov["models"])
    short = {a for a, n in excl["counts"].items() if n < excl["universe"]}
    assert short == {"clean", "esm1b"}


def test_the_excluded_proteins_are_exactly_the_ones_longer_than_esm1bs_context():
    """The claim M-12 puts in Methods, checked against sequence lengths.

    Methods must say "the evaluation cohort is Swiss-Prot proteins of <=1022 residues,
    set by the shortest context window among the compared models" -- NOT "we excluded
    14,010 proteins for cross-arm comparability". The first is only honest if the
    excluded set really is a pure length cut, so derive it a second way, from
    ``sprot.fasta.fai`` alone, and require the two to be identical.

    Verified 2026-09-15: symmetric difference 0. Skipped where the .fai is absent --
    it is a data file, not a repo artifact.
    """
    fai = Path(__file__).resolve().parents[1] / "data" / "raw" / "sprot_2024" / "sprot.fasta.fai"
    if not fai.exists():
        pytest.skip(f"{fai} not present (data files are not part of the checkout)")

    lengths = {}
    for line in fai.read_text().splitlines():
        name, length, *_ = line.split("\t")
        lengths[name] = int(length)

    by_length = {pid for pid, n in lengths.items() if 1022 < n <= 2000}
    assert by_length == set(_committed_exclusion()["excluded_ids"])
    # and the cohort the freeze leaves behind is the <=1022 set, stated the same way
    assert sum(n <= 1022 for n in lengths.values()) == 526871


def test_an_arm_that_was_never_cut_to_the_cohort_is_announced_not_silently_accepted(capsys):
    """The other half of the guarantee: subtracting the exclusion is not sufficient.

    A full 542,238-key deposit arm filtered by this freeze lands at 528,228, not
    526,871 -- the freeze names the 14,010 proteins of 1023-2000 aa, while the 1,357
    above 2000 were removed by the cohort *cut*, physically, and are not in the id
    list. So the arm comes out "filtered" and is still on its own test set. Measured
    on the real freeze: 540,881 + 1,357 = 542,238 in, 528,228 out.

    Warns rather than raises, because a deliberately-scoped subset run is a legitimate
    caller that also lands off the number.
    """
    expected = cohort_size()
    assert expected == 526871

    cohort_size(warn_if_not=528228, label="prostt5")
    out = capsys.readouterr().out
    assert "cohort WARNING [prostt5]" in out
    assert "528,228" in out and "526,871" in out and "+1,357" in out
    assert "never cut to the cohort" in out

    cohort_size(warn_if_not=526871, label="esm2_650m")
    assert "WARNING" not in capsys.readouterr().out
