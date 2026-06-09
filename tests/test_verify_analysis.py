"""Tests for evaluation.verify_analysis — the freeze-integrity gate (plan v3, Phase 0 item 2).

verify_analysis is the hard `afterok` precondition before any figure consumes an artifact
(Gate C). Unlike the presence-only upstream `verify_manifest`, it asserts *integrity*:

* the canonical FASTA on disk still hashes to the frozen ``canonical_content_sha256`` (the
  sequence set has not drifted out from under the analyses);
* the NEW-3 esm1b paired-stats policy is *locked* (non-null) before any paired stat runs;
* an analysis input's population matches the frozen id set (via ``assert_population``), with
  esm1b allowed to be its capped 267-subset.

It reuses the freeze manifest (``canonical_set.py``) and ``population.assert_population`` so
there is one definition of "the canonical set", not a drift-prone second copy.
"""
from __future__ import annotations

import json

import pytest

from evaluation.canonical_set import freeze_canonical_set, write_freeze
from evaluation.verify_analysis import (
    VerifyReport,
    load_manifest,
    main,
    verify_analysis,
    verify_fasta_unchanged,
    verify_policy_locked,
    verify_population,
)


# ── fixtures ────────────────────────────────────────────────────────────────────
def _fasta(tmp_path, name="c.fasta", body=">A\nMK\n>B\nGG\n>C\nWW\n"):
    p = tmp_path / name
    p.write_text(body)
    return p


@pytest.fixture
def frozen(tmp_path):
    """A written freeze (manifest + parquet) over a 3-protein set, esm1b capped to {A,B}."""
    fa = _fasta(tmp_path)
    m = freeze_canonical_set(
        fa, set_name="t", esm1b_keys=["A", "B"], cap_aa=1,
        esm1b_paired_policy="footnote_esm1b_out",
    )
    paths = write_freeze(m, tmp_path / "freeze", set_name="t")
    return {"fasta": fa, "manifest_path": paths["manifest"], "manifest": m}


# ── load_manifest ────────────────────────────────────────────────────────────────
def test_load_manifest_reads_valid(frozen):
    m = load_manifest(frozen["manifest_path"])
    assert m["canonical_content_sha256"] == frozen["manifest"]["canonical_content_sha256"]


def test_load_manifest_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.json")


def test_load_manifest_malformed_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError):
        load_manifest(bad)


def test_load_manifest_missing_required_key_raises(tmp_path):
    bad = tmp_path / "incomplete.json"
    bad.write_text(json.dumps({"set_name": "x"}))
    with pytest.raises(ValueError, match="missing"):
        load_manifest(bad)


def _valid_manifest_dict(**overrides):
    m = {
        "schema_version": 1,
        "canonical_content_sha256": "deadbeef",
        "ids": ["A", "B", "C"],
        "n_proteins": 3,
        "n_pairs": 3,
        "esm1b": None,
    }
    m.update(overrides)
    return m


def test_load_manifest_rejects_wrong_typed_esm1b(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps(_valid_manifest_dict(esm1b="oops")))
    with pytest.raises(ValueError, match="esm1b"):
        load_manifest(bad)


def test_load_manifest_rejects_ids_count_mismatch(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps(_valid_manifest_dict(n_proteins=5)))
    with pytest.raises(ValueError, match="inconsistent"):
        load_manifest(bad)


def test_load_manifest_rejects_bad_n_pairs(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps(_valid_manifest_dict(n_pairs=99)))
    with pytest.raises(ValueError, match="n_pairs"):
        load_manifest(bad)


# ── verify_fasta_unchanged ───────────────────────────────────────────────────────
def test_fasta_unchanged_ok(frozen):
    assert verify_fasta_unchanged(frozen["manifest"], frozen["fasta"]) == []


def test_fasta_unchanged_detects_residue_change(frozen, tmp_path):
    drifted = _fasta(tmp_path, name="drift.fasta", body=">A\nMK\n>B\nGG\n>C\nWY\n")
    reasons = verify_fasta_unchanged(frozen["manifest"], drifted)
    assert reasons and any("sha256" in r or "content" in r for r in reasons)


def test_fasta_unchanged_invariant_to_reformatting(frozen, tmp_path):
    # same set, re-wrapped + reordered + lowercased -> still matches the frozen content hash
    ref = _fasta(tmp_path, name="ref.fasta", body=">C\nww\n>B desc\nGG\n>A\nmk\n")
    assert verify_fasta_unchanged(frozen["manifest"], ref) == []


# ── verify_policy_locked ─────────────────────────────────────────────────────────
def test_policy_locked_ok(frozen):
    assert verify_policy_locked(frozen["manifest"]) == []


def test_policy_locked_flags_null_policy(tmp_path):
    fa = _fasta(tmp_path)
    m = freeze_canonical_set(fa, set_name="t", esm1b_keys=["A", "B"], cap_aa=1)
    reasons = verify_policy_locked(m)
    assert reasons and "esm1b_paired_policy" in reasons[0]


def test_policy_locked_noop_without_esm1b(tmp_path):
    fa = _fasta(tmp_path)
    m = freeze_canonical_set(fa, set_name="t")
    assert verify_policy_locked(m) == []


def test_policy_locked_flags_empty_esm1b_block():
    # Empty {} block must NOT silently pass via a falsy-dict short-circuit.
    reasons = verify_policy_locked({"esm1b": {}})
    assert reasons and "esm1b_paired_policy" in reasons[0]


# ── verify_population ────────────────────────────────────────────────────────────
def test_population_ok_full_set(frozen):
    assert verify_population(frozen["manifest"], ["A", "B", "C"], name="prott5") == []


def test_population_flags_foreign_id(frozen):
    reasons = verify_population(frozen["manifest"], ["A", "B", "C", "ZZ"], name="prott5")
    assert reasons


def test_population_flags_missing_when_not_capped(frozen):
    reasons = verify_population(frozen["manifest"], ["A", "B"], name="prott5")
    assert reasons


def test_population_allows_capped_subset(frozen):
    # esm1b carries only {A, B}; with capped=True that is allowed.
    assert verify_population(frozen["manifest"], ["A", "B"], name="esm1b", capped=True) == []


# ── verify_analysis (end to end) ─────────────────────────────────────────────────
def test_verify_analysis_ok(frozen):
    report = verify_analysis(frozen["manifest_path"], fasta_path=frozen["fasta"])
    assert report.ok
    assert report.failures == ()


def test_verify_analysis_detects_drift(frozen, tmp_path):
    drifted = _fasta(tmp_path, name="drift.fasta", body=">A\nMK\n>B\nGG\n>C\nWY\n")
    report = verify_analysis(frozen["manifest_path"], fasta_path=drifted)
    assert not report.ok
    assert report.failures


def test_verify_analysis_population_inputs_all_ok(frozen):
    report = verify_analysis(
        frozen["manifest_path"],
        fasta_path=frozen["fasta"],
        population_inputs={"prott5": ["A", "B", "C"], "esm1b": ["A", "B"]},
    )
    assert report.ok  # esm1b auto-capped to its 2-of-3 subset


def test_verify_analysis_population_inputs_flags_drifted_cohort(frozen):
    report = verify_analysis(
        frozen["manifest_path"],
        fasta_path=frozen["fasta"],
        population_inputs={"prott5": ["A", "B"]},  # missing C, not capped -> fail
    )
    assert not report.ok
    assert any(c.name == "population:prott5" for c in report.failures)


def test_verify_report_empty_is_not_ok():
    # A gate that ran zero checks must not report green.
    assert VerifyReport(checks=()).ok is False


# ── CLI ──────────────────────────────────────────────────────────────────────────
def test_main_exit_zero_on_match(frozen):
    rc = main(["--manifest", str(frozen["manifest_path"]), "--fasta", str(frozen["fasta"])])
    assert rc == 0


def test_main_exit_one_on_drift(frozen, tmp_path, capsys):
    drifted = _fasta(tmp_path, name="drift.fasta", body=">A\nMK\n>B\nGG\n>C\nWY\n")
    rc = main(["--manifest", str(frozen["manifest_path"]), "--fasta", str(drifted)])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().err


def test_main_exit_two_on_missing_manifest(tmp_path, capsys):
    rc = main(["--manifest", str(tmp_path / "nope.json"), "--fasta", str(_fasta(tmp_path))])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err
