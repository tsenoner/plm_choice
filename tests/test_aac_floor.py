"""Tests for evaluation.aac_floor — Unit 1: AAC-vector bridge.

Spec: docs/superpowers/specs/2026-06-11-aac-floor-design.md §3 Unit 1
Fan-fixes applied: I2 (parse_fasta returns list of tuples, not dict).

Real signatures confirmed from source:
  - data_preparation.aac.extract_aac(fasta_dict, normalize=True, include_other=True,
      reduce=True) -> {pid: ndarray}
    include_other=True  → (21,) float32
    include_other=False → (20,) float32
  - evaluation.canonical_set.parse_fasta(path) -> list[tuple[str, str]]
    (NOT a dict; bridge must do dict(parse_fasta(path)))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_fasta(tmp_path: Path, records: list[tuple[str, str]]) -> Path:
    """Write a minimal FASTA file and return its path."""
    fasta = tmp_path / "test.fasta"
    lines = []
    for pid, seq in records:
        lines.append(f">{pid}")
        lines.append(seq)
    fasta.write_text("\n".join(lines) + "\n")
    return fasta


# Canonical 20-d AA order: ACDEFGHIKLMNPQRSTVWY
_STD_AA = "ACDEFGHIKLMNPQRSTVWY"
_A_IDX = _STD_AA.index("A")   # 0
_C_IDX = _STD_AA.index("C")   # 1


# ---------------------------------------------------------------------------
# 1. Known-vector correctness — 20-d (include_other=False)
# ---------------------------------------------------------------------------

class TestKnownVectors20d:
    def test_all_alanine(self, tmp_path):
        """'AAAA' → A=1.0, all others 0."""
        fasta = _write_fasta(tmp_path, [("p1", "AAAA")])
        from evaluation.aac_floor import build_aac_embeddings

        result = build_aac_embeddings(fasta, expected_ids=["p1"])
        vec = result["p1"]
        assert vec.shape == (20,), f"expected (20,), got {vec.shape}"
        assert vec.dtype == np.float32
        assert vec[_A_IDX] == pytest.approx(1.0, abs=1e-6)
        assert vec[_C_IDX] == pytest.approx(0.0, abs=1e-6)

    def test_half_half(self, tmp_path):
        """'AACC' → A=0.5, C=0.5, all others 0."""
        fasta = _write_fasta(tmp_path, [("p1", "AACC")])
        from evaluation.aac_floor import build_aac_embeddings

        result = build_aac_embeddings(fasta, expected_ids=["p1"])
        vec = result["p1"]
        assert vec.shape == (20,)
        assert vec[_A_IDX] == pytest.approx(0.5, abs=1e-6)
        assert vec[_C_IDX] == pytest.approx(0.5, abs=1e-6)
        # all others are zero
        mask = np.ones(20, dtype=bool)
        mask[_A_IDX] = False
        mask[_C_IDX] = False
        assert np.all(vec[mask] == 0.0)

    def test_lowercase_sequence(self, tmp_path):
        """Lowercase sequence is treated case-insensitively (seq uppercased in featurizer)."""
        fasta = _write_fasta(tmp_path, [("p1", "aacc")])
        from evaluation.aac_floor import build_aac_embeddings

        result = build_aac_embeddings(fasta, expected_ids=["p1"])
        vec = result["p1"]
        assert vec[_A_IDX] == pytest.approx(0.5, abs=1e-6)
        assert vec[_C_IDX] == pytest.approx(0.5, abs=1e-6)

    def test_multiple_proteins(self, tmp_path):
        """Multiple proteins returned as separate keys."""
        fasta = _write_fasta(tmp_path, [("p1", "AAAA"), ("p2", "CCCC")])
        from evaluation.aac_floor import build_aac_embeddings

        result = build_aac_embeddings(fasta, expected_ids=["p1", "p2"])
        assert set(result.keys()) == {"p1", "p2"}
        assert result["p1"][_A_IDX] == pytest.approx(1.0, abs=1e-6)
        assert result["p2"][_C_IDX] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. include_other toggle: 20-d vs 21-d
# ---------------------------------------------------------------------------

class TestIncludeOtherToggle:
    def test_default_is_20d(self, tmp_path):
        """Default include_other=False → (20,) vector."""
        fasta = _write_fasta(tmp_path, [("p1", "AAAA")])
        from evaluation.aac_floor import build_aac_embeddings

        result = build_aac_embeddings(fasta, expected_ids=["p1"])
        assert result["p1"].shape == (20,)

    def test_include_other_true_gives_21d(self, tmp_path):
        """include_other=True → (21,) vector (21st bucket = non-standard AAs)."""
        fasta = _write_fasta(tmp_path, [("p1", "AAAA")])
        from evaluation.aac_floor import build_aac_embeddings

        result = build_aac_embeddings(fasta, expected_ids=["p1"], include_other=True)
        assert result["p1"].shape == (21,)

    def test_nonstandard_aa_goes_to_other_bucket(self, tmp_path):
        """'AAAB' with include_other=True → A=0.75, other=0.25; standard sum <=1 without."""
        # B is non-standard
        fasta = _write_fasta(tmp_path, [("p1", "AAAB")])
        from evaluation.aac_floor import build_aac_embeddings

        vec_20 = build_aac_embeddings(fasta, expected_ids=["p1"])["p1"]
        vec_21 = build_aac_embeddings(
            fasta, expected_ids=["p1"], include_other=True
        )["p1"]

        # 20-d: B dropped, 3 standard AAs → sum < 1
        assert vec_20.sum() == pytest.approx(0.75, abs=1e-6)
        # 21-d: B in bucket 20 → sum == 1
        assert vec_21.sum() == pytest.approx(1.0, abs=1e-6)
        assert vec_21[20] == pytest.approx(0.25, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Subset behaviour
# ---------------------------------------------------------------------------

class TestSubset:
    def test_extra_ids_in_fasta_are_dropped(self, tmp_path):
        """A FASTA with extra proteins beyond expected_ids → only expected_ids returned."""
        fasta = _write_fasta(
            tmp_path,
            [("p1", "AAAA"), ("p2", "CCCC"), ("p3", "DDDD")],
        )
        from evaluation.aac_floor import build_aac_embeddings

        result = build_aac_embeddings(fasta, expected_ids=["p1", "p2"])
        assert set(result.keys()) == {"p1", "p2"}
        assert "p3" not in result

    def test_expected_id_absent_from_fasta_raises(self, tmp_path):
        """A frozen id missing from the FASTA → ValueError (data fault)."""
        fasta = _write_fasta(tmp_path, [("p1", "AAAA")])
        from evaluation.aac_floor import build_aac_embeddings

        with pytest.raises(ValueError, match="p_missing"):
            build_aac_embeddings(fasta, expected_ids=["p1", "p_missing"])

    def test_empty_after_subset_raises(self, tmp_path):
        """FASTA has records but none in expected_ids → ValueError."""
        fasta = _write_fasta(tmp_path, [("p1", "AAAA")])
        from evaluation.aac_floor import build_aac_embeddings

        with pytest.raises(ValueError):
            build_aac_embeddings(fasta, expected_ids=["p_other"])

    def test_exact_subset_returns_correct_size(self, tmp_path):
        """Subsetting a 5-protein FASTA to 3 expected_ids returns exactly 3."""
        records = [(f"p{i}", "ACDE" * (i + 1)) for i in range(5)]
        fasta = _write_fasta(tmp_path, records)
        from evaluation.aac_floor import build_aac_embeddings

        result = build_aac_embeddings(fasta, expected_ids=["p0", "p2", "p4"])
        assert len(result) == 3


# ---------------------------------------------------------------------------
# 4. Sums-to-≤1 invariant (normalised frequencies)
# ---------------------------------------------------------------------------

class TestSumsInvariant:
    def test_all_standard_sums_to_one(self, tmp_path):
        """All-standard-AA sequence → sum == 1.0 (normalized)."""
        fasta = _write_fasta(tmp_path, [("p1", "ACDEFGHIKLMNPQRSTVWY")])
        from evaluation.aac_floor import build_aac_embeddings

        vec = build_aac_embeddings(fasta, expected_ids=["p1"])["p1"]
        assert float(vec.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_mixed_nonstandard_sums_leq_one(self, tmp_path):
        """Sequence with non-standard AAs (include_other=False) → sum ≤ 1."""
        fasta = _write_fasta(tmp_path, [("p1", "ACBXZ")])  # B, X, Z non-standard
        from evaluation.aac_floor import build_aac_embeddings

        vec = build_aac_embeddings(fasta, expected_ids=["p1"])["p1"]
        assert float(vec.sum()) <= 1.0 + 1e-6

    def test_include_other_true_sums_to_one(self, tmp_path):
        """With include_other=True, any protein should sum to exactly 1.0."""
        fasta = _write_fasta(tmp_path, [("p1", "ACBXZ")])
        from evaluation.aac_floor import build_aac_embeddings

        vec = build_aac_embeddings(
            fasta, expected_ids=["p1"], include_other=True
        )["p1"]
        assert float(vec.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_nonnegative(self, tmp_path):
        """All frequency values must be non-negative."""
        fasta = _write_fasta(tmp_path, [("p1", "ACDE")])
        from evaluation.aac_floor import build_aac_embeddings

        vec = build_aac_embeddings(fasta, expected_ids=["p1"])["p1"]
        assert np.all(vec >= 0.0)

    def test_dtype_is_float32(self, tmp_path):
        """Returned vectors must be float32 (as recalled by recall_at_first_fp)."""
        fasta = _write_fasta(tmp_path, [("p1", "ACDE")])
        from evaluation.aac_floor import build_aac_embeddings

        vec = build_aac_embeddings(fasta, expected_ids=["p1"])["p1"]
        assert vec.dtype == np.float32


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_aa_sequence(self, tmp_path):
        """Single-residue sequence → valid 20-d vector."""
        fasta = _write_fasta(tmp_path, [("p1", "A")])
        from evaluation.aac_floor import build_aac_embeddings

        vec = build_aac_embeddings(fasta, expected_ids=["p1"])["p1"]
        assert vec.shape == (20,)
        assert vec[_A_IDX] == pytest.approx(1.0, abs=1e-6)

    def test_return_type_is_dict(self, tmp_path):
        """Return type must be dict[str, np.ndarray]."""
        fasta = _write_fasta(tmp_path, [("p1", "AAAA")])
        from evaluation.aac_floor import build_aac_embeddings

        result = build_aac_embeddings(fasta, expected_ids=["p1"])
        assert isinstance(result, dict)
        assert isinstance(result["p1"], np.ndarray)

    def test_fasta_with_description_field(self, tmp_path):
        """FASTA header '>p1 some description' → id is 'p1' (first token)."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">p1 some long description\nAAAA\n")
        from evaluation.aac_floor import build_aac_embeddings

        result = build_aac_embeddings(fasta, expected_ids=["p1"])
        assert "p1" in result
        assert result["p1"][_A_IDX] == pytest.approx(1.0, abs=1e-6)

    def test_multiline_fasta_sequence(self, tmp_path):
        """Sequences split across multiple lines are handled correctly."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">p1\nAAAA\nAAAA\n")  # 8 A's total
        from evaluation.aac_floor import build_aac_embeddings

        result = build_aac_embeddings(fasta, expected_ids=["p1"])
        assert result["p1"][_A_IDX] == pytest.approx(1.0, abs=1e-6)

    def test_missing_ids_error_message_names_missing_ids(self, tmp_path):
        """ValueError for missing frozen ids must name the missing ids."""
        fasta = _write_fasta(tmp_path, [("p1", "AAAA")])
        from evaluation.aac_floor import build_aac_embeddings

        with pytest.raises(ValueError) as exc_info:
            build_aac_embeddings(fasta, expected_ids=["p1", "gone_id"])
        assert "gone_id" in str(exc_info.value)
