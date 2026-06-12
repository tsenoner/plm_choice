"""Unit 7 — cross-pLM agreement-matrix assembly + Holm over the 6 families.

Reads the per-pair sidecars cross_plm_report wrote, assembles the symmetric 15x15 (here
small-N) agreement matrices (ρ / R² / W₁-raw / W₁-z; diagonal ρ=1 / R²=1 / W₁=0), and applies
Holm-Bonferroni per ``(distance, metric)`` family over the C(n,2) unordered pLM pairs — **6
families** ``{ρ, R²} × {euclidean, cosine, manhattan}``. W₁ is descriptive only (no Holm).

The §9.3 guard order is load-bearing and tested explicitly:
1. **Pre-filter size assert** — each family must have EXACTLY C(n,2) entries *present* before
   any NaN handling (a missing/dead-job cell -> fail loud). A SEPARATE, PRIOR check from the
   NaN drop (not a circular "expected = C(n,2) - drops").
2. **NaN-p filter** — then drop cells whose perm-p is NaN/None for any reason; the dropped
   count is recorded.
3. **Holm** — ``stats.holm_bonferroni`` on the surviving p-vector (it RAISES on NaN, so step 2
   is mandatory).
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from evaluation.cross_plm_matrix import (
    HOLM_METRICS,
    MATRIX_METRICS,
    assemble_agreement_matrices,
)
from evaluation.stats import holm_bonferroni

PLMS = ["m1", "m2", "m3", "m4", "m5"]  # C(5,2) = 10 pairs per family


def _cell(a, b, dist, *, rho=(0.8, 0.01), r2=(0.64, 0.02), w1_raw=1.2, w1_z=0.3, rep="raw"):
    """A minimal cross-pLM sidecar dict (only the fields the assembler reads)."""
    return {
        "plm_a": a, "plm_b": b, "representation": rep, "distance": dist,
        "metrics": {
            "rho": {"point": rho[0], "perm_p": rho[1]},
            "r2": {"point": r2[0], "perm_p": r2[1]},
            "w1_raw": {"point": w1_raw, "perm_p": None},
            "w1_z": {"point": w1_z, "perm_p": None},
        },
        "n_pairs": 10, "path": "x",
    }


def _write_cell(d: Path, cell: dict, *, rep="raw"):
    a, b, dist = cell["plm_a"], cell["plm_b"], cell["distance"]
    (d / f"cross_plm_{a}__{b}_{rep}_{dist}.manifest.json").write_text(json.dumps(cell))


def _write_full_grid(d: Path, plms, distances, *, cell_fn=None):
    """Write a complete sidecar grid; cell_fn(a, b, dist, idx) -> kwargs override per cell."""
    for dist in distances:
        for idx, (a, b) in enumerate(combinations(plms, 2)):
            kw = cell_fn(a, b, dist, idx) if cell_fn else {}
            _write_cell(d, _cell(a, b, dist, **kw))


# ── matrices: symmetric, correct diagonal ──────────────────────────────────────
def test_matrices_symmetric_with_correct_diagonal(tmp_path):
    _write_full_grid(tmp_path, PLMS, ["cosine"])
    out = assemble_agreement_matrices(tmp_path, plms=PLMS, distances=["cosine"])
    n = len(PLMS)
    for metric in MATRIX_METRICS:
        M = np.array(out["matrices"]["cosine"][metric])
        assert M.shape == (n, n)
        assert np.allclose(M, M.T, equal_nan=True), f"{metric} not symmetric"
        diag = 0.0 if metric in ("w1_raw", "w1_z") else 1.0
        assert np.allclose(np.diag(M), diag), f"{metric} diagonal != {diag}"
    # off-diagonal point value round-trips
    rho_M = np.array(out["matrices"]["cosine"]["rho"])
    assert rho_M[0, 1] == pytest.approx(0.8) and rho_M[1, 0] == pytest.approx(0.8)
    w1_M = np.array(out["matrices"]["cosine"]["w1_raw"])
    assert w1_M[0, 1] == pytest.approx(1.2)


# ── 6 families ──────────────────────────────────────────────────────────────────
def test_six_holm_families_over_three_distances(tmp_path):
    distances = ["cosine", "euclidean", "manhattan"]
    _write_full_grid(tmp_path, PLMS, distances)
    out = assemble_agreement_matrices(tmp_path, plms=PLMS, distances=distances)
    assert set(out["families"]) == {
        f"{d}:{m}" for d in distances for m in ("rho", "r2")
    }
    assert len(out["families"]) == 6


def test_w1_metrics_have_no_holm_family(tmp_path):
    _write_full_grid(tmp_path, PLMS, ["cosine"])
    out = assemble_agreement_matrices(tmp_path, plms=PLMS, distances=["cosine"])
    assert HOLM_METRICS == ("rho", "r2")
    assert not any("w1" in key for key in out["families"])


# ── Holm wiring: rejections match a direct holm_bonferroni on the same p-vector ──
def test_holm_rejections_match_direct_computation(tmp_path):
    # A spread of perm-p so Holm rejects some and not others; the assembler must feed the
    # per-pair perm_p (in pair order) to holm_bonferroni and report the same rejections.
    pvals = [0.0001, 0.002, 0.4, 0.01, 0.9, 0.05, 0.3, 0.001, 0.2, 0.6]

    def cell_fn(a, b, dist, idx):
        return {"rho": (0.5, pvals[idx])}

    _write_full_grid(tmp_path, PLMS, ["cosine"], cell_fn=cell_fn)
    out = assemble_agreement_matrices(tmp_path, plms=PLMS, distances=["cosine"], alpha=0.05)
    fam = out["families"]["cosine:rho"]
    assert fam["n_present"] == 10 and fam["n_dropped"] == 0
    # records are in pair (combinations) order; compare against a direct Holm call.
    got_p = [r["perm_p"] for r in fam["records"]]
    assert got_p == pvals
    rejected, adjusted = holm_bonferroni(np.array(pvals), alpha=0.05)
    assert [r["rejected"] for r in fam["records"]] == list(map(bool, rejected))
    assert [r["adjusted_p"] for r in fam["records"]] == pytest.approx(list(adjusted))


# ── §9.3 step 1: pre-filter size assert (missing cell -> fail loud) ──────────────
def test_short_family_trips_size_assert(tmp_path):
    # Write all 10 cosine cells, then DELETE one -> 9 present != 10 expected -> loud failure
    # BEFORE any NaN handling (a dead-job cell must not be silently absorbed).
    _write_full_grid(tmp_path, PLMS, ["cosine"])
    a, b = list(combinations(PLMS, 2))[3]
    (tmp_path / f"cross_plm_{a}__{b}_raw_cosine.manifest.json").unlink()
    with pytest.raises(ValueError, match="cosine:rho"):
        assemble_agreement_matrices(tmp_path, plms=PLMS, distances=["cosine"])


def test_size_assert_is_separate_from_nan_drop(tmp_path):
    # A family with ALL 10 present but one NaN-p must NOT trip the size assert (present == 10);
    # it is dropped at step 2 instead. This distinguishes "missing cell" from "degenerate cell"
    # (the circular "expected = 10 - drops" flaw the fan caught would conflate them).
    def cell_fn(a, b, dist, idx):
        return {"rho": (0.5, None)} if idx == 2 else {}  # one NaN-p, all present

    _write_full_grid(tmp_path, PLMS, ["cosine"], cell_fn=cell_fn)
    out = assemble_agreement_matrices(tmp_path, plms=PLMS, distances=["cosine"])
    fam = out["families"]["cosine:rho"]
    assert fam["n_present"] == 10
    assert fam["n_dropped"] == 1
    assert len(fam["records"]) == 9  # the NaN-p cell dropped from Holm


# ── §9.3 step 2: NaN-p filter records the dropped pair ──────────────────────────
def test_nan_perm_p_dropped_and_recorded(tmp_path):
    dropped_pair = list(combinations(PLMS, 2))[4]

    def cell_fn(a, b, dist, idx):
        return {"r2": (0.5, None)} if idx == 4 else {}

    _write_full_grid(tmp_path, PLMS, ["cosine"], cell_fn=cell_fn)
    out = assemble_agreement_matrices(tmp_path, plms=PLMS, distances=["cosine"])
    fam = out["families"]["cosine:r2"]
    assert fam["n_dropped"] == 1
    assert [list(dropped_pair)] == fam["dropped_pairs"]
    # the dropped pair is absent from the surviving records
    surviving_pairs = {(r["a"], r["b"]) for r in fam["records"]}
    assert dropped_pair not in surviving_pairs


def test_nan_as_float_also_dropped(tmp_path):
    # A perm_p that is a literal float NaN (not JSON null) must also be filtered — holm raises
    # on NaN, so the filter must catch both None and float('nan').
    def cell_fn(a, b, dist, idx):
        return {"rho": (0.5, float("nan"))} if idx == 0 else {}

    # json.dumps(float('nan')) emits the bare token NaN; write via a custom dump to simulate a
    # sidecar that slipped a non-finite through (the assembler must still be robust).
    for idx, (a, b) in enumerate(combinations(PLMS, 2)):
        cell = _cell(a, b, "cosine", **({"rho": (0.5, float("nan"))} if idx == 0 else {}))
        (tmp_path / f"cross_plm_{a}__{b}_raw_cosine.manifest.json").write_text(
            json.dumps(cell)  # Python json emits NaN; json.loads reads it back as float nan
        )
    out = assemble_agreement_matrices(tmp_path, plms=PLMS, distances=["cosine"])
    fam = out["families"]["cosine:rho"]
    assert fam["n_dropped"] == 1
    assert all(math.isfinite(r["perm_p"]) for r in fam["records"])


# ── fan finding #19/#18: R² Holm path + combined descriptive-vs-Holm contract ───
def test_holm_rejections_match_direct_computation_r2(tmp_path):
    # The R² Holm family must be wired to holm_bonferroni exactly like ρ (a regression that
    # mis-fed R² perm-p — e.g. wrong pair order — would otherwise pass under the ρ-only test).
    pvals = [0.0003, 0.5, 0.01, 0.2, 0.8, 0.04, 0.001, 0.3, 0.7, 0.02]

    def cell_fn(a, b, dist, idx):
        return {"r2": (0.5, pvals[idx])}

    _write_full_grid(tmp_path, PLMS, ["cosine"], cell_fn=cell_fn)
    out = assemble_agreement_matrices(tmp_path, plms=PLMS, distances=["cosine"], alpha=0.05)
    fam = out["families"]["cosine:r2"]
    assert [r["perm_p"] for r in fam["records"]] == pvals
    rejected, adjusted = holm_bonferroni(np.array(pvals), alpha=0.05)
    assert [r["rejected"] for r in fam["records"]] == list(map(bool, rejected))
    assert [r["adjusted_p"] for r in fam["records"]] == pytest.approx(list(adjusted))


def test_families_are_exactly_the_six_rho_r2_while_matrices_keep_all_four(tmp_path):
    # Pin the "W₁ descriptive-only (in matrices, not families); ρ/R² in Holm" contract in ONE
    # assertion, so a regression that added W₁ to HOLM_METRICS (or dropped it from matrices)
    # is caught directly.
    distances = ["cosine", "euclidean", "manhattan"]
    _write_full_grid(tmp_path, PLMS, distances)
    out = assemble_agreement_matrices(tmp_path, plms=PLMS, distances=distances)
    assert set(out["families"]) == {f"{d}:{m}" for d in distances for m in ("rho", "r2")}
    for d in distances:
        assert set(out["matrices"][d]) == set(MATRIX_METRICS)


# ── fan finding #16: fail loud on <2 plms (no silent empty assembly) ────────────
def test_fewer_than_two_plms_raises(tmp_path):
    with pytest.raises(ValueError, match="at least 2 pLMs"):
        assemble_agreement_matrices(tmp_path, plms=["only_one"], distances=["cosine"])


# ── fan finding #9: a non-numeric perm_p is a malformed sidecar (typed raise) ───
def test_non_numeric_perm_p_raises_typed_error(tmp_path):
    # A corrupt sidecar whose perm_p is a list/string (not a number or null) must fail loud as
    # a malformed sidecar, not throw an untyped float() TypeError out of the assembler.
    for idx, (a, b) in enumerate(combinations(PLMS, 2)):
        cell = _cell(a, b, "cosine")
        if idx == 0:
            cell["metrics"]["rho"]["perm_p"] = ["not", "a", "number"]
        (tmp_path / f"cross_plm_{a}__{b}_raw_cosine.manifest.json").write_text(json.dumps(cell))
    with pytest.raises(ValueError, match="perm_p"):
        assemble_agreement_matrices(tmp_path, plms=PLMS, distances=["cosine"])


# ── representation axis ─────────────────────────────────────────────────────────
def test_only_requested_representation_read(tmp_path):
    # Write a raw grid + a stray ffn cell; the default (raw) assembly must ignore ffn.
    _write_full_grid(tmp_path, PLMS, ["cosine"])
    _write_cell(tmp_path, _cell("m1", "m2", "cosine", rep="ffn"), rep="ffn")
    out = assemble_agreement_matrices(tmp_path, plms=PLMS, distances=["cosine"])
    assert out["families"]["cosine:rho"]["n_present"] == 10
