"""Tests for evaluation.floor_comparison — Unit 4 of the AAC-floor arm.

Spec: docs/superpowers/specs/2026-06-11-aac-floor-design.md §3 Unit 4 + §6 +
§10 (C1/I3/I4/M7).

This is the net-new science of the arm: the one-sided pLM-vs-AAC-floor comparison
that ships the paper claim ("pLM X beats the AAC floor, one-sided, significant").
It consumes the recall-fp per-query parquet (pLM side) and the population-matched
AAC-floor per-query parquet (floor side), inner-joins on ``query_id``, computes
Δ = recall_plm − recall_aac per query, and runs the one-sided paired Wilcoxon +
a paired BCa CI on mean Δ.

Two correctness pillars:

* **Pillar 1 (direction):** pLM-dominates → p tiny, δ≈1, mean_delta>0, CI excludes 0;
  pLM==AAC → p≈1, δ≈0; pLM-worse → p≈1 under ``greater`` (direction enforced).
* **Pillar 2 (C1 paired-join):** a capped pLM compares only against its own queries
  (inner-join, no fabricated rows), and against the population-MATCHED AAC vs the
  wrong full-pop AAC gives a DIFFERENT Δ (population-matching is load-bearing).

I4: Δ ∈ [−1, 1]; its CI is NOT clipped to (0, 1) — a pLM that loses must show a
genuinely NEGATIVE delta_ci_lo (the floor-not-beaten signal must not be hidden).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.floor_comparison import (
    FLOOR_PARQUET_GUARDS,
    apply_holm_family,
    floor_comparison_report,
    main,
    main_holm,
)


# ── fixtures ─────────────────────────────────────────────────────────────────
def _write_per_query(path: Path, rows: dict[str, float]) -> Path:
    """Write a recall-fp-shaped per-query parquet (query_id, n_positives, recall,
    n_ties_at_first_fp) given {query_id: recall}."""
    df = pd.DataFrame(
        {
            "query_id": list(rows.keys()),
            "n_positives": [1] * len(rows),
            "recall": list(rows.values()),
            "n_ties_at_first_fp": [0] * len(rows),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _plm_dominates(tmp_path: Path):
    # pLM recall strictly > AAC recall on every query.
    ids = [f"Q{i}" for i in range(12)]
    plm = _write_per_query(
        tmp_path / "plm.parquet", {q: 0.9 for q in ids}
    )
    aac = _write_per_query(
        tmp_path / "aac.parquet", {q: 0.2 for q in ids}
    )
    return plm, aac, ids


# ── PILLAR 1: direction ──────────────────────────────────────────────────────
def test_plm_dominates_one_sided_significant(tmp_path):
    plm, aac, ids = _plm_dominates(tmp_path)
    m = floor_comparison_report(
        plm, aac, tmp_path / "out",
        plm="esm2", distance="euclidean", level="fold",
        n_boot=500, seed=42,
    )
    assert m["n_paired"] == len(ids)
    assert m["mean_delta"] == pytest.approx(0.7)
    assert m["mean_recall_plm"] == pytest.approx(0.9)
    assert m["mean_recall_aac"] == pytest.approx(0.2)
    assert m["p_one_sided"] < 0.01
    assert m["cliffs_delta"] == pytest.approx(1.0)
    # CI on a strictly-positive Δ excludes 0 (lower bound > 0).
    assert m["delta_ci_lo"] > 0.0
    assert m["alternative"] == "greater"


def test_plm_equals_aac_neutral(tmp_path):
    ids = [f"Q{i}" for i in range(10)]
    plm = _write_per_query(tmp_path / "plm.parquet", {q: 0.5 for q in ids})
    aac = _write_per_query(tmp_path / "aac.parquet", {q: 0.5 for q in ids})
    m = floor_comparison_report(
        plm, aac, tmp_path / "out",
        plm="esm2", distance="euclidean", level="fold", n_boot=200, seed=42,
    )
    # the np.all(a==b) neutral path: p == 1, delta == 0
    assert m["p_one_sided"] == pytest.approx(1.0)
    assert m["cliffs_delta"] == pytest.approx(0.0)
    assert m["mean_delta"] == pytest.approx(0.0)


def test_plm_worse_one_sided_not_significant_under_greater(tmp_path):
    # The direction discriminator: pLM is WORSE than AAC -> p ~ 1 under "greater"
    # (proves the test is directional, not symmetric).
    ids = [f"Q{i}" for i in range(12)]
    plm = _write_per_query(tmp_path / "plm.parquet", {q: 0.2 for q in ids})
    aac = _write_per_query(tmp_path / "aac.parquet", {q: 0.9 for q in ids})
    m = floor_comparison_report(
        plm, aac, tmp_path / "out",
        plm="esm2", distance="euclidean", level="fold", n_boot=200, seed=42,
    )
    assert m["mean_delta"] < 0.0
    assert m["p_one_sided"] > 0.99
    assert m["cliffs_delta"] == pytest.approx(-1.0)


# ── I4: a pLM that loses must have a GENUINELY NEGATIVE delta_ci_lo (un-clipped) ─
def test_i4_negative_lower_bound_not_clipped(tmp_path):
    # pLM loses on most queries -> mean Δ negative -> delta_ci_lo MUST be < 0.
    # If the CI were routed through bounded_mean_bca_ci (clip 0,1) the lower bound
    # would be truncated to 0 and the "floor not beaten" signal would be hidden.
    rng = np.random.default_rng(0)
    ids = [f"Q{i}" for i in range(40)]
    # pLM clearly below AAC, with spread so the bootstrap is non-degenerate.
    plm_vals = np.clip(0.2 + 0.05 * rng.standard_normal(len(ids)), 0, 1)
    aac_vals = np.clip(0.8 + 0.05 * rng.standard_normal(len(ids)), 0, 1)
    plm = _write_per_query(tmp_path / "plm.parquet", dict(zip(ids, plm_vals)))
    aac = _write_per_query(tmp_path / "aac.parquet", dict(zip(ids, aac_vals)))
    m = floor_comparison_report(
        plm, aac, tmp_path / "out",
        plm="esm2", distance="euclidean", level="fold", n_boot=2000, seed=42,
    )
    assert m["mean_delta"] < 0.0
    # the I4 guard: the lower bound is genuinely negative, NOT clipped to 0.
    assert m["delta_ci_lo"] < 0.0, (
        "delta_ci_lo was clipped to 0 — I4 violated (negative Δ CI hidden)"
    )
    assert m["delta_ci_hi"] < 0.0  # whole interval below 0 -> floor not beaten


# ── PILLAR 2: paired-join / capped cohort (C1 interaction) ───────────────────
def test_capped_plm_inner_joins_on_own_queries(tmp_path):
    # A capped pLM has only a query subset; the comparison must be over the
    # intersection only -> n_paired == the pLM's query count, no fabricated rows.
    aac_ids = [f"Q{i}" for i in range(12)]
    plm_ids = aac_ids[:7]  # capped: only 7 queries
    plm = _write_per_query(tmp_path / "plm.parquet", {q: 0.9 for q in plm_ids})
    aac = _write_per_query(tmp_path / "aac.parquet", {q: 0.3 for q in aac_ids})
    m = floor_comparison_report(
        plm, aac, tmp_path / "out",
        plm="esm1b", distance="euclidean", level="fold", n_boot=200, seed=42,
    )
    assert m["n_paired"] == 7  # the intersection, not 12
    cmp_df = pd.read_parquet(m["comparison_path"])
    assert len(cmp_df) == 7
    assert set(cmp_df["query_id"]) == set(plm_ids)  # no fabricated rows


def test_population_matching_changes_delta(tmp_path):
    # The C1 point: comparing the capped pLM against the WRONG (full-pop) AAC vs the
    # RIGHT (population-matched) AAC gives a DIFFERENT Δ. The full-pop AAC was scored
    # on a larger DB -> systematically different recall for the SAME queries.
    plm_ids = [f"Q{i}" for i in range(7)]
    plm = _write_per_query(tmp_path / "plm.parquet", {q: 0.9 for q in plm_ids})
    # matched AAC: scored on the 267-cohort -> higher recall (fewer FP candidates)
    aac_matched = _write_per_query(
        tmp_path / "aac_matched.parquet", {q: 0.5 for q in plm_ids}
    )
    # wrong full-pop AAC: scored on the full DB -> lower recall on the same queries
    full_ids = [f"Q{i}" for i in range(12)]
    aac_full = _write_per_query(
        tmp_path / "aac_full.parquet",
        {q: (0.2 if q in plm_ids else 0.4) for q in full_ids},
    )
    m_matched = floor_comparison_report(
        plm, aac_matched, tmp_path / "m",
        plm="esm1b", distance="euclidean", level="fold", n_boot=200, seed=42,
    )
    m_wrong = floor_comparison_report(
        plm, aac_full, tmp_path / "w",
        plm="esm1b", distance="euclidean", level="fold", n_boot=200, seed=42,
    )
    # both join down to the 7 pLM queries...
    assert m_matched["n_paired"] == 7 and m_wrong["n_paired"] == 7
    # ...but the Δ differs because the AAC recall differs (population-matching matters).
    assert m_matched["mean_delta"] != pytest.approx(m_wrong["mean_delta"])
    assert m_matched["mean_recall_aac"] == pytest.approx(0.5)
    assert m_wrong["mean_recall_aac"] == pytest.approx(0.2)


def test_disjoint_queries_raise(tmp_path):
    # No overlap at all -> nothing to compare -> ValueError (empty join).
    plm = _write_per_query(tmp_path / "plm.parquet", {"A1": 0.9, "A2": 0.9})
    aac = _write_per_query(tmp_path / "aac.parquet", {"B1": 0.3, "B2": 0.3})
    with pytest.raises(ValueError, match="no overlapping|empty"):
        floor_comparison_report(
            plm, aac, tmp_path / "out",
            plm="esm2", distance="euclidean", level="fold", seed=42,
        )


# ── comparison parquet + guards ──────────────────────────────────────────────
def test_comparison_parquet_shape_and_guards(tmp_path):
    plm, aac, ids = _plm_dominates(tmp_path)
    m = floor_comparison_report(
        plm, aac, tmp_path / "out",
        plm="esm2", distance="euclidean", level="fold", n_boot=200, seed=42,
    )
    df = pd.read_parquet(m["comparison_path"])
    assert set(df.columns) == {"query_id", "recall_plm", "recall_aac", "delta"}
    assert (df["delta"] == df["recall_plm"] - df["recall_aac"]).all()
    # the guards exist + have the right shape
    assert set(FLOOR_PARQUET_GUARDS["required_columns"]) == {
        "query_id", "recall_plm", "recall_aac", "delta",
    }
    assert FLOOR_PARQUET_GUARDS["unique_columns"] == ("query_id",)
    assert FLOOR_PARQUET_GUARDS["non_null_columns"] == ("query_id",)
    assert "delta" in FLOOR_PARQUET_GUARDS["finite_columns"]


def test_comparison_parquet_passes_the_real_barrier(tmp_path):
    from evaluation.analysis_barrier import ArtifactSpec, check_artifact

    plm, aac, ids = _plm_dominates(tmp_path)
    m = floor_comparison_report(
        plm, aac, tmp_path / "out",
        plm="esm2", distance="euclidean", level="fold", n_boot=200, seed=42,
    )
    spec = ArtifactSpec(
        label="floor_cmp:esm2:euclidean:fold",
        path=m["comparison_path"],
        expected_rows=m["n_paired"],
        **FLOOR_PARQUET_GUARDS,
    )
    status = check_artifact(spec)
    assert status.ok, status.reasons


def test_manifest_has_required_fields_and_ci_note(tmp_path):
    plm, aac, ids = _plm_dominates(tmp_path)
    m = floor_comparison_report(
        plm, aac, tmp_path / "out",
        plm="esm2", distance="cosine", level="superfamily", n_boot=200, seed=7,
    )
    for key in (
        "plm", "distance", "level", "n_paired", "mean_recall_plm",
        "mean_recall_aac", "mean_delta", "delta_ci_lo", "delta_ci_hi",
        "wilcoxon_statistic", "p_one_sided", "cliffs_delta", "alternative",
        "seed", "n_boot",
    ):
        assert key in m, f"manifest missing {key}"
    assert m["plm"] == "esm2"
    assert m["distance"] == "cosine"
    assert m["level"] == "superfamily"
    assert m["seed"] == 7
    assert isinstance(m["ci_note"], str) and m["ci_note"]


# ── Holm family (D10) ─────────────────────────────────────────────────────────
def _write_sidecar(path: Path, plm: str, p: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "plm": plm,
        "distance": "euclidean",
        "level": "fold",
        "n_paired": 100,
        "mean_recall_plm": 0.5,
        "mean_recall_aac": 0.4,
        "mean_delta": 0.1,
        "delta_ci_lo": 0.05,
        "delta_ci_hi": 0.15,
        "wilcoxon_statistic": 1.0,
        "p_one_sided": p,
        "cliffs_delta": 0.5,
        "alternative": "greater",
        "seed": 42,
        "n_boot": 10000,
    }
    path.write_text(json.dumps(manifest) + "\n")
    return path


def test_apply_holm_family_matches_hand_computed(tmp_path):
    # 4 raw p-values; Holm step-down at alpha=0.05.
    ps = {"a": 0.001, "b": 0.01, "c": 0.04, "d": 0.5}
    d = tmp_path / "fold"
    for plm, p in ps.items():
        _write_sidecar(d / f"floor_cmp_{plm}_raw.manifest.json", plm, p)
    out = apply_holm_family(d, distance="euclidean", level="fold", alpha=0.05)
    # hand Holm (n=4, sorted 0.001,0.01,0.04,0.5): adj = 0.004, 0.03, 0.08, 0.5
    verdict = {row["plm"]: row for row in out["verdicts"]}
    assert verdict["a"]["beats_floor"] is True   # 0.004 <= 0.05
    assert verdict["b"]["beats_floor"] is True   # 0.03  <= 0.05
    assert verdict["c"]["beats_floor"] is False  # 0.08  >  0.05
    assert verdict["d"]["beats_floor"] is False  # 0.5   >  0.05
    assert verdict["a"]["p_adj"] == pytest.approx(0.004)
    assert verdict["c"]["p_adj"] == pytest.approx(0.08)


def test_apply_holm_writes_verdict_file(tmp_path):
    ps = {"a": 0.001, "b": 0.2, "c": 0.9}
    d = tmp_path / "fold"
    for plm, p in ps.items():
        _write_sidecar(d / f"floor_cmp_{plm}_raw.manifest.json", plm, p)
    out = apply_holm_family(d, distance="euclidean", level="fold", alpha=0.05)
    vpath = Path(out["verdict_path"])
    assert vpath.exists()
    written = json.loads(vpath.read_text())
    assert written["distance"] == "euclidean"
    assert written["level"] == "fold"
    assert len(written["verdicts"]) == 3


# ── CLI exit-code matrix (per-cell comparison) ────────────────────────────────
def _cli_argv(plm, aac, out, *, plm_name="esm2", distance="euclidean", level="fold"):
    return [
        "--plm-per-query", str(plm),
        "--aac-per-query", str(aac),
        "--out-dir", str(out),
        "--plm", plm_name,
        "--distance", distance,
        "--level", level,
    ]


def test_cli_exit_0_writes_parquet_and_sidecar(tmp_path):
    plm, aac, ids = _plm_dominates(tmp_path)
    out = tmp_path / "out"
    rc = main(_cli_argv(plm, aac, out) + ["--n-boot", "200", "--seed", "42"])
    assert rc == 0
    assert (out / "floor_cmp_esm2_raw.manifest.json").exists()
    parquets = sorted(p.name for p in out.glob("*.parquet"))
    assert parquets == ["floor_cmp_esm2_fold.parquet"]


def test_cli_disjoint_population_exit_1(tmp_path):
    # An empty inner-join is the population-mismatch data failure -> exit 1.
    plm = _write_per_query(tmp_path / "plm.parquet", {"A1": 0.9})
    aac = _write_per_query(tmp_path / "aac.parquet", {"B1": 0.3})
    out = tmp_path / "out"
    rc = main(_cli_argv(plm, aac, out))
    assert rc == 1
    assert list(out.glob("*.parquet")) == []


def test_cli_missing_input_exit_2(tmp_path):
    aac = _write_per_query(tmp_path / "aac.parquet", {"Q0": 0.3})
    out = tmp_path / "out"
    rc = main(_cli_argv(tmp_path / "nope.parquet", aac, out))
    assert rc == 2


def test_cli_holm_mode_writes_verdict(tmp_path):
    ps = {"a": 0.001, "b": 0.2, "c": 0.9}
    d = tmp_path / "fold"
    for plm, p in ps.items():
        _write_sidecar(d / f"floor_cmp_{plm}_raw.manifest.json", plm, p)
    rc = main_holm([
        "--sidecar-dir", str(d),
        "--distance", "euclidean",
        "--level", "fold",
    ])
    assert rc == 0
    verdicts = sorted(d.glob("floor_family_verdict_*.json"))
    assert len(verdicts) == 1


def test_cli_dispatch_apply_holm_flag(tmp_path):
    # The single entrypoint dispatches to Holm mode on --apply-holm.
    ps = {"a": 0.001, "b": 0.9}
    d = tmp_path / "fold"
    for plm, p in ps.items():
        _write_sidecar(d / f"floor_cmp_{plm}_raw.manifest.json", plm, p)
    rc = main([
        "--apply-holm",
        "--sidecar-dir", str(d),
        "--distance", "euclidean",
        "--level", "fold",
    ])
    assert rc == 0
