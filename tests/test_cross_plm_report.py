"""Tests for evaluation.cross_plm_report — the cross-pLM agreement bridge + CLI.

Cross-pLM is a *pLM-pair* arm: one cell is a (plm_a, plm_b, representation, distance) tuple.
The bridge subsets BOTH pLMs to the frozen canonical set, asserts each population BEFORE
scoring (a silently-capped re-extract on either side must fail loud, not score a different
cohort), rejects degenerate embeddings, builds each pLM's square distance matrix over ONE
shared id order, computes the four symmetric agreement metrics (rho / r2 / w1_raw / w1_z)
with vertex-BCa CIs (+ a permutation p for rho/r2 ONLY — W₁ has no p by design), and
atomic-writes a per-pair parquet [pair_key, a, b, dist_a, dist_b].
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from evaluation.analysis_barrier import ArtifactSpec, check_artifact
from evaluation.population import PopulationError
from evaluation.cross_plm_report import (
    CROSS_PLM_METRICS,
    CROSS_PLM_PARQUET_GUARDS,
    CROSS_PLM_PER_PAIR_COLUMNS,
    cross_plm_report,
    main,
)

IDS = ["P1", "P2", "P3", "P4", "P5", "P6"]


def _emb_clean():
    # Two clusters offset from the origin (so no vector is zero-norm — rejected under cosine).
    return {
        "P1": np.array([1.0, 0.0], dtype=np.float32),
        "P2": np.array([1.1, 0.0], dtype=np.float32),
        "P3": np.array([1.2, 0.0], dtype=np.float32),
        "P4": np.array([5.0, 5.0], dtype=np.float32),
        "P5": np.array([5.1, 5.0], dtype=np.float32),
        "P6": np.array([5.2, 5.0], dtype=np.float32),
    }


def _emb_pair(n=12, noise=0.3, seed=0):
    """Two pLMs sharing a latent geometry + independent noise -> positive, non-perfect
    agreement (rho in (0,1), non-degenerate vertex-BCa CI)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 4)) + 3.0  # offset so no zero-norm under cosine
    ids = [f"Q{i}" for i in range(n)]
    a = {ids[i]: (base[i] + rng.normal(scale=noise, size=4)).astype(np.float32) for i in range(n)}
    b = {ids[i]: (base[i] + rng.normal(scale=noise, size=4)).astype(np.float32) for i in range(n)}
    return a, b, ids


def _parquets(out_dir):
    return sorted(p.name for p in out_dir.glob("*.parquet"))


# ── library: cross_plm_report ──────────────────────────────────────────────────
def test_identity_plms_trivial_agreement_and_degenerate_cis(tmp_path):
    emb = _emb_clean()
    m = cross_plm_report(
        emb, dict(emb), tmp_path, plm_a="prott5", plm_b="prott5_twin",
        expected_ids=IDS, distance="euclidean", n_boot=200, seed=42,
    )
    assert m["population_n_a"] == 6 and m["population_n_b"] == 6
    assert m["n_common"] == 6
    assert m["n_pairs"] == 15  # C(6, 2)
    assert set(m["metrics"]) == set(CROSS_PLM_METRICS)
    assert m["metrics"]["rho"]["point"] == pytest.approx(1.0, abs=1e-9)
    assert m["metrics"]["r2"]["point"] == pytest.approx(1.0, abs=1e-9)
    assert m["metrics"]["w1_raw"]["point"] == pytest.approx(0.0, abs=1e-9)
    assert m["metrics"]["w1_z"]["point"] == pytest.approx(0.0, abs=1e-9)
    assert m["metrics"]["w1_raw"]["ci_degenerate"] is True
    assert m["path"].endswith(".parquet")
    assert len(_parquets(tmp_path)) == 1


def test_w1_has_no_perm_p_but_rho_r2_do(tmp_path):
    a, b, ids = _emb_pair(n=12, noise=0.3, seed=1)
    m = cross_plm_report(
        a, b, tmp_path, plm_a="prott5", plm_b="esm2",
        expected_ids=ids, distance="euclidean", n_boot=200, n_perm=200, seed=3,
    )
    # rho / r2 carry a finite permutation p; W₁ (both variants) explicitly has perm_p = None.
    assert m["metrics"]["rho"]["perm_p"] is not None
    assert math.isfinite(m["metrics"]["rho"]["perm_p"])
    assert m["metrics"]["r2"]["perm_p"] is not None
    assert m["metrics"]["w1_raw"]["perm_p"] is None
    assert m["metrics"]["w1_z"]["perm_p"] is None


def test_r2_entry_carries_signed_r_ci(tmp_path):
    a, b, ids = _emb_pair(n=12, noise=0.3, seed=5)
    m = cross_plm_report(
        a, b, tmp_path, plm_a="prott5", plm_b="esm2",
        expected_ids=ids, distance="euclidean", n_boot=300, seed=7,
    )
    r2 = m["metrics"]["r2"]
    assert "r_point" in r2 and "r_ci_lo" in r2 and "r_ci_hi" in r2


def test_per_pair_parquet_passes_the_barrier(tmp_path):
    emb = _emb_clean()
    m = cross_plm_report(
        emb, dict(emb), tmp_path, plm_a="prott5", plm_b="prott5_twin",
        expected_ids=IDS, distance="euclidean", n_boot=200,
    )
    spec = ArtifactSpec(
        label="cross_plm:prott5:prott5_twin:raw:euclidean",
        path=m["path"], expected_rows=m["n_pairs"], **CROSS_PLM_PARQUET_GUARDS,
    )
    status = check_artifact(spec)
    assert status.ok, status.reasons
    df = pd.read_parquet(m["path"])
    assert list(df.columns) == list(CROSS_PLM_PER_PAIR_COLUMNS)
    assert list(df.columns) == ["pair_key", "a", "b", "dist_a", "dist_b"]


def test_population_asserted_for_pLM_a_before_scoring(tmp_path):
    emb = _emb_clean()
    with pytest.raises(PopulationError):
        cross_plm_report(
            emb, dict(emb), tmp_path, plm_a="broken_a", plm_b="ok",
            expected_ids=IDS + ["P7_missing"], distance="euclidean",
        )
    assert _parquets(tmp_path) == []


def test_population_asserted_for_pLM_b_before_scoring(tmp_path):
    a = _emb_clean()
    b = {k: v for k, v in _emb_clean().items() if k != "P6"}
    with pytest.raises(PopulationError):
        cross_plm_report(a, b, tmp_path, plm_a="ok", plm_b="broken_b",
                         expected_ids=IDS, distance="euclidean")
    assert _parquets(tmp_path) == []


def test_capped_pLM_b_reduces_n_common_and_pairs(tmp_path):
    a = _emb_clean()
    b = {k: v for k, v in _emb_clean().items() if k != "P6"}  # esm1b-style cap
    m = cross_plm_report(a, b, tmp_path, plm_a="prott5", plm_b="esm1b",
                         expected_ids=IDS, distance="euclidean",
                         allow_capped_b=True, n_boot=200)
    assert m["population_n_a"] == 6 and m["population_n_b"] == 5
    assert m["n_common"] == 5
    assert m["n_pairs"] == 10  # C(5, 2)
    # identical vectors on the 5 common ids -> perfect agreement on the intersection.
    assert m["metrics"]["rho"]["point"] == pytest.approx(1.0, abs=1e-9)


def test_capped_pLM_a_reduces_n_common(tmp_path):
    a = {k: v for k, v in _emb_clean().items() if k != "P6"}
    b = _emb_clean()
    m = cross_plm_report(a, b, tmp_path, plm_a="esm1b", plm_b="prott5",
                         expected_ids=IDS, distance="euclidean",
                         allow_capped_a=True, n_boot=200)
    assert m["population_n_a"] == 5 and m["population_n_b"] == 6
    assert m["n_common"] == 5
    with pytest.raises(PopulationError):
        cross_plm_report(a, b, tmp_path, plm_a="esm1b", plm_b="prott5",
                         expected_ids=IDS, distance="euclidean")


def test_zero_norm_vector_rejected_under_cosine_not_euclidean(tmp_path):
    a = _emb_clean()
    a["P2"] = np.array([0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="zero-norm"):
        cross_plm_report(a, _emb_clean(), tmp_path, plm_a="prott5", plm_b="esm2",
                         expected_ids=IDS, distance="cosine")
    assert _parquets(tmp_path) == []
    m = cross_plm_report(a, dict(a), tmp_path, plm_a="prott5", plm_b="esm2",
                         expected_ids=IDS, distance="euclidean", n_boot=200)
    assert m["n_common"] == 6


def test_nonfinite_embeddings_rejected_either_side(tmp_path):
    a = _emb_clean()
    a["P2"] = np.array([np.nan, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        cross_plm_report(a, _emb_clean(), tmp_path, plm_a="prott5", plm_b="esm2",
                         expected_ids=IDS, distance="euclidean")
    assert _parquets(tmp_path) == []
    b = _emb_clean()
    b["P3"] = np.array([np.inf, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        cross_plm_report(_emb_clean(), b, tmp_path, plm_a="prott5", plm_b="esm2",
                         expected_ids=IDS, distance="euclidean")
    assert _parquets(tmp_path) == []


def test_manhattan_distance_runs_end_to_end(tmp_path):
    # The third declared distance must actually score (a regression to a 2-distance axis
    # would never exercise manhattan — every other test uses euclidean/cosine).
    emb = _emb_clean()
    m = cross_plm_report(emb, dict(emb), tmp_path, plm_a="prott5", plm_b="esm2",
                         expected_ids=IDS, distance="manhattan", n_boot=200)
    assert m["distance"] == "manhattan"
    assert m["metrics"]["rho"]["point"] == pytest.approx(1.0, abs=1e-9)
    assert "manhattan" in m["path"]


def test_superset_embeddings_subset_to_frozen_set_both_sides(tmp_path):
    a, b = _emb_clean(), _emb_clean()
    a["PX_extra"] = np.array([99.0, 99.0], dtype=np.float32)
    b["PY_extra"] = np.array([-99.0, -99.0], dtype=np.float32)
    m = cross_plm_report(a, b, tmp_path, plm_a="prott5", plm_b="esm3",
                         expected_ids=IDS, distance="euclidean", n_boot=200)
    assert m["population_n_a"] == 6 and m["population_n_b"] == 6
    df = pd.read_parquet(m["path"])
    assert "PX_extra" not in df["a"].tolist() and "PX_extra" not in df["b"].tolist()
    assert "PY_extra" not in df["a"].tolist() and "PY_extra" not in df["b"].tolist()


def test_rerun_replaces_in_place_at_fixed_path(tmp_path):
    emb = _emb_clean()
    kw = dict(plm_a="prott5", plm_b="esm2", expected_ids=IDS, distance="euclidean", n_boot=200)
    first = cross_plm_report(emb, dict(emb), tmp_path, **kw)
    second = cross_plm_report(emb, dict(emb), tmp_path, **kw)
    assert first["path"] == second["path"]
    assert len(_parquets(tmp_path)) == 1


def test_distance_and_representation_in_filename_prevent_collision(tmp_path):
    emb = _emb_clean()
    base = dict(plm_a="prott5", plm_b="esm2", expected_ids=IDS, n_boot=200)
    euc = cross_plm_report(emb, dict(emb), tmp_path, distance="euclidean", representation="raw", **base)
    cos = cross_plm_report(emb, dict(emb), tmp_path, distance="cosine", representation="raw", **base)
    ffn = cross_plm_report(emb, dict(emb), tmp_path, distance="euclidean", representation="ffn", **base)
    assert len({euc["path"], cos["path"], ffn["path"]}) == 3
    assert "prott5__esm2_raw_euclidean" in euc["path"]
    assert "prott5__esm2_raw_cosine" in cos["path"]
    assert "prott5__esm2_ffn_euclidean" in ffn["path"]


def test_manifest_carries_ci_provenance(tmp_path):
    a, b, ids = _emb_pair(n=12, noise=0.3, seed=9)
    m = cross_plm_report(a, b, tmp_path, plm_a="prott5", plm_b="esm2",
                         expected_ids=ids, distance="euclidean", n_boot=400,
                         n_perm=200, ci_alpha=0.05, seed=1)
    assert m["ci_alpha"] == 0.05 and m["n_boot"] == 400 and m["n_perm"] == 200 and m["seed"] == 1
    assert "vertex" in m["ci_resample_unit"].lower()
    assert "BCa" in m["ci_method"]
    assert "i.i.d" in m["ci_note"].lower()
    assert m["per_pair_columns"] == list(CROSS_PLM_PER_PAIR_COLUMNS)


def test_seed_reproducible_non_degenerate(tmp_path):
    a, b, ids = _emb_pair(n=12, noise=0.4, seed=11)
    kw = dict(plm_a="prott5", plm_b="esm2", expected_ids=ids,
              distance="euclidean", n_boot=500, n_perm=200, seed=7)
    m1 = cross_plm_report(a, b, tmp_path / "x", **kw)
    m2 = cross_plm_report(a, b, tmp_path / "y", **kw)
    assert m1["metrics"]["rho"]["ci_degenerate"] is False
    assert (m1["metrics"]["rho"]["ci_lo"], m1["metrics"]["rho"]["ci_hi"]) == (
        m2["metrics"]["rho"]["ci_lo"], m2["metrics"]["rho"]["ci_hi"])
    assert m1["metrics"]["rho"]["perm_p"] == m2["metrics"]["rho"]["perm_p"]


# ── CLI (main) ───────────────────────────────────────────────────────────────
def _write_h5(path, embeddings):
    with h5py.File(path, "w") as f:
        for pid, vec in embeddings.items():
            f.create_dataset(pid, data=np.asarray(vec, dtype=np.float32))


def _write_freeze(path, ids):
    Path(path).write_text(json.dumps({"schema_version": 1, "set_name": "test",
                                      "ids": sorted(ids), "esm1b": None}))


def _cli_inputs(tmp_path, emb_a=None, emb_b=None, ids=None):
    emb_a = emb_a or _emb_clean()
    emb_b = emb_b or _emb_clean()
    ids = ids or IDS
    ha, hb = tmp_path / "prott5.h5", tmp_path / "esm2.h5"
    _write_h5(ha, emb_a)
    _write_h5(hb, emb_b)
    freeze = tmp_path / "freeze.json"
    _write_freeze(freeze, ids)
    return ha, hb, freeze, tmp_path / "out"


def _argv(ha, hb, freeze, out, *, distance="euclidean", rep="raw", extra=()):
    return [
        "--plm-a", "prott5", "--plm-b", "esm2",
        "--emb-h5-a", str(ha), "--emb-h5-b", str(hb),
        "--freeze", str(freeze), "--out-dir", str(out),
        "--distance", distance, "--representation", rep,
        "--n-boot", "200", "--n-perm", "200", *extra,
    ]


def test_cli_writes_parquet_and_sidecar(tmp_path):
    ha, hb, freeze, out = _cli_inputs(tmp_path)
    rc = main(_argv(ha, hb, freeze, out))
    assert rc == 0
    assert _parquets(out) == ["cross_plm_prott5__esm2_raw_euclidean.parquet"]
    sidecar = out / "cross_plm_prott5__esm2_raw_euclidean.manifest.json"
    assert sidecar.exists()
    m = json.loads(sidecar.read_text())
    assert m["plm_a"] == "prott5" and m["plm_b"] == "esm2"
    assert m["distance"] == "euclidean"
    assert m["population_n_a"] == 6 and m["population_n_b"] == 6
    assert m["metrics"]["rho"]["point"] == pytest.approx(1.0, abs=1e-9)


def test_cli_sidecar_path_points_at_real_parquet(tmp_path):
    ha, hb, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(ha, hb, freeze, out)) == 0
    m = json.loads((out / "cross_plm_prott5__esm2_raw_euclidean.manifest.json").read_text())
    assert Path(m["path"]).exists()


def test_cli_report_does_not_write_sidecar_but_cli_does(tmp_path):
    emb = _emb_clean()
    cross_plm_report(emb, dict(emb), tmp_path, plm_a="prott5", plm_b="esm2",
                     expected_ids=IDS, distance="euclidean", n_boot=200)
    assert list(tmp_path.glob("*.manifest.json")) == []


def test_cli_population_drift_returns_1_and_writes_nothing(tmp_path):
    ha, hb, _, out = _cli_inputs(tmp_path)
    freeze = tmp_path / "freeze7.json"
    _write_freeze(freeze, IDS + ["P7_missing"])
    rc = main(_argv(ha, hb, freeze, out))
    assert rc == 1
    assert _parquets(out) == []
    assert not (out / "cross_plm_prott5__esm2_raw_euclidean.manifest.json").exists()


def test_cli_allow_capped_b_ok(tmp_path):
    emb_b = {k: v for k, v in _emb_clean().items() if k != "P6"}
    ha, hb, freeze, out = _cli_inputs(tmp_path, emb_b=emb_b)
    rc = main(_argv(ha, hb, freeze, out, extra=("--allow-capped-b",)))
    assert rc == 0
    m = json.loads((out / "cross_plm_prott5__esm2_raw_euclidean.manifest.json").read_text())
    assert m["population_n_b"] == 5 and m["n_common"] == 5


def test_cli_allow_capped_a_ok(tmp_path):
    emb_a = {k: v for k, v in _emb_clean().items() if k != "P6"}
    ha, hb, freeze, out = _cli_inputs(tmp_path, emb_a=emb_a)
    rc = main(_argv(ha, hb, freeze, out, extra=("--allow-capped-a",)))
    assert rc == 0
    m = json.loads((out / "cross_plm_prott5__esm2_raw_euclidean.manifest.json").read_text())
    assert m["population_n_a"] == 5 and m["n_common"] == 5


def test_cli_disjoint_freeze_returns_1(tmp_path):
    ha, hb, _, out = _cli_inputs(tmp_path)
    freeze = tmp_path / "freeze_disjoint.json"
    _write_freeze(freeze, ["Z1", "Z2", "Z3", "Z4"])
    rc = main(_argv(ha, hb, freeze, out))
    assert rc == 1
    assert _parquets(out) == []


def test_cli_missing_emb_h5_returns_2(tmp_path):
    ha, hb, freeze, out = _cli_inputs(tmp_path)
    rc = main(_argv(tmp_path / "nope.h5", hb, freeze, out))
    assert rc == 2
    assert _parquets(out) == []


def test_cli_freeze_without_ids_returns_2(tmp_path):
    ha, hb, _, out = _cli_inputs(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1}))
    assert main(_argv(ha, hb, bad, out)) == 2


def test_cli_distance_axis_distinguishes_outputs_including_manhattan(tmp_path):
    ha, hb, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(ha, hb, freeze, out, distance="euclidean")) == 0
    assert main(_argv(ha, hb, freeze, out, distance="cosine")) == 0
    assert main(_argv(ha, hb, freeze, out, distance="manhattan")) == 0
    assert _parquets(out) == [
        "cross_plm_prott5__esm2_raw_cosine.parquet",
        "cross_plm_prott5__esm2_raw_euclidean.parquet",
        "cross_plm_prott5__esm2_raw_manhattan.parquet",
    ]


def test_cli_rejects_unknown_distance(tmp_path):
    ha, hb, freeze, out = _cli_inputs(tmp_path)
    with pytest.raises(SystemExit):  # argparse choices reject -> exit 2 via SystemExit
        main(_argv(ha, hb, freeze, out, distance="chebyshev"))


def test_cli_nonfinite_returns_2_and_writes_nothing(tmp_path):
    emb_a = _emb_clean()
    emb_a["P2"] = np.array([np.inf, 0.0], dtype=np.float32)
    ha, hb, freeze, out = _cli_inputs(tmp_path, emb_a=emb_a)
    assert main(_argv(ha, hb, freeze, out)) == 2
    assert _parquets(out) == []
    assert not (out / "cross_plm_prott5__esm2_raw_euclidean.manifest.json").exists()


def test_cli_sidecar_is_strict_json(tmp_path):
    # W₁ perm_p is None and metrics may be non-finite on tiny cohorts -> json_safe must keep
    # the sidecar strict-JSON valid (no bare NaN/Infinity tokens).
    ha, hb, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(ha, hb, freeze, out)) == 0
    text = (out / "cross_plm_prott5__esm2_raw_euclidean.manifest.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


def test_cli_seed_reproducible_through_main_non_degenerate(tmp_path):
    a, b, ids = _emb_pair(n=12, noise=0.4, seed=13)
    ha, hb, freeze, out1 = _cli_inputs(tmp_path, emb_a=a, emb_b=b, ids=ids)
    out2 = tmp_path / "out2"
    extra = ("--seed", "7",)
    assert main(_argv(ha, hb, freeze, out1, extra=extra)) == 0
    assert main(_argv(ha, hb, freeze, out2, extra=extra)) == 0
    m1 = json.loads((out1 / "cross_plm_prott5__esm2_raw_euclidean.manifest.json").read_text())
    m2 = json.loads((out2 / "cross_plm_prott5__esm2_raw_euclidean.manifest.json").read_text())
    assert m1["metrics"]["rho"]["ci_degenerate"] is False
    assert (m1["metrics"]["rho"]["ci_lo"], m1["metrics"]["rho"]["ci_hi"]) == (
        m2["metrics"]["rho"]["ci_lo"], m2["metrics"]["rho"]["ci_hi"])
