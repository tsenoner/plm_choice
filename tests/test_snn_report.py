"""Tests for evaluation.snn_report — the analysis-step bridge that turns the
in-memory cross-pLM SNN result into a barrier-checkable per-query parquet + sidecar.

SNN is a *cross-pLM* arm: one cell is a (plm_a, plm_b, representation, distance)
tuple. The bridge owns the glue the label-free ``snn.knn_jaccard_between_plms``
does not: subset BOTH pLMs to the frozen canonical set, assert each population
BEFORE scoring (so a silently-capped re-extract on either side fails loud rather
than scoring a different cohort), reject non-finite embeddings, compute a
degenerate-honest BCa CI on the per-query Jaccard, and atomic-write the parquet.
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
from evaluation.snn_report import SNN_PARQUET_GUARDS, main, snn_report

IDS = ["P1", "P2", "P3", "P4", "P5", "P6"]


def _emb_clean():
    # Two clean clusters {P1,P2,P3} near (1,0), {P4,P5,P6} near (5,5). Cluster 1 is
    # offset from the origin so no vector is zero-norm (a zero vector is rejected under
    # cosine — the metric is undefined there).
    return {
        "P1": np.array([1.0, 0.0], dtype=np.float32),
        "P2": np.array([1.1, 0.0], dtype=np.float32),
        "P3": np.array([1.2, 0.0], dtype=np.float32),
        "P4": np.array([5.0, 5.0], dtype=np.float32),
        "P5": np.array([5.1, 5.0], dtype=np.float32),
        "P6": np.array([5.2, 5.0], dtype=np.float32),
    }


def _emb_scrambled():
    # Different cluster membership {P1,P2,P4} / {P3,P5,P6} -> partial k-NN overlap
    # with _emb_clean, so per-query Jaccard varies in (0, 1) (non-degenerate CI).
    return {
        "P1": np.array([1.0, 0.0], dtype=np.float32),
        "P2": np.array([1.1, 0.0], dtype=np.float32),
        "P4": np.array([1.2, 0.0], dtype=np.float32),
        "P3": np.array([5.0, 5.0], dtype=np.float32),
        "P5": np.array([5.1, 5.0], dtype=np.float32),
        "P6": np.array([5.2, 5.0], dtype=np.float32),
    }


def _parquets(out_dir):
    return sorted(p.name for p in out_dir.glob("*.parquet"))


# ── library: snn_report ──────────────────────────────────────────────────────
def test_identity_plms_yield_jaccard_one_and_degenerate_ci(tmp_path):
    emb = _emb_clean()
    m = snn_report(
        emb, dict(emb), tmp_path, plm_a="prott5", plm_b="prott5_twin",
        expected_ids=IDS, distance="euclidean", k=2, n_boot=300, seed=42,
    )
    assert m["mean_jaccard"] == pytest.approx(1.0)
    assert m["n_common"] == 6
    assert m["population_n_a"] == 6 and m["population_n_b"] == 6
    assert m["ci_lo"] == 1.0 and m["ci_hi"] == 1.0 and m["ci_degenerate"] is True
    assert m["path"].endswith(".parquet")
    assert len(_parquets(tmp_path)) == 1


def test_scrambled_plms_below_one_and_ci_reproducible(tmp_path):
    a, b = _emb_clean(), _emb_scrambled()
    kw = dict(plm_a="prott5", plm_b="esm2", expected_ids=IDS,
              distance="euclidean", k=2, n_boot=800, seed=7)
    m1 = snn_report(a, b, tmp_path / "x", **kw)
    m2 = snn_report(a, b, tmp_path / "y", **kw)
    assert m1["mean_jaccard"] < 1.0                       # geometry differs
    assert m1["ci_degenerate"] is False                   # per-query Jaccard varies
    assert (m1["ci_lo"], m1["ci_hi"]) == (m2["ci_lo"], m2["ci_hi"])  # seed reproduces
    assert 0.0 <= m1["ci_lo"] <= m1["ci_hi"] <= 1.0


def test_per_query_parquet_passes_the_barrier(tmp_path):
    emb = _emb_clean()
    m = snn_report(
        emb, dict(emb), tmp_path, plm_a="prott5", plm_b="prott5_twin",
        expected_ids=IDS, distance="euclidean", k=2,
    )
    spec = ArtifactSpec(
        label="snn:prott5:prott5_twin:raw:euclidean",
        path=m["path"], expected_rows=m["n_common"], **SNN_PARQUET_GUARDS,
    )
    status = check_artifact(spec)
    assert status.ok, status.reasons
    df = pd.read_parquet(m["path"])
    assert list(df.columns) == ["query", "jaccard", "k_a", "k_b"]


def test_population_asserted_for_pLM_a_before_scoring(tmp_path):
    emb = _emb_clean()
    with pytest.raises(PopulationError):
        snn_report(
            emb, dict(emb), tmp_path, plm_a="broken_a", plm_b="ok",
            expected_ids=IDS + ["P7_missing"], distance="euclidean", k=2,
        )
    assert _parquets(tmp_path) == []


def test_population_asserted_for_pLM_b_before_scoring(tmp_path):
    # pLM B silently missing a frozen id must fail just as loudly as pLM A.
    a = _emb_clean()
    b = {k: v for k, v in _emb_clean().items() if k != "P6"}  # B drops P6
    with pytest.raises(PopulationError):
        snn_report(a, b, tmp_path, plm_a="ok", plm_b="broken_b",
                   expected_ids=IDS, distance="euclidean", k=2)
    assert _parquets(tmp_path) == []


def test_capped_pLM_allowed_as_subset_reduces_n_common(tmp_path):
    a = _emb_clean()
    b = {k: v for k, v in _emb_clean().items() if k != "P6"}  # esm1b-style cap
    m = snn_report(a, b, tmp_path, plm_a="prott5", plm_b="esm1b",
                   expected_ids=IDS, distance="euclidean", k=2,
                   allow_capped_b=True)
    assert m["population_n_a"] == 6
    assert m["population_n_b"] == 5          # capped subset reported separately
    assert m["n_common"] == 5               # scored over the intersection only
    # a and b carry IDENTICAL vectors on the 5 common ids, so the agreement is perfect
    # ONLY if both rank against the same common cohort. The pre-fix asymmetric code
    # (A's k-NN DB = 6, B's = 5) would deflate this below 1.0.
    assert m["mean_jaccard"] == pytest.approx(1.0)


def test_zero_norm_vector_rejected_under_cosine_not_euclidean(tmp_path):
    a = _emb_clean()
    a["P2"] = np.array([0.0, 0.0], dtype=np.float32)  # finite but zero-norm
    with pytest.raises(ValueError, match="zero-norm"):
        snn_report(a, _emb_clean(), tmp_path, plm_a="prott5", plm_b="esm2",
                   expected_ids=IDS, distance="cosine", k=2)
    assert _parquets(tmp_path) == []
    # The same zero vector is a valid point (the origin) under euclidean — not rejected.
    m = snn_report(a, dict(a), tmp_path, plm_a="prott5", plm_b="esm2",
                   expected_ids=IDS, distance="euclidean", k=2)
    assert m["n_common"] == 6


def test_capped_pLM_a_allowed_as_subset(tmp_path):
    # A is asserted BEFORE B, so the A-side allow_capped path needs its own coverage:
    # a swapped/missing allow_capped_a would be invisible to the B-side test above.
    a = {k: v for k, v in _emb_clean().items() if k != "P6"}  # esm1b on the A side
    b = _emb_clean()
    m = snn_report(a, b, tmp_path, plm_a="esm1b", plm_b="prott5",
                   expected_ids=IDS, distance="euclidean", k=2,
                   allow_capped_a=True)
    assert m["population_n_a"] == 5 and m["population_n_b"] == 6
    assert m["n_common"] == 5
    # without the flag, the A-side cap must fail loud (population drift), nothing written.
    with pytest.raises(PopulationError):
        snn_report(a, b, tmp_path, plm_a="esm1b", plm_b="prott5",
                   expected_ids=IDS, distance="euclidean", k=2)


def test_manhattan_distance_runs_end_to_end(tmp_path):
    # The third declared distance must actually score (a typo in the metric passthrough
    # for manhattan would otherwise be uncaught — every other test uses euclidean/cosine).
    emb = _emb_clean()
    m = snn_report(emb, dict(emb), tmp_path, plm_a="prott5", plm_b="esm2",
                   expected_ids=IDS, distance="manhattan", k=2)
    assert m["distance"] == "manhattan"
    assert m["mean_jaccard"] == pytest.approx(1.0)
    assert "manhattan" in m["path"]


def test_superset_embeddings_subset_to_frozen_set_both_sides(tmp_path):
    a, b = _emb_clean(), _emb_clean()
    a["PX_extra"] = np.array([99.0, 99.0], dtype=np.float32)
    b["PY_extra"] = np.array([-99.0, -99.0], dtype=np.float32)
    m = snn_report(a, b, tmp_path, plm_a="prott5", plm_b="esm3",
                   expected_ids=IDS, distance="euclidean", k=2)
    assert m["population_n_a"] == 6 and m["population_n_b"] == 6
    df = pd.read_parquet(m["path"])
    assert "PX_extra" not in df["query"].tolist()
    assert "PY_extra" not in df["query"].tolist()


def test_rerun_replaces_in_place_at_fixed_path(tmp_path):
    emb = _emb_clean()
    kw = dict(plm_a="prott5", plm_b="esm2", expected_ids=IDS, distance="euclidean", k=2)
    first = snn_report(emb, dict(emb), tmp_path, **kw)
    second = snn_report(emb, dict(emb), tmp_path, **kw)
    assert first["path"] == second["path"]
    assert len(_parquets(tmp_path)) == 1


def test_distance_and_representation_in_filename_prevent_collision(tmp_path):
    emb = _emb_clean()
    base = dict(plm_a="prott5", plm_b="esm2", expected_ids=IDS, k=2)
    euc = snn_report(emb, dict(emb), tmp_path, distance="euclidean", representation="raw", **base)
    cos = snn_report(emb, dict(emb), tmp_path, distance="cosine", representation="raw", **base)
    ffn = snn_report(emb, dict(emb), tmp_path, distance="euclidean", representation="ffn", **base)
    paths = {euc["path"], cos["path"], ffn["path"]}
    assert len(paths) == 3  # distance and representation both disambiguate
    assert "prott5__esm2_raw_euclidean" in euc["path"]
    assert "prott5__esm2_raw_cosine" in cos["path"]
    assert "prott5__esm2_ffn_euclidean" in ffn["path"]


def test_nonfinite_embeddings_rejected_either_side(tmp_path):
    a = _emb_clean()
    a["P2"] = np.array([np.nan, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        snn_report(a, _emb_clean(), tmp_path, plm_a="prott5", plm_b="esm2",
                   expected_ids=IDS, distance="euclidean", k=2)
    assert _parquets(tmp_path) == []

    b = _emb_clean()
    b["P3"] = np.array([np.inf, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        snn_report(_emb_clean(), b, tmp_path, plm_a="prott5", plm_b="esm2",
                   expected_ids=IDS, distance="euclidean", k=2)
    assert _parquets(tmp_path) == []


def test_manifest_carries_ci_provenance(tmp_path):
    emb = _emb_clean()
    m = snn_report(emb, _emb_scrambled(), tmp_path, plm_a="prott5", plm_b="esm2",
                   expected_ids=IDS, distance="euclidean", k=2, n_boot=400,
                   ci_alpha=0.05, seed=1)
    assert m["ci_alpha"] == 0.05 and m["n_boot"] == 400 and m["seed"] == 1
    assert m["ci_resample_unit"] == "query"
    assert "BCa" in m["ci_method"]
    assert "i.i.d" in m["ci_note"].lower()
    assert m["k"] == 2 and m["distance"] == "euclidean"


# ── CLI (main) ───────────────────────────────────────────────────────────────
def _write_h5(path, embeddings):
    with h5py.File(path, "w") as f:
        for pid, vec in embeddings.items():
            f.create_dataset(pid, data=np.asarray(vec, dtype=np.float32))


def _write_freeze(path, ids):
    Path(path).write_text(json.dumps({"schema_version": 1, "set_name": "test",
                                      "ids": sorted(ids), "esm1b": None}))


def _cli_inputs(tmp_path, emb_a=None, emb_b=None):
    emb_a = emb_a or _emb_clean()
    emb_b = emb_b or _emb_clean()
    ha, hb = tmp_path / "prott5.h5", tmp_path / "esm2.h5"
    _write_h5(ha, emb_a)
    _write_h5(hb, emb_b)
    freeze = tmp_path / "freeze.json"
    _write_freeze(freeze, IDS)
    return ha, hb, freeze, tmp_path / "out"


def _argv(ha, hb, freeze, out, *, distance="euclidean", rep="raw", extra=()):
    return [
        "--plm-a", "prott5", "--plm-b", "esm2",
        "--emb-h5-a", str(ha), "--emb-h5-b", str(hb),
        "--freeze", str(freeze), "--out-dir", str(out),
        "--distance", distance, "--representation", rep, "--k", "2", *extra,
    ]


def test_cli_writes_parquet_and_sidecar(tmp_path):
    ha, hb, freeze, out = _cli_inputs(tmp_path)
    rc = main(_argv(ha, hb, freeze, out))
    assert rc == 0
    assert _parquets(out) == ["snn_prott5__esm2_raw_euclidean.parquet"]
    sidecar = out / "snn_prott5__esm2_raw_euclidean.manifest.json"
    assert sidecar.exists()
    m = json.loads(sidecar.read_text())
    assert m["plm_a"] == "prott5" and m["plm_b"] == "esm2"
    assert m["distance"] == "euclidean" and m["k"] == 2
    assert m["population_n_a"] == 6 and m["population_n_b"] == 6
    assert m["mean_jaccard"] == pytest.approx(1.0)


def test_cli_sidecar_path_points_at_real_parquet(tmp_path):
    ha, hb, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(ha, hb, freeze, out)) == 0
    m = json.loads((out / "snn_prott5__esm2_raw_euclidean.manifest.json").read_text())
    assert Path(m["path"]).exists()


def test_cli_report_does_not_write_sidecar_but_cli_does(tmp_path):
    emb = _emb_clean()
    snn_report(emb, dict(emb), tmp_path, plm_a="prott5", plm_b="esm2",
               expected_ids=IDS, distance="euclidean", k=2)
    assert list(tmp_path.glob("*.manifest.json")) == []


def test_cli_population_drift_returns_1_and_writes_nothing(tmp_path):
    ha, hb, _, out = _cli_inputs(tmp_path)
    freeze = tmp_path / "freeze7.json"
    _write_freeze(freeze, IDS + ["P7_missing"])
    rc = main(_argv(ha, hb, freeze, out))
    assert rc == 1
    assert _parquets(out) == []
    assert not (out / "snn_prott5__esm2_raw_euclidean.manifest.json").exists()


def test_cli_allow_capped_b_ok(tmp_path):
    emb_b = {k: v for k, v in _emb_clean().items() if k != "P6"}
    ha, hb, freeze, out = _cli_inputs(tmp_path, emb_b=emb_b)
    rc = main(_argv(ha, hb, freeze, out, extra=("--allow-capped-b",)))
    assert rc == 0
    m = json.loads((out / "snn_prott5__esm2_raw_euclidean.manifest.json").read_text())
    assert m["population_n_b"] == 5 and m["n_common"] == 5


def test_cli_allow_capped_a_ok(tmp_path):
    emb_a = {k: v for k, v in _emb_clean().items() if k != "P6"}
    ha, hb, freeze, out = _cli_inputs(tmp_path, emb_a=emb_a)
    rc = main(_argv(ha, hb, freeze, out, extra=("--allow-capped-a",)))
    assert rc == 0
    m = json.loads((out / "snn_prott5__esm2_raw_euclidean.manifest.json").read_text())
    assert m["population_n_a"] == 5 and m["n_common"] == 5


def test_cli_disjoint_freeze_returns_1_and_writes_nothing(tmp_path):
    # Freeze entirely disjoint from the H5s -> after subsetting, an EMPTY cohort ->
    # PopulationError -> exit 1 (data failure). Pins the empty-intersection branch the
    # missing-id drift test does not reach.
    ha, hb, _, out = _cli_inputs(tmp_path)
    freeze = tmp_path / "freeze_disjoint.json"
    _write_freeze(freeze, ["Q1", "Q2", "Q3", "Q4"])
    rc = main(_argv(ha, hb, freeze, out))
    assert rc == 1
    assert _parquets(out) == []
    assert not (out / "snn_prott5__esm2_raw_euclidean.manifest.json").exists()


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


def test_cli_distance_axis_distinguishes_outputs(tmp_path):
    ha, hb, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(ha, hb, freeze, out, distance="euclidean")) == 0
    assert main(_argv(ha, hb, freeze, out, distance="cosine")) == 0
    assert _parquets(out) == [
        "snn_prott5__esm2_raw_cosine.parquet",
        "snn_prott5__esm2_raw_euclidean.parquet",
    ]
    assert (out / "snn_prott5__esm2_raw_cosine.manifest.json").exists()
    assert (out / "snn_prott5__esm2_raw_euclidean.manifest.json").exists()


def test_cli_mean_pools_2d_embeddings(tmp_path):
    # Per-residue (L, D) on the A side is mean-pooled to a protein vector; the B side is
    # the clean 1-D reference. The two rows of each A protein are chosen so the MEAN keeps
    # the cluster GROUPING ({P1,P2,P3} / {P4,P5,P6}, matching B -> jaccard 1.0), while
    # row-0 alone MIXES the groups (P1's row-0 sits in B's other cluster, P4's row-0 in the
    # first) so P1's row-0 k-NN are {P5,P6} not {P2,P3} -> jaccard < 1. The assertion thus
    # fails for row-0 / slice / max reductions and passes only for mean-pooling: not vacuous
    # (k-NN Jaccard is invariant to where a cluster sits, so the fixture must break grouping,
    # not just translate it — feeding the same array to both sides would be vacuous).
    emb_a2d = {
        "P1": np.array([[5.0, 5.0], [-3.0, -5.0]], dtype=np.float32),   # mean [1.0, 0.0]
        "P2": np.array([[1.1, 0.0], [1.1, 0.0]], dtype=np.float32),     # mean [1.1, 0.0]
        "P3": np.array([[1.2, 0.0], [1.2, 0.0]], dtype=np.float32),     # mean [1.2, 0.0]
        "P4": np.array([[1.0, 0.0], [9.0, 10.0]], dtype=np.float32),    # mean [5.0, 5.0]
        "P5": np.array([[5.1, 5.0], [5.1, 5.0]], dtype=np.float32),     # mean [5.1, 5.0]
        "P6": np.array([[5.2, 5.0], [5.2, 5.0]], dtype=np.float32),     # mean [5.2, 5.0]
    }
    ha, hb, freeze, out = _cli_inputs(tmp_path, emb_a=emb_a2d, emb_b=_emb_clean())
    assert main(_argv(ha, hb, freeze, out)) == 0
    m = json.loads((out / "snn_prott5__esm2_raw_euclidean.manifest.json").read_text())
    assert m["mean_jaccard"] == pytest.approx(1.0)


def test_cli_rerun_replaces_in_place(tmp_path):
    ha, hb, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(ha, hb, freeze, out)) == 0
    assert main(_argv(ha, hb, freeze, out)) == 0
    assert _parquets(out) == ["snn_prott5__esm2_raw_euclidean.parquet"]
    assert sorted(p.name for p in out.glob("*.manifest*.json")) == [
        "snn_prott5__esm2_raw_euclidean.manifest.json"
    ]


def test_cli_nonfinite_returns_2_and_writes_nothing(tmp_path):
    emb_a = _emb_clean()
    emb_a["P2"] = np.array([np.inf, 0.0], dtype=np.float32)
    ha, hb, freeze, out = _cli_inputs(tmp_path, emb_a=emb_a)
    assert main(_argv(ha, hb, freeze, out)) == 2
    assert _parquets(out) == []
    assert not (out / "snn_prott5__esm2_raw_euclidean.manifest.json").exists()


def test_cli_sidecar_is_strict_json(tmp_path):
    ha, hb, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(ha, hb, freeze, out, extra=("--seed", "42", "--n-boot", "300"))) == 0
    text = (out / "snn_prott5__esm2_raw_euclidean.manifest.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)  # strict round-trip


def test_cli_seed_reproducible_through_main_non_degenerate(tmp_path):
    ha, hb, freeze, out1 = _cli_inputs(tmp_path)
    # scrambled B so the CI is non-degenerate and actually exercises the RNG.
    _write_h5(hb, _emb_scrambled())
    out2 = tmp_path / "out2"
    extra = ("--seed", "7", "--n-boot", "800")
    assert main(_argv(ha, hb, freeze, out1, extra=extra)) == 0
    assert main(_argv(ha, hb, freeze, out2, extra=extra)) == 0
    m1 = json.loads((out1 / "snn_prott5__esm2_raw_euclidean.manifest.json").read_text())
    m2 = json.loads((out2 / "snn_prott5__esm2_raw_euclidean.manifest.json").read_text())
    assert m1["ci_degenerate"] is False
    assert (m1["ci_lo"], m1["ci_hi"]) == (m2["ci_lo"], m2["ci_hi"])
