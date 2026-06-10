"""Tests for evaluation.analysis_io — the shared loaders/serialisers every analysis
arm uses (embedding H5 -> dict, canonical-set freeze -> ids, JSON-safe manifest).

These were first written privately inside ``recall_fp_report``; consolidating them
here is the single source of truth so the SNN / EC / AAC / pdb-TM bridges all subset,
assert, and serialise the same way (no drift-prone second copy).
"""
from __future__ import annotations

import json
import math

import h5py
import numpy as np
import pytest

from evaluation.analysis_io import json_safe, load_embeddings_h5, load_frozen_ids


def _write_h5(path, embeddings):
    with h5py.File(path, "w") as f:
        for pid, vec in embeddings.items():
            f.create_dataset(pid, data=np.asarray(vec, dtype=np.float32))


def test_load_embeddings_1d_roundtrip(tmp_path):
    emb = {"P1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
           "P2": np.array([4.0, 5.0, 6.0], dtype=np.float32)}
    h5 = tmp_path / "e.h5"
    _write_h5(h5, emb)
    out = load_embeddings_h5(h5)
    assert set(out) == {"P1", "P2"}
    assert out["P1"].dtype == np.float32
    np.testing.assert_allclose(out["P2"], [4.0, 5.0, 6.0])


def test_load_embeddings_mean_pools_2d(tmp_path):
    # A per-residue (L, D) dataset is mean-pooled over residues to (D,).
    emb = {"P1": np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32)}  # mean [1.0, 2.0]
    h5 = tmp_path / "e.h5"
    _write_h5(h5, emb)
    out = load_embeddings_h5(h5)
    assert out["P1"].shape == (2,)
    np.testing.assert_allclose(out["P1"], [1.0, 2.0])


def test_load_embeddings_scalar_safe(tmp_path):
    # `[()]` reads scalar + array datasets alike (regression: f[k][:] fails on a scalar).
    h5 = tmp_path / "e.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("S", data=np.float32(3.0))  # 0-d scalar dataset
    out = load_embeddings_h5(h5)
    assert out["S"].dtype == np.float32


def test_load_frozen_ids_reads_ids(tmp_path):
    freeze = tmp_path / "f.json"
    freeze.write_text(json.dumps({"set_name": "t", "ids": ["P3", "P1", "P2"]}))
    assert load_frozen_ids(freeze) == ["P3", "P1", "P2"]  # order preserved, not sorted


def test_load_frozen_ids_missing_ids_raises(tmp_path):
    freeze = tmp_path / "f.json"
    freeze.write_text(json.dumps({"set_name": "t", "n_proteins": 0}))
    with pytest.raises(ValueError, match="ids"):
        load_frozen_ids(freeze)


def test_load_frozen_ids_empty_list_raises(tmp_path):
    freeze = tmp_path / "f.json"
    freeze.write_text(json.dumps({"ids": []}))
    with pytest.raises(ValueError):
        load_frozen_ids(freeze)


def test_json_safe_maps_nonfinite_to_none():
    obj = {"a": float("nan"), "b": 1.5, "c": float("inf"), "d": -float("inf"),
           "nested": {"x": float("nan"), "y": [1.0, float("nan"), 3.0]}}
    safe = json_safe(obj)
    assert safe["a"] is None and safe["c"] is None and safe["d"] is None
    assert safe["b"] == 1.5
    assert safe["nested"]["x"] is None
    assert safe["nested"]["y"] == [1.0, None, 3.0]
    # round-trips through strict JSON (no bare NaN token)
    text = json.dumps(safe)
    assert "NaN" not in text and "Infinity" not in text


def test_json_safe_leaves_finite_and_non_floats():
    obj = {"s": "hi", "i": 3, "f": 2.0, "b": True, "n": None}
    assert json_safe(obj) == obj


def test_json_safe_coerces_numpy_scalars():
    # np.float32 is NOT a Python float subclass, so a manifest value that skipped a
    # float(...) wrap at its source would slip through as a non-finite/unserialisable
    # numpy scalar and break strict JSON. json_safe must coerce numpy scalars first.
    obj = {
        "f32_nan": np.float32("nan"),
        "f32_val": np.float32(0.25),
        "f64_inf": np.float64("inf"),
        "i64": np.int64(7),
        "nested": [np.float32("nan"), np.float64(1.5)],
    }
    safe = json_safe(obj)
    assert safe["f32_nan"] is None and safe["f64_inf"] is None
    assert safe["f32_val"] == pytest.approx(0.25) and isinstance(safe["f32_val"], float)
    assert safe["i64"] == 7 and isinstance(safe["i64"], int)
    assert safe["nested"][0] is None and safe["nested"][1] == 1.5
    text = json.dumps(safe)  # strict round-trip — no bare NaN/Infinity, no numpy type error
    assert "NaN" not in text and "Infinity" not in text


def test_json_safe_does_not_mutate_input():
    obj = {"x": float("nan")}
    json_safe(obj)
    assert math.isnan(obj["x"])  # original untouched


def test_json_safe_reproduces_legacy_shallow_on_zero_query_recall_manifest():
    # Anti-drift: the recursive json_safe replaced recall_fp_report's deleted SHALLOW
    # _json_safe_manifest, which nulled only levels[*].{mean_recall_1stFP,ci_lo,ci_hi}.
    # On a representative 0-query recall-fp manifest the new function must null exactly
    # those (and leave finite siblings / strings / ints / nested structure intact), and
    # round-trip through strict JSON.
    manifest = {
        "pLM": "prott5", "representation": "raw", "distance": "euclidean",
        "population_n": 4, "ci_alpha": 0.05, "n_boot": 500, "seed": 42,
        "ci_method": "BCa bootstrap, query-level resample",
        "levels": {
            "fold": {
                "path": "/tmp/recall_fp_prott5_raw_fold.parquet",
                "n_queries_with_positives": 0, "n_scored": 0,
                "mean_recall_1stFP": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "ci_degenerate": True,
            },
            "superfamily": {
                "path": "/tmp/recall_fp_prott5_raw_superfamily.parquet",
                "n_queries_with_positives": 4, "n_scored": 4,
                "mean_recall_1stFP": 1.0, "ci_lo": 1.0, "ci_hi": 1.0,
                "ci_degenerate": True,
            },
        },
    }
    safe = json_safe(manifest)
    fold = safe["levels"]["fold"]
    assert fold["mean_recall_1stFP"] is None
    assert fold["ci_lo"] is None and fold["ci_hi"] is None
    assert fold["n_queries_with_positives"] == 0  # int untouched
    assert fold["path"].endswith(".parquet")      # str untouched
    sf = safe["levels"]["superfamily"]
    assert sf["mean_recall_1stFP"] == 1.0 and sf["ci_lo"] == 1.0  # finite untouched
    assert safe["pLM"] == "prott5" and safe["population_n"] == 4 and safe["ci_alpha"] == 0.05
    text = json.dumps(safe)  # strict round-trip, no bare NaN token
    assert "NaN" not in text
