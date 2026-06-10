"""Tests for evaluation.spec_merge.merge_specs — fold N per-arm barrier specs into one."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.analysis_barrier import _load_specs, run_barrier
from evaluation.spec_merge import SpecBuildError, main, merge_specs


def _spec(*labels, reconstructed=(), population=None):
    # guard fields here are synthetic; merge_specs passes artifacts through opaquely.
    population = {} if population is None else population
    arts = [{"label": L, "path": f"/x/{L}.parquet", "expected_rows": 3, "kind": "parquet",
             "required_columns": ["query"], "unique_columns": ["query"],
             "non_null_columns": ["query"], "finite_columns": ["jaccard"]} for L in labels]
    return {"artifacts": arts,
            "_meta": {"n_cells": len(arts), "n_cells_without_sidecar": len(reconstructed),
                      "reconstructed_cells": list(reconstructed),
                      "population_n": population}}


def test_merge_two_arms_concatenates_in_order():
    m = merge_specs([_spec("recall_fp:a:raw:fold"), _spec("snn:a:b:raw:cosine")])
    assert [a["label"] for a in m["artifacts"]] == [
        "recall_fp:a:raw:fold", "snn:a:b:raw:cosine"]
    assert m["_meta"]["n_cells"] == 2


def test_merge_three_arms_is_a_real_reduce():
    m = merge_specs([_spec("a:1"), _spec("b:1"), _spec("c:1")])
    assert m["_meta"]["n_cells"] == 3
    assert len(m["_meta"]["arms"]) == 3


def test_artifacts_passed_through_opaquely():
    # A future arm's extra field (e.g. expected_dim for an H5 artifact) must survive.
    spec = {"artifacts": [{"label": "h5:x", "path": "/x.h5", "kind": "h5",
                           "expected_dim": 1024, "require_positive_norm": True}],
            "_meta": {"n_cells": 1, "n_cells_without_sidecar": 0,
                      "reconstructed_cells": []}}
    m = merge_specs([spec])
    assert m["artifacts"][0]["expected_dim"] == 1024
    assert m["artifacts"][0]["require_positive_norm"] is True


def test_meta_sums_and_concats_and_keeps_population_verbatim():
    s1 = _spec("a:1", reconstructed=["a:1"], population={"a:raw": 267})
    s2 = _spec("b:1", "b:2", reconstructed=["b:2"],
               population={"x__y:raw:cosine": {"a": 6, "b": 5}})
    m = merge_specs([s1, s2], names=["recall", "snn"])
    assert m["_meta"]["n_cells"] == 3
    assert m["_meta"]["n_cells_without_sidecar"] == 2
    assert m["_meta"]["reconstructed_cells"] == ["a:1", "b:2"]
    arms = {e["source"]: e["meta"] for e in m["_meta"]["arms"]}
    assert arms["recall"]["population_n"] == {"a:raw": 267}        # capped cohort survives
    assert arms["snn"]["population_n"] == {"x__y:raw:cosine": {"a": 6, "b": 5}}


def test_ordering_swaps_with_input():
    m = merge_specs([_spec("snn:1"), _spec("recall:1")])
    assert [a["label"] for a in m["artifacts"]] == ["snn:1", "recall:1"]
    assert [e["source"] for e in m["_meta"]["arms"]] == ["arm_0", "arm_1"]


def test_duplicate_label_across_specs_raises_naming_both_sources():
    with pytest.raises(SpecBuildError) as exc:
        merge_specs([_spec("dup"), _spec("dup")])
    msg = str(exc.value)
    assert "duplicate artifact label 'dup'" in msg
    assert "#0" in msg and "#1" in msg  # both source indices named


def test_duplicate_label_within_one_spec_also_caught():
    s = {"artifacts": [{"label": "d", "path": "/d.parquet"},
                       {"label": "d", "path": "/d2.parquet"}],
         "_meta": {"n_cells": 2, "n_cells_without_sidecar": 0, "reconstructed_cells": []}}
    with pytest.raises(SpecBuildError, match="duplicate artifact label 'd'"):
        merge_specs([s])


def test_collision_is_label_based_not_identity():
    a = _spec("same")
    b = _spec("same")  # distinct dict objects, same label
    with pytest.raises(SpecBuildError, match="duplicate"):
        merge_specs([a, b])


def test_empty_specs_fails_closed():
    with pytest.raises(SpecBuildError, match="no specs"):
        merge_specs([])


def test_single_spec_is_faithful_passthrough():
    m = merge_specs([_spec("a:1", "a:2")])
    assert m["_meta"]["n_cells"] == 2 and len(m["_meta"]["arms"]) == 1


def test_missing_artifacts_raises():
    with pytest.raises(SpecBuildError, match="artifacts"):
        merge_specs([{"_meta": {}}])


def test_non_dict_spec_raises_with_index():
    with pytest.raises(SpecBuildError, match=r"spec #1"):
        merge_specs([_spec("a:1"), ["not", "a", "dict"]])


def test_missing_meta_tolerated():
    m = merge_specs([{"artifacts": [{"label": "a:1", "path": "/a.parquet"}]}])
    assert m["_meta"]["n_cells"] == 1
    assert m["_meta"]["arms"][0]["meta"] is None


def test_corrupt_n_without_sidecar_count_fails_closed():
    # A non-int count must exit-2 cleanly, not escape as an uncaught ValueError.
    bad = {"artifacts": [{"label": "a:1", "path": "/a.parquet"}],
           "_meta": {"n_cells_without_sidecar": "oops", "reconstructed_cells": []}}
    with pytest.raises(SpecBuildError, match="n_cells_without_sidecar"):
        merge_specs([bad])


def test_bool_n_without_sidecar_count_fails_closed():
    # bool is an int subclass; it must be rejected, not silently summed.
    bad = {"artifacts": [{"label": "a:1", "path": "/a.parquet"}],
           "_meta": {"n_cells_without_sidecar": False, "reconstructed_cells": []}}
    with pytest.raises(SpecBuildError, match="n_cells_without_sidecar"):
        merge_specs([bad])


def test_artifact_without_label_or_path_raises():
    with pytest.raises(SpecBuildError, match="label"):
        merge_specs([{"artifacts": [{"path": "/x.parquet"}],
                      "_meta": {"reconstructed_cells": []}}])
    with pytest.raises(SpecBuildError, match="path"):
        merge_specs([{"artifacts": [{"label": "L"}],
                      "_meta": {"reconstructed_cells": []}}])


def test_merged_spec_round_trips_through_real_barrier_loader(tmp_path):
    from evaluation.barrier_spec_base import write_barrier_spec
    m = merge_specs([_spec("recall_fp:a:raw:fold"), _spec("snn:a:b:raw:cosine")])
    out = tmp_path / "merged.json"
    write_barrier_spec(m, out)
    specs = _load_specs(out)  # the REAL barrier loader, no SpecError
    assert len(specs) == 2


# ---- CLI ---------------------------------------------------------------------
def test_cli_merges_files_returns_0(tmp_path):
    from evaluation.barrier_spec_base import write_barrier_spec
    a, b = tmp_path / "recall.json", tmp_path / "snn.json"
    write_barrier_spec(_spec("recall_fp:a:raw:fold"), a)
    write_barrier_spec(_spec("snn:a:b:raw:cosine"), b)
    out = tmp_path / "merged.json"
    rc = main(["--specs", str(a), str(b), "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["_meta"]["n_cells"] == 2
    assert [e["source"] for e in payload["_meta"]["arms"]] == ["recall", "snn"]  # filename stems


def test_cli_collision_returns_2_and_writes_nothing(tmp_path):
    from evaluation.barrier_spec_base import write_barrier_spec
    a, b = tmp_path / "x.json", tmp_path / "y.json"
    write_barrier_spec(_spec("dup"), a)
    write_barrier_spec(_spec("dup"), b)
    out = tmp_path / "merged.json"
    rc = main(["--specs", str(a), str(b), "--out", str(out)])
    assert rc == 2
    assert not out.exists()


def _make_real_recall(tmp_path):
    import h5py
    from evaluation.recall_fp_report import main as recall_main
    emb = {"P1": [0.0, 0.0], "P2": [0.1, 0.0], "P3": [5.0, 5.0], "P4": [5.1, 5.0]}
    h5 = tmp_path / "prott5.h5"
    with h5py.File(h5, "w") as f:
        for pid, v in emb.items():
            f.create_dataset(pid, data=np.array(v, dtype=np.float32))
    tsv = tmp_path / "cath.tsv"
    tsv.write_text("Entry\tGene3D\nP1\t3.30.70.10\nP2\t3.30.70.10\nP3\t1.10.10.10\nP4\t1.10.10.10\n")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"ids": ["P1", "P2", "P3", "P4"], "n_proteins": 4}))
    out = tmp_path / "recall_out"
    assert recall_main(["--plm", "prott5", "--emb-h5", str(h5), "--cath-tsv", str(tsv),
                        "--freeze", str(freeze), "--out-dir", str(out),
                        "--distance", "euclidean", "--representation", "raw"]) == 0
    return out


def _make_real_snn_cell(tmp_path):
    d = tmp_path / "snn_out"
    d.mkdir()
    from evaluation.snn_report import SNN_PER_QUERY_COLUMNS
    n = 4
    pq = d / "snn_prott5__esm2_raw_cosine.parquet"
    df = {c: (np.arange(n) if c != "query" else [f"Q{i}" for i in range(n)])
          for c in SNN_PER_QUERY_COLUMNS}
    pd.DataFrame(df).to_parquet(pq, index=False)
    (d / "snn_prott5__esm2_raw_cosine.manifest.json").write_text(json.dumps({
        "plm_a": "prott5", "plm_b": "esm2", "n_common": n,
        "population_n_a": n, "population_n_b": n,
        "per_query_columns": list(SNN_PER_QUERY_COLUMNS), "path": str(pq),
    }))
    return d


def test_cross_arm_merge_passes_real_barrier_and_attributes_failures(tmp_path):
    from evaluation.barrier_spec import build_recall_fp_barrier_spec
    from evaluation.snn_barrier_spec import build_snn_barrier_spec
    from evaluation.barrier_spec_base import write_barrier_spec

    rdir = _make_real_recall(tmp_path)
    sdir = _make_real_snn_cell(tmp_path)
    rspec = build_recall_fp_barrier_spec(rdir, plms=["prott5"], representations=["raw"])
    sspec = build_snn_barrier_spec(sdir, pairs=[("prott5", "esm2")],
                                   representations=["raw"], distances=["cosine"])
    merged = merge_specs([rspec, sspec], names=["recall", "snn"])
    out = tmp_path / "merged.json"
    write_barrier_spec(merged, out)

    specs = _load_specs(out)
    assert len(specs) == len(rspec["artifacts"]) + len(sspec["artifacts"])
    assert run_barrier(specs).ok, run_barrier(specs).format_report()

    # Bite test: delete ONE recall parquet -> only the recall label fails (guard
    # contracts stayed arm-specific through the merge; SNN's unique=('query',) was
    # NOT applied to the recall artifact whose id column is 'query_id').
    Path(rspec["artifacts"][0]["path"]).unlink()
    report = run_barrier(_load_specs(out))
    assert not report.ok
    failing = [s.label for s in report.failures]
    assert rspec["artifacts"][0]["label"] in failing
    assert all(not lbl.startswith("snn:") for lbl in failing)
