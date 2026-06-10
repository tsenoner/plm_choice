"""Tests for evaluation.recall_fp_report — the analysis-step bridge that turns
the in-memory recall-FP result into a barrier-checkable on-disk parquet.

The bridge owns the error-prone glue the per-function tests don't cover:
subset the pLM embeddings to the frozen canonical set, assert population BEFORE
scoring (S3), score each available CATH level with the set-intersection
predicate, and atomic-write the per-query parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.analysis_barrier import ArtifactSpec, check_artifact
from evaluation.population import PopulationError
from evaluation.recall_fp_report import recall_fp_report


def _db():
    # Two folds on a line: {P1,P2} near 0, {P3,P4} near 5 -> clean retrieval.
    embeddings = {
        "P1": np.array([0.0, 0.0], dtype=np.float32),
        "P2": np.array([0.1, 0.0], dtype=np.float32),
        "P3": np.array([5.0, 5.0], dtype=np.float32),
        "P4": np.array([5.1, 5.0], dtype=np.float32),
    }
    labels = pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3", "P4"],
            "fold": [
                frozenset({"a"}),
                frozenset({"a"}),
                frozenset({"b"}),
                frozenset({"b"}),
            ],
            "superfamily": [
                frozenset({"a1"}),
                frozenset({"a1"}),
                frozenset({"b1"}),
                frozenset({"b1"}),
            ],
            "family": [None, None, None, None],
        }
    )
    return embeddings, labels


def _parquets(out_dir):
    return sorted(p.name for p in out_dir.glob("*.parquet"))


def test_writes_one_parquet_per_level_and_returns_manifest(tmp_path):
    emb, labels = _db()
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    assert set(manifest["levels"]) == {"fold", "superfamily"}
    for level in ("fold", "superfamily"):
        info = manifest["levels"][level]
        assert info["n_queries_with_positives"] == 4
        assert info["mean_recall_1stFP"] == pytest.approx(1.0)
        assert info["path"].endswith(".parquet")
    # two parquet files actually landed on disk
    assert len(_parquets(tmp_path)) == 2


def test_default_levels_exclude_family(tmp_path):
    emb, labels = _db()
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    assert "family" not in manifest["levels"]
    # note: "superfamily" contains the substring "family" -> match the level suffix
    assert not any(n.endswith("_family.parquet") for n in _parquets(tmp_path))


def test_output_parquet_passes_the_barrier(tmp_path):
    emb, labels = _db()
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    path = manifest["levels"]["fold"]["path"]
    spec = ArtifactSpec(
        label="recall_fp:prott5:fold",
        path=path,
        expected_rows=4,
        required_columns=("query_id", "n_positives", "recall", "n_ties_at_first_fp"),
        finite_columns=("recall",),
        unique_columns=("query_id",),
        non_null_columns=("query_id",),
    )
    status = check_artifact(spec)
    assert status.ok, status.reasons


def test_population_asserted_before_scoring_writes_nothing_on_drift(tmp_path):
    # A pLM silently missing a frozen id (and not flagged capped) must raise
    # BEFORE any parquet is written.
    emb, labels = _db()
    with pytest.raises(PopulationError):
        recall_fp_report(
            emb, labels, tmp_path, pLM="broken",
            expected_ids=["P1", "P2", "P3", "P4", "P5_missing"],
            distance="euclidean",
        )
    assert _parquets(tmp_path) == []


def test_capped_plm_allowed_as_subset(tmp_path):
    emb, labels = _db()
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="esm1b",
        expected_ids=["P1", "P2", "P3", "P4", "P5_missing"],
        allow_capped=True, distance="euclidean",
    )
    assert manifest["population_n"] == 4
    assert manifest["levels"]["fold"]["n_queries_with_positives"] == 4


def test_superset_embeddings_subset_to_frozen_set(tmp_path):
    # prott5/esm3 carry a 1225-key pool; the bridge must subset to the frozen
    # set rather than score (or population-fail on) the extras.
    emb, labels = _db()
    emb = dict(emb)
    emb["PX_extra"] = np.array([99.0, 99.0], dtype=np.float32)
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    assert manifest["population_n"] == 4
    df = pd.read_parquet(manifest["levels"]["fold"]["path"])
    assert "PX_extra" not in df["query_id"].tolist()


def test_rerun_replaces_in_place_at_fixed_path(tmp_path):
    # The barrier validates a FIXED spec path, so a re-run must atomically
    # replace that path (default overwrite=True), not orphan the fresh result at
    # a timestamped sibling the barrier never checks.
    emb, labels = _db()
    first = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    second = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    assert first["levels"]["fold"]["path"] == second["levels"]["fold"]["path"]
    assert len(_parquets(tmp_path)) == 2  # fold + superfamily, replaced in place


def test_no_overwrite_keeps_prior_via_timestamped_sibling(tmp_path):
    # The explicit opt-out still works for ad-hoc never-clobber use.
    emb, labels = _db()
    first = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    second = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
        overwrite=False,
    )
    assert first["levels"]["fold"]["path"] != second["levels"]["fold"]["path"]
    assert len(_parquets(tmp_path)) == 4


def test_representation_axis_in_filename_prevents_collision(tmp_path):
    # raw and ffn recall-FP of the same pLM/level must land on distinct paths.
    emb, labels = _db()
    raw = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5", representation="raw",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    ffn = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5", representation="ffn",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    assert raw["levels"]["fold"]["path"] != ffn["levels"]["fold"]["path"]
    assert "prott5_raw_fold" in raw["levels"]["fold"]["path"]
    assert "prott5_ffn_fold" in ffn["levels"]["fold"]["path"]
    assert len(_parquets(tmp_path)) == 4  # 2 reps x 2 levels, no collision


def test_n_scored_distinguishes_cohort_from_labelled(tmp_path):
    # A canonical protein with no CATH label is in the cohort (population_n) but
    # not ranked (n_scored) -> the two denominators must differ, not be conflated.
    emb, labels = _db()
    emb = dict(emb)
    emb["P5_nolabel"] = np.array([0.2, 0.0], dtype=np.float32)
    manifest = recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4", "P5_nolabel"], distance="euclidean",
    )
    assert manifest["population_n"] == 5
    assert manifest["levels"]["fold"]["n_scored"] == 4  # P5_nolabel not in labels


def test_recall_fp_report_rejects_nonfinite_embeddings(tmp_path):
    # A corrupt/degenerate embedding (NaN/Inf) must fail loudly BEFORE scoring, not
    # produce a finite-but-meaningless recall the barrier would pass.
    emb, labels = _db()
    emb = dict(emb)
    emb["P2"] = np.array([np.nan, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        recall_fp_report(
            emb, labels, tmp_path, pLM="prott5",
            expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
        )
    assert _parquets(tmp_path) == []  # nothing written


# ── CLI (main) — the analysis-DAG recall-fp step ─────────────────────────────
# The CLI is the thin wrapper the DAG calls per (pLM, representation) cell: it
# loads the pLM H5 -> dict, the cath_labels TSV, and the committed canonical-set
# freeze, then calls recall_fp_report and persists the returned manifest as a
# sidecar JSON (recall_fp_report itself deliberately does NOT write the sidecar —
# the spec-builder reads it). Exit-code contract mirrors the other DAG mains:
# 0 = ok, 1 = data failure (population drift), 2 = operator/config fault.
import json
from pathlib import Path

import h5py

from evaluation.recall_fp_report import main


def _write_h5(path, embeddings):
    with h5py.File(path, "w") as f:
        for pid, vec in embeddings.items():
            f.create_dataset(pid, data=np.asarray(vec, dtype=np.float32))


def _load_embeddings_for_test(path):
    with h5py.File(path, "r") as f:
        return {k: np.asarray(f[k][()]) for k in f.keys()}


def _write_cath_tsv(path, gene3d_by_id):
    # Minimal UniProt-style export: Entry \t Gene3D (what load_cath_labels reads).
    lines = ["Entry\tGene3D"] + [f"{pid}\t{code}" for pid, code in gene3d_by_id.items()]
    Path(path).write_text("\n".join(lines) + "\n")


def _write_freeze(path, ids):
    n = len(ids)
    Path(path).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "set_name": "test",
                "n_proteins": n,
                "n_pairs": n * (n - 1) // 2,
                "ids": sorted(ids),
                "esm1b": None,
            }
        )
    )


def _cli_inputs(tmp_path):
    # Same two clean folds as _db(), but staged on disk as the CLI consumes them.
    emb = {
        "P1": np.array([0.0, 0.0], dtype=np.float32),
        "P2": np.array([0.1, 0.0], dtype=np.float32),
        "P3": np.array([5.0, 5.0], dtype=np.float32),
        "P4": np.array([5.1, 5.0], dtype=np.float32),
    }
    h5 = tmp_path / "prott5.h5"
    _write_h5(h5, emb)
    tsv = tmp_path / "cath.tsv"
    _write_cath_tsv(
        tsv,
        {"P1": "3.30.70.10", "P2": "3.30.70.10", "P3": "1.10.10.10", "P4": "1.10.10.10"},
    )
    freeze = tmp_path / "freeze.json"
    _write_freeze(freeze, ["P1", "P2", "P3", "P4"])
    out = tmp_path / "out"
    return h5, tsv, freeze, out


def _argv(h5, tsv, freeze, out, *, plm="prott5", rep="raw", distance="euclidean", extra=()):
    return [
        "--plm", plm, "--emb-h5", str(h5), "--cath-tsv", str(tsv),
        "--freeze", str(freeze), "--out-dir", str(out),
        "--distance", distance, "--representation", rep, *extra,
    ]


def test_cli_writes_parquets_and_sidecar(tmp_path):
    h5, tsv, freeze, out = _cli_inputs(tmp_path)
    rc = main(_argv(h5, tsv, freeze, out))
    assert rc == 0
    assert _parquets(out) == [
        "recall_fp_prott5_raw_fold.parquet",
        "recall_fp_prott5_raw_superfamily.parquet",
    ]
    sidecar = out / "recall_fp_prott5_raw.manifest.json"
    assert sidecar.exists()
    manifest = json.loads(sidecar.read_text())
    assert manifest["pLM"] == "prott5"
    assert manifest["representation"] == "raw"
    assert manifest["distance"] == "euclidean"
    assert manifest["population_n"] == 4
    assert set(manifest["levels"]) == {"fold", "superfamily"}
    assert manifest["levels"]["fold"]["n_queries_with_positives"] == 4
    assert manifest["levels"]["fold"]["mean_recall_1stFP"] == pytest.approx(1.0)


def test_cli_sidecar_paths_point_at_real_parquets(tmp_path):
    h5, tsv, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(h5, tsv, freeze, out)) == 0
    manifest = json.loads((out / "recall_fp_prott5_raw.manifest.json").read_text())
    for level in ("fold", "superfamily"):
        assert Path(manifest["levels"][level]["path"]).exists()


def test_cli_report_does_not_write_sidecar_but_cli_does(tmp_path):
    # Guard the contract split: recall_fp_report() writes only parquets; the
    # sidecar JSON is exclusively the CLI's responsibility.
    emb, labels = _db()
    recall_fp_report(
        emb, labels, tmp_path, pLM="prott5",
        expected_ids=["P1", "P2", "P3", "P4"], distance="euclidean",
    )
    assert list(tmp_path.glob("*.manifest.json")) == []


def test_cli_population_drift_returns_1_and_writes_nothing(tmp_path):
    h5, tsv, _, out = _cli_inputs(tmp_path)
    freeze = tmp_path / "freeze5.json"
    _write_freeze(freeze, ["P1", "P2", "P3", "P4", "P5_missing"])
    rc = main(_argv(h5, tsv, freeze, out))
    assert rc == 1
    assert _parquets(out) == []
    assert not (out / "recall_fp_prott5_raw.manifest.json").exists()


def test_cli_allow_capped_subset_ok(tmp_path):
    h5, tsv, _, out = _cli_inputs(tmp_path)  # H5 covers P1..P4
    freeze = tmp_path / "freeze5.json"
    _write_freeze(freeze, ["P1", "P2", "P3", "P4", "P5_missing"])
    rc = main(_argv(h5, tsv, freeze, out, extra=("--allow-capped",)))
    assert rc == 0
    manifest = json.loads((out / "recall_fp_prott5_raw.manifest.json").read_text())
    # n reflects the present subset (4), not the frozen count (5) -> a capped pLM
    # is never folded into a bare cross-pLM mean over the wrong denominator.
    assert manifest["population_n"] == 4
    assert manifest["levels"]["fold"]["n_queries_with_positives"] == 4


def test_cli_missing_emb_h5_returns_2(tmp_path):
    h5, tsv, freeze, out = _cli_inputs(tmp_path)
    rc = main(_argv(tmp_path / "nope.h5", tsv, freeze, out))
    assert rc == 2
    assert _parquets(out) == []


def test_cli_freeze_without_ids_returns_2(tmp_path):
    h5, tsv, _, out = _cli_inputs(tmp_path)
    bad = tmp_path / "bad_freeze.json"
    bad.write_text(json.dumps({"schema_version": 1, "n_proteins": 0}))
    rc = main(_argv(h5, tsv, bad, out))
    assert rc == 2


def test_cli_representation_axis_distinguishes_sidecars(tmp_path):
    h5, tsv, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(h5, tsv, freeze, out, rep="raw")) == 0
    assert main(_argv(h5, tsv, freeze, out, rep="ffn")) == 0
    assert (out / "recall_fp_prott5_raw.manifest.json").exists()
    assert (out / "recall_fp_prott5_ffn.manifest.json").exists()
    # raw/ffn must also land on distinct PARQUET paths, not just distinct sidecars
    # (a regression passing a constant representation to the parquet writer would
    # collide the data even while naming the sidecars correctly).
    assert _parquets(out) == [
        "recall_fp_prott5_ffn_fold.parquet",
        "recall_fp_prott5_ffn_superfamily.parquet",
        "recall_fp_prott5_raw_fold.parquet",
        "recall_fp_prott5_raw_superfamily.parquet",
    ]


def test_cli_mean_pools_2d_embeddings(tmp_path):
    # Per-residue (L, D) embeddings are mean-pooled to a protein-level vector,
    # matching distance_computation's loader. The two rows of each protein DIFFER
    # and are chosen so the MEAN lands the protein in its correct fold cluster
    # (mean P1,P2 ~ origin; mean P3,P4 ~ (5,5) -> clean retrieval, recall 1.0),
    # while row-0 alone scrambles the clusters (P1's row-0 nearest neighbour is
    # P4, a different fold). So this assertion fails for row-0 / slice / max
    # reductions and passes only for mean-pooling -> not a vacuous test.
    emb2d = {
        "P1": np.array([[-3.0, -3.0], [3.0, 3.0]], dtype=np.float32),   # mean [0.0, 0.0]
        "P2": np.array([[3.1, 3.0], [-2.9, -3.0]], dtype=np.float32),   # mean [0.1, 0.0]
        "P3": np.array([[8.0, 8.0], [2.0, 2.0]], dtype=np.float32),     # mean [5.0, 5.0]
        "P4": np.array([[2.1, 2.0], [8.1, 8.0]], dtype=np.float32),     # mean [5.1, 5.0]
    }
    h5 = tmp_path / "prott5.h5"
    _write_h5(h5, emb2d)
    tsv = tmp_path / "cath.tsv"
    _write_cath_tsv(
        tsv,
        {"P1": "3.30.70.10", "P2": "3.30.70.10", "P3": "1.10.10.10", "P4": "1.10.10.10"},
    )
    freeze = tmp_path / "freeze.json"
    _write_freeze(freeze, ["P1", "P2", "P3", "P4"])
    out = tmp_path / "out"
    rc = main(_argv(h5, tsv, freeze, out))
    assert rc == 0
    manifest = json.loads((out / "recall_fp_prott5_raw.manifest.json").read_text())
    assert manifest["levels"]["fold"]["mean_recall_1stFP"] == pytest.approx(1.0)


def test_cli_rerun_replaces_in_place(tmp_path):
    # The DAG CLI always replaces at the canonical path (no --no-overwrite footgun):
    # a re-run leaves exactly the canonical sidecar + 2 parquets, no siblings.
    h5, tsv, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(h5, tsv, freeze, out)) == 0
    assert main(_argv(h5, tsv, freeze, out)) == 0
    assert _parquets(out) == [
        "recall_fp_prott5_raw_fold.parquet",
        "recall_fp_prott5_raw_superfamily.parquet",
    ]
    assert sorted(p.name for p in out.glob("*.manifest*.json")) == [
        "recall_fp_prott5_raw.manifest.json"
    ]


def test_cli_nonfinite_embedding_returns_2_and_writes_nothing(tmp_path):
    emb = {
        "P1": np.array([0.0, 0.0], dtype=np.float32),
        "P2": np.array([np.inf, 0.0], dtype=np.float32),  # degenerate
        "P3": np.array([5.0, 5.0], dtype=np.float32),
        "P4": np.array([5.1, 5.0], dtype=np.float32),
    }
    h5 = tmp_path / "prott5.h5"
    _write_h5(h5, emb)
    tsv = tmp_path / "cath.tsv"
    _write_cath_tsv(
        tsv,
        {"P1": "3.30.70.10", "P2": "3.30.70.10", "P3": "1.10.10.10", "P4": "1.10.10.10"},
    )
    freeze = tmp_path / "freeze.json"
    _write_freeze(freeze, ["P1", "P2", "P3", "P4"])
    out = tmp_path / "out"
    assert main(_argv(h5, tsv, freeze, out)) == 2  # ValueError -> operator/config fault
    assert _parquets(out) == []
    assert not (out / "recall_fp_prott5_raw.manifest.json").exists()


def test_cli_family_level_never_fabricates_positive(tmp_path):
    # family labels are an unmet input -> parse_cath_from_gene3d sets family=None.
    # Scoring --levels family must NOT fabricate recall=1.0 (the None==None footgun).
    h5, tsv, freeze, out = _cli_inputs(tmp_path)
    rc = main(_argv(h5, tsv, freeze, out, extra=("--levels", "family")))
    if rc == 0:
        fam = json.loads((out / "recall_fp_prott5_raw.manifest.json").read_text())[
            "levels"
        ]["family"]
        assert fam["n_queries_with_positives"] == 0
        assert fam["mean_recall_1stFP"] is None
        assert len(pd.read_parquet(fam["path"])) == 0
    else:
        assert rc == 2  # fail-closed (e.g. no labelled proteins) is equally acceptable


def test_cli_multidomain_set_intersection_through_bridge(tmp_path):
    # P1 carries TWO Gene3D domains, sharing ONE with P2. The set-intersection
    # predicate must score P1 as a positive of P2 (share ANY domain). Under naive
    # set-EQUALITY, P1's 2-element set != P2's 1-element set -> P1 would have no
    # positive and n_queries_with_positives would drop below 4. Asserting ==4
    # distinguishes intersection (correct) from equality (the bug the predicate fixes).
    emb = {
        "P1": np.array([0.0, 0.0], dtype=np.float32),
        "P2": np.array([0.1, 0.0], dtype=np.float32),
        "P3": np.array([5.0, 5.0], dtype=np.float32),
        "P4": np.array([5.1, 5.0], dtype=np.float32),
    }
    h5 = tmp_path / "prott5.h5"
    _write_h5(h5, emb)
    tsv = tmp_path / "cath.tsv"
    _write_cath_tsv(
        tsv,
        {
            "P1": "3.30.70.10;1.10.10.10",  # two domains
            "P2": "3.30.70.10",             # shares 3.30.70 with P1
            "P3": "2.40.50.10",
            "P4": "2.40.50.10",
        },
    )
    freeze = tmp_path / "freeze.json"
    _write_freeze(freeze, ["P1", "P2", "P3", "P4"])
    out = tmp_path / "out"
    assert main(_argv(h5, tsv, freeze, out)) == 0
    fold = json.loads((out / "recall_fp_prott5_raw.manifest.json").read_text())[
        "levels"
    ]["fold"]
    assert fold["n_queries_with_positives"] == 4  # all four share a domain with someone
    assert fold["mean_recall_1stFP"] == pytest.approx(1.0)


def test_cli_parquet_passes_the_real_barrier(tmp_path):
    # The CLI is the path that gets barrier-validated in production, and it differs
    # from the direct recall_fp_report() call (it loads from H5 + TSV + freeze).
    # Arm an ArtifactSpec from the shared PARQUET_GUARDS contract + the sidecar's
    # reported path/expected_rows and confirm the CLI-emitted parquet passes.
    from evaluation.recall_fp_report import PARQUET_GUARDS

    h5, tsv, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(h5, tsv, freeze, out)) == 0
    manifest = json.loads((out / "recall_fp_prott5_raw.manifest.json").read_text())
    info = manifest["levels"]["fold"]
    spec = ArtifactSpec(
        label="recall_fp:prott5:raw:fold",
        path=info["path"],
        expected_rows=info["n_queries_with_positives"],
        **PARQUET_GUARDS,
    )
    status = check_artifact(spec)
    assert status.ok, status.reasons


def test_cli_zero_positive_level_emits_0row_parquet_and_null_mean(tmp_path):
    # Every protein in a distinct fold -> no two share a domain -> every query has
    # 0 positives. The level must emit a 0-row parquet and a NaN mean that the
    # sidecar serialises as JSON null (not the invalid bare `NaN` token), so a
    # strict reader (the spec-builder) can round-trip it.
    emb = {
        "P1": np.array([0.0, 0.0], dtype=np.float32),
        "P2": np.array([1.0, 0.0], dtype=np.float32),
        "P3": np.array([5.0, 5.0], dtype=np.float32),
        "P4": np.array([6.0, 5.0], dtype=np.float32),
    }
    h5 = tmp_path / "prott5.h5"
    _write_h5(h5, emb)
    tsv = tmp_path / "cath.tsv"
    _write_cath_tsv(
        tsv,
        {  # four DISTINCT folds (and superfamilies) -> zero shared domains
            "P1": "3.30.70.10",
            "P2": "1.10.10.10",
            "P3": "2.40.50.10",
            "P4": "3.40.50.20",
        },
    )
    freeze = tmp_path / "freeze.json"
    _write_freeze(freeze, ["P1", "P2", "P3", "P4"])
    out = tmp_path / "out"
    rc = main(_argv(h5, tsv, freeze, out))
    assert rc == 0
    # the raw sidecar text must be standards-valid JSON (no bare NaN token)
    sidecar_text = (out / "recall_fp_prott5_raw.manifest.json").read_text()
    assert "NaN" not in sidecar_text
    manifest = json.loads(sidecar_text)  # strict round-trip
    fold = manifest["levels"]["fold"]
    assert fold["n_queries_with_positives"] == 0
    assert fold["mean_recall_1stFP"] is None  # NaN -> null
    # the parquet exists and has zero rows (the barrier rejects it by design)
    df = pd.read_parquet(fold["path"])
    assert len(df) == 0


def test_cli_levels_override_scores_only_requested(tmp_path):
    h5, tsv, freeze, out = _cli_inputs(tmp_path)
    rc = main(_argv(h5, tsv, freeze, out, extra=("--levels", "fold")))
    assert rc == 0
    manifest = json.loads((out / "recall_fp_prott5_raw.manifest.json").read_text())
    assert set(manifest["levels"]) == {"fold"}
    assert _parquets(out) == ["recall_fp_prott5_raw_fold.parquet"]


def test_cli_canonical_id_without_cath_label_is_cohort_not_scored(tmp_path):
    # A frozen protein present in the H5 but absent from the cath TSV is in the
    # cohort (population_n) yet not ranked (n_scored) -> the CLI's distinct label
    # (TSV) and population (freeze) loaders must keep the two denominators apart.
    emb = {
        "P1": np.array([0.0, 0.0], dtype=np.float32),
        "P2": np.array([0.1, 0.0], dtype=np.float32),
        "P3": np.array([5.0, 5.0], dtype=np.float32),
        "P4": np.array([5.1, 5.0], dtype=np.float32),
        "P5": np.array([0.2, 0.0], dtype=np.float32),  # no CATH label below
    }
    h5 = tmp_path / "prott5.h5"
    _write_h5(h5, emb)
    tsv = tmp_path / "cath.tsv"
    _write_cath_tsv(
        tsv,
        {"P1": "3.30.70.10", "P2": "3.30.70.10", "P3": "1.10.10.10", "P4": "1.10.10.10"},
    )
    freeze = tmp_path / "freeze.json"
    _write_freeze(freeze, ["P1", "P2", "P3", "P4", "P5"])
    out = tmp_path / "out"
    rc = main(_argv(h5, tsv, freeze, out))
    assert rc == 0
    manifest = json.loads((out / "recall_fp_prott5_raw.manifest.json").read_text())
    assert manifest["population_n"] == 5  # cohort
    assert manifest["levels"]["fold"]["n_scored"] == 4  # P5 unlabelled


def test_cli_superset_h5_subset_to_frozen_set(tmp_path):
    # The real prott5/esm3 pools carry ~1225 keys; the CLI loader + bridge must
    # subset to the frozen set rather than score (or population-fail on) the extras.
    h5, tsv, freeze, out = _cli_inputs(tmp_path)  # freeze = P1..P4
    emb = _load_embeddings_for_test(h5)
    emb["PX_extra"] = np.array([99.0, 99.0], dtype=np.float32)
    _write_h5(h5, emb)
    rc = main(_argv(h5, tsv, freeze, out))
    assert rc == 0
    manifest = json.loads((out / "recall_fp_prott5_raw.manifest.json").read_text())
    assert manifest["population_n"] == 4
    df = pd.read_parquet(manifest["levels"]["fold"]["path"])
    assert "PX_extra" not in df["query_id"].tolist()


def test_cli_disjoint_population_returns_1_and_writes_nothing(tmp_path):
    # Freeze entirely disjoint from the H5 -> after subsetting, an EMPTY cohort ->
    # PopulationError -> exit 1 (data failure), nothing written. Pins the empty-
    # intersection branch the missing-id drift test doesn't reach.
    h5, tsv, _, out = _cli_inputs(tmp_path)  # H5 = P1..P4
    freeze = tmp_path / "freeze_disjoint.json"
    _write_freeze(freeze, ["Q1", "Q2", "Q3", "Q4"])
    rc = main(_argv(h5, tsv, freeze, out))
    assert rc == 1
    assert _parquets(out) == []
    assert not (out / "recall_fp_prott5_raw.manifest.json").exists()
