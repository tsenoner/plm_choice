"""Unit 4 — orphan report library + 3-exit-code CLI (clone of ec_report).

The arm's value ships here: score_orphan_pairs -> per-pair parquet (D9 contract) +
orphan_auroc_vertex_bca_ci -> AUROC point + vertex BCa CI, plus the naive i.i.d.-pair
CI as a FLAGGED comparison, plus the two Spearman ρ. The CLI writes the sidecar; the
library writes only the parquet (the EC split).
"""
import json
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

from evaluation.orphan_report import (
    ORPHAN_PARQUET_GUARDS,
    ORPHAN_PER_PAIR_COLUMNS,
    OrphanPopulationError,
    main as orphan_main,
    orphan_correlation_report,
)


# ── fixtures ──────────────────────────────────────────────────────────────────────────
def _clean_monotone(n=24, seed=1):
    """A cohort where sibling cos is strictly above non-sibling cos -> AUROC 1.0."""
    rng = np.random.default_rng(seed)
    ids = [f"P{i:02d}" for i in range(n)]
    # siblings get parallel vectors; non-siblings orthogonal.
    emb = {pid: rng.normal(size=8).astype(np.float32) for pid in ids}
    rows = []
    # siblings: copy one vector to its partner (cos ~ 1)
    for i in range(0, n - 1, 2):
        emb[ids[i + 1]] = emb[ids[i]].copy()
        rows.append((ids[i], ids[i + 1], 0.9, 0.8, True))
    # non-siblings: orthogonalise
    for i in range(0, n - 2, 4):
        a = emb[ids[i]]
        b = emb[ids[i + 2]]
        # remove the a-component from b so they are ~orthogonal -> cos ~ 0
        b = b - (np.dot(a, b) / (np.dot(a, a) + 1e-9)) * a
        emb[ids[i + 2]] = b.astype(np.float32)
        rows.append((ids[i], ids[i + 2], 0.1, 0.2, False))
    pairs = pd.DataFrame(rows, columns=["p1", "p2", "tm", "snn", "sibling"])
    return emb, pairs, ids


def _write_h5(path, emb, *, root_attrs=None):
    with h5py.File(path, "w") as f:
        if root_attrs:
            for k, v in root_attrs.items():
                f.attrs[k] = v
        for k, v in emb.items():
            f.create_dataset(k, data=np.asarray(v, dtype=np.float32))


def _write_pairs_tsv(path, pairs):
    """Write a Bromberg-schema pairs TSV (p1 p2 TM SNN siblings pident)."""
    lines = ["p1\tp2\tTM\tSNN\tsiblings\tpident"]
    for r in pairs.itertuples():
        lines.append(f"{r.p1}\t{r.p2}\t{r.tm}\t{r.snn}\t{r.sibling}\t50.0")
    path.write_text("\n".join(lines) + "\n")


def _write_freeze(path, ids):
    path.write_text(json.dumps({"set_name": "orphan_bromberg", "ids": ids,
                                "n_proteins": len(ids)}))


# ── library ───────────────────────────────────────────────────────────────────────────
def test_guards_contract_shape():
    assert ORPHAN_PER_PAIR_COLUMNS == ("pair_key", "p1", "p2", "cos", "snn", "tm", "sibling")
    assert ORPHAN_PARQUET_GUARDS["required_columns"] == ORPHAN_PER_PAIR_COLUMNS
    assert ORPHAN_PARQUET_GUARDS["unique_columns"] == ("pair_key",)
    assert ORPHAN_PARQUET_GUARDS["non_null_columns"] == ("pair_key", "p1", "p2", "sibling")
    assert ORPHAN_PARQUET_GUARDS["finite_columns"] == ("cos", "snn", "tm")


def test_report_writes_parquet_and_returns_manifest(tmp_path):
    emb, pairs, ids = _clean_monotone()
    manifest = orphan_correlation_report(
        emb, pairs, tmp_path, plm="toyplm", seed=42, n_boot=200, ci_alpha=0.1,
    )
    pq = tmp_path / "orphan_toyplm_raw_cosine.parquet"
    assert pq.exists()
    df = pd.read_parquet(pq)
    assert list(df.columns) == list(ORPHAN_PER_PAIR_COLUMNS)
    assert df["pair_key"].is_unique
    # pair_key == p1 + tab + p2
    assert (df["pair_key"] == df["p1"] + "\t" + df["p2"]).all()
    # manifest fields
    assert manifest["plm"] == "toyplm"
    assert manifest["representation"] == "raw"
    assert manifest["distance"] == "cosine"
    assert manifest["siblings_AUROC"] == pytest.approx(1.0)
    assert "ci_lo" in manifest and "ci_hi" in manifest
    assert "ci_degenerate" in manifest
    assert "percentile_diverged" in manifest
    assert "n_boot_undefined" in manifest
    assert "naive_ci_lo" in manifest and "naive_ci_hi" in manifest
    assert "spearman_cos_vs_SNN" in manifest and "spearman_cos_vs_TM" in manifest
    assert manifest["n_pairs"] == len(pairs)
    assert manifest["n_siblings"] == int(pairs["sibling"].sum())
    assert manifest["n_proteins"] == len(ids)
    assert manifest["population_n"] == len(ids)
    assert manifest["per_pair_columns"] == list(ORPHAN_PER_PAIR_COLUMNS)
    assert manifest["ci_note"]  # non-empty framing string
    assert "path" in manifest


def test_report_population_drift_raises(tmp_path):
    emb, pairs, ids = _clean_monotone()
    del emb[ids[0]]  # drop a frozen id from the embeddings
    with pytest.raises(OrphanPopulationError, match="missing"):
        orphan_correlation_report(
            emb, pairs, tmp_path, plm="toyplm", expected_ids=ids,
            seed=42, n_boot=100, ci_alpha=0.1,
        )


def test_report_allow_capped_tolerates_missing(tmp_path):
    emb, pairs, ids = _clean_monotone()
    del emb[ids[0]]
    # with allow_capped, a missing frozen id is tolerated (scored on the present subset)
    manifest = orphan_correlation_report(
        emb, pairs, tmp_path, plm="toyplm", expected_ids=ids, allow_capped=True,
        seed=42, n_boot=100, ci_alpha=0.1,
    )
    assert manifest["population_n"] < len(ids)


# ── relaxed provenance guard (R1): WARN if cap marker present, silent if absent ──────────
def test_provenance_warns_when_cap_marker_present(tmp_path):
    emb, pairs, ids = _clean_monotone()
    h5 = tmp_path / "toyplm.h5"
    _write_h5(h5, emb, root_attrs={"max_length_cap": 1024})
    freeze = tmp_path / "freeze.json"; _write_freeze(freeze, ids)
    pairs_tsv = tmp_path / "pairs.tsv"; _write_pairs_tsv(pairs_tsv, pairs)
    with pytest.warns(UserWarning, match="max_length_cap"):
        rc = orphan_main([
            "--plm", "toyplm", "--emb-h5", str(h5), "--pairs", str(pairs_tsv),
            "--freeze", str(freeze), "--out-dir", str(tmp_path),
            "--n-boot", "100", "--ci-alpha", "0.1",
        ])
    assert rc == 0  # WARN, not fail-closed


def test_provenance_silent_when_marker_absent(tmp_path):
    emb, pairs, ids = _clean_monotone()
    h5 = tmp_path / "toyplm.h5"
    _write_h5(h5, emb)  # NO root attrs -> marker absent
    freeze = tmp_path / "freeze.json"; _write_freeze(freeze, ids)
    pairs_tsv = tmp_path / "pairs.tsv"; _write_pairs_tsv(pairs_tsv, pairs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        rc = orphan_main([
            "--plm", "toyplm", "--emb-h5", str(h5), "--pairs", str(pairs_tsv),
            "--freeze", str(freeze), "--out-dir", str(tmp_path),
            "--n-boot", "100", "--ci-alpha", "0.1",
        ])
    assert rc == 0  # proceeds normally, no warning


# ── NEW-1 kernel input-sensitivity (a UNIT test, NOT a run gate, per R1) ─────────────────
def test_truncated_vs_uncapped_changes_auroc(tmp_path):
    # A truncated embedding (tail zeroed) changes the cosine -> changes AUROC. Pins that
    # the kernel is input-sensitive; it does NOT gate the run (design R1). Use a fixture
    # whose AUROC is NOT a saturated 1.0, so a single-vector perturbation can move it.
    rng = np.random.default_rng(7)
    n = 16
    ids = [f"P{i:02d}" for i in range(n)]
    emb = {pid: rng.normal(size=8).astype(np.float32) for pid in ids}
    rows = []
    # a mix of sibling/non-sibling pairs over noisy embeddings -> AUROC strictly inside (0,1)
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.4:
                rows.append((ids[a], ids[b], 0.5, 0.5, bool(rng.integers(0, 2))))
    pairs = pd.DataFrame(rows, columns=["p1", "p2", "tm", "snn", "sibling"])
    m_full = orphan_correlation_report(
        emb, pairs, tmp_path / "full", plm="toyplm", seed=42, n_boot=50, ci_alpha=0.1)
    assert 0.0 < m_full["siblings_AUROC"] < 1.0  # non-saturated -> sensitive

    # "truncate" the highest-degree protein: zero the tail half of its vector.
    deg = {pid: 0 for pid in ids}
    for r in pairs.itertuples():
        deg[r.p1] += 1
        deg[r.p2] += 1
    victim = max(deg, key=deg.get)
    emb_trunc = {k: v.copy() for k, v in emb.items()}
    tv = emb_trunc[victim].copy()
    tv[len(tv) // 2:] = 0.0
    emb_trunc[victim] = tv
    m_trunc = orphan_correlation_report(
        emb_trunc, pairs, tmp_path / "trunc", plm="toyplm", seed=42, n_boot=50, ci_alpha=0.1)
    assert m_full["siblings_AUROC"] != pytest.approx(m_trunc["siblings_AUROC"], abs=1e-9)


# ── CLI 3-exit-code battery ──────────────────────────────────────────────────────────────
def test_cli_exit0_writes_sidecar(tmp_path):
    emb, pairs, ids = _clean_monotone()
    h5 = tmp_path / "toyplm.h5"; _write_h5(h5, emb)
    freeze = tmp_path / "freeze.json"; _write_freeze(freeze, ids)
    pairs_tsv = tmp_path / "pairs.tsv"; _write_pairs_tsv(pairs_tsv, pairs)
    rc = orphan_main([
        "--plm", "toyplm", "--emb-h5", str(h5), "--pairs", str(pairs_tsv),
        "--freeze", str(freeze), "--out-dir", str(tmp_path),
        "--n-boot", "100", "--ci-alpha", "0.1",
    ])
    assert rc == 0
    sidecar = tmp_path / "orphan_toyplm_raw_cosine.manifest.json"
    assert sidecar.exists()
    m = json.loads(sidecar.read_text())
    assert m["plm"] == "toyplm" and m["distance"] == "cosine"
    # sidecar is json_safe-valid: a strict json reader (no NaN/Infinity tokens) parses it.
    json.loads(sidecar.read_text(), parse_constant=_reject_constants)


def _reject_constants(tok):
    raise AssertionError(f"non-JSON-spec constant in sidecar: {tok}")


def test_cli_exit2_on_missing_input(tmp_path):
    rc = orphan_main([
        "--plm", "x", "--emb-h5", str(tmp_path / "nope.h5"),
        "--pairs", str(tmp_path / "nope.tsv"), "--freeze", str(tmp_path / "nope.json"),
        "--out-dir", str(tmp_path),
    ])
    assert rc == 2


def test_cli_exit2_on_malformed_freeze(tmp_path):
    emb, pairs, ids = _clean_monotone()
    h5 = tmp_path / "toyplm.h5"; _write_h5(h5, emb)
    bad_freeze = tmp_path / "bad.json"; bad_freeze.write_text(json.dumps({"ids": []}))
    pairs_tsv = tmp_path / "pairs.tsv"; _write_pairs_tsv(pairs_tsv, pairs)
    rc = orphan_main([
        "--plm", "toyplm", "--emb-h5", str(h5), "--pairs", str(pairs_tsv),
        "--freeze", str(bad_freeze), "--out-dir", str(tmp_path),
    ])
    assert rc == 2


def test_cli_exit1_on_population_drift(tmp_path):
    emb, pairs, ids = _clean_monotone()
    del emb[ids[0]]  # drop a frozen id from the embeddings
    h5 = tmp_path / "toyplm.h5"; _write_h5(h5, emb)
    freeze = tmp_path / "freeze.json"; _write_freeze(freeze, ids)
    pairs_tsv = tmp_path / "pairs.tsv"; _write_pairs_tsv(pairs_tsv, pairs)
    rc = orphan_main([
        "--plm", "toyplm", "--emb-h5", str(h5), "--pairs", str(pairs_tsv),
        "--freeze", str(freeze), "--out-dir", str(tmp_path),
    ])
    assert rc == 1


def test_cli_parquet_honours_guards(tmp_path):
    # The written parquet must satisfy ORPHAN_PARQUET_GUARDS via the barrier's check_artifact.
    from evaluation.analysis_barrier import ArtifactSpec, check_artifact

    emb, pairs, ids = _clean_monotone()
    h5 = tmp_path / "toyplm.h5"; _write_h5(h5, emb)
    freeze = tmp_path / "freeze.json"; _write_freeze(freeze, ids)
    pairs_tsv = tmp_path / "pairs.tsv"; _write_pairs_tsv(pairs_tsv, pairs)
    rc = orphan_main([
        "--plm", "toyplm", "--emb-h5", str(h5), "--pairs", str(pairs_tsv),
        "--freeze", str(freeze), "--out-dir", str(tmp_path),
        "--n-boot", "100", "--ci-alpha", "0.1",
    ])
    assert rc == 0
    pq = tmp_path / "orphan_toyplm_raw_cosine.parquet"
    spec = ArtifactSpec(label="orphan:toyplm:cosine", path=str(pq), kind="parquet",
                        **ORPHAN_PARQUET_GUARDS)
    status = check_artifact(spec)
    assert status.ok, status
