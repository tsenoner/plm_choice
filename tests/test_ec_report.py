import json

import numpy as np
import pandas as pd
import pytest

from evaluation.ec_report import (
    ec_dist_histogram,
    stratify_by_class,
)


def _pairs():
    return pd.DataFrame(
        {
            "a": ["P1", "P1", "P2"],
            "b": ["P2", "P3", "P3"],
            "dist": [0.1, 0.9, 0.5],
            "ec_dist": [0.0, 4.0, 4.0],
        }
    )


def _ec_class():
    # first EC field per protein
    return {"P1": "1", "P2": "1", "P3": "2"}


def test_histogram_counts_integer_bins():
    h = ec_dist_histogram(_pairs())
    assert h[0] == 1 and h[4] == 2


def test_stratify_within_vs_across_class():
    out = stratify_by_class(_pairs(), _ec_class())
    # within-class pairs: P1-P2 (both class 1). across: P1-P3, P2-P3.
    assert out["n_within"] == 1
    assert out["n_across"] == 2


def test_stratify_returns_real_stratum_tau_b():
    # A stratum with enough pairs to actually compute a correlation (the count-only
    # tests above never exercise the stratified statistic itself). Within class "1":
    # four proteins, embedding distance monotone in ec_dist -> within-class tau_b > 0.
    pairs = pd.DataFrame({
        "a": ["Q1", "Q1", "Q1", "Q2", "Q2", "Q3"],
        "b": ["Q2", "Q3", "Q4", "Q3", "Q4", "Q4"],
        "dist": [1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
        "ec_dist": [1.0, 2.0, 3.0, 1.0, 2.0, 1.0],  # == dist -> perfect monotone
    })
    ec_class = {"Q1": "1", "Q2": "1", "Q3": "1", "Q4": "1"}  # all same class
    out = stratify_by_class(pairs, ec_class)
    assert out["n_within"] == 6 and out["n_across"] == 0
    assert out["tau_b_within"] == pytest.approx(1.0)   # the real stratified statistic
    assert np.isnan(out["tau_b_across"])               # empty stratum -> NaN, not 0


from evaluation.ec_report import stratify_by_superfamily


def test_superfamily_within_across_and_nonhomologous_restriction():
    pairs = pd.DataFrame({
        "a": ["P1", "P1", "P2"],
        "b": ["P2", "P3", "P3"],
        "dist": [0.1, 0.9, 0.5],
        "ec_dist": [0.0, 4.0, 4.0],
    })
    # superfamily sets per protein (CATH multi-domain frozensets)
    sfam = {
        "P1": frozenset({"3.40.50.300"}),
        "P2": frozenset({"3.40.50.300"}),   # shares with P1 -> homologous
        "P3": frozenset({"1.10.10.10"}),
    }
    out = stratify_by_superfamily(pairs, sfam)
    assert out["n_within_superfamily"] == 1   # P1-P2
    assert out["n_across_superfamily"] == 2    # P1-P3, P2-P3
    # non-homologous restriction == across-superfamily subset
    assert out["n_nonhomologous"] == 2


from evaluation.ec_report import PopulationError, _build_matrices


def test_build_matrices_aligned_and_symmetric():
    emb = {
        "P1": np.array([0.0, 0.0], dtype=np.float32),
        "P2": np.array([3.0, 4.0], dtype=np.float32),
        "P3": np.array([0.0, 1.0], dtype=np.float32),
    }
    ec_labels = pd.DataFrame({
        "protein_id": ["P1", "P2", "P3"],
        "ec_set": [frozenset({"1.1.1.1"}), frozenset({"1.1.1.1"}), frozenset({"2.7.11.1"})],
    })
    ids, dist, ec, pairs = _build_matrices(
        emb, ec_labels, ["P1", "P2", "P3"], distance="euclidean", ec_set_agg="min")
    assert ids == ["P1", "P2", "P3"]
    assert dist.shape == (3, 3) and ec.shape == (3, 3)
    assert np.allclose(dist, dist.T) and np.allclose(ec, ec.T)
    assert dist[0, 1] == pytest.approx(5.0)   # P1-P2
    assert ec[0, 1] == 0.0                      # share 1.1.1.1
    assert ec[0, 2] == 4.0                      # class differs
    # pairs frame carries the per-pair long form
    assert set(pairs.columns) >= {"a", "b", "dist", "ec_dist"}


def test_build_matrices_population_drift_raises():
    emb = {"P1": np.array([0.0]), "P2": np.array([1.0])}  # missing P3
    ec_labels = pd.DataFrame({
        "protein_id": ["P1", "P2", "P3"],
        "ec_set": [frozenset({"1.1.1.1"})] * 3,
    })
    with pytest.raises(PopulationError, match="missing"):
        _build_matrices(emb, ec_labels, ["P1", "P2", "P3"],
                        distance="euclidean", ec_set_agg="min", allow_capped=False)


from evaluation.ec_report import ec_correlation_report


def _monotone_cohort(n=24, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.integers(0, 5, size=n)
    emb = {f"P{i:02d}": np.array([float(scores[i]), 0.0], dtype=np.float32) for i in range(n)}
    ec_labels = pd.DataFrame({
        "protein_id": [f"P{i:02d}" for i in range(n)],
        "ec_set": [frozenset({f"{scores[i]+1}.1.1.1"}) for i in range(n)],
    })
    return emb, ec_labels, [f"P{i:02d}" for i in range(n)]


def test_report_writes_parquet_and_returns_manifest(tmp_path):
    emb, ec_labels, ids = _monotone_cohort()
    manifest = ec_correlation_report(
        emb, ec_labels, tmp_path, plm="toyplm", distance="euclidean",
        ec_set_agg="min", wildcard_policy="exclude", statistic="tau_b",
        expected_ec_ids=ids, seed=42, n_boot=200, n_perm=100, ci_alpha=0.1,
    )
    # parquet written
    pq = tmp_path / "ec_toyplm_raw_euclidean.parquet"
    assert pq.exists()
    df = pd.read_parquet(pq)
    assert list(df.columns)[:5] == list(__import__("evaluation.ec_report", fromlist=["EC_PER_PAIR_COLUMNS"]).EC_PER_PAIR_COLUMNS)
    assert df["pair_key"].is_unique
    # manifest fields
    assert manifest["plm"] == "toyplm"
    assert manifest["statistic"] == "tau_b"
    assert manifest["ec_set_agg"] == "min"
    assert manifest["wildcard_policy"] == "exclude"
    assert manifest["tau_b"] > 0.5            # monotone cohort
    assert "ci_lo" in manifest and "ci_hi" in manifest
    assert "perm_p_value" in manifest
    assert "ec_dist_histogram" in manifest
    assert "sensitivity" in manifest and set(manifest["sensitivity"]) == {"min", "mean", "hausdorff"}
    assert manifest["per_pair_columns"] == list(manifest["per_pair_columns"])
    assert manifest["n_ec_proteins"] == len(ids)
    assert "path" in manifest


import h5py
from evaluation.ec_report import main as ec_main


def _write_h5(path, emb):
    with h5py.File(path, "w") as f:
        for k, v in emb.items():
            f.create_dataset(k, data=np.asarray(v, dtype=np.float32))


def _write_freeze(path, ids):
    path.write_text(json.dumps({"set_name": "ec_positive_subset", "ids": ids,
                                "n_proteins": len(ids)}))


def test_cli_exit0_writes_sidecar(tmp_path, capsys):
    emb, ec_labels, ids = _monotone_cohort()
    h5 = tmp_path / "toyplm.h5"; _write_h5(h5, emb)
    freeze = tmp_path / "ec_freeze.json"; _write_freeze(freeze, ids)
    tsv = tmp_path / "labels.tsv"
    # Build a UniProt-style TSV the CLI's parse_ec reads.
    pd.DataFrame({
        "Entry": ids,
        "Protein names": [f"enzyme (EC {list(s)[0]})" for s in ec_labels["ec_set"]],
    }).to_csv(tsv, sep="\t", index=False)
    rc = ec_main([
        "--plm", "toyplm", "--emb-h5", str(h5), "--freeze", str(freeze),
        "--ec-tsv", str(tsv), "--out-dir", str(tmp_path),
        "--distance", "euclidean", "--n-boot", "200", "--n-perm", "100", "--ci-alpha", "0.1",
    ])
    assert rc == 0
    sidecar = tmp_path / "ec_toyplm_raw_euclidean.manifest.json"
    assert sidecar.exists()
    m = json.loads(sidecar.read_text())
    assert m["plm"] == "toyplm" and m["statistic"] == "tau_b"


def test_cli_exit2_on_missing_input(tmp_path):
    rc = ec_main([
        "--plm", "x", "--emb-h5", str(tmp_path / "nope.h5"),
        "--freeze", str(tmp_path / "nope.json"), "--ec-tsv", str(tmp_path / "nope.tsv"),
        "--out-dir", str(tmp_path), "--distance", "euclidean",
    ])
    assert rc == 2


def test_cli_exit1_on_population_drift(tmp_path):
    emb, ec_labels, ids = _monotone_cohort()
    del emb[ids[0]]  # drop a frozen id from the embeddings
    h5 = tmp_path / "toyplm.h5"; _write_h5(h5, emb)
    freeze = tmp_path / "ec_freeze.json"; _write_freeze(freeze, ids)
    tsv = tmp_path / "labels.tsv"
    pd.DataFrame({"Entry": ids,
                  "Protein names": [f"enzyme (EC {list(s)[0]})" for s in ec_labels["ec_set"]]
                  }).to_csv(tsv, sep="\t", index=False)
    rc = ec_main(["--plm", "toyplm", "--emb-h5", str(h5), "--freeze", str(freeze),
                  "--ec-tsv", str(tsv), "--out-dir", str(tmp_path), "--distance", "euclidean"])
    assert rc == 1


def test_cli_exit2_on_malformed_freeze(tmp_path):
    # A freeze with an empty 'ids' list -> load_frozen_ids raises ValueError -> exit 2
    # (the ValueError arm of the exit-code matrix, distinct from the I/O FileNotFound arm).
    emb, ec_labels, ids = _monotone_cohort()
    h5 = tmp_path / "toyplm.h5"; _write_h5(h5, emb)
    bad_freeze = tmp_path / "bad.json"; bad_freeze.write_text(json.dumps({"ids": []}))
    tsv = tmp_path / "labels.tsv"
    pd.DataFrame({"Entry": ids,
                  "Protein names": [f"enzyme (EC {list(s)[0]})" for s in ec_labels["ec_set"]]
                  }).to_csv(tsv, sep="\t", index=False)
    rc = ec_main(["--plm", "toyplm", "--emb-h5", str(h5), "--freeze", str(bad_freeze),
                  "--ec-tsv", str(tsv), "--out-dir", str(tmp_path), "--distance", "euclidean"])
    assert rc == 2


def test_cli_with_ec_col_and_superfamily_populates_strata(tmp_path):
    # Integration: the structured --ec-col path AND the --superfamily-source homology
    # control are both reachable from the CLI and populate the manifest strata (D8 + D9).
    emb, ec_labels, ids = _monotone_cohort()
    h5 = tmp_path / "toyplm.h5"; _write_h5(h5, emb)
    freeze = tmp_path / "ec_freeze.json"; _write_freeze(freeze, ids)
    # Structured EC column (not the name regex).
    tsv = tmp_path / "labels.tsv"
    pd.DataFrame({"Entry": ids,
                  "EC number": [list(s)[0] for s in ec_labels["ec_set"]]}).to_csv(
        tsv, sep="\t", index=False)
    # CATH labels TSV (Gene3D column) for the superfamily control: give two superfamilies.
    cath = tmp_path / "cath.tsv"
    pd.DataFrame({
        "Entry": ids,
        "Gene3D": ["3.40.50.300" if i % 2 == 0 else "1.10.10.10" for i in range(len(ids))],
    }).to_csv(cath, sep="\t", index=False)
    rc = ec_main([
        "--plm", "toyplm", "--emb-h5", str(h5), "--freeze", str(freeze),
        "--ec-tsv", str(tsv), "--ec-col", "EC number",
        "--superfamily-source", str(cath), "--out-dir", str(tmp_path),
        "--distance", "euclidean", "--n-boot", "200", "--n-perm", "100", "--ci-alpha", "0.1",
    ])
    assert rc == 0
    m = json.loads((tmp_path / "ec_toyplm_raw_euclidean.manifest.json").read_text())
    assert m["stratify_superfamily"] != {}          # homology control actually ran
    assert "n_within_superfamily" in m["stratify_superfamily"]
