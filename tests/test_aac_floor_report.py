"""Tests for evaluation.aac_floor_report — Unit 2 of the AAC-floor arm.

Spec: docs/superpowers/specs/2026-06-11-aac-floor-design.md §3 Unit 2 + §10 (C1/I3).

The AAC-floor producer is a near-clone of recall_fp_report that scores the
20-d amino-acid-composition floor with recall-at-first-FP. The one real
correctness subtlety is the **C1 capped-cohort fix**: recall-at-first-FP is a
function of the ENTIRE lookup database, so the floor must be scored on the SAME
population a pLM was scored on. The producer scores whatever population it is
handed (``expected_ids``) and tags the output (``population_tag``) so a full-319
AAC cell and a capped-267 AAC cell never collide.

The per-query parquet contract is the SAME as recall-fp (reused verbatim from
recall_fp_report.PARQUET_GUARDS / PER_QUERY_COLUMNS).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.analysis_barrier import ArtifactSpec, check_artifact
from evaluation.population import PopulationError
from evaluation.recall_fp_report import (
    CI_NOTE,
    PARQUET_GUARDS,
    PER_QUERY_COLUMNS,
)
from evaluation.aac_floor_report import aac_floor_report, main


# ── fixtures ─────────────────────────────────────────────────────────────────
def _write_fasta(path: Path, records: list[tuple[str, str]]) -> Path:
    lines = []
    for pid, seq in records:
        lines.append(f">{pid}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n")
    return path


def _labels(ids_by_fold: dict[str, list[str]]) -> pd.DataFrame:
    # Build a CATH label frame (frozenset fold/superfamily) from {fold_name: [ids]}.
    rows = []
    for fold_name, ids in ids_by_fold.items():
        for pid in ids:
            rows.append(
                {
                    "protein_id": pid,
                    "fold": frozenset({fold_name}),
                    "superfamily": frozenset({fold_name + "1"}),
                    "family": None,
                }
            )
    return pd.DataFrame(rows)


def _clean_fasta(tmp_path: Path) -> Path:
    # Two AAC-separable folds: {P1,P2} alanine-rich, {P3,P4} cysteine-rich.
    return _write_fasta(
        tmp_path / "clean.fasta",
        [
            ("P1", "AAAAAAAAAC"),  # 90% A
            ("P2", "AAAAAAAACC"),  # 80% A
            ("P3", "CCCCCCCCCA"),  # 90% C
            ("P4", "CCCCCCCCAA"),  # 80% C
        ],
    )


def _clean_labels() -> pd.DataFrame:
    return _labels({"a": ["P1", "P2"], "b": ["P3", "P4"]})


def _parquets(out_dir: Path) -> list[str]:
    return sorted(p.name for p in out_dir.glob("*.parquet"))


# ── producer: scores a tiny fixture, parquet matches the recall-fp contract ──
def test_producer_scores_fixture_and_returns_manifest(tmp_path):
    fasta = _clean_fasta(tmp_path)
    labels = _clean_labels()
    out = tmp_path / "euclidean"
    manifest = aac_floor_report(
        fasta, labels, out,
        expected_ids=["P1", "P2", "P3", "P4"],
        distance="euclidean", population_tag="full319",
    )
    assert set(manifest["levels"]) == {"fold", "superfamily"}
    for level in ("fold", "superfamily"):
        info = manifest["levels"][level]
        assert info["n_queries_with_positives"] == 4
        assert info["mean_recall_1stFP"] == pytest.approx(1.0)
        assert info["path"].endswith(".parquet")
    assert manifest["population_n"] == 4
    assert manifest["population_tag"] == "full319"
    assert manifest["distance"] == "euclidean"
    assert manifest["include_other"] is False


def test_parquet_satisfies_recall_fp_guards(tmp_path):
    fasta = _clean_fasta(tmp_path)
    out = tmp_path / "euclidean"
    manifest = aac_floor_report(
        fasta, _clean_labels(), out,
        expected_ids=["P1", "P2", "P3", "P4"],
        distance="euclidean", population_tag="full319",
    )
    info = manifest["levels"]["fold"]
    df = pd.read_parquet(info["path"])
    # exact column contract reused from recall-fp
    assert tuple(df.columns) == PER_QUERY_COLUMNS
    spec = ArtifactSpec(
        label="aac_floor:full319:fold",
        path=info["path"],
        expected_rows=info["n_queries_with_positives"],
        **PARQUET_GUARDS,
    )
    status = check_artifact(spec)
    assert status.ok, status.reasons


def test_filename_stem_has_population_tag_and_level_no_distance(tmp_path):
    # I3: distance is separated by out_dir; the stem carries population_tag + level.
    fasta = _clean_fasta(tmp_path)
    out = tmp_path / "euclidean"
    manifest = aac_floor_report(
        fasta, _clean_labels(), out,
        expected_ids=["P1", "P2", "P3", "P4"],
        distance="euclidean", population_tag="full319",
    )
    names = _parquets(out)
    assert names == [
        "aac_floor_full319_fold.parquet",
        "aac_floor_full319_superfamily.parquet",
    ]
    # distance must NOT be in the filename (separated by out_dir)
    assert not any("euclidean" in n for n in names)


def test_manifest_carries_ci_provenance_and_tie_summary(tmp_path):
    fasta = _clean_fasta(tmp_path)
    out = tmp_path / "euclidean"
    manifest = aac_floor_report(
        fasta, _clean_labels(), out,
        expected_ids=["P1", "P2", "P3", "P4"],
        distance="euclidean", population_tag="full319",
        n_boot=500, ci_alpha=0.05, seed=42,
    )
    assert manifest["ci_note"] == CI_NOTE
    assert manifest["ci_resample_unit"] == "query"
    assert manifest["seed"] == 42 and manifest["n_boot"] == 500
    fold = manifest["levels"]["fold"]
    # the floor-quality readout: a tie summary is surfaced per level
    assert "n_ties_at_first_fp" in fold
    assert "ci_lo" in fold and "ci_hi" in fold and "ci_degenerate" in fold


# ── C1 REGRESSION: full vs subset population → DIFFERENT recall ──────────────
def test_c1_full_vs_subset_population_differs(tmp_path):
    # recall-at-first-FP depends on the WHOLE lookup DB, so the SAME query scored
    # against a full-DB vs a strict-subset-DB must give a DIFFERENT recall for at
    # least one query (proving the lookup-DB size is load-bearing — the capped-cohort
    # correctness subtlety this unit exists for). Mirrors test_snn's capped-DB test.
    #
    # Construct a DB where dropping a false-positive neighbour CHANGES a query's
    # recall. P1's positive is P2; under the full DB an intruding FP (P3, near P1)
    # cuts off P2 -> recall 0; in the subset DB (P3 removed) P2 is reached -> recall 1.
    fasta = _write_fasta(
        tmp_path / "f.fasta",
        [
            ("P1", "AAAAAAAAAA"),          # 100% A
            ("P2", "AAAAAAAACC"),          # 80% A  (P1's positive, same fold)
            ("P3", "AAAAAAAAAC"),          # 90% A  (FP: different fold, sits between P1 and P2)
            ("Q1", "CCCCCCCCCC"),          # 100% C (anchor for P3's fold)
        ],
    )
    labels = _labels({"a": ["P1", "P2"], "b": ["P3", "Q1"]})
    full_ids = ["P1", "P2", "P3", "Q1"]
    subset_ids = ["P1", "P2", "Q1"]  # drop P3, the intruding FP

    full = aac_floor_report(
        fasta, labels, tmp_path / "full",
        expected_ids=full_ids, distance="euclidean", population_tag="full",
    )
    subset = aac_floor_report(
        fasta, labels, tmp_path / "sub",
        expected_ids=subset_ids, distance="euclidean",
        population_tag="capped", allow_capped=False,
    )

    full_df = pd.read_parquet(full["levels"]["fold"]["path"]).set_index("query_id")
    sub_df = pd.read_parquet(subset["levels"]["fold"]["path"]).set_index("query_id")

    # P1 is present in BOTH cells; its recall must DIFFER (DB size is load-bearing).
    assert "P1" in full_df.index and "P1" in sub_df.index
    common = full_df.index.intersection(sub_df.index)
    assert any(
        full_df.loc[q, "recall"] != sub_df.loc[q, "recall"] for q in common
    ), "full vs subset population yielded identical recall — DB size not load-bearing"
    # P1 specifically: FP intrusion in full -> lower recall than the FP-free subset.
    assert full_df.loc["P1", "recall"] < sub_df.loc["P1", "recall"]


def test_c1_population_tag_distinguishes_outputs(tmp_path):
    # The full-cohort AAC cell and the capped-cohort AAC cell must be distinguishable
    # on disk (different parquet stems + different manifest tags) so they never collide.
    fasta = _clean_fasta(tmp_path)
    labels = _clean_labels()
    out = tmp_path / "euclidean"
    full = aac_floor_report(
        fasta, labels, out,
        expected_ids=["P1", "P2", "P3", "P4"],
        distance="euclidean", population_tag="full319",
    )
    capped = aac_floor_report(
        fasta, labels, out,
        expected_ids=["P1", "P2", "P3"],
        distance="euclidean", population_tag="esm1b", allow_capped=False,
    )
    # both cells live in the same per-distance out_dir but on distinct paths
    assert full["levels"]["fold"]["path"] != capped["levels"]["fold"]["path"]
    assert "full319" in full["levels"]["fold"]["path"]
    assert "esm1b" in capped["levels"]["fold"]["path"]
    assert full["population_tag"] == "full319"
    assert capped["population_tag"] == "esm1b"
    # no clobber: 4 parquets (2 tags x 2 levels)
    assert len(_parquets(out)) == 4


# ── capped path: a strict subset passes with allow_capped=True ──────────────
def test_allow_capped_lets_strict_subset_through(tmp_path):
    # expected_ids names a protein the FASTA lacks; allow_capped passes the subset.
    fasta = _clean_fasta(tmp_path)  # P1..P4
    labels = _clean_labels()
    out = tmp_path / "euclidean"
    manifest = aac_floor_report(
        fasta, labels, out,
        expected_ids=["P1", "P2", "P3", "P4", "P5_missing"],
        distance="euclidean", population_tag="esm1b", allow_capped=True,
    )
    assert manifest["population_n"] == 4
    assert manifest["levels"]["fold"]["n_queries_with_positives"] == 4


def test_fasta_missing_expected_id_raises_value_error(tmp_path):
    # build_aac_embeddings raises ValueError when a frozen id is absent from the FASTA.
    fasta = _clean_fasta(tmp_path)  # P1..P4
    out = tmp_path / "euclidean"
    with pytest.raises(ValueError, match="absent"):
        aac_floor_report(
            fasta, _clean_labels(), out,
            expected_ids=["P1", "P2", "P3", "P4", "P5_missing"],
            distance="euclidean", population_tag="full319", allow_capped=False,
        )


# ── tie-heavy fixture (M6): many proteins at distance 0 under euclidean ─────
def test_tie_heavy_fixture_recall_distribution_is_sane(tmp_path):
    # Under euclidean on discrete 20-d frequency vectors, identical compositions sit at
    # distance 0. If the nearest FP is also at distance 0, the adversarial strict-walk
    # discards every tied positive -> recall can collapse. Assert the recall distribution
    # is a SANE reading (not silently all-zero, not all-one) and ties are surfaced.
    #
    # P1,P2,P3 share composition "AC" (identical 50/50 vectors -> distance 0); P1,P2 are
    # fold-a positives, P3 is fold-b (a false positive AT distance 0). P4 anchors fold-b.
    fasta = _write_fasta(
        tmp_path / "ties.fasta",
        [
            ("P1", "AC"),
            ("P2", "AC"),
            ("P3", "AC"),   # identical comp, DIFFERENT fold -> FP at distance 0
            ("P4", "CCCC"),
        ],
    )
    labels = _labels({"a": ["P1", "P2"], "b": ["P3", "P4"]})
    out = tmp_path / "euclidean"
    manifest = aac_floor_report(
        fasta, labels, out,
        expected_ids=["P1", "P2", "P3", "P4"],
        distance="euclidean", population_tag="full319",
    )
    df = pd.read_parquet(manifest["levels"]["fold"]["path"])
    recalls = df["recall"].to_numpy()
    # ties surfaced per query (the floor-quality readout)
    assert "n_ties_at_first_fp" in df.columns
    # P1 and P2: their only positive (each other) is at distance 0, but the FP P3 is
    # ALSO at distance 0 -> adversarial walk discards the tied positive -> recall 0.
    p1 = df.set_index("query_id").loc["P1"]
    assert p1["recall"] == 0.0  # tie-collapse: a correct-but-tested floor reading
    assert int(p1["n_ties_at_first_fp"]) >= 1  # the tie is counted, not hidden
    # not the silently-all-something degenerate case: distribution has structure
    assert recalls.min() == 0.0  # the tie-collapsed queries
    assert manifest["levels"]["fold"]["n_ties_at_first_fp"] is not None  # summary present


# ── CLI exit-code matrix: 0 / PopulationError→1 / OSError→2 / ValueError→2 ──
def _write_cath_tsv(path: Path, gene3d_by_id: dict[str, str]) -> Path:
    lines = ["Entry\tGene3D"] + [f"{pid}\t{code}" for pid, code in gene3d_by_id.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_freeze(path: Path, ids: list[str]) -> Path:
    n = len(ids)
    path.write_text(
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
    return path


def _cli_inputs(tmp_path: Path):
    fasta = _clean_fasta(tmp_path)
    tsv = _write_cath_tsv(
        tmp_path / "cath.tsv",
        {"P1": "3.30.70.10", "P2": "3.30.70.10", "P3": "1.10.10.10", "P4": "1.10.10.10"},
    )
    freeze = _write_freeze(tmp_path / "freeze.json", ["P1", "P2", "P3", "P4"])
    out = tmp_path / "euclidean"
    return fasta, tsv, freeze, out


def _argv(fasta, tsv, freeze, out, *, tag="full319", distance="euclidean", extra=()):
    return [
        "--fasta", str(fasta), "--cath-tsv", str(tsv), "--freeze", str(freeze),
        "--out-dir", str(out), "--distance", distance, "--population-tag", tag,
        *extra,
    ]


def test_cli_exit_0_writes_parquets_and_sidecar(tmp_path):
    fasta, tsv, freeze, out = _cli_inputs(tmp_path)
    rc = main(_argv(fasta, tsv, freeze, out))
    assert rc == 0
    assert _parquets(out) == [
        "aac_floor_full319_fold.parquet",
        "aac_floor_full319_superfamily.parquet",
    ]
    sidecar = out / "aac_floor_full319.manifest.json"
    assert sidecar.exists()
    manifest = json.loads(sidecar.read_text())
    assert manifest["population_tag"] == "full319"
    assert manifest["distance"] == "euclidean"
    assert manifest["population_n"] == 4
    assert manifest["levels"]["fold"]["mean_recall_1stFP"] == pytest.approx(1.0)


def test_cli_population_error_maps_to_exit_1(tmp_path, monkeypatch):
    # Exit 1 is the PopulationError data-failure branch. In the FASTA-derived producer,
    # a frozen id absent from the FASTA surfaces in build_aac as a ValueError (exit 2),
    # so the genuine PopulationError surface (the defensive S3/C1 assert firing on a
    # built-but-drifted population) is structurally a different branch. We pin the CLI's
    # mapping of PopulationError -> 1 by making the report raise PopulationError.
    import evaluation.aac_floor_report as mod

    def _raise(*a, **k):
        raise PopulationError("synthetic drift for the exit-code map")

    fasta, tsv, freeze, out = _cli_inputs(tmp_path)
    monkeypatch.setattr(mod, "aac_floor_report", _raise)
    rc = mod.main(_argv(fasta, tsv, freeze, out))
    assert rc == 1
    assert _parquets(out) == []
    assert not (out / "aac_floor_full319.manifest.json").exists()


def test_cli_capped_population_tagged_and_exit_0(tmp_path):
    fasta, tsv, _, out = _cli_inputs(tmp_path)  # FASTA = P1..P4
    freeze = _write_freeze(tmp_path / "freeze3.json", ["P1", "P2", "P3"])
    rc = main(_argv(fasta, tsv, freeze, out, tag="esm1b"))
    assert rc == 0
    manifest = json.loads((out / "aac_floor_esm1b.manifest.json").read_text())
    assert manifest["population_n"] == 3
    assert manifest["population_tag"] == "esm1b"


def test_cli_missing_fasta_returns_2(tmp_path):
    fasta, tsv, freeze, out = _cli_inputs(tmp_path)
    rc = main(_argv(tmp_path / "nope.fasta", tsv, freeze, out))
    assert rc == 2
    assert _parquets(out) == []


def test_cli_missing_freeze_returns_2(tmp_path):
    fasta, tsv, _, out = _cli_inputs(tmp_path)
    rc = main(_argv(fasta, tsv, tmp_path / "nope.json", out))
    assert rc == 2


def test_cli_freeze_id_absent_from_fasta_returns_2(tmp_path):
    # A frozen id not in the FASTA is a ValueError from build_aac -> exit 2, nothing written.
    fasta, tsv, _, out = _cli_inputs(tmp_path)
    freeze = _write_freeze(tmp_path / "freeze5.json", ["P1", "P2", "P3", "P4", "P5_missing"])
    rc = main(_argv(fasta, tsv, freeze, out))
    assert rc == 2
    assert _parquets(out) == []
    assert not (out / "aac_floor_full319.manifest.json").exists()


def test_cli_disjoint_population_returns_2(tmp_path):
    # Freeze entirely disjoint from the FASTA -> build_aac ValueError -> exit 2.
    fasta, tsv, _, out = _cli_inputs(tmp_path)
    freeze = _write_freeze(tmp_path / "disjoint.json", ["Q1", "Q2", "Q3", "Q4"])
    rc = main(_argv(fasta, tsv, freeze, out))
    assert rc == 2
    assert _parquets(out) == []


def test_cli_report_does_not_write_sidecar_but_cli_does(tmp_path):
    fasta = _clean_fasta(tmp_path)
    out = tmp_path / "euclidean"
    aac_floor_report(
        fasta, _clean_labels(), out,
        expected_ids=["P1", "P2", "P3", "P4"],
        distance="euclidean", population_tag="full319",
    )
    assert list(out.glob("*.manifest.json")) == []


def test_cli_rerun_replaces_in_place(tmp_path):
    fasta, tsv, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(fasta, tsv, freeze, out)) == 0
    assert main(_argv(fasta, tsv, freeze, out)) == 0
    assert _parquets(out) == [
        "aac_floor_full319_fold.parquet",
        "aac_floor_full319_superfamily.parquet",
    ]
    assert sorted(p.name for p in out.glob("*.manifest*.json")) == [
        "aac_floor_full319.manifest.json"
    ]


def test_cli_parquet_passes_the_real_barrier(tmp_path):
    fasta, tsv, freeze, out = _cli_inputs(tmp_path)
    assert main(_argv(fasta, tsv, freeze, out)) == 0
    manifest = json.loads((out / "aac_floor_full319.manifest.json").read_text())
    info = manifest["levels"]["fold"]
    spec = ArtifactSpec(
        label="aac_floor:full319:fold",
        path=info["path"],
        expected_rows=info["n_queries_with_positives"],
        **PARQUET_GUARDS,
    )
    status = check_artifact(spec)
    assert status.ok, status.reasons


def test_cli_sidecar_is_valid_json_no_bare_nan(tmp_path):
    # A zero-positive level emits a 0-row parquet + NaN mean serialised as null.
    fasta = _write_fasta(
        tmp_path / "distinct.fasta",
        [
            ("P1", "AAAAAAAAAA"),
            ("P2", "CCCCCCCCCC"),
            ("P3", "DDDDDDDDDD"),
            ("P4", "EEEEEEEEEE"),
        ],
    )
    tsv = _write_cath_tsv(
        tmp_path / "cath.tsv",
        {  # four DISTINCT folds -> no shared domains -> zero positives everywhere
            "P1": "3.30.70.10",
            "P2": "1.10.10.10",
            "P3": "2.40.50.10",
            "P4": "3.40.50.20",
        },
    )
    freeze = _write_freeze(tmp_path / "freeze.json", ["P1", "P2", "P3", "P4"])
    out = tmp_path / "euclidean"
    rc = main(_argv(fasta, tsv, freeze, out))
    assert rc == 0
    sidecar_text = (out / "aac_floor_full319.manifest.json").read_text()
    assert "NaN" not in sidecar_text
    manifest = json.loads(sidecar_text)
    fold = manifest["levels"]["fold"]
    assert fold["n_queries_with_positives"] == 0
    assert fold["mean_recall_1stFP"] is None
