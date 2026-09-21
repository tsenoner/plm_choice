"""
End-to-end run of the experimental-TM pipeline against synthetic reference data.

The defect the 2026-09-18 rewrite fixed was not a crash: the SIFTS parser came
back empty, the job logged "0 pairs have structures for both proteins" and
EXITED 0. Every unit test in tests/test_pdb_tmscore.py would have passed over a
pipeline that still did that, because none of them runs main(). So this file
runs main() and asserts a row comes out the far end with the right provenance.

Two things are stubbed and nothing else:
  * US-align -> a shell script that answers -v and prints one -outfmt 2 row, so
    the test needs no binary;
  * ProcessPoolExecutor -> a serial shim, so the suite does not spawn processes
    (the pool is not what is under test here).
Both mmCIF caches are pre-seeded, so the test never touches the network.
"""

import gzip
import json
import os
import stat

import polars as pl
import pytest

import data_preparation.pdb_tmscore as mod

_FAKE_USALIGN = """#!/bin/sh
if [ "$1" = "-v" ]; then
  echo "*********************************************************************"
  echo " * US-align (Version 20260527) *"
  exit 0
fi
printf '#PDBchain1\\tPDBchain2\\tTM1\\tTM2\\tRMSD\\tID1\\tID2\\tIDali\\tL1\\tL2\\tLali\\n'
printf 'a\\tb\\t0.8000\\t0.6000\\t2.00\\t0.50\\t0.40\\t0.30\\t30\\t30\\t28\\n'
"""

_CIF_HEADER = """data_TEST
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""


class _SerialPool:
    """Stand-in for ProcessPoolExecutor: same surface, no processes."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, iterable, chunksize=None):
        return map(fn, iterable)


def _write_cif(path, n_res=30):
    rows = [
        f"ATOM {i} C CA . ALA A {i} {float(i):.3f} 0.000 0.000 1.00 10.00 {i} A 1\n"
        for i in range(1, n_res + 1)
    ]
    with gzip.open(path, "wt") as fh:
        fh.write(_CIF_HEADER + "".join(rows) + "#\n")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A complete miniature of every input the pipeline reads."""
    monkeypatch.setattr(mod, "ProcessPoolExecutor", _SerialPool)

    # No test may reach the network. Every four-character id here is also a REAL
    # PDB entry, so without this the "structure is missing" case quietly fetches
    # the genuine 2xyz from RCSB and tests something else entirely.
    def _no_network(*a, **k):
        raise OSError("network disabled in tests")

    monkeypatch.setattr(mod.urllib.request, "urlretrieve", _no_network)

    usalign = tmp_path / "USalign"
    usalign.write_text(_FAKE_USALIGN)
    usalign.chmod(usalign.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    pairs = tmp_path / "pairs.parquet"
    pl.DataFrame(
        {
            "query": ["P12345", "P12345"],
            "target": ["Q99999", "P00000"],  # P00000 has no PDB entry -> dropped
            "fident": [0.42, 0.31],
            "hfsp": [30.0, 12.0],
            "alntmscore": [0.71, 0.55],
        }
    ).write_parquet(pairs)

    entry_sifts = tmp_path / "uniprot_pdb.tsv"
    entry_sifts.write_text(
        "# 2026/09/01 - SIFTS\n"
        "SP_PRIMARY\tPDB\n"
        "P12345\t1abc;3def\n"
        "Q99999\t2xyz\n"
    )

    # P12345 has two candidate chains: 1abc covers 100 residues at 2.5 A, 3def
    # covers 95 at 1.2 A. Which one wins is decided by --coverage_guard, which
    # is what makes this fixture able to change the pick WITHOUT changing the
    # pair -- exactly the situation a checkpoint must not paper over.
    chain_sifts = tmp_path / "pdb_chain_uniprot.tsv"
    chain_sifts.write_text(
        "# 2026/09/01 - SIFTS\n"
        "PDB\tCHAIN\tSP_PRIMARY\tRES_BEG\tRES_END\tPDB_BEG\tPDB_END\tSP_BEG\tSP_END\n"
        "1abc\tA\tP12345\t1\t100\t1\t100\t1\t100\n"
        "3def\tA\tP12345\t1\t95\t1\t95\t1\t95\n"
        "2xyz\tA\tQ99999\t1\t80\t1\t80\t1\t80\n"
    )

    entries_idx = tmp_path / "entries.idx"
    entries_idx.write_text(
        "IDCODE\tHEADER\tDATE\tCOMPOUND\tSOURCE\tAUTHORS\tRESOLUTION\tEXPERIMENT TYPE\n"
        "------\t------\t----\t--------\t------\t-------\t----------\t---------------\n"
        "1ABC\th\td\tc\ts\ta\t2.50\t\n"
        "3DEF\th\td\tc\ts\ta\t1.20\t\n"
        "2XYZ\th\td\tc\ts\ta\t2.00\t\n"
    )

    work = tmp_path / "work"
    cif_dir = work / "cache" / "cif"
    cif_dir.mkdir(parents=True)
    for pdb_id in ("1abc", "3def", "2xyz"):
        _write_cif(cif_dir / f"{pdb_id}.cif.gz")

    return {
        "tmp": tmp_path,
        "usalign": usalign,
        "pairs": pairs,
        "entry_sifts": entry_sifts,
        "chain_sifts": chain_sifts,
        "entries_idx": entries_idx,
        "work": work,
    }


def _run(world, monkeypatch, out_name="out.parquet", coverage_guard="0.90"):
    out = world["tmp"] / out_name
    monkeypatch.setattr(
        "sys.argv",
        [
            "pdb_tmscore.py",
            "--pairs_parquet", str(world["pairs"]),
            "--output_parquet", str(out),
            "--work_dir", str(world["work"]),
            "--entry_sifts", str(world["entry_sifts"]),
            "--chain_sifts", str(world["chain_sifts"]),
            "--entries_idx", str(world["entries_idx"]),
            "--usalign_path", str(world["usalign"]),
            "--resolution_cutoff", "3.0",
            "--coverage_guard", coverage_guard,
            "--max_workers", "1",
            "--download_workers", "1",
        ],
    )
    mod.main()
    attrition = json.loads((world["work"] / "attrition.json").read_text())
    return pl.read_parquet(out), attrition


def test_pipeline_produces_a_scored_row_with_full_provenance(world, monkeypatch):
    df, attrition = _run(world, monkeypatch)

    # The original failure mode was zero rows and exit 0.
    assert df.height == 1
    row = df.row(0, named=True)
    assert (row["query"], row["target"]) == ("P12345", "Q99999")

    # min(TM1, TM2) / max(TM1, TM2) / length-weighted mean of the stub's output.
    assert row["tmscore_exp"] == pytest.approx(0.60)
    assert row["tmscore_exp_short"] == pytest.approx(0.80)
    assert row["tmscore_exp_avg"] == pytest.approx(0.70)

    # Provenance, and the target column joined back from the pair table.
    assert row["q_pdb"] == "3def"  # 1.2 A beats 2.5 A inside the 0.90 guard
    assert row["q_chain"] == "A"
    assert row["q_method"] == "X-RAY DIFFRACTION"
    assert row["q_resolution"] == pytest.approx(1.20)
    assert row["q_n_ca"] == 30
    assert row["q_whole_chain_fallback"] is False
    assert row["t_pdb"] == "2xyz"
    assert row["alntmscore"] == pytest.approx(0.71)

    # The funnel, and the version/source stamps that let a rerun be compared.
    steps = dict(tuple(s) for s in attrition["steps"])
    assert steps["test-split pairs"] == 2
    assert steps["both accessions in entry-level SIFTS"] == 1
    assert steps["pairs with a US-align score"] == 1
    assert "20260527" in attrition["usalign_version"]
    assert attrition["resolution_cutoff"] == 3.0
    assert set(attrition["sources"]) == {
        "pairs_parquet", "entry_sifts", "chain_sifts", "entries_idx",
    }
    assert attrition["sources"]["chain_sifts"]["header"].startswith("# 2026/09/01")
    assert attrition["candidate_entry_methods"] == {"X-RAY DIFFRACTION": 3}


def test_rerun_reuses_the_checkpoint_when_nothing_changed(world, monkeypatch, caplog):
    _run(world, monkeypatch)
    with caplog.at_level("INFO"):
        df, attrition = _run(world, monkeypatch, out_name="out2.parquet")
    assert df.height == 1
    assert attrition["checkpoint_pairs_rescored_as_stale"] == 0
    assert "Resuming: 1 reusable pairs" in caplog.text
    assert "US-align on 0 pairs" in caplog.text


def test_rerun_rescores_when_the_representative_chain_changed(world, monkeypatch, caplog):
    """The one defect that could silently corrupt a published number.

    checkpoint.jsonl is keyed on (query, target). Tightening --coverage_guard
    makes select_representative_chain pick 1abc instead of 3def for P12345 --
    same pair, different structure. Without the stamp the cached 3def score is
    reused while the parquet's q_pdb/q_resolution columns describe 1abc, and
    nothing anywhere says so.
    """
    first, _ = _run(world, monkeypatch)
    assert first.row(0, named=True)["q_pdb"] == "3def"

    with caplog.at_level("WARNING"):
        second, attrition = _run(
            world, monkeypatch, out_name="out2.parquet", coverage_guard="0.99"
        )

    row = second.row(0, named=True)
    assert row["q_pdb"] == "1abc"  # 95 < 0.99 * 100, so 3def is out
    assert row["q_resolution"] == pytest.approx(2.50)
    assert attrition["checkpoint_pairs_rescored_as_stale"] == 1
    assert "different chains or a different US-align build" in caplog.text

    # Every checkpoint line carries the structures it was computed from.
    lines = [
        json.loads(ln)
        for ln in (world["work"] / "checkpoint.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    assert [rec["stamp"].split("|")[0] for rec in lines] == ["3def_A", "1abc_A"]


def test_an_empty_funnel_exits_nonzero_and_still_writes_the_attrition(world, monkeypatch):
    """The failure this module exists to not repeat: "0 pairs", exit 0.

    With one structure unobtainable no pair survives. The run must say so with a
    non-zero exit, and attrition.json -- the only record of WHERE the pairs went
    -- must be on disk, even though the parquet write is precisely the step that
    cannot complete.
    """
    os.remove(world["work"] / "cache" / "cif" / "2xyz.cif.gz")
    with pytest.raises(SystemExit) as excinfo:
        _run(world, monkeypatch)
    assert excinfo.value.code == 1

    attrition = json.loads((world["work"] / "attrition.json").read_text())
    steps = dict(tuple(s) for s in attrition["steps"])
    assert steps["pairs with a chain file for BOTH proteins"] == 0
    assert steps["pairs with a US-align score"] == 0
    assert attrition["reject_reasons"]["cif_download_failed"] == 1
    assert not (world["tmp"] / "out.parquet").exists()
