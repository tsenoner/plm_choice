"""Tests for evaluation.canonical_set — the canonical-set freeze (plan v3, Phase 0 item 1 + NEW-3).

The whole pLM comparison is defined over one frozen protein set (the canonical 319,
``2024_novelSeqs2.fasta``). This module produces the freeze that every analysis asserts
against and that all pairwise metrics share as a common pair index. The freeze must:

* hash the set in a *normalization-invariant* way (sequence-set drift, not cosmetic
  reformatting — line-wrap width, header whitespace, record order — must change the hash);
* emit the sorted canonical id list (``assert_population``'s ``expected``);
* emit the frozen common pair index (all C(n, 2) unordered pairs, ``id_a < id_b``);
* record esm1b's architecture-capped coverage (NEW-3): a strict 267/319 subset whose 52
  absent ids are all > 1022 aa, so a caller can ``allow_capped`` esm1b and report n=267.
"""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from evaluation.canonical_set import (
    build_pair_index,
    canonical_content_sha256,
    freeze_canonical_set,
    main,
    parse_fasta,
    raw_file_sha256,
    write_freeze,
)


# ── fixtures ────────────────────────────────────────────────────────────────────
def _write(path, text):
    path.write_text(text)
    return path


@pytest.fixture
def tiny_fasta(tmp_path):
    # ids deliberately out of sorted order; B has a wrapped multi-line sequence.
    return _write(
        tmp_path / "tiny.fasta",
        ">C\nMKVL\n>A extra description here\nGGGG\n>B\nMKWL\nKKPP\n",
    )


# ── parse_fasta ───────────────────────────────────────────────────────────────
def test_parse_fasta_reads_id_and_sequence(tiny_fasta):
    recs = dict(parse_fasta(tiny_fasta))
    assert recs == {"C": "MKVL", "A": "GGGG", "B": "MKWLKKPP"}


def test_parse_fasta_id_is_first_whitespace_token(tiny_fasta):
    ids = [i for i, _ in parse_fasta(tiny_fasta)]
    assert ids == ["C", "A", "B"]  # "A extra description" -> "A"


def test_parse_fasta_empty_file_returns_empty(tmp_path):
    assert parse_fasta(_write(tmp_path / "e.fasta", "")) == []


def test_parse_fasta_tolerates_crlf(tmp_path):
    fa = _write(tmp_path / "crlf.fasta", ">A desc\r\nMK\r\nVL\r\n>B\r\nGG\r\n")
    assert dict(parse_fasta(fa)) == {"A": "MKVL", "B": "GG"}


def test_parse_fasta_raises_on_header_with_no_id(tmp_path):
    with pytest.raises(ValueError, match="no id"):
        parse_fasta(_write(tmp_path / "bad.fasta", ">\nMKVL\n"))


def test_parse_fasta_raises_on_sequence_before_first_header(tmp_path):
    with pytest.raises(ValueError, match="before the first FASTA header"):
        parse_fasta(_write(tmp_path / "lead.fasta", "MKVL\n>A\nGGGG\n"))


# ── canonical_content_sha256 ────────────────────────────────────────────────────
def test_content_hash_invariant_to_record_order():
    a = canonical_content_sha256([("A", "GGGG"), ("B", "MKWL")])
    b = canonical_content_sha256([("B", "MKWL"), ("A", "GGGG")])
    assert a == b


def test_content_hash_invariant_to_case():
    assert canonical_content_sha256([("A", "ggg")]) == canonical_content_sha256(
        [("A", "GGG")]
    )


def test_content_hash_changes_when_a_sequence_changes():
    a = canonical_content_sha256([("A", "GGGG"), ("B", "MKWL")])
    b = canonical_content_sha256([("A", "GGGC"), ("B", "MKWL")])
    assert a != b


def test_content_hash_changes_when_an_id_changes():
    a = canonical_content_sha256([("A", "GGGG")])
    b = canonical_content_sha256([("Z", "GGGG")])
    assert a != b


def test_content_hash_is_deterministic_known_value():
    # Pin the exact recipe so a future refactor that changes the wire format is caught.
    expected = hashlib.sha256(b"A\tGGGG\nB\tMKWL\n").hexdigest()
    assert canonical_content_sha256([("B", "MKWL"), ("A", "GGGG")]) == expected


def test_content_hash_raises_on_duplicate_id():
    with pytest.raises(ValueError, match="duplicate"):
        canonical_content_sha256([("A", "GGGG"), ("A", "MKWL")])


# ── raw_file_sha256 ─────────────────────────────────────────────────────────────
def test_raw_file_sha256_matches_hashlib(tiny_fasta):
    expected = hashlib.sha256(tiny_fasta.read_bytes()).hexdigest()
    assert raw_file_sha256(tiny_fasta) == expected


# ── build_pair_index ────────────────────────────────────────────────────────────
def test_pair_index_count_is_n_choose_2():
    df = build_pair_index(["A", "B", "C", "D"])
    assert len(df) == 6  # C(4, 2)


def test_pair_index_columns_and_ordering():
    df = build_pair_index(["B", "A", "C"])
    assert list(df.columns) == ["id_a", "id_b"]
    # every pair is canonicalised id_a < id_b ...
    assert (df["id_a"] < df["id_b"]).all()
    # ... and rows are sorted deterministically by (id_a, id_b).
    assert df.to_records(index=False).tolist() == [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    ]


def test_pair_index_no_duplicate_pairs():
    df = build_pair_index(["A", "B", "C", "D", "E"])
    assert not df.duplicated().any()


def test_pair_index_raises_on_duplicate_input_id():
    with pytest.raises(ValueError, match="duplicate"):
        build_pair_index(["A", "B", "A"])


def test_pair_index_single_id_is_empty():
    df = build_pair_index(["A"])
    assert len(df) == 0
    assert list(df.columns) == ["id_a", "id_b"]


# ── freeze_canonical_set ─────────────────────────────────────────────────────────
def test_freeze_manifest_core_fields(tiny_fasta):
    m = freeze_canonical_set(tiny_fasta, set_name="tiny")
    assert m["set_name"] == "tiny"
    assert m["n_proteins"] == 3
    assert m["n_pairs"] == 3  # C(3, 2)
    assert m["source_uri"] is None
    assert m["ids"] == ["A", "B", "C"]  # sorted
    assert m["raw_file_sha256"] == raw_file_sha256(tiny_fasta)
    assert m["canonical_content_sha256"] == canonical_content_sha256(
        parse_fasta(tiny_fasta)
    )
    assert m["source_fasta"] == "tiny.fasta"


def test_freeze_records_source_uri_when_given(tiny_fasta):
    m = freeze_canonical_set(tiny_fasta, set_name="tiny", source_uri="lrz:/path/x.fasta")
    assert m["source_uri"] == "lrz:/path/x.fasta"


def test_freeze_raises_on_empty_fasta(tmp_path):
    with pytest.raises(ValueError, match="cannot be empty"):
        freeze_canonical_set(_write(tmp_path / "e.fasta", ""), set_name="e")


def test_freeze_n_pairs_matches_built_index(tmp_path):
    # The barrier rides on manifest n_pairs == len(pair_index); pin it past the n=3 case.
    fa = _write(tmp_path / "five.fasta", "".join(f">{c}\nGG\n" for c in "ABCDE"))
    m = freeze_canonical_set(fa, set_name="five")
    assert m["n_pairs"] == 10  # C(5, 2)
    assert len(build_pair_index(m["ids"])) == m["n_pairs"]


def test_freeze_raises_on_duplicate_id(tmp_path):
    dup = _write(tmp_path / "dup.fasta", ">A\nGGGG\n>A\nMKWL\n")
    with pytest.raises(ValueError, match="duplicate"):
        freeze_canonical_set(dup, set_name="dup")


def test_freeze_records_esm1b_capped_coverage(tiny_fasta):
    # esm1b covers only {A, B}; C is "missing". Provide lengths so the freeze can
    # attribute the absence to the > cap_aa architecture cap.
    m = freeze_canonical_set(
        tiny_fasta,
        set_name="tiny",
        esm1b_keys=["A", "B"],
        cap_aa=4,  # "C" would need to be > 4 aa to be cap-explained; here it is 4 -> not
    )
    e = m["esm1b"]
    assert e["n_covered"] == 2
    assert e["n_missing"] == 1
    assert e["missing_ids"] == ["C"]
    assert e["cap_aa"] == 4


def test_freeze_esm1b_missing_all_over_cap_flag(tmp_path):
    fa = _write(
        tmp_path / "len.fasta",
        ">short\n" + "G" * 3 + "\n>long\n" + "G" * 10 + "\n",
    )
    # esm1b covers only "short"; "long" (10 aa) is absent and is > cap_aa=5.
    m = freeze_canonical_set(fa, set_name="len", esm1b_keys=["short"], cap_aa=5)
    e = m["esm1b"]
    assert e["missing_ids"] == ["long"]
    assert e["missing_all_over_cap"] is True
    assert e["missing_len_min"] == 10
    assert e["missing_len_max"] == 10


def test_freeze_raises_on_foreign_esm1b_id(tiny_fasta):
    # esm1b must be a subset of the canonical set; a foreign id is a real bug.
    with pytest.raises(ValueError, match="foreign|not in"):
        freeze_canonical_set(tiny_fasta, set_name="tiny", esm1b_keys=["A", "ZZZ"])


def test_freeze_without_esm1b_has_no_esm1b_block(tiny_fasta):
    m = freeze_canonical_set(tiny_fasta, set_name="tiny")
    assert m.get("esm1b") is None


def test_freeze_esm1b_policy_defaults_to_null(tiny_fasta):
    m = freeze_canonical_set(tiny_fasta, set_name="tiny", esm1b_keys=["A", "B"], cap_aa=1)
    assert m["esm1b"]["esm1b_paired_policy"] is None


def test_freeze_records_locked_esm1b_policy(tiny_fasta):
    m = freeze_canonical_set(
        tiny_fasta,
        set_name="tiny",
        esm1b_keys=["A", "B"],
        cap_aa=1,
        esm1b_paired_policy="footnote_esm1b_out",
    )
    assert m["esm1b"]["esm1b_paired_policy"] == "footnote_esm1b_out"


def test_freeze_rejects_unknown_esm1b_policy(tiny_fasta):
    with pytest.raises(ValueError, match="esm1b_paired_policy"):
        freeze_canonical_set(
            tiny_fasta, set_name="tiny", esm1b_keys=["A"], esm1b_paired_policy="bogus"
        )


# ── write_freeze (I/O via atomic_write) ──────────────────────────────────────────
def test_write_freeze_emits_manifest_and_pair_index(tiny_fasta, tmp_path):
    import json

    m = freeze_canonical_set(tiny_fasta, set_name="tiny")
    out = tmp_path / "freeze"
    paths = write_freeze(m, out, set_name="tiny")

    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["canonical_content_sha256"] == m["canonical_content_sha256"]
    assert manifest["ids"] == ["A", "B", "C"]

    df = pd.read_parquet(paths["pair_index"])
    assert list(df.columns) == ["id_a", "id_b"]
    assert len(df) == m["n_pairs"] == 3
    # pair index is consistent with the frozen ids
    assert set(df["id_a"]) | set(df["id_b"]) == {"A", "B", "C"}


def test_write_freeze_leaves_no_tmp_files(tiny_fasta, tmp_path):
    m = freeze_canonical_set(tiny_fasta, set_name="tiny")
    out = tmp_path / "freeze"
    write_freeze(m, out, set_name="tiny")
    assert list(out.glob("*.tmp*")) == []


def test_write_freeze_refuses_to_clobber_by_default(tiny_fasta, tmp_path):
    m = freeze_canonical_set(tiny_fasta, set_name="tiny")
    out = tmp_path / "freeze"
    write_freeze(m, out, set_name="tiny")
    with pytest.raises(FileExistsError, match="already exist"):
        write_freeze(m, out, set_name="tiny")  # no sibling, no silent stale path


def test_write_freeze_overwrite_replaces_in_place(tiny_fasta, tmp_path):
    m = freeze_canonical_set(tiny_fasta, set_name="tiny")
    out = tmp_path / "freeze"
    first = write_freeze(m, out, set_name="tiny")
    second = write_freeze(m, out, set_name="tiny", overwrite=True)
    # canonical path, not a timestamped sibling
    assert second["manifest"] == first["manifest"]
    assert second["pair_index"] == first["pair_index"]
    assert sorted(p.name for p in out.glob("canonical_set_*.json")) == [
        "canonical_set_tiny.json"
    ]


def test_freeze_round_trip_manifest_to_pair_index(tiny_fasta, tmp_path):
    import json

    m = freeze_canonical_set(tiny_fasta, set_name="tiny")
    out = tmp_path / "freeze"
    paths = write_freeze(m, out, set_name="tiny")
    # Reload the committed manifest and regenerate the index from its ids alone —
    # the regenerable-not-committed parquet must match byte-for-byte in content.
    reloaded = json.loads(paths["manifest"].read_text())
    rebuilt = build_pair_index(reloaded["ids"])
    on_disk = pd.read_parquet(paths["pair_index"])
    pd.testing.assert_frame_equal(rebuilt, on_disk)


# ── CLI main() ───────────────────────────────────────────────────────────────────
def test_main_happy_path_writes_and_returns_zero(tiny_fasta, tmp_path, capsys):
    out = tmp_path / "freeze"
    rc = main(["--fasta", str(tiny_fasta), "--set-name", "tiny", "--out-dir", str(out)])
    assert rc == 0
    assert (out / "canonical_set_tiny.json").exists()
    assert (out / "pair_index_tiny.parquet").exists()


def test_main_missing_fasta_returns_two(tmp_path, capsys):
    out = tmp_path / "freeze"
    rc = main(["--fasta", str(tmp_path / "nope.fasta"), "--set-name", "x", "--out-dir", str(out)])
    assert rc == 2
    assert "I/O ERROR" in capsys.readouterr().err


def test_main_refuses_clobber_returns_two(tiny_fasta, tmp_path, capsys):
    out = tmp_path / "freeze"
    main(["--fasta", str(tiny_fasta), "--set-name", "tiny", "--out-dir", str(out)])
    rc = main(["--fasta", str(tiny_fasta), "--set-name", "tiny", "--out-dir", str(out)])
    assert rc == 2
    assert "CLOBBER" in capsys.readouterr().err
