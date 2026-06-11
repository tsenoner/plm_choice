"""Unit 1 — orphan pairs loader (re-home of run_pipeline._load_pairs)."""
import gzip

import pandas as pd
import pytest

from evaluation.orphan_io import OrphanPairsError, load_orphan_pairs

# The published header (run_pipeline._load_pairs reads exactly this): tab-separated
# `p1 p2 TM SNN siblings pident`. `pident` is dropped; `TM`/`SNN`/`siblings` kept.
_HEADER = "p1\tp2\tTM\tSNN\tsiblings\tpident\n"
_ROWS = [
    "A\tB\t0.90\t0.80\tTrue\t55.0\n",
    "A\tC\t0.10\t0.20\tFalse\t12.0\n",
    "B\tC\t0.50\t0.40\tTrue\t30.0\n",
]


def _write(path, header=_HEADER, rows=_ROWS, *, gz=False):
    text = header + "".join(rows)
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)
    return path


def test_loads_plain_tsv_typed(tmp_path):
    p = _write(tmp_path / "pairs.tsv")
    df = load_orphan_pairs(p)
    assert list(df.columns) == ["p1", "p2", "tm", "snn", "sibling"]
    assert df.shape == (3, 5)
    # types: ids are str, tm/snn float, sibling bool
    assert df["p1"].tolist() == ["A", "A", "B"]
    assert df["sibling"].tolist() == [True, False, True]
    assert df["tm"].tolist() == pytest.approx([0.90, 0.10, 0.50])
    assert df["snn"].tolist() == pytest.approx([0.80, 0.20, 0.40])
    assert df["sibling"].dtype == bool


def test_gzip_round_trip_matches_plain(tmp_path):
    plain = load_orphan_pairs(_write(tmp_path / "pairs.tsv"))
    gzd = load_orphan_pairs(_write(tmp_path / "pairs.tsv.gz", gz=True))
    pd.testing.assert_frame_equal(plain, gzd)


def test_bad_header_raises(tmp_path):
    # A header that does not match the published 6-column schema must fail loud.
    p = _write(tmp_path / "pairs.tsv", header="a\tb\tc\td\te\tf\n")
    with pytest.raises(OrphanPairsError, match="header"):
        load_orphan_pairs(p)


def test_malformed_rows_counted_not_silently_dropped(tmp_path):
    rows = _ROWS + ["D\tE\t0.5\n"]  # too few columns
    p = _write(tmp_path / "pairs.tsv", rows=rows)
    with pytest.raises(OrphanPairsError, match="malformed"):
        load_orphan_pairs(p)


def test_malformed_count_surfaced_when_tolerated(tmp_path):
    # When `strict=False` the loader returns (df, n_malformed, n_self) so the caller
    # can log them.
    rows = _ROWS + ["D\tE\t0.5\n", "F\n"]
    p = _write(tmp_path / "pairs.tsv", rows=rows)
    df, n_malformed, n_self = load_orphan_pairs(p, strict=False)
    assert n_malformed == 2
    assert n_self == 0
    assert df.shape[0] == 3  # only the well-formed rows survive


def test_self_pair_dropped_and_counted(tmp_path):
    # A p1 == p2 row violates the u != v invariant the boot weighting count(u)*count(v)
    # and the incremental jackknife both assume. It must be dropped + counted, never
    # scored — Bromberg pairs are distinct orphans.
    rows = _ROWS + ["A\tA\t0.7\t0.6\tFalse\t40.0\n"]  # self-pair
    p = _write(tmp_path / "pairs.tsv", rows=rows)
    df, n_malformed, n_self = load_orphan_pairs(p, strict=False)
    assert n_self == 1
    assert n_malformed == 0
    assert df.shape[0] == 3                      # the 3 well-formed distinct-orphan rows
    assert not (df["p1"] == df["p2"]).any()      # no self-pair survived


def test_self_pair_raises_when_strict(tmp_path):
    rows = _ROWS + ["A\tA\t0.7\t0.6\tFalse\t40.0\n"]
    p = _write(tmp_path / "pairs.tsv", rows=rows)
    with pytest.raises(OrphanPairsError, match="self-pair"):
        load_orphan_pairs(p)
