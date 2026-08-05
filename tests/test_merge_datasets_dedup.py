"""Canonical-pair deduplication in ``merge_datasets``.

``run_mmseqs_all_vs_all.sh`` / ``run_foldseek_all_vs_all.sh`` use
``--num-iterations 3``, i.e. an iterative *profile* search, which is inherently
directional: a pair may be reported as (A,B), as (B,A), or as both, and the two
reports can carry different alignments. Nothing downstream collapsed them, so on
the published table 61.5% of unordered sequence pairs (74.9% of structural ones)
carried double weight while the rest carried single weight -- and the duplicated
ones were measurably the *more similar* pairs (mean fident 0.5158 vs 0.4142,
mean alntmscore 0.8304 vs 0.6319). That silently up-weights strong homologues in
every pair-level statistic, and it feeds conflicting targets to the *symmetric*
probe arms (``linear_distance``, ``euclidean``) while the concatenating arms
(``fnn``, ``linear``) never see the conflict.

The two sides get deliberately different aggregation rules:

* sequence -- keep the single alignment with the **lowest E-value**. HFSP is a
  non-linear function of (PIDE, L), so averaging those would fabricate an
  alignment that never existed; one real alignment keeps fident/nident/mismatch/
  qcov/tcov mutually consistent.
* structure -- **mean** ``alntmscore``. It is a lone scalar with no dependent
  columns, and it is symmetric by definition (normalised over the alignment, not
  by query or target length -- verified empirically: 41.7% of bidirectional pairs
  with *different* protein lengths carry identical scores, versus 12.6% matching
  a target-normalised model). Taking max instead would re-introduce an upward
  bias on exactly the duplicated subset we are de-weighting.
"""

from __future__ import annotations

import polars as pl

from data_preparation.merge_datasets import ProteinAnalysisPipeline


def _pipe() -> ProteinAnalysisPipeline:
    """The dedup helpers do not touch ``self``; bypass the on-disk __init__."""
    return object.__new__(ProteinAnalysisPipeline)


def _mmseqs_frame(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query": [r[0] for r in rows],
            "target": [r[1] for r in rows],
            "fident": [r[2] for r in rows],
            "evalue": [r[3] for r in rows],
            "nident": [r[4] for r in rows],
            "mismatch": [r[5] for r in rows],
            "qcov": [r[6] for r in rows],
            "tcov": [r[7] for r in rows],
        },
        schema_overrides={"nident": pl.Int64, "mismatch": pl.Int64},
    )


def test_canonicalises_pair_orientation():
    """(B,A) is rewritten to (A,B) so both orientations share a key."""
    df = _mmseqs_frame([("P2", "P1", 0.5, 1e-40, 100, 50, 0.9, 0.9)])
    out = _pipe()._dedupe_mmseqs_pairs(df)
    assert out.height == 1
    assert out["query"][0] == "P1"
    assert out["target"][0] == "P2"


def test_mmseqs_keeps_the_lower_evalue_alignment_intact():
    """Both directions collapse to one row, carrying the better alignment's values."""
    df = _mmseqs_frame(
        [
            ("P1", "P2", 0.40, 1e-20, 100, 60, 0.85, 0.85),  # worse
            ("P2", "P1", 0.55, 1e-60, 120, 40, 0.95, 0.95),  # better -> kept
        ]
    )
    out = _pipe()._dedupe_mmseqs_pairs(df)
    assert out.height == 1
    row = out.row(0, named=True)
    # every column comes from the SAME surviving alignment - no blending
    assert row["fident"] == 0.55
    assert row["evalue"] == 1e-60
    assert row["nident"] == 120
    assert row["mismatch"] == 40
    assert row["qcov"] == 0.95


def test_mmseqs_single_direction_pair_is_preserved():
    """A pair reported only one way survives unchanged apart from orientation."""
    df = _mmseqs_frame([("P1", "P2", 0.42, 1e-33, 90, 30, 0.88, 0.81)])
    out = _pipe()._dedupe_mmseqs_pairs(df)
    assert out.height == 1
    assert out.row(0, named=True)["fident"] == 0.42


def test_mmseqs_distinct_pairs_are_not_merged():
    """Deduplication is per unordered pair, not global."""
    df = _mmseqs_frame(
        [
            ("P1", "P2", 0.40, 1e-20, 100, 60, 0.9, 0.9),
            ("P2", "P1", 0.55, 1e-60, 120, 40, 0.9, 0.9),
            ("P1", "P3", 0.31, 1e-10, 80, 70, 0.9, 0.9),
        ]
    )
    out = _pipe()._dedupe_mmseqs_pairs(df).sort("target")
    assert out.height == 2
    assert out["target"].to_list() == ["P2", "P3"]


def test_foldseek_averages_alntmscore_across_directions():
    """alntmscore is symmetric by definition, so the two reports are averaged."""
    df = pl.DataFrame(
        {
            "query": ["P1", "P2"],
            "target": ["P2", "P1"],
            "min_cov": [0.80, 0.90],
            "alntmscore": [0.60, 0.80],
        }
    )
    out = _pipe()._dedupe_foldseek_pairs(df)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["query"] == "P1" and row["target"] == "P2"
    assert abs(row["alntmscore"] - 0.70) < 1e-12
    assert abs(row["min_cov"] - 0.85) < 1e-12


def test_hfsp_is_computed_from_the_retained_alignment():
    """Dedup runs before HFSP, so HFSP reflects the surviving alignment only."""
    pipe = _pipe()
    df = _mmseqs_frame(
        [
            ("P1", "P2", 0.40, 1e-20, 100, 60, 0.9, 0.9),
            ("P2", "P1", 0.55, 1e-60, 120, 40, 0.9, 0.9),
        ]
    )
    out = pipe._compute_hfsp_scores(pipe._dedupe_mmseqs_pairs(df))
    assert out.height == 1
    # surviving alignment: fident=0.55, L = 120 + 40 = 160
    assert out["ungapped_len"][0] == 160
    expected = 0.55 * 100 - 770 * 160 ** (-0.33 * (1 + pow(2.718281828459045, -0.160)))
    assert abs(out["hfsp"][0] - expected) < 1e-4
