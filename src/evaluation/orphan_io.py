"""Typed loader for the published Bromberg orphan pairs file.

A tested re-home of ``run_pipeline._load_pairs`` (an untested inline helper in the
legacy orphan path). The pairs file ``orphan_sibling_score.tsv.gz`` is an external
Bromberg artifact with the fixed tab-separated header ``p1 p2 TM SNN siblings pident``
(309,549 rows, 6,219 ``siblings == True``). This module:

* validates the header against that exact schema (a silently-reshaped file fails loud);
* drops ``pident`` (unused by the orphan metric — the AUROC depends only on
  ``(p1, p2, siblings)``, with ``TM``/``SNN`` feeding the secondary ρ);
* counts malformed rows rather than silently dropping them.

The returned frame's columns are renamed to the lower-case ``[p1, p2, tm, snn, sibling]``
convention used by the scoring kernel + the freeze.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

# The published schema, exactly. Order matters: the loader positionally maps the
# tab-split fields, so a header that does not match (renamed/reordered/extra cols)
# means the file is not the Bromberg artifact this code was written for.
EXPECTED_HEADER: tuple[str, ...] = ("p1", "p2", "TM", "SNN", "siblings", "pident")
OUTPUT_COLUMNS: tuple[str, ...] = ("p1", "p2", "tm", "snn", "sibling")


class OrphanPairsError(ValueError):
    """The orphan pairs file is malformed (bad header or unparseable rows)."""


def _open_text(path: Path):
    """Open a plain or gzip-compressed text file (sniff by suffix)."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def load_orphan_pairs(
    path: Path | str, *, strict: bool = True
) -> pd.DataFrame | tuple[pd.DataFrame, int, int]:
    """Load the orphan pairs TSV into a typed ``[p1, p2, tm, snn, sibling]`` frame.

    Gzip-aware (``.gz`` suffix → ``gzip.open``). The header MUST equal
    :data:`EXPECTED_HEADER` or :class:`OrphanPairsError` is raised. Two row-level
    pathologies are dropped-and-counted:

    * **malformed** rows (wrong field count, unparseable float);
    * **self-pairs** (``p1 == p2``). Bromberg pairs are pairs of *distinct* orphans;
      the downstream vertex-bootstrap weighting ``count(u)·count(v)`` and the incremental
      leave-one-orphan-out jackknife both assume ``u != v``, so a self-pair violates an
      invariant the CI machinery depends on. It is the loader's job to enforce it (the gate).

    * ``strict=True`` (default): any malformed row OR any self-pair raises
      :class:`OrphanPairsError`.
    * ``strict=False``: returns ``(df, n_malformed, n_self_pairs)`` with the well-formed,
      distinct-orphan rows only, so a caller can log the counts (the legacy path silently
      kept/dropped them).

    ``sibling`` is parsed from the literal string ``True``/``False`` (the pairs file's
    own boolean encoding — no custom cutoff), exactly as ``run_pipeline._load_pairs``.
    """
    path = Path(path)
    p1_l: list[str] = []
    p2_l: list[str] = []
    tm_l: list[float] = []
    snn_l: list[float] = []
    sib_l: list[bool] = []
    n_malformed = 0
    n_self_pairs = 0

    with _open_text(path) as fh:
        header_line = fh.readline().rstrip("\n")
        header = tuple(header_line.split("\t"))
        if header != EXPECTED_HEADER:
            raise OrphanPairsError(
                f"unexpected header {header!r}; expected {EXPECTED_HEADER!r} "
                f"(the Bromberg orphan_sibling_score schema)"
            )
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != len(EXPECTED_HEADER):
                n_malformed += 1
                continue
            a, b, t, s, sb, _pident = fields
            if a == b:
                n_self_pairs += 1  # u != v invariant — never score a self-pair
                continue
            try:
                tm_v = float(t)
                snn_v = float(s)
            except ValueError:
                n_malformed += 1
                continue
            p1_l.append(a)
            p2_l.append(b)
            tm_l.append(tm_v)
            snn_l.append(snn_v)
            sib_l.append(sb == "True")

    if strict and n_malformed:
        raise OrphanPairsError(
            f"{n_malformed} malformed row(s) in {path}; pass strict=False to tolerate"
        )
    if strict and n_self_pairs:
        raise OrphanPairsError(
            f"{n_self_pairs} self-pair row(s) (p1==p2) in {path}; the orphan metric "
            f"requires distinct orphans (u != v). Pass strict=False to drop + count them."
        )

    df = pd.DataFrame(
        {
            "p1": pd.Series(p1_l, dtype="object"),
            "p2": pd.Series(p2_l, dtype="object"),
            "tm": pd.Series(tm_l, dtype="float64"),
            "snn": pd.Series(snn_l, dtype="float64"),
            "sibling": pd.Series(sib_l, dtype="bool"),
        }
    )
    if strict:
        return df
    return df, n_malformed, n_self_pairs
