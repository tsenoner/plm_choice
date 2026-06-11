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
import warnings
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
) -> pd.DataFrame | tuple[pd.DataFrame, int, int, int, int]:
    """Load the orphan pairs TSV into a typed ``[p1, p2, tm, snn, sibling]`` frame.

    Gzip-aware (``.gz`` suffix → ``gzip.open``). The header MUST equal
    :data:`EXPECTED_HEADER` or :class:`OrphanPairsError` is raised. Four row-level
    pathologies are handled-and-counted:

    * **malformed** rows (wrong field count, unparseable float) — dropped + counted;
    * **self-pairs** (``p1 == p2``). Bromberg pairs are pairs of *distinct* orphans;
      the downstream vertex-bootstrap weighting ``count(u)·count(v)`` and the incremental
      leave-one-orphan-out jackknife both assume ``u != v``, so a self-pair violates an
      invariant the CI machinery depends on — dropped + counted;
    * **non-canonical (reordered) endpoints**. The orphan arm consumes an EXTERNAL
      Bromberg pairs file (not EC's by-construction-canonical ``pairwise_distance_long``).
      The cosine metric and the ``sibling`` label are SYMMETRIC, so ``(A, B)`` and
      ``(B, A)`` are the SAME pair; un-canonicalized they yield two distinct ``pair_key``s
      and the AUROC double-counts. Each row is canonicalized to sorted ``(min, max)``
      endpoint order (``tm``/``snn``/``sibling`` carried with it — they are symmetric);
      rows whose endpoints were out of order are counted as ``n_reordered``;
    * **exact post-canonical duplicate** ``(p1, p2)`` rows — dropped + counted as
      ``n_dup_dropped`` (so ``pair_key`` is genuinely unique after load).

    * ``strict=True`` (default): any malformed row, self-pair, reorder, OR duplicate
      raises :class:`OrphanPairsError`. A well-formed canonical benchmark file has zero
      of all four, so on the real Bromberg file this is a NO-OP; the guard only fires
      (loudly) on a malformed / directed file.
    * ``strict=False``: returns
      ``(df, n_malformed, n_self_pairs, n_reordered, n_dup_dropped)`` with the
      well-formed, distinct-orphan, canonical, de-duplicated rows only, so a caller can
      log the counts. A :class:`UserWarning` is emitted if any reorder or duplicate
      occurred.

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
    n_reordered = 0
    n_dup_dropped = 0
    seen: set[tuple[str, str]] = set()

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
            # Canonicalize to sorted endpoint order: cosine + sibling are symmetric, so
            # (A,B) and (B,A) are the same pair. The carried tm/snn/sibling are symmetric.
            if a > b:
                a, b = b, a
                n_reordered += 1
            key = (a, b)
            if key in seen:
                n_dup_dropped += 1  # exact post-canonical duplicate -> pair_key collision
                continue
            seen.add(key)
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
    if strict and n_reordered:
        raise OrphanPairsError(
            f"{n_reordered} non-canonical (reordered-endpoint) row(s) in {path}; a "
            f"well-formed Bromberg file is already canonical (sorted p1<p2). The cosine "
            f"metric + sibling label are symmetric so (A,B)==(B,A). Pass strict=False to "
            f"canonicalize + count them."
        )
    if strict and n_dup_dropped:
        raise OrphanPairsError(
            f"{n_dup_dropped} duplicate (post-canonical (p1,p2)) row(s) in {path}; "
            f"pair_key must be unique. Pass strict=False to drop + count them."
        )

    if not strict and (n_reordered or n_dup_dropped):
        warnings.warn(
            f"orphan pairs file {path} was not canonical: {n_reordered} reordered "
            f"endpoint(s) + {n_dup_dropped} duplicate(s) dropped. A well-formed Bromberg "
            f"file has zero of both; pair_key uniqueness has been enforced.",
            UserWarning,
            stacklevel=2,
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
    return df, n_malformed, n_self_pairs, n_reordered, n_dup_dropped
