"""Differential test for the Leaf-C promotion of ``_pivot_long_to_matrix``.

Leaf C lifts the pure ``_pivot_long_to_matrix`` helper out of ``ec_report.py:103`` into
``analysis_io.py`` so both the EC arm and the cross-pLM arm share one long->symmetric-matrix
builder (``analysis_io`` is the established home for the shared matrix/long helpers —
``pairwise_distance_long`` already lives there). The move must be behaviour-preserving:

* ``analysis_io._pivot_long_to_matrix`` reproduces the frozen inline body byte-for-byte on a
  fixture battery (square symmetry, diagonal untouched, id order = ``ids``, float coercion);
* ``ec_report._pivot_long_to_matrix`` is the SAME object (the EC function body is replaced by
  an import, not a copy that can drift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import evaluation.analysis_io as analysis_io
import evaluation.ec_report as ec_report


def _frozen_pivot(long_df: pd.DataFrame, ids: list[str], value_col: str) -> np.ndarray:
    """The exact pre-promotion ec_report._pivot_long_to_matrix body (ec_report.py:103-111)."""
    pos = {pid: i for i, pid in enumerate(ids)}
    n = len(ids)
    mat = np.zeros((n, n), dtype=float)
    for a, b, v in zip(long_df["a"], long_df["b"], long_df[value_col]):
        i, j = pos[a], pos[b]
        mat[i, j] = mat[j, i] = float(v)
    return mat


# Long frames spanning: a 3-id complete triangle, a 4-id frame with non-sorted id order,
# a single-pair frame, and a value column named something other than "dist".
def _frame(rows, value_col):
    return pd.DataFrame(rows, columns=["a", "b", value_col])


_FIXTURES = [
    (_frame([("p1", "p2", 1.5), ("p1", "p3", 2.0), ("p2", "p3", 3.0)], "dist"),
     ["p1", "p2", "p3"], "dist"),
    # id order deliberately NOT lexicographic — the matrix must follow `ids`, not sorted().
    (_frame([("a", "b", 0.1), ("a", "c", 0.2), ("a", "d", 0.3),
             ("b", "c", 0.4), ("b", "d", 0.5), ("c", "d", 0.6)], "ec_dist"),
     ["d", "b", "a", "c"], "ec_dist"),
    (_frame([("x", "y", 7.0)], "value"), ["x", "y"], "value"),
]


def test_analysis_io_pivot_matches_frozen_inline():
    for long_df, ids, value_col in _FIXTURES:
        got = analysis_io._pivot_long_to_matrix(long_df, ids, value_col)
        expected = _frozen_pivot(long_df, ids, value_col)
        assert np.array_equal(got, expected), f"{ids}: {got!r} != {expected!r}"


def test_analysis_io_pivot_is_symmetric_zero_diagonal():
    long_df, ids, value_col = _FIXTURES[0]
    mat = analysis_io._pivot_long_to_matrix(long_df, ids, value_col)
    assert np.array_equal(mat, mat.T)
    assert np.array_equal(np.diag(mat), np.zeros(len(ids)))


def test_ec_report_pivot_is_the_promoted_object():
    # The EC function body is deleted in favour of the import — the two names must bind the
    # SAME object so the EC arm cannot drift from the promoted helper.
    assert ec_report._pivot_long_to_matrix is analysis_io._pivot_long_to_matrix
