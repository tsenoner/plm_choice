"""Set maths behind the embedding-coverage UpSet figure.

The figure exists because the 16 embedding arms do NOT cover the same proteins, and
``src/shared/datasets.py:99-105`` drops a pair when *either* protein is missing from
that arm's HDF5 -- so a coverage hole costs pairs quadratically. Getting the set
arithmetic wrong here would misstate which arm is deficient, so the pure functions
are pinned separately from the rendering.
"""

from __future__ import annotations

import pytest

from visualization.plot_embedding_coverage_upset import (
    membership_patterns,
    pattern_rows,
    set_sizes_from_patterns,
)


def test_membership_patterns_counts_each_distinct_combination():
    sets = {
        "a": {"p1", "p2", "p3"},
        "b": {"p1", "p2"},
        "c": {"p1"},
    }
    pats = membership_patterns(sets)
    # p1 in all three, p2 in a+b, p3 in a only
    assert pats[frozenset({"a", "b", "c"})] == 1
    assert pats[frozenset({"a", "b"})] == 1
    assert pats[frozenset({"a"})] == 1
    assert sum(pats.values()) == 3


def test_membership_patterns_ignores_absent_only_combination():
    """A protein in no set cannot exist -- the universe is the union."""
    sets = {"a": {"p1"}, "b": {"p1"}}
    pats = membership_patterns(sets)
    assert frozenset() not in pats
    assert pats[frozenset({"a", "b"})] == 1


def test_set_sizes_are_recoverable_from_the_patterns():
    """Each set total must equal the sum of every pattern containing it."""
    sets = {"a": {"p1", "p2", "p3"}, "b": {"p1", "p2"}, "c": {"p1"}}
    pats = membership_patterns(sets)
    sizes = set_sizes_from_patterns(pats, ["a", "b", "c"])
    assert sizes == {"a": 3, "b": 2, "c": 1}


def test_pattern_rows_sorted_by_count_descending():
    pats = {
        frozenset({"a"}): 5,
        frozenset({"a", "b"}): 100,
        frozenset({"b"}): 50,
    }
    rows = pattern_rows(pats, ["a", "b"])
    assert [r.count for r in rows] == [100, 50, 5]


def test_pattern_rows_marks_the_complete_intersection():
    pats = {frozenset({"a", "b"}): 10, frozenset({"a"}): 2}
    rows = pattern_rows(pats, ["a", "b"])
    assert rows[0].complete is True
    assert rows[1].complete is False
    assert rows[1].missing == ("b",)


def test_pattern_rows_can_be_truncated_and_reports_the_remainder():
    pats = {frozenset({"a"}): 10, frozenset({"b"}): 5, frozenset({"a", "b"}): 1}
    rows, dropped = pattern_rows(pats, ["a", "b"], max_rows=2, return_dropped=True)
    assert len(rows) == 2
    # truncation must be reported, never silent
    assert dropped == 1


def test_pattern_rows_rejects_a_set_name_not_in_the_order():
    pats = {frozenset({"a", "zzz"}): 1}
    with pytest.raises(ValueError, match="zzz"):
        pattern_rows(pats, ["a"])
