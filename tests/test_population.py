"""Tests for src/evaluation/population.py — the canonical-set guard."""
from __future__ import annotations

import pytest

from evaluation.population import PopulationError, assert_population

FROZEN = {"P1", "P2", "P3", "P4"}


def test_exact_match_passes():
    assert assert_population(["P1", "P2", "P3", "P4"], FROZEN) is None
    # Order and duplicates do not matter.
    assert assert_population(["P4", "P1", "P1", "P2", "P3"], FROZEN) is None


def test_foreign_id_raises():
    with pytest.raises(PopulationError, match="not in the frozen set"):
        assert_population(["P1", "P2", "P3", "P4", "X"], FROZEN)


def test_missing_id_raises_by_default():
    with pytest.raises(PopulationError, match="missing"):
        assert_population(["P1", "P2", "P3"], FROZEN)


def test_missing_id_allowed_when_capped():
    assert assert_population(["P1", "P2", "P3"], FROZEN, allow_capped=True) is None


def test_capped_still_rejects_foreign_id():
    with pytest.raises(PopulationError, match="not in the frozen set"):
        assert_population(["P1", "P2", "X"], FROZEN, allow_capped=True)


def test_empty_population_raises_even_when_capped():
    with pytest.raises(PopulationError, match="empty"):
        assert_population([], FROZEN, allow_capped=True)


def test_error_message_includes_name():
    with pytest.raises(PopulationError, match="recall-fp/esm2_650m"):
        assert_population(["P1"], FROZEN, name="recall-fp/esm2_650m")
