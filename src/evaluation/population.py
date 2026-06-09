"""Canonical-set guard for the revision analyses.

Every analysis intersects its inputs (embeddings, labels, pair tables) before
computing a metric. Without a guard, a pLM that is silently missing proteins —
an architecture cap (e.g. ESM-1b skips sequences > 1022 aa), a truncated
re-extract, a join that dropped rows — produces a metric over a *different*
population than its peers, and any "mean over pLMs" silently mixes cohorts.

Call :func:`assert_population` at the top of every analysis with the frozen
canonical id set (e.g. the 319) as ``expected``. A pLM that is legitimately
capped passes ``allow_capped=True`` so it may be a strict *subset* of the
frozen set — but never contain ids outside it, and never be empty. Report its
per-cell ``n`` separately so a capped pLM is never folded into a bare mean.
"""
from __future__ import annotations

from typing import Iterable


class PopulationError(AssertionError):
    """Raised when an analysis input does not match the frozen population."""


def assert_population(
    observed: Iterable[str],
    expected: Iterable[str],
    *,
    name: str = "analysis",
    allow_capped: bool = False,
) -> None:
    """Assert the observed protein set matches the frozen ``expected`` set.

    Parameters
    ----------
    observed
        Protein ids present in this analysis input (order/duplicates ignored).
    expected
        The frozen canonical set (the single source of truth).
    name
        Label for the analysis/pLM, surfaced in the error so a failure points
        at the exact cell.
    allow_capped
        If True, ``observed`` may be a strict subset of ``expected`` (an
        architecture-capped pLM). It still may not contain foreign ids or be
        empty.

    Raises
    ------
    PopulationError
        On any foreign id, on missing ids when ``allow_capped`` is False, or on
        an empty population.
    """
    obs = set(observed)
    exp = set(expected)

    foreign = obs - exp
    if foreign:
        sample = sorted(foreign)[:5]
        raise PopulationError(
            f"{name}: {len(foreign)} id(s) not in the frozen set "
            f"(e.g. {sample}); inputs are not aligned to the canonical population."
        )

    if not obs:
        raise PopulationError(f"{name}: empty population (zero proteins present).")

    missing = exp - obs
    if missing and not allow_capped:
        sample = sorted(missing)[:5]
        raise PopulationError(
            f"{name}: {len(missing)} frozen id(s) missing (population drift; "
            f"e.g. {sample}). Pass allow_capped=True only for an "
            f"architecture-capped pLM, and report its per-cell n separately."
        )
    return None
