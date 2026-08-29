"""One protein cohort shared by every embedding arm.

**The problem.** The 15 pLM embedding sets do not cover the same proteins
(``freeze/embedding_key_coverage.json``): only 422,972 of 542,238 are present in
every arm, and after the interrupted ``esm2_3b`` run is completed, 526,871.
``shared.datasets._load_and_filter_data`` drops a pair when *either* protein is
missing from that arm's HDF5, so the loss is **quadratic** in coverage -- measured
on the published grid, ``esm2_3b`` was scored on 558,947 test pairs where ten
other arms got 872,572, yet is reported at rank #10. A cross-pLM ranking whose
rows were scored on different data is not a ranking.

**Why exclusion and not completion.** The gaps have different causes, and one of
them cannot be fixed: ESM-1b's learned positional embeddings cap at 1022 tokens
(``embedding_generation.py:88``), so ~2.8% of the cohort can never be embedded by
it, and ``clean`` inherits exactly that set by construction. Topping arms up can
therefore never produce a uniform test set. Restricting all arms to the
intersection is the only construction that gives every model identical data.

**Why a load-time filter and not deleting datasets.** The ``.h5`` files are the
md5-verified Zenodo deposit. Deleting from them is irreversible and would make
each file stop matching its published checksum. A filter over a committed id list
is reversible, reviewable, and reproducible from the deposit as published.

**Why the freeze stores the excluded ids.** The exclusion is ~34x smaller than the
inclusion (15,367 vs 526,871 ids), so it is the compact half to commit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Committed exclusion list. Absent freeze => empty exclusion => unchanged behaviour.
DEFAULT_EXCLUSION_FREEZE = (
    Path(__file__).resolve().parents[2] / "freeze" / "embedding_excluded_proteins.json"
)


@dataclass(frozen=True)
class ExclusionSummary:
    """What restricting one arm to the cohort actually did."""

    kept: int
    removed: int
    not_present: int

    def describe(self, label: str) -> str:
        return (
            f"cohort filter [{label}]: kept {self.kept:,}, removed {self.removed:,}"
            + (
                f" ({self.not_present:,} excluded id(s) were already absent here)"
                if self.not_present
                else ""
            )
        )


def load_excluded_proteins(path: Path | str | None = None) -> frozenset[str]:
    """Read the committed exclusion list.

    A missing freeze returns an empty set rather than raising: the filter is then a
    no-op and behaviour is identical to before the cohort was introduced. That makes
    adopting it an explicit act (commit the freeze) rather than an accident.
    """
    freeze = Path(path) if path is not None else DEFAULT_EXCLUSION_FREEZE
    if not freeze.exists():
        return frozenset()
    blob = json.loads(freeze.read_text())
    return frozenset(blob["excluded_ids"])


def restrict_to_cohort(keys: set[str], excluded: frozenset[str]) -> set[str]:
    """Remove the excluded proteins from one arm's key set."""
    if not excluded:
        return keys
    return set(keys) - excluded


def exclusion_summary(keys: set[str], excluded: frozenset[str]) -> ExclusionSummary:
    """Account for the filter: kept, removed, and excluded-but-already-absent.

    The third number matters. An arm that was *already* missing an excluded protein
    contributes nothing to ``removed``; conflating the two would make the filter look
    like it did more work on the deficient arms than it did.
    """
    present_and_excluded = {k for k in keys if k in excluded}
    return ExclusionSummary(
        kept=len(keys) - len(present_and_excluded),
        removed=len(present_and_excluded),
        not_present=len(excluded) - len(present_and_excluded),
    )
