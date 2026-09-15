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

import argparse
import functools
import hashlib
import json
import operator
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1

def _default_freeze_path() -> Path:
    """Locate ``freeze/embedding_excluded_proteins.json`` without assuming the repo layout.

    ``parents[2]`` only works from a source checkout. The wheel ships ``src/*`` and NOT
    ``freeze/`` (``pyproject.toml [tool.hatch.build.targets.wheel]``), so from
    ``site-packages/shared/`` that expression points two levels above site-packages and
    the freeze is never found -- and because an absent freeze is a deliberate no-op, the
    cohort filter would switch itself off with no visible sign. Walk up for the repo
    marker instead, then fall back to a cwd-relative ``freeze/``, which is the convention
    ``ec_freeze`` and ``orphan_freeze`` already use.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / "freeze" / FREEZE_NAME
    return Path("freeze") / FREEZE_NAME


FREEZE_NAME = "embedding_excluded_proteins.json"

#: Committed exclusion list. Absent freeze => empty exclusion => unchanged behaviour,
#: but the absence is announced (see ``load_excluded_proteins``) rather than silent.
DEFAULT_EXCLUSION_FREEZE = None  # resolved lazily by _default_freeze_path()


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


@functools.cache
def load_excluded_proteins(path: Path | str | None = None) -> frozenset[str]:
    """Read the committed exclusion list.

    A missing freeze returns an empty set rather than raising: the filter is then a
    no-op and behaviour is identical to before the cohort was introduced. That makes
    adopting it an explicit act (commit the freeze) rather than an accident.

    Cached because every loader in a run reads the same freeze -- one parse per
    process, not one per arm per split.

    The manifest is verified HERE, not only where it is written. A freeze is hand-
    editable on disk between runs; at write time it was just derived in-process and
    cannot have drifted. Verifying only on write puts the guard in the one place it
    cannot help.
    """
    freeze = Path(path) if path is not None else _default_freeze_path()
    if not freeze.exists():
        # Announce it: an absent freeze is a legitimate no-op, but a silent one is
        # indistinguishable from a filter that ran and found nothing to remove.
        print(f"cohort filter: no exclusion freeze at {freeze} -- every arm keeps its own keys")
        return frozenset()
    blob = json.loads(freeze.read_text())
    verify_exclusion(blob)
    return frozenset(blob["excluded_ids"])


def restrict_to_cohort(
    keys: set[str], excluded: frozenset[str]
) -> tuple[set[str], ExclusionSummary]:
    """Remove the excluded proteins from one arm's key set, and account for it.

    Filtering and reporting come back together because they are the same
    intersection. Computing them separately walked the ~542k key set twice and
    copied it once; ``keys & excluded`` walks the ~15k exclusion list instead
    (CPython iterates the smaller operand) -- measured 36.6 ms -> 8.2 ms per arm.

    ``not_present`` matters and is why the summary is not just ``len``s. An arm that
    was *already* missing an excluded protein contributes nothing to ``removed``;
    conflating the two would make the filter look like it did more work on the
    deficient arms than it did.
    """
    removed = keys & excluded
    return keys - removed, ExclusionSummary(
        kept=len(keys) - len(removed),
        removed=len(removed),
        not_present=len(excluded) - len(removed),
    )


# --------------------------------------------------------------------------- #
# Derivation + freeze writing
# --------------------------------------------------------------------------- #
#
# Until this existed, ``DEFAULT_EXCLUSION_FREEZE`` named a file that nothing in the
# repo produced, so the filter above was a permanent no-op and the "one cohort shared
# by every arm" guarantee was aspirational. The exclusion is not recoverable from
# ``freeze/embedding_key_coverage*.json`` -- that stores pattern *counts*, not ids --
# so deriving it has to go back to the HDF5 key sets.


def content_hash(excluded_ids: list[str]) -> str:
    """SHA-256 of the sorted id list. Changes iff the excluded *set* changes."""
    payload = json.dumps(sorted(excluded_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_exclusion(keysets: dict[str, set]) -> dict:
    """Build the exclusion manifest from one key set per arm.

    Excluded = union - intersection: every protein that at least one arm is missing.
    Dropping them is what makes the arms comparable, and it is done on the *pair*
    loader rather than in the HDF5 files because those are the md5-verified Zenodo
    deposit (see the module docstring).
    """
    if not keysets:
        raise ValueError("no key sets given -- nothing to derive a cohort from")
    arms = sorted(keysets)
    universe: set[str] = set().union(*(keysets[a] for a in arms))
    # reduce(&) rather than set.intersection(*...): the unbound method rejects a
    # frozenset as `self`, while the set().union() above accepts one -- an asymmetry
    # that shows up only at a caller that happens to hold immutable key sets.
    intersection: set[str] = functools.reduce(operator.and_, (keysets[a] for a in arms))
    excluded = sorted(universe - intersection)
    return {
        "schema_version": SCHEMA_VERSION,
        "arms": arms,
        "counts": {a: len(keysets[a]) for a in arms},
        "universe": len(universe),
        "intersection_all": len(intersection),
        "n_excluded": len(excluded),
        "content_sha256": content_hash(excluded),
        "excluded_ids": excluded,
    }


def verify_exclusion(manifest: dict) -> bool:
    """Re-derive the hash and the counts from ``excluded_ids`` alone.

    Independent of :func:`derive_exclusion` on purpose: a hand-edited freeze whose
    ``n_excluded`` or hash no longer matches its own id list must fail here rather
    than quietly filter a different cohort than it claims to.
    """
    ids = list(manifest["excluded_ids"])
    if len(set(ids)) != len(ids):
        raise ValueError("exclusion freeze contains duplicate ids")
    stated_n, actual_n = manifest.get("n_excluded"), len(ids)
    if stated_n is not None and stated_n != actual_n:
        raise ValueError(f"n_excluded {stated_n} != {actual_n} ids present")
    stated_u, stated_i = manifest.get("universe"), manifest.get("intersection_all")
    if stated_u is not None and stated_i is not None and stated_u - stated_i != actual_n:
        raise ValueError(
            f"universe - intersection_all = {stated_u - stated_i}, but {actual_n} ids listed"
        )
    stated_h, actual_h = manifest.get("content_sha256"), content_hash(ids)
    if stated_h is not None and stated_h != actual_h:
        raise ValueError(f"content drift: manifest {stated_h!r} != re-derived {actual_h!r}")
    return True


def write_exclusion_freeze(
    manifest: dict,
    path: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the manifest, refusing to clobber an existing freeze without intent."""
    from shared.atomic_io import atomic_write

    verify_exclusion(manifest)
    out = Path(path) if path is not None else _default_freeze_path()
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace")
    out.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write(
        out,
        lambda p: p.write_text(json.dumps(manifest, indent=2) + "\n"),
        mode="replace",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive the shared-cohort exclusion freeze from the embedding HDF5 files.",
    )
    parser.add_argument(
        "--h5-dir", type=Path, required=True, help="Directory of per-arm .h5 files."
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help=f"Output path (default: {_default_freeze_path()})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing freeze.")
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore the .keys.txt sidecars when scanning."
    )
    args = parser.parse_args(argv)

    from shared.h5_keys import load_h5_keysets

    keysets = load_h5_keysets(args.h5_dir, use_cache=not args.no_cache)
    if not keysets:
        print(f"no .h5 files in {args.h5_dir}", file=sys.stderr)
        return 1

    manifest = derive_exclusion(keysets)
    for arm in manifest["arms"]:
        n = manifest["counts"][arm]
        print(f"  {arm:14s} {n:>9,}  (missing {manifest['universe'] - n:>6,})")
    print(
        f"universe {manifest['universe']:,} | in every arm {manifest['intersection_all']:,} "
        f"| excluded {manifest['n_excluded']:,}"
    )
    out = write_exclusion_freeze(manifest, args.out, overwrite=args.overwrite)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
