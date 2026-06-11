"""Fan-in barrier spec-builder for the AAC-floor grid (population_tag x level).

The AAC analogue of :mod:`evaluation.barrier_spec` / :mod:`evaluation.ec_barrier_spec`:
walks the ``population_tag × level`` grid (for one per-distance ``sidecar_dir``),
reads each per-population sidecar that :func:`evaluation.aac_floor_report.main` wrote,
and emits the ``{"artifacts": [...]}`` payload the generic
:mod:`evaluation.analysis_barrier` validates.

Three correctness properties (mirrors recall-fp and EC siblings):

1. **Full grid, no silent gaps.** One artifact per (population_tag, level) cell is
   emitted even when a sidecar is absent — using the canonical reconstructed parquet
   path with ``expected_rows=None`` — so a dead fan-out job is surfaced MISSING by
   the barrier rather than dropped from the spec.
2. **Sidecar-path-authoritative.** When a sidecar exists, the parquet path comes
   from what the producing run actually recorded (``levels[level]["path"]``). The
   canonical name is only a fallback for a sidecar-less cell.
3. **Single source of truth for the column contract.** Drift is detected via
   :func:`evaluation.barrier_spec_base.check_per_query_columns_drift` against
   :data:`evaluation.recall_fp_report.PER_QUERY_COLUMNS` — the AAC per-query parquet
   uses the same schema as recall-fp (D13); one record drifting from the contract
   raises ``SpecBuildError`` rather than passing silently.

**Filename contract (I3 / aac_floor_report.py):**

  * sidecar: ``<sidecar_dir>/aac_floor_<population_tag>.manifest.json``
  * parquet:  ``<sidecar_dir>/aac_floor_<population_tag>_<level>.parquet``

The ``sidecar_dir`` is already scoped to ONE distance by the caller (I3 — distance
is the outer directory, not encoded in the filename). The ``population_tag`` (e.g.
``"full319"``, ``"esm1b"``) distinguishes distinct scoring populations (C1 fix: the
capped-267 esm1b AAC cell must not collide with the full-319 cell).

CLI exit codes mirror the sibling builders: ``0`` — spec written; ``2`` —
operator/config fault (empty grid, malformed/contradictory sidecar, unwritable
output). There is no exit 1 — this step builds a spec, not a data validation.

CLI::

    python -m evaluation.aac_floor_barrier_spec --sidecar-dir <dir> \\
        --population-tags full319 esm1b --out aac_floor_barrier_spec.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from evaluation.barrier_spec_base import (
    SpecBuildError,
    check_per_query_columns_drift,
    dedup,
    emit_cell,
    read_sidecar_dict,
    require_grid_size,
    write_barrier_spec,
)
from evaluation.recall_fp_report import DEFAULT_LEVELS, PARQUET_GUARDS, PER_QUERY_COLUMNS

DEFAULT_POPULATION_TAGS: tuple[str, ...] = ("full319", "esm1b")


def _sidecar_path(sidecar_dir: Path, population_tag: str) -> Path:
    return sidecar_dir / f"aac_floor_{population_tag}.manifest.json"


def _canonical_parquet(sidecar_dir: Path, population_tag: str, level: str) -> Path:
    return sidecar_dir / f"aac_floor_{population_tag}_{level}.parquet"


def _load_sidecar(path: Path) -> dict:
    """Read + validate one AAC-floor sidecar. Raises SpecBuildError on any fault.

    The sidecar shape mirrors recall-fp: a ``levels`` map (per-level dicts with a
    non-empty ``path`` field) plus a top-level ``per_query_columns`` drift guard.
    Structural read is delegated to :func:`read_sidecar_dict`; this adds the
    AAC-specific shape assertions before the drift check.
    """
    manifest = read_sidecar_dict(path)
    if "levels" not in manifest:
        raise SpecBuildError(f"sidecar missing 'levels' object: {path}")
    levels = manifest["levels"]
    if not isinstance(levels, dict):
        raise SpecBuildError(f"sidecar 'levels' must be an object: {path}")
    for lvl, info in levels.items():
        if not isinstance(info, dict):
            raise SpecBuildError(
                f"sidecar level {lvl!r} block must be an object: {path}"
            )
        p = info.get("path")
        if not isinstance(p, str) or not p:
            raise SpecBuildError(
                f"sidecar level {lvl!r} has no non-empty string 'path': {path}"
            )
    # Drift guard: AAC per-query parquet shares the recall-fp column contract (D13).
    check_per_query_columns_drift(manifest, PER_QUERY_COLUMNS, path)
    return manifest


def build_aac_floor_barrier_spec(
    sidecar_dir: Path | str,
    *,
    population_tags: Sequence[str] = DEFAULT_POPULATION_TAGS,
    levels: Sequence[str] = DEFAULT_LEVELS,
    use_expected_rows: bool = True,
    expected_n_population_tags: int | None = None,
) -> dict:
    """Build the ``{"artifacts": [...]}`` barrier spec over the AAC-floor grid.

    Parameters
    ----------
    sidecar_dir
        Directory holding the ``aac_floor_<population_tag>.manifest.json`` sidecars
        (and the parquets they reference). **Already scoped to one distance** by the
        caller (I3 — distance is the outer directory, not encoded in the filenames).
    population_tags
        The population tags that define the grid (e.g. ``["full319", "esm1b"]``).
        Required and non-empty. De-duplicated, order preserved (C1 — each population
        tag corresponds to a distinct scoring cohort; full-319 and capped-267 cells
        produce genuinely different AAC recall numbers because the retrieval DB size
        is load-bearing).
    levels
        CATH levels (default Topology + Homologous-SF; family deferred, W3).
    use_expected_rows
        If True (default), set each cell's ``expected_rows`` from the sidecar's
        ``n_queries_with_positives`` so a truncated parquet fails the barrier on row
        count.
    expected_n_population_tags
        Optional silent-under-coverage guard: if given, the number of unique
        population tags must equal this exactly (an operator-supplied count, e.g. 2
        for full319 + esm1b).

    Returns
    -------
    dict
        ``{"artifacts": [...], "_meta": {...}}``. ``_meta`` records ``n_cells``,
        ``n_cells_without_sidecar``, ``reconstructed_cells``, and ``population_n``
        per population tag (so a downstream consumer can distinguish the capped-267
        and full-319 floor numbers after this step).

    Raises
    ------
    SpecBuildError
        Empty ``population_tags``, an ``expected_n_population_tags`` mismatch, an
        orphan parquet (canonical parquet present with no sidecar — stale/partial
        artifact), or a sidecar that is malformed / contradicts the column contract.
    """
    sidecar_dir = Path(sidecar_dir)
    population_tags = dedup(list(population_tags))
    levels = dedup(list(levels))
    if not population_tags:
        raise SpecBuildError("no population_tags given; the AAC-floor grid cannot be empty.")
    require_grid_size(
        population_tags, expected_n_population_tags,
        singular="population_tag", axis_label="population_tags",
    )

    artifacts: list[dict] = []
    reconstructed: list[str] = []
    population: dict[str, int | None] = {}
    for pop_tag in population_tags:
        sc_path = _sidecar_path(sidecar_dir, pop_tag)
        manifest = _load_sidecar(sc_path) if sc_path.exists() else None
        if manifest is not None:
            population[pop_tag] = manifest.get("population_n")
        for level in levels:
            label = f"aac_floor:{pop_tag}:{level}"
            covered = manifest is not None and level in manifest["levels"]
            canonical = _canonical_parquet(sidecar_dir, pop_tag, level)

            # `_lvl=level` freezes the loop var at definition time (Python binds
            # closure names late); `manifest` is stable across the level loop.
            def _path_rows(_lvl=level):
                info = manifest["levels"][_lvl]
                rows = info.get("n_queries_with_positives") if use_expected_rows else None
                return info["path"], rows

            art, recon = emit_cell(
                label, covered=covered, get_path_rows=_path_rows,
                canonical_parquet=canonical, guards=PARQUET_GUARDS,
            )
            artifacts.append(art)
            if recon:
                reconstructed.append(label)

    return {
        "artifacts": artifacts,
        "_meta": {
            "n_cells": len(artifacts),
            "n_cells_without_sidecar": len(reconstructed),
            "reconstructed_cells": reconstructed,
            "population_n": population,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="aac_floor_barrier_spec",
        description="Build the fan-in barrier spec for the AAC-floor grid from the "
        "per-population sidecar manifests. The sidecar-dir is already scoped to one "
        "distance (I3); the grid is population_tag x level.",
    )
    ap.add_argument(
        "--sidecar-dir", required=True,
        help="Directory of aac_floor_<population_tag>.manifest.json sidecars "
        "(scoped to one distance by the caller).",
    )
    ap.add_argument(
        "--population-tags", nargs="+", default=list(DEFAULT_POPULATION_TAGS),
        help=f"Population tags defining the grid (default: {' '.join(DEFAULT_POPULATION_TAGS)}).",
    )
    ap.add_argument(
        "--levels", nargs="+", default=list(DEFAULT_LEVELS),
        choices=("fold", "superfamily", "family"),
        help=f"CATH levels (default: {' '.join(DEFAULT_LEVELS)}; family deferred, W3).",
    )
    ap.add_argument("--out", required=True, help="Output barrier spec JSON path.")
    ap.add_argument(
        "--expected-n-population-tags", type=int, default=None,
        help="Guard against silent under-coverage: fail unless the grid has exactly "
        "this many unique population tags (e.g. 2 for full319 + esm1b).",
    )
    ap.add_argument(
        "--no-expected-rows", action="store_true",
        help="Do not set expected_rows from the sidecars; rely on the barrier's "
        "0-row/unique/non-null/finite guards only.",
    )
    args = ap.parse_args(argv)

    try:
        spec = build_aac_floor_barrier_spec(
            args.sidecar_dir,
            population_tags=args.population_tags,
            levels=args.levels,
            use_expected_rows=not args.no_expected_rows,
            expected_n_population_tags=args.expected_n_population_tags,
        )
        written = write_barrier_spec(spec, args.out, overwrite=True)
    except SpecBuildError as e:
        print(f"aac_floor_barrier_spec: CONFIG ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except OSError as e:
        print(f"aac_floor_barrier_spec: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    meta = spec["_meta"]
    print(f"aac_floor_barrier_spec: {meta['n_cells']} cell(s) -> {written}", flush=True)
    if meta["n_cells_without_sidecar"]:
        print(
            f"  WARNING: {meta['n_cells_without_sidecar']} cell(s) had no sidecar; "
            f"emitted reconstructed-path specs (the barrier will report them missing): "
            f"{meta['reconstructed_cells']}",
            file=sys.stderr,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
