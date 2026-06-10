"""Fan-in barrier spec-builder for the recall-fp analysis grid (plan v3, B6 wiring).

:mod:`evaluation.analysis_barrier` is generic by design — it imports no project
modules and validates whatever :class:`~evaluation.analysis_barrier.ArtifactSpec`
grid the caller hands it. This module IS that caller for the recall-fp step: it
walks the ``pLM × representation × CATH-level`` grid, reads the per-(pLM,
representation) sidecar manifests that :func:`evaluation.recall_fp_report.main`
wrote, and emits the ``barrier_spec.json`` the barrier consumes
(``{"artifacts": [ {<ArtifactSpec fields>}, ... ]}``).

Three correctness properties it guarantees:

1. **Full grid, no silent gaps.** One artifact per (pLM, rep, level) cell is
   emitted even when a sidecar is *absent* — using the canonical reconstructed
   parquet name with ``expected_rows=None`` — so a fan-out cell whose job died is
   surfaced by the barrier (missing file) rather than dropped from the spec. The
   count of sidecar-less cells is recorded in ``_meta`` (logged by the CLI).
2. **Sidecar-path-authoritative.** When a sidecar exists, the parquet path comes
   from what the producing run actually recorded (``levels[level]["path"]``),
   which survives ``--no-overwrite`` timestamping. The canonical name is only a
   fallback for an absent sidecar (where there is nothing to read).
3. **Single source of truth for the contract.** The column/guard fields are
   transcribed from :data:`evaluation.recall_fp_report.PARQUET_GUARDS`, and a
   sidecar whose ``per_query_columns`` disagree with
   :data:`~evaluation.recall_fp_report.PER_QUERY_COLUMNS` is a task-1/task-2 drift
   signal that fails loud (``SpecBuildError``) rather than being silently
   transcribed under a stale contract.

CLI exit codes mirror the sibling DAG mains (``analysis_barrier`` /
``verify_analysis``): ``0`` — spec written; ``2`` — operator/config fault (no
pLMs, malformed/contradictory sidecar, unwritable output). There is no exit 1:
this step builds a spec, it does not validate data — that is the barrier's job.

CLI::

    python -m evaluation.barrier_spec --sidecar-dir <dir> \\
        --plms prott5 esm2 ... --representations raw ffn --out barrier_spec.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from evaluation.recall_fp_report import DEFAULT_LEVELS, PARQUET_GUARDS, PER_QUERY_COLUMNS
# Shared base helpers. SpecBuildError + write_barrier_spec are also part of this
# module's public surface (tests import them from here).
from evaluation.barrier_spec_base import (
    SpecBuildError,
    check_per_query_columns_drift,
    dedup,
    emit_cell,
    read_sidecar_dict,
    require_grid_size,
    write_barrier_spec,
)


def _sidecar_path(sidecar_dir: Path, plm: str, rep: str) -> Path:
    return sidecar_dir / f"recall_fp_{plm}_{rep}.manifest.json"


def _canonical_parquet(sidecar_dir: Path, plm: str, rep: str, level: str) -> Path:
    return sidecar_dir / f"recall_fp_{plm}_{rep}_{level}.parquet"


def _load_sidecar(path: Path) -> dict:
    """Read + validate one recall-fp sidecar. Raises SpecBuildError on any fault.

    JSON parsing and the top-level object check are delegated to ``read_sidecar_dict``;
    this validates the recall-fp-specific shape (the ``levels`` map, per-level block
    objects, non-empty path fields) and then the ``per_query_columns`` drift guard.
    """
    manifest = read_sidecar_dict(path)
    if "levels" not in manifest:
        raise SpecBuildError(f"sidecar missing 'levels' object: {path}")
    levels = manifest["levels"]
    if not isinstance(levels, dict):
        raise SpecBuildError(f"sidecar 'levels' must be an object: {path}")
    for lvl, info in levels.items():
        if not isinstance(info, dict):
            raise SpecBuildError(f"sidecar level {lvl!r} block must be an object: {path}")
        p = info.get("path")
        if not isinstance(p, str) or not p:
            raise SpecBuildError(
                f"sidecar level {lvl!r} has no non-empty string 'path': {path}"
            )
    check_per_query_columns_drift(manifest, PER_QUERY_COLUMNS, path)
    return manifest


def build_recall_fp_barrier_spec(
    sidecar_dir: Path | str,
    *,
    plms: Sequence[str],
    representations: Sequence[str] = ("raw", "ffn"),
    levels: Sequence[str] = DEFAULT_LEVELS,
    use_expected_rows: bool = True,
    expected_n_plms: int | None = None,
) -> dict:
    """Build the ``{"artifacts": [...]}`` barrier spec over the recall-fp grid.

    Parameters
    ----------
    sidecar_dir
        Directory holding the ``recall_fp_<plm>_<rep>.manifest.json`` sidecars (and,
        by convention, the parquets they reference).
    plms
        The pLM names that define the grid (e.g. the 15 trained pLMs). Required and
        non-empty — an empty grid is an operator fault. De-duplicated, order preserved.
    representations
        Representation axis (default raw + ffn).
    levels
        CATH levels (default Topology + Homologous-SF; family excluded — W3).
    use_expected_rows
        If True (default), set each cell's ``expected_rows`` from the sidecar's
        ``n_queries_with_positives`` so a truncated parquet fails the barrier on row
        count. If False (or for a sidecar-less cell), leave ``expected_rows`` None and
        rely on the barrier's 0-row + unique/non-null/finite guards.
    expected_n_plms
        Optional guard against silent under-coverage: if given, the number of *unique*
        pLMs must equal it (the DAG submit script passes the intended grid size, e.g.
        15), else a typo'd / truncated ``plms`` list that quietly drops pLMs is caught
        here rather than producing a spec the barrier happily passes over a subset.

    Returns
    -------
    dict
        ``{"artifacts": [...], "_meta": {...}}``. ``artifacts`` is the barrier payload
        (the barrier ignores ``_meta``); ``_meta`` records ``n_cells``,
        ``n_cells_without_sidecar``, the reconstructed-cell labels, and per-(pLM,rep)
        ``population_n`` (so a downstream step can keep a capped pLM, e.g. esm1b at 267,
        out of a bare cross-pLM mean — the cap is otherwise lost after this step).

    Raises
    ------
    SpecBuildError
        Empty ``plms``, an ``expected_n_plms`` mismatch, an *orphan* parquet (a canonical
        parquet present with no sidecar — a stale/partial artifact), or a sidecar that is
        malformed / contradicts the column contract.
    """
    sidecar_dir = Path(sidecar_dir)
    plms = dedup(plms)
    representations = dedup(representations)
    levels = dedup(levels)
    if not plms:
        raise SpecBuildError("no pLMs given; the recall-fp grid cannot be empty.")
    require_grid_size(plms, expected_n_plms, singular="pLM", axis_label="plms")

    artifacts: list[dict] = []
    reconstructed: list[str] = []
    population: dict[str, int | None] = {}
    for plm in plms:
        for rep in representations:
            sc_path = _sidecar_path(sidecar_dir, plm, rep)
            manifest = _load_sidecar(sc_path) if sc_path.exists() else None
            if manifest is not None:
                population[f"{plm}:{rep}"] = manifest.get("population_n")
            for level in levels:
                label = f"recall_fp:{plm}:{rep}:{level}"
                covered = manifest is not None and level in manifest["levels"]
                canonical = _canonical_parquet(sidecar_dir, plm, rep, level)

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
        prog="barrier_spec",
        description="Build the fan-in barrier spec for the recall-fp grid from the "
        "per-(pLM,representation) sidecar manifests.",
    )
    ap.add_argument("--sidecar-dir", required=True, help="Directory of recall_fp_*.manifest.json.")
    ap.add_argument("--plms", nargs="+", required=True, help="pLM names defining the grid.")
    ap.add_argument(
        "--representations", nargs="+", default=["raw", "ffn"],
        help="Representation axis (default: raw ffn).",
    )
    ap.add_argument(
        "--levels", nargs="+", default=list(DEFAULT_LEVELS),
        choices=("fold", "superfamily", "family"),
        help=f"CATH levels (default: {' '.join(DEFAULT_LEVELS)}; family deferred, W3).",
    )
    ap.add_argument("--out", required=True, help="Output barrier_spec.json path.")
    ap.add_argument(
        "--expected-n-plms", type=int, default=None,
        help="Guard against silent under-coverage: fail unless the grid has exactly "
        "this many unique pLMs (the DAG passes the intended size, e.g. 15).",
    )
    ap.add_argument(
        "--no-expected-rows", action="store_true",
        help="Do not set expected_rows from the sidecars; rely on the barrier's "
        "0-row/unique/non-null/finite guards only.",
    )
    args = ap.parse_args(argv)

    try:
        spec = build_recall_fp_barrier_spec(
            args.sidecar_dir,
            plms=args.plms,
            representations=args.representations,
            levels=args.levels,
            use_expected_rows=not args.no_expected_rows,
            expected_n_plms=args.expected_n_plms,
        )
        # The spec is a regenerable build product; the DAG always replaces in place.
        written = write_barrier_spec(spec, args.out, overwrite=True)
    except SpecBuildError as e:
        print(f"barrier_spec: CONFIG ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except OSError as e:
        print(f"barrier_spec: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    meta = spec["_meta"]
    print(
        f"barrier_spec: {meta['n_cells']} cell(s) -> {written}", flush=True
    )
    if meta["n_cells_without_sidecar"]:
        # "no silent caps": surface the gaps the barrier will report as missing.
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
