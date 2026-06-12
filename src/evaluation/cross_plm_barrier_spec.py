"""Fan-in barrier spec-builder for the cross-pLM agreement grid (pLM-pair x rep x distance).

The cross-pLM analogue of :mod:`evaluation.snn_barrier_spec`: walks the
``pLM-pair x representation x distance`` grid, reads the per-cell sidecar manifests that
:func:`evaluation.cross_plm_report.main` wrote, and emits the ``{"artifacts": [...]}`` payload
the generic :mod:`evaluation.analysis_barrier` consumes. Same three guarantees as the sibling
builders: full grid (no silent gaps), sidecar-path-authoritative, single-source-of-truth
contract (columns/guards transcribed from
:data:`evaluation.cross_plm_report.CROSS_PLM_PARQUET_GUARDS` /
:data:`~evaluation.cross_plm_report.CROSS_PLM_PER_PAIR_COLUMNS`).

This is a HYBRID of the two prior templates: the **pLM-pair grid + per-cell
``population_n {a, b}``** comes from the SNN builder; the **``n_pairs`` -> expected_rows and
``per_pair_columns`` drift key** come from the EC builder (a cross-pLM parquet is per-PAIR, not
per-query). It is cloned from ``snn_barrier_spec`` **specifically for its 3-element
``DEFAULT_DISTANCES``**: cloning ``ec_barrier_spec`` (2-element ``{euclidean, cosine}``) would
silently drop the entire manhattan column — the barrier does not enforce the U7 family count,
so a 2-element default here is invisible.

CLI exit codes mirror the sibling DAG mains: ``0`` — spec written; ``2`` — operator/config
fault. There is no exit 1 (this builds a spec, it does not validate data).

CLI::

    python -m evaluation.cross_plm_barrier_spec --sidecar-dir <dir> \\
        --pairs prott5,esm2 prott5,esm3 ... --representations raw \\
        --distances cosine euclidean manhattan --out cross_plm_barrier_spec.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from evaluation.cross_plm_report import CROSS_PLM_PARQUET_GUARDS, CROSS_PLM_PER_PAIR_COLUMNS
from evaluation.barrier_spec_base import (
    SpecBuildError,
    check_per_query_columns_drift,
    dedup,
    emit_cell,
    read_sidecar_dict,
    require_grid_size,
    write_barrier_spec,
)

# 3-element by design (cloned from snn_barrier_spec, NOT ec_barrier_spec) — a 2-element default
# would silently drop the manhattan column.
DEFAULT_DISTANCES: tuple[str, ...] = ("cosine", "euclidean", "manhattan")


def _stem(plm_a: str, plm_b: str, rep: str, distance: str) -> str:
    return f"cross_plm_{plm_a}__{plm_b}_{rep}_{distance}"


def _sidecar_path(d: Path, a: str, b: str, rep: str, distance: str) -> Path:
    return d / f"{_stem(a, b, rep, distance)}.manifest.json"


def _canonical_parquet(d: Path, a: str, b: str, rep: str, distance: str) -> Path:
    return d / f"{_stem(a, b, rep, distance)}.parquet"


def _load_sidecar(path: Path) -> dict:
    """Read + validate one cross-pLM sidecar. Raises SpecBuildError on any fault.

    Validates the cross-pLM flat shape: a non-empty string ``path``; integer ``n_pairs``
    (-> expected_rows) and ``population_n_a``/``population_n_b`` (-> _meta), so a corrupt
    sidecar surfaces as a clean operator fault here rather than a confusing barrier-time error.
    The ``per_pair_columns`` drift guard is called last to preserve error precedence (shape
    checks fire before the drift check). bool is an int subclass -> reject it explicitly.
    """
    manifest = read_sidecar_dict(path)
    p = manifest.get("path")
    if not isinstance(p, str) or not p:
        raise SpecBuildError(f"sidecar has no non-empty string 'path': {path}")
    for key in ("n_pairs", "population_n_a", "population_n_b"):
        v = manifest.get(key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
            raise SpecBuildError(
                f"sidecar {key!r} must be an integer or null, got {v!r} ({path})."
            )
    check_per_query_columns_drift(
        manifest, CROSS_PLM_PER_PAIR_COLUMNS, path, key="per_pair_columns")
    return manifest


def build_cross_plm_barrier_spec(
    sidecar_dir: Path | str,
    *,
    pairs: Sequence[tuple[str, str]],
    representations: Sequence[str] = ("raw",),
    distances: Sequence[str] = DEFAULT_DISTANCES,
    use_expected_rows: bool = True,
    expected_n_pairs: int | None = None,
) -> dict:
    """Build the ``{"artifacts": [...]}`` barrier spec over the cross-pLM agreement grid.

    Parameters
    ----------
    sidecar_dir
        Directory holding the ``cross_plm_<a>__<b>_<rep>_<distance>.manifest.json`` sidecars
        (and, by convention, the parquets they reference).
    pairs
        The (plm_a, plm_b) pairs defining the grid (e.g. all C(15,2) unordered pairs).
        Required and non-empty. De-duplicated, order preserved.
    representations
        Representation axis (default ``("raw",)`` — cross-pLM has no FFN producer).
    distances
        Distance axis (default cosine + euclidean + manhattan).
    use_expected_rows
        If True (default), set each cell's ``expected_rows`` from the sidecar's ``n_pairs``
        so a truncated parquet fails the barrier on row count.
    expected_n_pairs
        Optional guard against silent under-coverage: if given, the number of *unique* pairs
        must equal it (the DAG submit script passes the intended size, e.g. 105).

    Returns
    -------
    dict
        ``{"artifacts": [...], "_meta": {...}}``. ``_meta`` records ``n_cells``,
        ``n_cells_without_sidecar``, the reconstructed-cell labels, and per-cell
        ``population_n`` ``{"a": n_a, "b": n_b}`` (so a capped pLM's cap survives this step).

    Raises
    ------
    SpecBuildError
        Empty ``pairs``, an ``expected_n_pairs`` mismatch, an orphan parquet (canonical
        parquet present with no sidecar), or a malformed/contradictory sidecar.
    """
    sidecar_dir = Path(sidecar_dir)
    pairs = dedup([tuple(p) for p in pairs])
    representations = dedup(representations)
    distances = dedup(distances)
    if not pairs:
        raise SpecBuildError("no pairs given; the cross-pLM grid cannot be empty.")
    require_grid_size(pairs, expected_n_pairs, singular="pair", axis_label="pairs")

    artifacts: list[dict] = []
    reconstructed: list[str] = []
    population: dict[str, dict] = {}
    for a, b in pairs:
        for rep in representations:
            for dist in distances:
                label = f"cross_plm:{a}:{b}:{rep}:{dist}"
                sc_path = _sidecar_path(sidecar_dir, a, b, rep, dist)
                canonical = _canonical_parquet(sidecar_dir, a, b, rep, dist)
                covered = sc_path.exists()
                manifest = _load_sidecar(sc_path) if covered else None
                if manifest is not None:
                    population[f"{a}__{b}:{rep}:{dist}"] = {
                        "a": manifest.get("population_n_a"),
                        "b": manifest.get("population_n_b"),
                    }

                # manifest is captured directly: emit_cell calls this synchronously within the
                # same iteration, so there is no late-binding concern.
                def _path_rows(_m=manifest):
                    rows = _m.get("n_pairs") if use_expected_rows else None
                    return _m["path"], rows

                art, recon = emit_cell(
                    label, covered=covered, get_path_rows=_path_rows,
                    canonical_parquet=canonical, guards=CROSS_PLM_PARQUET_GUARDS,
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


def _parse_pair(s: str) -> tuple[str, str]:
    parts = s.split(",")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise SpecBuildError(f"--pairs entry {s!r} must be 'plm_a,plm_b'")
    return parts[0], parts[1]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cross_plm_barrier_spec",
        description="Build the fan-in barrier spec for the cross-pLM agreement grid from the "
        "per-cell sidecar manifests.",
    )
    ap.add_argument("--sidecar-dir", required=True, help="Directory of cross_plm_*.manifest.json.")
    ap.add_argument(
        "--pairs", nargs="*", default=[],
        help="pLM pairs 'plm_a,plm_b' defining the grid (e.g. prott5,esm2 prott5,esm3).",
    )
    ap.add_argument(
        "--representations", nargs="+", default=["raw"],
        help="Representation axis (default: raw).",
    )
    ap.add_argument(
        "--distances", nargs="+", default=list(DEFAULT_DISTANCES),
        choices=DEFAULT_DISTANCES,
        help=f"Distance axis (default: {' '.join(DEFAULT_DISTANCES)}).",
    )
    ap.add_argument("--out", required=True, help="Output barrier spec JSON path.")
    ap.add_argument(
        "--expected-n-pairs", type=int, default=None,
        help="Guard against silent under-coverage: fail unless the grid has exactly this many "
        "unique pairs (the DAG passes the intended size, e.g. 105).",
    )
    ap.add_argument(
        "--no-expected-rows", action="store_true",
        help="Do not set expected_rows from the sidecars; rely on the barrier's "
        "0-row/unique/non-null/finite guards only.",
    )
    args = ap.parse_args(argv)

    try:
        pairs = [_parse_pair(s) for s in args.pairs]
        spec = build_cross_plm_barrier_spec(
            args.sidecar_dir,
            pairs=pairs,
            representations=args.representations,
            distances=args.distances,
            use_expected_rows=not args.no_expected_rows,
            expected_n_pairs=args.expected_n_pairs,
        )
        written = write_barrier_spec(spec, args.out, overwrite=True)
    except SpecBuildError as e:
        print(f"cross_plm_barrier_spec: CONFIG ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except OSError as e:
        print(f"cross_plm_barrier_spec: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    meta = spec["_meta"]
    print(f"cross_plm_barrier_spec: {meta['n_cells']} cell(s) -> {written}", flush=True)
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
