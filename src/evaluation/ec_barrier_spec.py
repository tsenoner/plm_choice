"""Fan-in barrier spec-builder for the EC grid (pLM x distance).

The EC analogue of snn_barrier_spec: walks the plm x distance grid, reads each
per-cell sidecar that ec_report.main wrote, and emits the {"artifacts": [...]} payload
the generic analysis_barrier consumes. Same three guarantees as the sibling builders:
full grid (no silent gaps), sidecar-path-authoritative, single-source-of-truth contract
(columns transcribed from ec_report.EC_PARQUET_GUARDS / EC_PER_PAIR_COLUMNS).

CLI::

    python -m evaluation.ec_barrier_spec --sidecar-dir <dir> \\
        --plms prott5 esm2 ... --distances euclidean cosine --out ec_barrier_spec.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from evaluation.ec_report import EC_PARQUET_GUARDS, EC_PER_PAIR_COLUMNS
from evaluation.barrier_spec_base import (
    SpecBuildError,
    check_per_query_columns_drift,
    dedup,
    emit_cell,
    read_sidecar_dict,
    require_grid_size,
    write_barrier_spec,
)

DEFAULT_DISTANCES: tuple[str, ...] = ("euclidean", "cosine")


def _stem(plm: str, rep: str, distance: str) -> str:
    return f"ec_{plm}_{rep}_{distance}"


def _load_sidecar(path: Path) -> dict:
    manifest = read_sidecar_dict(path)
    p = manifest.get("path")
    if not isinstance(p, str) or not p:
        raise SpecBuildError(f"sidecar has no non-empty string 'path': {path}")
    v = manifest.get("n_pairs")
    if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
        raise SpecBuildError(f"sidecar 'n_pairs' must be int or null, got {v!r} ({path}).")
    # GENUINE reuse of the shared base drift guard, pointed at the EC manifest's column
    # key (the EC sidecar records columns under 'per_pair_columns', not 'per_query_columns').
    # This requires the one-line base generalization in Step 0 below.
    check_per_query_columns_drift(manifest, EC_PER_PAIR_COLUMNS, path, key="per_pair_columns")
    return manifest


def build_ec_barrier_spec(
    sidecar_dir: Path | str,
    *,
    plms: Sequence[str],
    distances: Sequence[str] = DEFAULT_DISTANCES,
    representation: str = "raw",
    use_expected_rows: bool = True,
    expected_n_plms: int | None = None,
) -> dict:
    """Build the EC barrier spec over the plm x distance grid."""
    sidecar_dir = Path(sidecar_dir)
    plms = dedup(list(plms))
    distances = dedup(list(distances))
    if not plms:
        raise SpecBuildError("no plms given; the EC grid cannot be empty.")
    require_grid_size(plms, expected_n_plms, singular="pLM", axis_label="plms")

    artifacts: list[dict] = []
    reconstructed: list[str] = []
    population: dict[str, int] = {}
    for plm in plms:
        for dist in distances:
            label = f"ec:{plm}:{dist}"
            stem = _stem(plm, representation, dist)
            sc_path = sidecar_dir / f"{stem}.manifest.json"
            canonical = sidecar_dir / f"{stem}.parquet"
            covered = sc_path.exists()
            manifest = _load_sidecar(sc_path) if covered else None
            if manifest is not None:
                population[f"{plm}:{dist}"] = manifest.get("population_n")

            def _path_rows(_m=manifest):
                rows = _m.get("n_pairs") if use_expected_rows else None
                return _m["path"], rows

            art, recon = emit_cell(
                label, covered=covered, get_path_rows=_path_rows,
                canonical_parquet=canonical, guards=EC_PARQUET_GUARDS,
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
        prog="ec_barrier_spec",
        description="Build the fan-in barrier spec for the EC grid from per-cell sidecars.",
    )
    ap.add_argument("--sidecar-dir", required=True)
    ap.add_argument("--plms", nargs="+", required=True)
    ap.add_argument("--distances", nargs="+", default=list(DEFAULT_DISTANCES),
                    choices=DEFAULT_DISTANCES)
    ap.add_argument("--representation", default="raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-n-plms", type=int, default=None)
    ap.add_argument("--no-expected-rows", action="store_true")
    args = ap.parse_args(argv)

    try:
        spec = build_ec_barrier_spec(
            args.sidecar_dir, plms=args.plms, distances=args.distances,
            representation=args.representation,
            use_expected_rows=not args.no_expected_rows,
            expected_n_plms=args.expected_n_plms,
        )
        written = write_barrier_spec(spec, args.out, overwrite=True)
    except SpecBuildError as e:
        print(f"ec_barrier_spec: CONFIG ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except OSError as e:
        print(f"ec_barrier_spec: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    meta = spec["_meta"]
    print(f"ec_barrier_spec: {meta['n_cells']} cell(s) -> {written}", flush=True)
    if meta["n_cells_without_sidecar"]:
        print(f"  WARNING: {meta['n_cells_without_sidecar']} cell(s) had no sidecar: "
              f"{meta['reconstructed_cells']}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
