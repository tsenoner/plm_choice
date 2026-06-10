"""Merge per-arm fan-in barrier specs into one ``barrier_spec.json``.

The unified analysis DAG (``submit_analysis_dag.sh``) runs each arm's spec-builder
(recall-fp, SNN, EC, ...), each emitting ``{"artifacts": [...], "_meta": {...}}``,
then merges them so a single :mod:`evaluation.analysis_barrier` gates the whole
fan-in. Merge concatenates artifacts opaquely (preserving every field for future
H5/guard kinds), fails closed on a duplicate label across arms (two arms claiming
one cell would be double-validated), and keeps each arm's ``_meta`` verbatim under
``arms`` so heterogeneous ``population_n`` shapes (and the capped-cohort signal)
survive. ``n_cells`` is recomputed authoritatively from the merged artifacts.

CLI exit codes mirror the sibling DAG mains: ``0`` — merged spec written; ``2`` —
operator/config fault (empty input, malformed spec, label collision, unwritable out).

CLI::

    python -m evaluation.spec_merge --specs recall.json snn.json ... --out barrier_spec.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from evaluation.barrier_spec_base import SpecBuildError, write_barrier_spec


def merge_specs(specs: Sequence[dict], *, names: "Sequence[str] | None" = None) -> dict:
    """Fold per-arm specs into one ``{"artifacts": [...], "_meta": {...}}`` payload.

    Parameters
    ----------
    specs
        Per-arm spec dicts (each ``{"artifacts": [...], "_meta": {...}}``). Required
        and non-empty — an empty merge would make the barrier vacuously pass.
    names
        Optional per-arm source labels (e.g. filename stems), parallel to ``specs``;
        defaults to ``arm_0``, ``arm_1``, ... Recorded under ``_meta.arms[].source``.

    Raises
    ------
    SpecBuildError
        Empty ``specs``; a ``names``/``specs`` length mismatch; a spec that is not a
        dict / lacks an ``artifacts`` list; an artifact lacking a non-empty
        ``label``/``path``; or a duplicate ``label`` anywhere across the merged set.
    """
    if not specs:
        raise SpecBuildError("no specs to merge; the merged barrier grid cannot be empty.")
    if names is not None and len(names) != len(specs):
        raise SpecBuildError(
            f"names length {len(names)} != specs length {len(specs)}."
        )

    merged: list[dict] = []
    seen: dict[str, int] = {}  # label -> first source index
    arms: list[dict] = []
    n_without = 0
    reconstructed: list[str] = []

    for i, spec in enumerate(specs):
        src = names[i] if names is not None else f"arm_{i}"
        if not isinstance(spec, dict):
            raise SpecBuildError(
                f"spec #{i} ({src}) is not an object: {type(spec).__name__}."
            )
        arts = spec.get("artifacts")
        if not isinstance(arts, list):
            raise SpecBuildError(f"spec #{i} ({src}) has no 'artifacts' list.")
        for a in arts:
            if not isinstance(a, dict):
                raise SpecBuildError(
                    f"spec #{i} ({src}) artifact is not an object: {type(a).__name__}."
                )
            label = a.get("label")
            if not isinstance(label, str) or not label:
                raise SpecBuildError(
                    f"spec #{i} ({src}) artifact has no non-empty 'label'."
                )
            path = a.get("path")
            if not isinstance(path, str) or not path:
                raise SpecBuildError(
                    f"spec #{i} ({src}) artifact {label!r} has no non-empty 'path'."
                )
            if label in seen:
                raise SpecBuildError(
                    f"duplicate artifact label {label!r}: appears in spec #{seen[label]} "
                    f"and #{i} ({src}); two arms claiming the same cell would be "
                    f"double-validated by the barrier."
                )
            seen[label] = i
            merged.append(a)            # opaque passthrough — preserve every field

        meta = spec.get("_meta")
        if isinstance(meta, dict):
            nws = meta.get("n_cells_without_sidecar", 0)
            if nws is None:
                nws = 0  # explicit null -> tolerate as 0 (optional metadata)
            # A present non-int (incl. bool, which is an int subclass) is a corrupt
            # sidecar -> fail closed (exit 2), not an uncaught ValueError downstream.
            if isinstance(nws, bool) or not isinstance(nws, int):
                raise SpecBuildError(
                    f"spec #{i} ({src}) _meta.n_cells_without_sidecar must be an int, "
                    f"got {nws!r}."
                )
            n_without += nws
            rc = meta.get("reconstructed_cells", [])
            if rc is None:
                rc = []
            if not isinstance(rc, list):
                raise SpecBuildError(
                    f"spec #{i} ({src}) _meta.reconstructed_cells must be a list, "
                    f"got {rc!r}."
                )
            reconstructed.extend(rc)
            arms.append({"source": src, "meta": meta})
        else:
            arms.append({"source": src, "meta": None})

    return {
        "artifacts": merged,
        "_meta": {
            "n_cells": len(merged),
            "n_cells_without_sidecar": n_without,
            "reconstructed_cells": reconstructed,
            "arms": arms,
        },
    }


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        prog="spec_merge",
        description="Merge per-arm barrier specs into one barrier_spec.json.",
    )
    ap.add_argument("--specs", nargs="+", required=True,
                    help="Per-arm barrier spec JSON files.")
    ap.add_argument("--out", required=True, help="Output merged barrier_spec.json path.")
    args = ap.parse_args(argv)

    try:
        loaded: list[dict] = []
        names: list[str] = []
        for s in args.specs:
            p = Path(s)
            try:
                payload = json.loads(p.read_text())
            except json.JSONDecodeError as e:
                raise SpecBuildError(f"spec file is not valid JSON: {p}: {e}") from e
            except OSError as e:
                raise SpecBuildError(f"spec file unreadable: {p}: {e}") from e
            loaded.append(payload)
            names.append(p.stem)
        merged = merge_specs(loaded, names=names)
        written = write_barrier_spec(merged, args.out, overwrite=True)
    except SpecBuildError as e:
        print(f"spec_merge: CONFIG ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except OSError as e:
        print(f"spec_merge: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    meta = merged["_meta"]
    print(f"spec_merge: {meta['n_cells']} cell(s) from {len(loaded)} arm(s) -> {written}",
          flush=True)
    if meta["n_cells_without_sidecar"]:
        print(
            f"  WARNING: {meta['n_cells_without_sidecar']} cell(s) had no sidecar; "
            f"emitted reconstructed-path specs (the barrier will report them missing): "
            f"{meta['reconstructed_cells']}",
            file=sys.stderr, flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
