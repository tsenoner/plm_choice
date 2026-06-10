"""Shared base for the per-arm fan-in barrier spec-builders.

Each analysis arm (recall-fp, SNN, EC, AAC, cross-pLM, pdb-TM, orphan) ships a
``<arm>_barrier_spec.py`` that walks its own grid and reads its own sidecar shape,
then emits the ``{"artifacts": [...], "_meta": {...}}`` payload the generic
:mod:`evaluation.analysis_barrier` validates. The grid nesting and sidecar shape
are genuinely arm-specific (recall-fp's sidecar is per-(pLM,rep) with a ``levels``
map; SNN's is per-cell flat), so they stay in each arm. This module owns the parts
that are identical across arms: the error type, order-preserving de-dup, the
ArtifactSpec-shaped artifact dict (with a guard-field contract check), a
structural-only sidecar reader, the ``per_query_columns`` drift guard, the
silent-under-coverage grid-size guard, the per-cell orphan/reconstruct *tail*
(:func:`emit_cell` — the single riskiest shared decision), and the atomic writer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence, TypeVar

_T = TypeVar("_T")

from shared.atomic_io import atomic_write

# The ArtifactSpec parquet-guard field names make_artifact accepts. A guards dict
# with any other key would be silently dropped by analysis_barrier._spec_from_dict
# (it .get()s known keys only), so a typo'd guard key must fail loud here instead.
_GUARD_FIELDS = frozenset(
    {"required_columns", "unique_columns", "non_null_columns", "finite_columns"}
)


class SpecBuildError(Exception):
    """A barrier spec cannot be built (operator/config fault -> exit 2)."""


def dedup(seq: Sequence[_T]) -> list[_T]:
    """Order-preserving de-duplication (so a duplicated grid axis can't mask a gap).

    Works on any hashable elements (pLM-name strings and (plm_a, plm_b) tuples).
    """
    seen: set = set()
    out: list = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def make_artifact(label: str, path: Path | str, expected_rows: int | None,
                  guards: dict) -> dict:
    """One ArtifactSpec-shaped dict armed with an arm's parquet guard contract.

    ``guards`` keys must be a subset of the ArtifactSpec parquet-guard fields; an
    unknown key (typo) raises rather than silently vanishing at validation time.
    """
    bad = set(guards) - _GUARD_FIELDS
    if bad:
        raise SpecBuildError(
            f"unknown guard field(s) {sorted(bad)} in make_artifact; "
            f"valid fields are {sorted(_GUARD_FIELDS)}."
        )
    return {
        "label": label,
        "path": str(path),
        "expected_rows": expected_rows,
        "kind": "parquet",
        **{key: list(cols) for key, cols in guards.items()},
    }


def read_sidecar_dict(path: Path | str) -> dict:
    """Read a sidecar manifest with STRUCTURAL validation only: readable + JSON object.

    Semantic validation (levels-map vs flat, required fields, drift) is arm-specific
    and stays in each builder's ``_load_sidecar``. Keeping this reader-only preserves
    each arm's error precedence (the arm-shape check fires before the drift check).
    """
    try:
        manifest = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise SpecBuildError(f"sidecar is not valid JSON: {path}: {e}") from e
    except OSError as e:
        raise SpecBuildError(f"sidecar unreadable: {path}: {e}") from e
    if not isinstance(manifest, dict):
        raise SpecBuildError(f"sidecar must be a JSON object: {path}")
    return manifest


def check_per_query_columns_drift(manifest: dict, contract: Sequence[str],
                                  path: Path | str) -> None:
    """Fail loud if the sidecar's per_query_columns disagree with the arm's contract.

    A sidecar predating the field (key absent or value ``None``) legitimately skips
    the check.
    """
    cols = manifest.get("per_query_columns")
    if cols is not None and (not isinstance(cols, list) or tuple(cols) != tuple(contract)):
        raise SpecBuildError(
            f"sidecar per_query_columns {cols!r} disagree with the contract "
            f"{list(contract)} ({path}); schema drift."
        )


def require_grid_size(axis: Sequence, expected: int | None, *,
                      singular: str, plural_key: str) -> None:
    """Silent-under-coverage guard: fail unless the deduped axis has ``expected`` items."""
    if expected is not None and len(axis) != expected:
        raise SpecBuildError(
            f"grid has {len(axis)} unique {singular}(s) but expected {expected}; "
            f"refusing to build a spec over an under/over-specified grid "
            f"(silent under-coverage guard). {plural_key}={axis}"
        )


def emit_cell(label: str, *, covered: bool,
              get_path_rows: Callable[[], tuple[str, int | None]],
              canonical_parquet: Path | str, guards: dict) -> tuple[dict, bool]:
    """Resolve one grid cell to ``(artifact, reconstructed)``.

    ``covered`` — does an authoritative sidecar cover this cell? When True,
    ``get_path_rows()`` supplies the sidecar-authoritative (path, expected_rows).
    When False, a canonical parquet sitting there with no sidecar is an orphan
    (stale/partial artifact, sidecar is written last) -> fail closed; otherwise the
    cell is genuinely absent -> emit the reconstructed canonical path so the barrier
    reports it MISSING rather than the gap going unnoticed.
    """
    if covered:
        path, rows = get_path_rows()
        return make_artifact(label, path, rows, guards), False
    if Path(canonical_parquet).exists():
        raise SpecBuildError(
            f"orphan parquet without a sidecar: {canonical_parquet} (cell {label}); "
            f"a parquet present with no manifest is a stale/partial artifact "
            f"— remove it or re-run the cell with --overwrite."
        )
    return make_artifact(label, canonical_parquet, None, guards), True


def write_barrier_spec(spec: dict, out_path: Path | str, *,
                       overwrite: bool = True) -> Path:
    """Atomically write the barrier spec JSON; return where it landed.

    The spec is a regenerable build product (rebuilt each DAG submission), so the
    default is atomic in-place replacement (``overwrite=True``); pass
    ``overwrite=False`` for a never-clobber timestamped sibling.
    """
    return atomic_write(
        Path(out_path),
        lambda p: p.write_text(json.dumps(spec, indent=2) + "\n"),
        mode="replace" if overwrite else "timestamp",
    )
