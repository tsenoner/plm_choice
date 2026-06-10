"""Shared I/O for the analysis-DAG bridges (embeddings, freeze, JSON serialisation).

Every analysis arm (recall-FP, SNN, EC, AAC floor, pdb-TM, ...) does the same three
boundary operations: load a per-protein embedding H5 into ``{id: 1-D vector}``, read
the committed canonical-set freeze for the expected population, and serialise a
manifest sidecar as standards-valid JSON (non-finite floats → ``null``). These were
first written privately inside ``recall_fp_report``; consolidating them here keeps a
single source of truth so the bridges cannot drift on how they subset, assert, and
serialise.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def load_embeddings_h5(path: Path | str) -> dict[str, np.ndarray]:
    """Load a per-protein embedding H5 into ``{protein_id: 1-D np.ndarray}``.

    Each dataset is one protein. A 2-D ``(L, D)`` per-residue dataset is mean-pooled
    over residues to a protein-level vector (matching
    :class:`data_preparation.distance_computation`'s loader), so a per-residue H5 is
    accepted as well as the reduced per-protein H5 the extract step writes.
    """
    import h5py

    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arr = np.asarray(f[key][()])  # [()] reads scalar + array datasets alike
            if arr.ndim > 1:
                arr = arr.mean(axis=0)  # (L, D) per-residue -> (D,) protein-level
            out[key] = np.asarray(arr, dtype=np.float32)
    return out


def load_frozen_ids(freeze_path: Path | str) -> list[str]:
    """Read the committed canonical-set freeze and return its ``ids`` list.

    The freeze (``canonical_set_<name>.json``) is the single source of truth for the
    analysis population — the caller must pass it rather than reconstruct the id set.
    Raises ``ValueError`` if the manifest carries no non-empty ``ids`` list (an
    operator/config fault → CLI exit 2).
    """
    data = json.loads(Path(freeze_path).read_text())
    ids = data.get("ids") if isinstance(data, dict) else None
    if not isinstance(ids, list) or not ids:
        raise ValueError(
            f"freeze {freeze_path} has no non-empty 'ids' list; pass the committed "
            f"canonical_set_<name>.json"
        )
    return ids


def json_safe(obj):
    """Recursively copy ``obj`` rendering every non-finite float (NaN/Inf) as ``None``.

    ``json.dumps`` would emit the bare tokens ``NaN`` / ``Infinity`` — accepted by
    Python's ``json.loads`` but invalid per the JSON spec, so a strict / non-Python
    reader (the barrier spec-builder) rejects the sidecar. The contract a manifest
    relies on is "a null metric == an undefined/0-population cell". The input is never
    mutated.

    NumPy scalars are coerced first: ``np.float32`` is **not** a Python ``float``
    subclass (``isinstance(np.float32('nan'), float)`` is False), so a manifest value
    that skipped a ``float(...)`` wrap at its source would otherwise slip through as a
    non-finite/unserialisable numpy scalar. ``np.float64`` IS a float subclass, but we
    normalise both for one consistent code path. This hardening matters for future arms
    that may forget the wrap; the current arms already wrap every metric in ``float(...)``.
    """
    if isinstance(obj, np.floating):
        obj = float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
