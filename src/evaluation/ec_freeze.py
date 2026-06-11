"""Derive + verify the EC-positive cohort freeze.

The EC arm's frozen population is the subset of the canonical set that carries a
valid EC number (after the wildcard policy). This is a *named, overwrite-with-intent*
artifact committed to the tracked ``freeze/`` directory (NOT ``data/freeze/`` —
``data/`` is gitignored, so a freeze written there would never be committed and the
verify gate would have nothing to check). The loader is the shared
``analysis_io.load_frozen_ids`` (reads only ``ids``); this module owns derivation +
an independent verification path so the anti-tautology test cannot self-confirm.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1


def _canonical_records(labels: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Sorted ``(protein_id, sorted-ec-list)`` records — the hash + ids input."""
    if not {"protein_id", "ec_set"}.issubset(labels.columns):
        raise KeyError("labels must have columns protein_id and ec_set")
    recs = [
        (str(pid), sorted(str(e) for e in ec_set))
        for pid, ec_set in zip(labels["protein_id"], labels["ec_set"])
        if ec_set  # non-empty EC set == EC-positive
    ]
    recs.sort(key=lambda r: r[0])
    return recs


def content_hash(records: list[tuple[str, list[str]]]) -> str:
    """SHA-256 of the canonical records (stable JSON, sorted)."""
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_ec_freeze(
    labels: pd.DataFrame,
    *,
    derived_from: str,
    source_tsv: str,
    wildcard_policy: str = "exclude",
    ec_col: str | None = None,
) -> dict:
    """Build the EC-positive freeze manifest (pure; no manifest I/O)."""
    records = _canonical_records(labels)
    ids = [pid for pid, _ in records]
    return {
        "schema_version": SCHEMA_VERSION,
        "set_name": "ec_positive_subset",
        "derived_from": derived_from,
        "source_tsv": source_tsv,
        "parser": "label_adapters.parse_ec",
        "parser_params": {"ec_col": ec_col, "wildcard_policy": wildcard_policy},
        "ids": ids,
        "n_proteins": len(ids),
        "content_sha256": content_hash(records),
    }


def verify_ec_freeze(manifest: dict, labels: pd.DataFrame) -> bool:
    """Independently re-derive the hash from ``labels`` and compare to the manifest.

    Raises ``ValueError`` on drift (hash mismatch) — a second code path from
    :func:`derive_ec_freeze` so the freeze can't validate itself trivially.
    """
    expected = manifest.get("content_sha256")
    actual = content_hash(_canonical_records(labels))
    if expected != actual:
        raise ValueError(
            f"EC freeze content drift: manifest {expected!r} != re-derived {actual!r}"
        )
    return True


def write_ec_freeze(manifest: dict, out_dir: Path | str = "freeze",
                    *, overwrite: bool = False) -> Path:
    """Write the manifest to ``<out_dir>/ec_positive_subset_319.json`` (tracked dir)."""
    from shared.atomic_io import atomic_write

    path = Path(out_dir) / "ec_positive_subset_319.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace")
    return atomic_write(
        path,
        lambda p: p.write_text(json.dumps(manifest, indent=2) + "\n"),
        mode="replace",
    )
