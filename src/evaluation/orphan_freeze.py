"""Derive + verify the orphan-cohort freeze (the Bromberg pairs file).

The orphan arm's frozen cohort is **the external Bromberg pairs file**, not a subset of
the canonical set — the metric is pair-level. A lightweight content-hash freeze guards
against a silently swapped / re-sorted / re-scored pairs file, mirroring ``ec_freeze``'s
anti-tautology hash.

Two-tier hash (§10-I3 of the design):

* the **headline cohort identity** is a SHA-256 over the sorted ``(p1, p2, siblings)``
  rows ONLY — the orphan AUROC depends on exactly those columns (``pident`` is dropped by
  the loader; ``TM``/``SNN`` feed only the secondary ρ). A re-scored TM column must NOT
  false-alarm the headline freeze;
* ``SNN`` / ``TM`` column hashes are recorded as **separate, non-gating** provenance
  fields so a swap there is *visible* without breaking the cohort-identity gate.

:func:`verify_orphan_freeze` re-derives the headline hash *independently* from a fresh
pairs frame (EC's anti-self-confirmation property) rather than re-reading the stored
value, so the freeze cannot validate itself trivially.

The freeze CODE ships + is fixture-tested offline; the committed *artifact*
``freeze/orphan_bromberg_11444.json`` is derived on LRZ at run time (the 309k-row pairs
file is LRZ-only — §10-I4/R3 of the design). The orphan ids are the union of ``p1``∪``p2``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1


def _require_columns(pairs: pd.DataFrame) -> None:
    needed = {"p1", "p2", "tm", "snn", "sibling"}
    missing = needed - set(pairs.columns)
    if missing:
        raise KeyError(
            f"pairs frame missing columns {sorted(missing)} "
            f"(expected the load_orphan_pairs schema)"
        )


def _headline_records(pairs: pd.DataFrame) -> list[tuple[str, str, bool]]:
    """Sorted ``(p1, p2, sibling)`` records — the cohort-identity hash input.

    Sorted by ``(p1, p2)`` so a re-ordered pairs file produces the identical hash
    (normalisation invariance, exactly ``ec_freeze._canonical_records``).
    """
    _require_columns(pairs)
    recs = [
        (str(a), str(b), bool(s))
        for a, b, s in zip(pairs["p1"], pairs["p2"], pairs["sibling"])
    ]
    recs.sort(key=lambda r: (r[0], r[1]))
    return recs


def headline_content_hash(records: list[tuple[str, str, bool]]) -> str:
    """SHA-256 of the sorted ``(p1, p2, sibling)`` records (stable JSON)."""
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _column_hash(pairs: pd.DataFrame, column: str) -> str:
    """SHA-256 of a single column, ordered by the sorted ``(p1, p2)`` key.

    Non-gating provenance: ties the column's *values* to the cohort's pair order, so a
    re-scored TM/SNN column is detectable without poisoning the headline cohort identity.
    """
    _require_columns(pairs)
    order = sorted(
        range(len(pairs)),
        key=lambda i: (str(pairs["p1"].iloc[i]), str(pairs["p2"].iloc[i])),
    )
    vals = [pairs[column].iloc[i] for i in order]
    payload = json.dumps([_jsonable(v) for v in vals], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(v):
    if isinstance(v, bool):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def derive_orphan_freeze(
    pairs: pd.DataFrame,
    *,
    derived_from: str,
    source_tsv: str,
) -> dict:
    """Build the orphan-cohort freeze manifest (pure; no I/O).

    ``ids`` is the sorted union of orphan ids appearing in either endpoint.
    ``content_sha256`` is the headline (cohort-identity) hash over ``(p1,p2,sibling)``;
    ``snn_sha256`` / ``tm_sha256`` are non-gating provenance.
    """
    records = _headline_records(pairs)
    ids = sorted(set(pairs["p1"].astype(str)) | set(pairs["p2"].astype(str)))
    n_siblings = int(pairs["sibling"].sum())
    return {
        "schema_version": SCHEMA_VERSION,
        "set_name": "orphan_bromberg",
        "derived_from": derived_from,
        "source_tsv": source_tsv,
        "loader": "orphan_io.load_orphan_pairs",
        "ids": ids,
        "n_proteins": len(ids),
        "n_pairs": int(len(pairs)),
        "n_siblings": n_siblings,
        "content_sha256": headline_content_hash(records),
        "snn_sha256": _column_hash(pairs, "snn"),
        "tm_sha256": _column_hash(pairs, "tm"),
    }


def verify_orphan_freeze(manifest: dict, pairs: pd.DataFrame) -> bool:
    """Independently re-derive the HEADLINE hash from ``pairs`` and compare.

    Raises :class:`ValueError` on cohort-identity drift (a changed ``p1``/``p2``/
    ``sibling``). A changed ``SNN``/``TM`` column does NOT raise (it is non-gating
    provenance) — re-deriving the column hash is left to the caller if it wants to warn.
    A second code path from :func:`derive_orphan_freeze`, so the freeze can't self-confirm.
    """
    expected = manifest.get("content_sha256")
    actual = headline_content_hash(_headline_records(pairs))
    if expected != actual:
        raise ValueError(
            f"orphan freeze content drift: manifest {expected!r} != re-derived {actual!r}"
        )
    return True


def write_orphan_freeze(
    manifest: dict, out_dir: Path | str = "freeze", *, overwrite: bool = False
) -> Path:
    """Write the manifest to ``<out_dir>/orphan_bromberg_11444.json`` (tracked dir)."""
    from shared.atomic_io import atomic_write

    path = Path(out_dir) / "orphan_bromberg_11444.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace")
    return atomic_write(
        path,
        lambda p: p.write_text(json.dumps(manifest, indent=2) + "\n"),
        mode="replace",
    )
