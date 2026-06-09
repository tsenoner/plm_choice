"""Canonical-set freeze for the revision analyses (plan v3, Phase 0 item 1 + NEW-3).

The whole pLM comparison is defined over one frozen protein set — the canonical 319
(``2024_novelSeqs2.fasta``). Two correctness controls hang off this freeze:

* :func:`~evaluation.population.assert_population` asserts every analysis input against
  the frozen id set, so a pLM silently missing proteins never gets mean-pooled with its
  peers over a different population.
* every pairwise metric is computed over ONE frozen pair index, so all pLMs are compared
  on identical pairs (W4) rather than on whatever each pLM happened to retain.

This module produces that freeze and is the single source of truth ``verify_analysis``
checks against. The freeze records:

* ``canonical_content_sha256`` — a normalization-invariant hash of the ``(id, sequence)``
  set: records are sorted by id, sequences upper-cased, emitted as one ``id\\tseq\\n`` line
  each. It changes iff the *sequence set* changes — not on cosmetic reformatting (line-wrap
  width, header description, record order). This is the hash ``verify_analysis`` asserts.
* ``raw_file_sha256`` — sha256 of the exact FASTA bytes; informational, flags any re-write.
* ``ids`` — the sorted canonical id list (``assert_population``'s ``expected``).
* ``n_pairs`` — ``C(n, 2)``; the frozen pair index (built by :func:`build_pair_index`) is
  the common index every pairwise metric uses.
* ``esm1b`` (NEW-3) — ESM-1b is architecture-capped at 1022 aa, so it covers a strict
  subset (267/319; the 52 absent are all > 1022 aa). The freeze records the absent ids and
  attributes them to the cap, so a caller can ``assert_population(..., allow_capped=True)``
  for esm1b and report its per-cell ``n`` (267) separately, never folding it into a bare mean.
  ``esm1b_paired_policy`` is left ``None`` here — the choice between *common-267-for-all* and
  *footnote-esm1b-out* changes the Holm denominator and per-cell N, so it is a co-PI decision
  recorded by editing the frozen manifest, not inferred by this builder.

FASTA parsing is dependency-free (no pyfaidx) so the freeze is hermetic — it writes no
``.fai`` sidecar next to the source and is trivially unit-testable with synthetic fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = 1

# The two legal esm1b paired-stats policies (NEW-3); see the freeze README. ``None`` = unlocked.
ESM1B_PAIRED_POLICIES = ("common_267_for_all", "footnote_esm1b_out")


# ── FASTA parsing ───────────────────────────────────────────────────────────────
def parse_fasta(path: Path | str) -> list[tuple[str, str]]:
    """Parse a FASTA file into ``[(id, sequence), ...]`` in file order.

    The id is the first whitespace-delimited token after ``>`` (UniProt accession for the
    canonical set). Sequence lines are concatenated verbatim (case preserved here;
    normalization happens in :func:`canonical_content_sha256`). CRLF (``\\r\\n``) line
    endings are tolerated.

    A *malformed* canonical file fails loudly rather than hashing to a confident-but-wrong
    value: a ``>`` header with no id, or any non-blank sequence content before the first
    header, raises ``ValueError``.

    Deliberately not :func:`data_preparation.embeddings.embedding_generation.read_fasta_sequences`,
    which uses ``pyfaidx`` and writes a ``.fai`` sidecar next to the source — the freeze
    must stay hermetic (no mutation of the canonical input directory).
    """
    records: list[tuple[str, str]] = []
    cur_id: str | None = None
    cur_seq: list[str] = []
    with open(path, "r") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\r\n")
            if line.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, "".join(cur_seq)))
                tokens = line[1:].split()
                if not tokens:
                    raise ValueError(
                        f"FASTA header with no id at line {lineno}: {line!r}"
                    )
                cur_id = tokens[0]
                cur_seq = []
            elif line.strip() == "":
                continue  # blank lines are harmless padding
            elif cur_id is not None:
                cur_seq.append(line.strip())
            else:
                raise ValueError(
                    f"sequence content before the first FASTA header "
                    f"at line {lineno}: {line!r}"
                )
    if cur_id is not None:
        records.append((cur_id, "".join(cur_seq)))
    return records


# ── hashing ─────────────────────────────────────────────────────────────────────
def canonical_content_sha256(records: Iterable[tuple[str, str]]) -> str:
    """Normalization-invariant sha256 of an ``(id, sequence)`` set.

    Records are sorted by id and emitted as ``f"{id}\\t{seq.upper()}\\n"``, so the hash is
    invariant to record order, line-wrap width, and sequence case, but changes if any id or
    residue changes. Raises ``ValueError`` on a duplicate id (an ambiguous canonical set).
    """
    seen: set[str] = set()
    items: list[tuple[str, str]] = []
    for pid, seq in records:
        if pid in seen:
            raise ValueError(f"duplicate id in canonical set: {pid!r}")
        seen.add(pid)
        items.append((pid, seq.upper()))
    items.sort(key=lambda t: t[0])
    h = hashlib.sha256()
    for pid, seq in items:
        h.update(f"{pid}\t{seq}\n".encode("utf-8"))
    return h.hexdigest()


def raw_file_sha256(path: Path | str) -> str:
    """sha256 of the exact file bytes (informational; flags any re-write)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── pair index ──────────────────────────────────────────────────────────────────
def build_pair_index(ids: Sequence[str]):
    """All ``C(n, 2)`` unordered id pairs as a DataFrame (``id_a``, ``id_b``).

    Each pair is canonicalised ``id_a < id_b`` and rows are sorted by ``(id_a, id_b)`` so
    the index is deterministic across runs and machines — the frozen common index every
    pairwise metric joins onto (W4). Raises ``ValueError`` on a duplicate input id.
    """
    import pandas as pd

    if len(set(ids)) != len(ids):
        raise ValueError("duplicate id in pair-index input; ids must be unique")
    ordered = sorted(ids)
    pairs = list(itertools.combinations(ordered, 2))  # already (a < b), (a,b)-sorted
    return pd.DataFrame(pairs, columns=["id_a", "id_b"])


# ── freeze ──────────────────────────────────────────────────────────────────────
def freeze_canonical_set(
    fasta_path: Path | str,
    *,
    set_name: str,
    esm1b_keys: Iterable[str] | None = None,
    cap_aa: int = 1022,
    source_uri: str | None = None,
    esm1b_paired_policy: str | None = None,
) -> dict:
    """Build the freeze manifest for ``fasta_path`` (pure; no I/O of the manifest itself).

    Parameters
    ----------
    fasta_path
        The canonical FASTA (e.g. ``2024_novelSeqs2.fasta``).
    set_name
        Logical name for the frozen set (e.g. ``"canonical_319"``).
    esm1b_keys
        If given, the protein ids esm1b actually embedded. The manifest then records esm1b's
        capped coverage (NEW-3): covered count, absent ids, and whether every absent id is
        attributable to the > ``cap_aa`` architecture cap. Raises ``ValueError`` if any
        esm1b key is foreign to the canonical set.
    cap_aa
        ESM-1b's residue cap (1022); used only to attribute esm1b absences.
    source_uri
        Optional machine-readable provenance (e.g. the LRZ source path) recorded verbatim
        in the manifest. Kept deterministic — no wall-clock/git stamp — so re-freezing the
        same input reproduces the same manifest byte-for-byte.

    Returns
    -------
    dict — the manifest (see module docstring for the schema).
    """
    if esm1b_paired_policy is not None and esm1b_paired_policy not in ESM1B_PAIRED_POLICIES:
        raise ValueError(
            f"esm1b_paired_policy must be one of {ESM1B_PAIRED_POLICIES} or None, "
            f"got {esm1b_paired_policy!r}"
        )
    fasta_path = Path(fasta_path)
    records = parse_fasta(fasta_path)
    if not records:
        raise ValueError(
            f"no FASTA records in {fasta_path}; a canonical set cannot be empty."
        )
    ids = [pid for pid, _ in records]
    dups = sorted(p for p, c in Counter(ids).items() if c > 1)
    if dups:
        raise ValueError(f"duplicate id(s) in canonical set: {dups[:5]}")
    lengths = {pid: len(seq) for pid, seq in records}
    sorted_ids = sorted(ids)

    manifest: dict = {
        "schema_version": SCHEMA_VERSION,
        "set_name": set_name,
        "source_fasta": fasta_path.name,
        "source_uri": source_uri,
        "n_proteins": len(sorted_ids),
        "n_pairs": len(sorted_ids) * (len(sorted_ids) - 1) // 2,
        "raw_file_sha256": raw_file_sha256(fasta_path),
        "canonical_content_sha256": canonical_content_sha256(records),
        "ids": sorted_ids,
        "esm1b": None,
    }

    if esm1b_keys is not None:
        keys = set(esm1b_keys)
        canonical = set(sorted_ids)
        foreign = keys - canonical
        if foreign:
            raise ValueError(
                f"{len(foreign)} esm1b key(s) foreign to the canonical set "
                f"(e.g. {sorted(foreign)[:5]}); not in the frozen population."
            )
        missing = sorted(canonical - keys)
        missing_lens = [lengths[m] for m in missing]
        manifest["esm1b"] = {
            "n_covered": len(canonical & keys),
            "n_missing": len(missing),
            "cap_aa": cap_aa,
            "missing_ids": missing,
            "missing_all_over_cap": bool(missing) and all(l > cap_aa for l in missing_lens),
            "missing_len_min": min(missing_lens) if missing_lens else None,
            "missing_len_max": max(missing_lens) if missing_lens else None,
            # Co-PI decision (NEW-3); validated against ESM1B_PAIRED_POLICIES.
            # Allowed: "common_267_for_all" | "footnote_esm1b_out" | None (unlocked).
            "esm1b_paired_policy": esm1b_paired_policy,
        }

    return manifest


# ── I/O driver ──────────────────────────────────────────────────────────────────
def write_freeze(
    manifest: dict,
    out_dir: Path | str,
    *,
    set_name: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write the manifest JSON and the derived pair-index parquet to their canonical paths.

    The manifest is the committed source of truth; the pair index is the (regenerable)
    frozen common index for pairwise metrics. A freeze is a *named, overwrite-with-intent*
    artifact — never a timestamped sibling — so a regenerated freeze always lands at
    ``canonical_set_<name>.json`` / ``pair_index_<name>.parquet`` and ``verify_analysis``
    can never read a stale file while believing the freeze was refreshed.

    Each file is written via :func:`shared.atomic_io.atomic_write` with ``mode="replace"``
    (tmp-file + ``os.replace``) so a killed write never leaves a valid-looking partial (B7),
    but the *target path is always canonical*. If a target already exists, this raises
    ``FileExistsError`` unless ``overwrite=True`` — so an accidental re-freeze is caught,
    while a deliberate refresh (CLI ``--overwrite``) replaces in place atomically.

    Returns ``{"manifest": Path, "pair_index": Path}`` — the (canonical) paths written.
    """
    from shared.atomic_io import atomic_write

    out_dir = Path(out_dir)
    manifest_path = out_dir / f"canonical_set_{set_name}.json"
    pair_index_path = out_dir / f"pair_index_{set_name}.parquet"

    existing = [p for p in (manifest_path, pair_index_path) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"freeze target(s) already exist: {[str(p) for p in existing]}; "
            f"pass overwrite=True (CLI --overwrite) to atomically replace them."
        )

    df = build_pair_index(manifest["ids"])

    written_manifest = atomic_write(
        manifest_path,
        lambda p: p.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n"),
        mode="replace",
    )
    written_pairs = atomic_write(
        pair_index_path,
        lambda p: df.to_parquet(p, index=False),
        mode="replace",
    )
    return {"manifest": written_manifest, "pair_index": written_pairs}


def _read_h5_keys(h5_path: Path | str) -> list[str]:
    import h5py

    with h5py.File(h5_path, "r") as f:
        return list(f.keys())


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="canonical_set",
        description="Freeze a canonical protein set (sha256 + ids + pair index + esm1b coverage).",
    )
    ap.add_argument("--fasta", required=True, help="Canonical FASTA to freeze.")
    ap.add_argument("--set-name", required=True, help="Logical name, e.g. canonical_319.")
    ap.add_argument("--out-dir", required=True, help="Directory for the manifest + parquet.")
    ap.add_argument(
        "--esm1b-h5",
        default=None,
        help="Optional esm1b embeddings H5; its keys define esm1b's capped coverage (NEW-3).",
    )
    ap.add_argument("--cap-aa", type=int, default=1022, help="ESM-1b residue cap (default 1022).")
    ap.add_argument(
        "--source-uri",
        default=None,
        help="Optional provenance string (e.g. the LRZ source path) recorded in the manifest.",
    )
    ap.add_argument(
        "--esm1b-paired-policy",
        choices=ESM1B_PAIRED_POLICIES,
        default=None,
        help="Lock the NEW-3 esm1b paired-stats policy (default: unlocked/null).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing freeze (default: refuse to clobber).",
    )
    args = ap.parse_args(argv)

    # Mirror analysis_barrier's exit-code contract: 2 = operator/input fault (clean stderr,
    # no traceback), distinct from a successful run.
    try:
        esm1b_keys = _read_h5_keys(args.esm1b_h5) if args.esm1b_h5 else None
        manifest = freeze_canonical_set(
            args.fasta,
            set_name=args.set_name,
            esm1b_keys=esm1b_keys,
            cap_aa=args.cap_aa,
            source_uri=args.source_uri,
            esm1b_paired_policy=args.esm1b_paired_policy,
        )
        paths = write_freeze(
            manifest, args.out_dir, set_name=args.set_name, overwrite=args.overwrite
        )
    except FileExistsError as e:
        print(f"canonical_set: REFUSING TO CLOBBER: {e}", file=sys.stderr, flush=True)
        return 2
    except (FileNotFoundError, OSError) as e:
        print(f"canonical_set: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except ValueError as e:
        print(f"canonical_set: INPUT ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    print(f"froze {manifest['n_proteins']} proteins, {manifest['n_pairs']} pairs", flush=True)
    print(f"  content sha256: {manifest['canonical_content_sha256']}", flush=True)
    if manifest.get("esm1b"):
        e = manifest["esm1b"]
        print(
            f"  esm1b: {e['n_covered']} covered, {e['n_missing']} capped "
            f"(all > {e['cap_aa']} aa: {e['missing_all_over_cap']})",
            flush=True,
        )
    print(f"  manifest:   {paths['manifest']}", flush=True)
    print(f"  pair_index: {paths['pair_index']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
