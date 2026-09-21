#!/usr/bin/env python3
"""
PDB Experimental TM-Score Pipeline (task B6, reviewer R2.2)

Computes pairwise TM-scores from EXPERIMENTAL PDB structures for pairs whose
two UniProt accessions both map into the PDB, so that the AlphaFold-derived
``alntmscore`` used as a training target can be checked against experimental
structural similarity.

Pipeline
--------
1. Map UniProt accessions to PDB *chains* via SIFTS ``pdb_chain_uniprot.tsv``
   (the residue-level flat file; the entry-level ``uniprot_pdb.tsv`` carries no
   chain column -- see HISTORY below).
2. Attach experiment method + resolution per PDB entry from the wwPDB
   ``derived_data/index/entries.idx`` index.
3. Pick ONE representative chain per accession with a deterministic rule
   (``select_representative_chain``), applying the method allowlist and the
   resolution cutoff -- both of which are now actually enforced.
4. Download the selected entries from RCSB, extract the SIFTS-mapped residues
   of the selected chain as a CA-only PDB file.
5. Run US-align on each pair and write a parquet with ``tmscore_exp`` plus full
   provenance.

HISTORY -- defects found in the original version of this module (2026-09-18).
None of them had ever been exercised: nothing in the repo had run this file
end to end.

  (a) FATAL, silent: ``load_sifts_mapping`` read ``uniprot_pdb.tsv`` and did
      ``chain = parts[header.get("CHAIN", 2)]``. That file has exactly two
      columns (SP_PRIMARY, PDB), so ``parts[2]`` raised IndexError on EVERY
      data row, the bare ``except (IndexError, KeyError): continue`` swallowed
      it, and the mapping came back empty for every input. The pipeline would
      have reported "0 pairs have structures for both proteins" and exited 0.
      It also would have used the whole semicolon-joined PDB list
      ("6kvc;6kv9") as a single ``pdb_id`` had it got that far.
  (b) ``--resolution_cutoff`` was accepted, threaded into
      ``load_sifts_mapping`` and never read.  NOW WIRED (see
      ``select_representative_chain``).
  (c) ``EXPERIMENTAL_METHODS`` was defined and never referenced.  NOW WIRED.
  (d) ``select_best_structure`` returned ``entries[0]`` -- i.e. SIFTS file
      order -- despite a docstring promising "prefer highest-resolution X-ray".
      Replaced by ``select_representative_chain``.
  (e) ``extract_chain_pdb`` copied raw space-delimited mmCIF ``_atom_site``
      lines into a file named ``.pdb``. That is not PDB format, keeps every
      model of an NMR ensemble, keeps both altlocs, and cannot express a
      multi-character ``auth_asym_id``. Replaced by ``extract_chain_ca_pdb``,
      which emits fixed-width CA-only PDB, first model only, altloc '.'/'A'
      only, renumbered 1..N, chain relabelled 'A'.
  (f) TMalign is not available on the target machine and is no longer the
      reference implementation; US-align replaces it. NOTE: ``-outfmt 2``
      cannot be combined with ``-a``/``-u``/``-L``/``-d`` (US-align errors
      out), so this module parses TM1/TM2 from the tabular output and derives
      the length-weighted average itself.

Usage
-----
    PYTHONPATH=src python src/data_preparation/pdb_tmscore.py \
        --pairs_parquet   data/processed/sprot_pre2024/sets/test.parquet \
        --entry_sifts     data/reference/sifts/uniprot_pdb.tsv \
        --work_dir        <results>/ \
        --output_parquet  <results>/tmscore_exp.parquet \
        --usalign_path    USalign \
        --resolution_cutoff 3.0

    PYTHONPATH=src python src/evaluation/analyze_experimental_tm.py \
        --parquet         <results>/tmscore_exp.parquet \
        --attrition       <results>/attrition.json \
        --results_dir     <results>/

``--entry_sifts`` comes from scripts/download_reference_data.sh. ``--chain_sifts``
and ``--entries_idx`` are downloaded into ``<work_dir>/cache`` when omitted; both
are WEEKLY-ROLLING files at a fixed URL, so ``attrition.json`` records the size,
mtime and (for SIFTS) the release-date header line of whatever was actually read.

US-align is NOT in this repo and NOT fetched by download_reference_data.sh: build
it from https://zhanggroup.org/US-align/ and put it on PATH, or point
``--usalign_path`` at the binary. Its version string is recorded in
``attrition.json`` and stamped into every checkpoint record.

Created: 2026-03-19 (Ivan infrastructure for pLM Choice revision)
Rewritten: 2026-09-18 (task B6)
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import subprocess
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Remote sources (all three are also accepted as local paths on the CLI).
CHAIN_SIFTS_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/"
    "pdb_chain_uniprot.tsv.gz"
)
ENTRIES_IDX_URL = "https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

# Experimental methods we accept. WIRED as of 2026-09-18: applied as a hard
# filter in select_representative_chain(). Entries whose EXPERIMENT TYPE string
# is not exactly one of these (e.g. "ELECTRON CRYSTALLOGRAPHY", or the combined
# "X-RAY DIFFRACTION, NEUTRON DIFFRACTION") are dropped as candidates.
EXPERIMENTAL_METHODS = {
    "X-RAY DIFFRACTION",
    "NEUTRON DIFFRACTION",
    "ELECTRON MICROSCOPY",
    "SOLUTION NMR",
    "SOLID-STATE NMR",
}

# Preference order inside the allowlist (lower is preferred).
METHOD_RANK = {
    "X-RAY DIFFRACTION": 0,
    "NEUTRON DIFFRACTION": 1,
    "ELECTRON MICROSCOPY": 2,
    "SOLUTION NMR": 3,
    "SOLID-STATE NMR": 4,
}

# Methods that report a diffraction/reconstruction resolution and are therefore
# subject to --resolution_cutoff. NMR entries carry no resolution and are exempt
# (they are already ranked last by METHOD_RANK).
RESOLUTION_BEARING_METHODS = {
    "X-RAY DIFFRACTION",
    "NEUTRON DIFFRACTION",
    "ELECTRON MICROSCOPY",
}

# A candidate chain is only considered if its SIFTS-mapped coverage is at least
# this fraction of the best coverage available for that accession. Without it,
# the stated preference order (method, then resolution, then length) hands 6.0%
# of accessions a chain covering less than half of the best available segment --
# a high-resolution fragment beats a full-length structure. Measured on this
# protein set, 2026-09-18.
COVERAGE_GUARD = 0.90


# --------------------------------------------------------------------------- #
#                            REFERENCE TABLES
# --------------------------------------------------------------------------- #


def _fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, str(dest))
    return dest


def _open_maybe_gz(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, errors="replace")


def _source_stamp(path: Path) -> dict[str, str | int]:
    """
    Provenance for one input file, for attrition.json.

    SIFTS and entries.idx are refreshed weekly at a FIXED url, so two runs months
    apart read different content through the same CLI flags and can legitimately
    pick different representative chains. Without a stamp there is no way to tell
    afterwards which release produced a number. The SIFTS flatfiles carry their
    release date on line 1 as a '#' comment; entries.idx does not, so size and
    mtime stand in.
    """
    st = path.stat()
    stamp: dict[str, str | int] = {
        "path": str(path),
        "bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(),
    }
    name = str(path)
    if name.endswith((".tsv", ".tsv.gz", ".idx", ".txt")):
        with _open_maybe_gz(path) as fh:
            first = fh.readline().strip()
        if first.startswith("#"):
            stamp["header"] = first[:200]
    return stamp


def load_entry_pdb_map(entry_sifts: Path, targets: set[str]) -> dict[str, set[str]]:
    """
    Parse the entry-level SIFTS map (SP_PRIMARY, PDB) shipped in the repo.

    Line 1 is a '#' date comment, line 2 is the header, PDB is a semicolon-
    separated list of entry ids. This file defines which pairs count as
    "both proteins map into the PDB", so it is kept as the gate even though the
    chains come from the residue-level file.
    """
    out: dict[str, set[str]] = {}
    with _open_maybe_gz(entry_sifts) as f:
        for i, line in enumerate(f):
            if i == 0 or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] == "SP_PRIMARY":
                continue
            acc = parts[0].strip()
            if acc not in targets:
                continue
            out[acc] = {p.strip().lower() for p in parts[1].split(";") if p.strip()}
    logger.info(
        "Entry-level SIFTS: %d/%d target accessions map to >=1 PDB entry",
        len(out),
        len(targets),
    )
    return out


def load_sifts_chain_mapping(
    chain_sifts: Path, targets: set[str]
) -> dict[str, dict[tuple[str, str], list[tuple[int, int, str, str]]]]:
    """
    Parse residue-level SIFTS (PDB, CHAIN, SP_PRIMARY, RES_BEG, RES_END,
    PDB_BEG, PDB_END, SP_BEG, SP_END).

    Returns acc -> {(pdb_id, auth_chain): [(sp_beg, sp_end, pdb_beg, pdb_end)]}.
    PDB_BEG/PDB_END are author residue labels and may carry insertion codes, so
    they are kept as strings.
    """
    out: dict[str, dict[tuple[str, str], list[tuple[int, int, str, str]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    with _open_maybe_gz(chain_sifts) as f:
        for i, line in enumerate(f):
            if i < 2 or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            acc = p[2].strip()
            if acc not in targets:
                continue
            try:
                sp_beg, sp_end = int(p[7]), int(p[8])
            except ValueError:
                continue
            out[acc][(p[0].strip().lower(), p[1].strip())].append(
                (sp_beg, sp_end, p[5].strip(), p[6].strip())
            )
    logger.info(
        "Residue-level SIFTS: %d/%d target accessions have >=1 mapped chain",
        len(out),
        len(targets),
    )
    return {a: dict(d) for a, d in out.items()}


def load_entry_metadata(entries_idx: Path) -> dict[str, tuple[str, float | None]]:
    """
    Parse the wwPDB entries index -> pdb_id -> (experiment_type, resolution).

    Columns: IDCODE, HEADER, ACCESSION DATE, COMPOUND, SOURCE, AUTHOR LIST,
    RESOLUTION, EXPERIMENT TYPE (IF NOT X-RAY). An empty EXPERIMENT TYPE means
    X-ray, per the file's own header. Resolutions <= 0 are recorded as None
    (NMR and friends).
    """
    meta: dict[str, tuple[str, float | None]] = {}
    n_short = 0
    with _open_maybe_gz(entries_idx) as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                # An X-ray entry has an EMPTY EXPERIMENT TYPE, so it only reaches
                # 8 fields if wwPDB writes the trailing tab. If it ever stops
                # doing so, every X-ray entry silently loses its metadata, fails
                # the method allowlist and the cohort collapses to NMR/EM. Loud
                # rather than silent.
                n_short += 1
                continue
            pdb_id = p[0].strip().lower()
            try:
                res: float | None = float(p[6].strip())
            except ValueError:
                res = None
            if res is not None and res <= 0:
                res = None
            method = p[7].strip().upper() or "X-RAY DIFFRACTION"
            meta[pdb_id] = (method, res)
    if n_short:
        logger.warning(
            "entries.idx: %d rows had fewer than 8 tab-separated fields and were "
            "skipped -- those entries have NO method/resolution and will be "
            "dropped by the allowlist. Check the file layout before trusting the "
            "cohort.",
            n_short,
        )
    logger.info("Entry metadata: %d PDB entries", len(meta))
    return meta


# --------------------------------------------------------------------------- #
#                         REPRESENTATIVE CHAIN CHOICE
# --------------------------------------------------------------------------- #


def select_representative_chain(
    accession: str,
    chain_candidates: dict[tuple[str, str], list[tuple[int, int, str, str]]],
    allowed_entries: set[str],
    entry_meta: dict[str, tuple[str, float | None]],
    resolution_cutoff: float,
    coverage_guard: float = COVERAGE_GUARD,
) -> tuple[dict | None, str]:
    """
    Deterministic representative-chain rule. Returns (pick, reject_reason).

    Filters, in order:
      F1  the entry must also be listed for this accession in the entry-level
          SIFTS map that defines the pair set;
      F2  EXPERIMENT TYPE must be in EXPERIMENTAL_METHODS;
      F3  for resolution-bearing methods, resolution must be <= cutoff
          (entries with no resolution, i.e. NMR, are exempt);
      F4  SIFTS-mapped coverage must be >= coverage_guard x the best coverage
          surviving F1-F3 for this accession.

    Then sorts ascending by
          (METHOD_RANK, resolution or +inf, -coverage, pdb_id, chain)
    and takes the first -- i.e. prefer X-ray, then best (lowest) resolution,
    then longest SIFTS-mapped segment, with a fully deterministic id tie-break.
    """
    rows = []
    for (pdb_id, chain), segs in chain_candidates.items():
        if pdb_id not in allowed_entries:  # F1
            continue
        method, resolution = entry_meta.get(pdb_id, ("UNKNOWN", None))
        if method not in EXPERIMENTAL_METHODS:  # F2
            continue
        if (  # F3
            method in RESOLUTION_BEARING_METHODS
            and resolution is not None
            and resolution > resolution_cutoff
        ):
            continue
        coverage = sum(e - b + 1 for b, e, _, _ in segs)
        rows.append(
            {
                "accession": accession,
                "pdb_id": pdb_id,
                "chain": chain,
                "method": method,
                "resolution": resolution,
                "mapped_len": coverage,
                "segments": sorted(segs),
            }
        )

    if not rows:
        return None, "no_candidate_after_filters"

    best_cov = max(r["mapped_len"] for r in rows)
    rows = [r for r in rows if r["mapped_len"] >= coverage_guard * best_cov]  # F4

    rows.sort(
        key=lambda r: (
            METHOD_RANK[r["method"]],
            r["resolution"] if r["resolution"] is not None else float("inf"),
            -r["mapped_len"],
            r["pdb_id"],
            r["chain"],
        )
    )
    return rows[0], ""


# --------------------------------------------------------------------------- #
#                        STRUCTURE DOWNLOAD / EXTRACTION
# --------------------------------------------------------------------------- #


def download_cif(pdb_id: str, cache_dir: Path, retries: int = 3) -> Path | None:
    """Download <pdb_id>.cif.gz from RCSB into cache_dir. Kept gzipped."""
    dest = cache_dir / f"{pdb_id}.cif.gz"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    tmp = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, str(tmp))
            tmp.rename(dest)
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if attempt == retries - 1:
                logger.debug("download failed %s: %s", pdb_id, exc)
    return None


def _split_cif_line(line: str) -> list[str]:
    """Whitespace split that respects mmCIF single/double quoting."""
    out: list[str] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c in "'\"":
            j = i + 1
            while j < n:
                if line[j] == c and (j + 1 >= n or line[j + 1] in " \t\r\n"):
                    break
                j += 1
            out.append(line[i + 1 : j])
            i = j + 1
        else:
            j = i
            while j < n and line[j] not in " \t\r\n":
                j += 1
            out.append(line[i:j])
            i = j
    return out


_AUTH_SEQ_RE = re.compile(r"([+-]?\d+)(.*)$")


def _auth_seq_key(label: str) -> tuple[int, str] | None:
    """'27A' -> (27, 'A'); '-3' -> (-3, ''); 'None'/'' -> None."""
    label = label.strip()
    if not label or label in {".", "?", "None", "null"}:
        return None
    m = _AUTH_SEQ_RE.match(label)
    if m is None:
        return None
    return int(m.group(1)), m.group(2).strip()


def extract_chain_ca_pdb(
    cif_gz: Path,
    chain_id: str,
    segments: Sequence[tuple[int, int, str, str]],
    out_pdb: Path,
) -> tuple[int, bool]:
    """
    Write the CA atoms of one author chain to a fixed-width PDB file.

    Only CA atoms are written: US-align/TM-align use the CA trace alone for
    protein alignment, and CA-only files are ~20x smaller and faster to parse.

    Rules, all of which the previous implementation got wrong:
      * first ``pdbx_PDB_model_num`` only (NMR ensembles otherwise stack every
        model into one "chain");
      * altloc must be '.', '?' or 'A';
      * ATOM records, plus HETATM whose comp_id is MSE (selenomethionine is a
        genuine residue of the chain and is deposited as HETATM);
      * residues restricted to the SIFTS PDB_BEG..PDB_END author ranges when
        those parse; if none parse, the whole chain is kept and the caller is
        told via the second return value;
      * chain relabelled 'A' and residues renumbered 1..N, because PDB format
        has one column for the chain id and four for the residue number while
        mmCIF ``auth_asym_id``/``auth_seq_id`` have neither limit. TM-align is
        sequence-order independent, so renumbering cannot change the score.

    Returns (n_ca_written, used_whole_chain_fallback).
    """
    ranges: list[tuple[tuple[int, str], tuple[int, str]]] = []
    for _, _, pdb_beg, pdb_end in segments:
        b, e = _auth_seq_key(pdb_beg), _auth_seq_key(pdb_end)
        if b is not None and e is not None and (b[0], b[1]) <= (e[0], e[1]):
            ranges.append((b, e))

    def in_range(key: tuple[int, str]) -> bool:
        return any(b <= key <= e for b, e in ranges)

    cols: dict[str, int] = {}
    in_loop = False
    model0: str | None = None
    kept: list[tuple[str, float, float, float, float, float, str]] = []
    # Every CA of the chain, before the SIFTS range filter. Collected in the same
    # pass so the whole-chain fallback below does not have to re-read the gzip.
    chain_ca: list[tuple[str, float, float, float, float, float, str]] = []

    with gzip.open(cif_gz, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("_atom_site."):
                cols[line.strip().split(".", 1)[1]] = len(cols)
                in_loop = True
                continue
            if not in_loop:
                continue
            if line.startswith("#") or not line.strip():
                if chain_ca:
                    break
                in_loop = False
                cols = {}
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            p = _split_cif_line(line)
            if len(p) < len(cols):
                continue
            try:
                if p[cols["label_atom_id"]] != "CA":
                    continue
                comp = p[cols["label_comp_id"]]
                if line.startswith("HETATM") and comp != "MSE":
                    continue
                # Indexed, not .get(): a missing column must raise KeyError into
                # the guard below (row skipped like any other unparseable row).
                # cols.get(...) returns None on an _atom_site loop that has
                # neither column, and p[None] raises TypeError, which is NOT in
                # the guard -- it escapes to the caller's except Exception and
                # the whole protein is dropped as chain_extraction_error with the
                # cause logged at DEBUG.
                ch_col = cols["auth_asym_id"] if "auth_asym_id" in cols else cols["label_asym_id"]
                if p[ch_col] != chain_id:
                    continue
                model = p[cols["pdbx_PDB_model_num"]] if "pdbx_PDB_model_num" in cols else "1"
                if model0 is None:
                    model0 = model
                if model != model0:
                    continue
                alt = p[cols["label_alt_id"]] if "label_alt_id" in cols else "."
                if alt not in {".", "?", "A"}:
                    continue
                seq_col = cols["auth_seq_id"] if "auth_seq_id" in cols else cols["label_seq_id"]
                key = _auth_seq_key(p[seq_col])
                if key is None:
                    continue
                atom = (
                    comp,
                    float(p[cols["Cartn_x"]]),
                    float(p[cols["Cartn_y"]]),
                    float(p[cols["Cartn_z"]]),
                    float(p[cols["occupancy"]]) if "occupancy" in cols else 1.0,
                    float(p[cols["B_iso_or_equiv"]]) if "B_iso_or_equiv" in cols else 0.0,
                    p[cols["type_symbol"]] if "type_symbol" in cols else "C",
                )
                chain_ca.append(atom)
                if ranges and not in_range(key):
                    continue
                kept.append(atom)
            except (KeyError, IndexError, ValueError):
                continue

    if not kept and chain_ca:
        # Author ranges did not intersect anything (renumbered entry, odd
        # insertion codes); fall back to the whole chain rather than dropping.
        return _write_ca_pdb(chain_ca, out_pdb), True

    if not kept:
        return 0, False

    return _write_ca_pdb(kept, out_pdb), False


# PDB's resSeq field is four characters wide, so a residue past 9999 cannot be
# expressed in fixed-width PDB at all. No deposited protein chain comes close.
# But truncating silently would align a partial chain while the parquet reported
# the full length, so the cut is logged and the returned count is what is
# actually on disk.
MAX_PDB_RESSEQ = 9999


def _write_ca_pdb(rows, out_pdb: Path) -> int:
    """Write the CA rows as fixed-width PDB. Returns how many were written."""
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) > MAX_PDB_RESSEQ:
        logger.warning(
            "%s: chain has %d CA atoms, writing only the first %d (PDB resSeq is "
            "4 columns wide)",
            out_pdb.name,
            len(rows),
            MAX_PDB_RESSEQ,
        )
        rows = rows[:MAX_PDB_RESSEQ]
    with open(out_pdb, "w") as fh:
        for i, (comp, x, y, z, occ, b, el) in enumerate(rows, start=1):
            fh.write(
                f"ATOM  {i:>5d}  CA  {comp[:3]:>3} A{i:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{b:>6.2f}"
                f"          {el[:2].rjust(2):>2}\n"
            )
        fh.write("TER\nEND\n")
    return len(rows)


# --------------------------------------------------------------------------- #
#                            US-ALIGN EXECUTION
# --------------------------------------------------------------------------- #


def run_usalign(
    structure_a: Path, structure_b: Path, usalign_path: str, timeout: int = 120
) -> dict[str, float] | None:
    """
    Run US-align in tabular mode and return the raw columns.

    ``-outfmt 2`` gives: PDBchain1 PDBchain2 TM1 TM2 RMSD ID1 ID2 IDali L1 L2 Lali
    where TM1 is normalised by L1 and TM2 by L2.

    GOTCHA (verified against US-align 20260527 on 2026-09-18): ``-outfmt 2``
    refuses to run together with ``-a``, ``-u``, ``-L`` or ``-d`` -- it prints
    "-outfmt 2 cannot be used with -a, -u, -L, -d" and exits. The
    average-length normalisation is therefore derived here as the
    length-weighted mean (TM1*L1 + TM2*L2)/(L1+L2). That is an approximation of
    US-align's own "-a T" value because d0 depends on the normalisation length;
    on a spot check (1ubq vs 1ttn) it gave 0.68911 against US-align's 0.68922.
    """
    try:
        proc = subprocess.run(
            [usalign_path, str(structure_a), str(structure_b), "-outfmt", "2", "-mol", "prot"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 11:
            continue
        try:
            tm1, tm2 = float(f[2]), float(f[3])
            l1, l2, lali = int(f[8]), int(f[9]), int(f[10])
        except ValueError:
            continue
        if l1 <= 0 or l2 <= 0:
            return None
        return {
            "tm1": tm1,
            "tm2": tm2,
            "rmsd": float(f[4]),
            "seqid_ali": float(f[7]),
            "len1": l1,
            "len2": l2,
            "len_ali": lali,
            # normalised by the LONGER chain: symmetric under query/target swap
            # and the conservative choice (a short domain matching part of a
            # long protein cannot inflate it).
            "tmscore_exp": min(tm1, tm2),
            # normalised by the SHORTER chain.
            "tmscore_exp_short": max(tm1, tm2),
            "tmscore_exp_avg": (tm1 * l1 + tm2 * l2) / (l1 + l2),
        }
    return None


def _score_one(item):
    q, t, pa, pb, exe = item
    return q, t, run_usalign(Path(pa), Path(pb), exe)


# --------------------------------------------------------------------------- #
#                                  MAIN
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pairwise TM-scores from experimental PDB structures (US-align).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pairs_parquet", type=Path, required=True)
    ap.add_argument("--output_parquet", type=Path, required=True)
    ap.add_argument("--work_dir", type=Path, required=True,
                    help="Cache + checkpoints. Keep this OUTSIDE the git tree.")
    ap.add_argument("--entry_sifts", type=Path, required=True,
                    help="uniprot_pdb.tsv (SP_PRIMARY, PDB) -- defines the pair set.")
    ap.add_argument("--chain_sifts", type=Path, default=None,
                    help="pdb_chain_uniprot.tsv[.gz]; downloaded into work_dir if absent.")
    ap.add_argument("--entries_idx", type=Path, default=None,
                    help="wwPDB entries.idx; downloaded into work_dir if absent.")
    ap.add_argument("--usalign_path", type=str, default="USalign")
    ap.add_argument("--resolution_cutoff", type=float, default=3.0,
                    help="APPLIED (since 2026-09-18) to X-ray/neutron/EM entries. "
                         "NMR entries report no resolution and are exempt.")
    ap.add_argument("--coverage_guard", type=float, default=COVERAGE_GUARD)
    ap.add_argument("--max_entries", type=int, default=None,
                    help="Cap on distinct PDB entries downloaded. Proteins whose "
                         "representative entry falls outside the cap are dropped "
                         "and counted. Default: no cap.")
    ap.add_argument("--max_workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--download_workers", type=int, default=8)
    ap.add_argument("--checkpoint_every", type=int, default=2000)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    cache = args.work_dir / "cache"
    cif_dir = cache / "cif"
    chain_dir = cache / "chains"
    for d in (cache, cif_dir, chain_dir):
        d.mkdir(parents=True, exist_ok=True)

    exe = os.path.expanduser(args.usalign_path)
    probe = subprocess.run([exe, "-v"], capture_output=True, text=True)
    # The banner's first line is a row of '*', so pick the line that names the
    # version rather than whatever comes first.
    version = "?"
    for ln in (probe.stdout + probe.stderr).splitlines():
        if "Version" in ln:
            version = ln.strip().strip("*").strip()
            break
    logger.info("US-align: %s (%s)", exe, version)

    attrition: list[tuple[str, int]] = []

    pairs = pl.read_parquet(args.pairs_parquet)
    attrition.append(("test-split pairs", pairs.height))

    targets = set(pairs["query"].unique().to_list()) | set(pairs["target"].unique().to_list())
    entry_map = load_entry_pdb_map(args.entry_sifts, targets)

    pairs = pairs.filter(
        pl.col("query").is_in(list(entry_map)) & pl.col("target").is_in(list(entry_map))
    )
    attrition.append(("both accessions in entry-level SIFTS", pairs.height))

    proteins = set(pairs["query"].unique().to_list()) | set(pairs["target"].unique().to_list())
    attrition.append(("distinct proteins in those pairs", len(proteins)))

    chain_sifts = args.chain_sifts or _fetch(CHAIN_SIFTS_URL, cache / "pdb_chain_uniprot.tsv.gz")
    entries_idx = args.entries_idx or _fetch(ENTRIES_IDX_URL, cache / "entries.idx")
    chain_map = load_sifts_chain_mapping(chain_sifts, proteins)
    entry_meta = load_entry_metadata(entries_idx)

    # Surface what the EXPERIMENTAL_METHODS allowlist (filter F2) throws away.
    # entries.idx writes EXPERIMENT TYPE as free text, so a spelling this module
    # does not list -- a bare "NMR" where it expects "SOLUTION NMR", say -- drops
    # a whole class of structures into the undifferentiated
    # no_candidate_after_filters bucket with no error and a quietly narrower
    # cohort than the docstring claims.
    candidate_entries = {pdb for cands in chain_map.values() for pdb, _ in cands}
    method_hist = Counter(
        entry_meta.get(e, ("NOT_IN_ENTRIES_IDX", None))[0] for e in candidate_entries
    )
    rejected_methods = {m: n for m, n in method_hist.items() if m not in EXPERIMENTAL_METHODS}
    if rejected_methods:
        logger.warning(
            "F2 drops %d of %d candidate entries on EXPERIMENT TYPE: %s",
            sum(rejected_methods.values()),
            len(candidate_entries),
            dict(sorted(rejected_methods.items(), key=lambda kv: -kv[1])),
        )

    picks: dict[str, dict] = {}
    reject: dict[str, str] = {}
    for acc in sorted(proteins):
        cands = chain_map.get(acc)
        if not cands:
            reject[acc] = "no_chain_in_residue_level_sifts"
            continue
        pick, why = select_representative_chain(
            acc, cands, entry_map.get(acc, set()), entry_meta,
            args.resolution_cutoff, args.coverage_guard,
        )
        if pick is None:
            reject[acc] = why
        else:
            picks[acc] = pick
    logger.info("Representative chains: %d picked, %d rejected", len(picks), len(reject))
    attrition.append(("proteins with a representative chain", len(picks)))

    wanted_entries = sorted({p["pdb_id"] for p in picks.values()})
    if args.max_entries is not None and len(wanted_entries) > args.max_entries:
        keep = set(wanted_entries[: args.max_entries])
        dropped = {a for a, p in picks.items() if p["pdb_id"] not in keep}
        logger.warning(
            "--max_entries=%d caps %d entries -> dropping %d proteins",
            args.max_entries, len(wanted_entries), len(dropped),
        )
        for a in dropped:
            reject[a] = "max_entries_cap"
            picks.pop(a)
        wanted_entries = sorted(keep)
    logger.info("Distinct PDB entries to fetch: %d", len(wanted_entries))

    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futs = {pool.submit(download_cif, e, cif_dir): e for e in wanted_entries}
        ok = 0
        for fut in tqdm(as_completed(futs), total=len(futs), desc="RCSB download", unit="cif"):
            ok += fut.result() is not None
    logger.info("Downloaded/cached %d/%d entries", ok, len(wanted_entries))

    chain_files: dict[str, Path] = {}
    extract_stats: dict[str, dict] = {}
    for acc, p in tqdm(sorted(picks.items()), desc="Extract chains", unit="chain"):
        cif = cif_dir / f"{p['pdb_id']}.cif.gz"
        if not cif.exists():
            reject[acc] = "cif_download_failed"
            continue
        out = chain_dir / f"{acc}_{p['pdb_id']}_{p['chain']}.pdb"
        try:
            n, fb = extract_chain_ca_pdb(cif, p["chain"], p["segments"], out)
        except Exception as exc:  # noqa: BLE001
            # WARNING, not DEBUG: this branch means the mmCIF did not parse the
            # way this module assumes, which is exactly the class of surprise
            # that must not be invisible at the module's INFO default.
            logger.warning("chain extraction failed for %s (%s): %s", acc, p["pdb_id"], exc)
            reject[acc] = "chain_extraction_error"
            continue
        if n < 20:
            reject[acc] = f"too_few_ca:{n}"
            continue
        chain_files[acc] = out
        extract_stats[acc] = {"n_ca": n, "whole_chain_fallback": fb}
    logger.info("Usable chain files: %d", len(chain_files))
    attrition.append(("proteins with a usable CA chain file", len(chain_files)))

    have = set(chain_files)
    pairs = pairs.filter(pl.col("query").is_in(list(have)) & pl.col("target").is_in(list(have)))
    attrition.append(("pairs with a chain file for BOTH proteins", pairs.height))

    # A cached score is only reusable if it was produced from the SAME two
    # structures by the SAME binary. Changing --resolution_cutoff or
    # --coverage_guard, or letting the weekly SIFTS/entries.idx refresh through,
    # changes what select_representative_chain picks WITHOUT changing the pair --
    # so a checkpoint keyed on (query, target) alone would hand old scores to new
    # q_pdb/q_chain/q_resolution provenance columns and say nothing about it.
    # Stamping each record makes a stale entry recomputable instead of silently
    # wrong. Records written before this stamp existed have no "stamp" key and
    # are therefore treated as stale, which is the safe reading.
    def _stamp(q: str, t: str) -> str:
        pq, pt = picks[q], picks[t]
        return f"{pq['pdb_id']}_{pq['chain']}|{pt['pdb_id']}_{pt['chain']}|{version}"

    ckpt = args.work_dir / "checkpoint.jsonl"
    done: dict[tuple[str, str], dict | None] = {}
    n_stale = 0
    if ckpt.exists():
        with open(ckpt) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (rec["query"], rec["target"])
                if key[0] not in picks or key[1] not in picks:
                    n_stale += 1
                    continue
                if rec.get("stamp") != _stamp(*key):
                    n_stale += 1
                    continue
                done[key] = rec["res"]
        logger.info("Resuming: %d reusable pairs in %s", len(done), ckpt)
        if n_stale:
            logger.warning(
                "%d checkpointed pairs came from different chains or a different "
                "US-align build and are being rescored, not reused.",
                n_stale,
            )

    pair_ids = list(zip(pairs["query"].to_list(), pairs["target"].to_list(), strict=True))
    work = [
        (q, t, str(chain_files[q]), str(chain_files[t]), exe)
        for q, t in pair_ids
        if (q, t) not in done
    ]
    logger.info("US-align on %d pairs (%d workers)", len(work), args.max_workers)

    if work:
        with open(ckpt, "a") as fh, ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            n = 0
            for q, t, res in tqdm(
                pool.map(_score_one, work, chunksize=16), total=len(work),
                desc="US-align", unit="pair",
            ):
                done[(q, t)] = res
                fh.write(
                    json.dumps(
                        {"query": q, "target": t, "stamp": _stamp(q, t), "res": res}
                    )
                    + "\n"
                )
                n += 1
                if n % args.checkpoint_every == 0:
                    fh.flush()

    rows = []
    for q, t in pair_ids:
        res = done.get((q, t))
        pq, pt = picks[q], picks[t]
        row = {
            "query": q, "target": t,
            "q_pdb": pq["pdb_id"], "q_chain": pq["chain"],
            "q_method": pq["method"], "q_resolution": pq["resolution"],
            "q_mapped_len": pq["mapped_len"], "q_n_ca": extract_stats[q]["n_ca"],
            "q_whole_chain_fallback": extract_stats[q]["whole_chain_fallback"],
            "t_pdb": pt["pdb_id"], "t_chain": pt["chain"],
            "t_method": pt["method"], "t_resolution": pt["resolution"],
            "t_mapped_len": pt["mapped_len"], "t_n_ca": extract_stats[t]["n_ca"],
            "t_whole_chain_fallback": extract_stats[t]["whole_chain_fallback"],
        }
        for k in ("tmscore_exp", "tmscore_exp_short", "tmscore_exp_avg",
                  "tm1", "tm2", "rmsd", "seqid_ali", "len1", "len2", "len_ali"):
            row[k] = res[k] if res else None
        rows.append(row)

    attrition.append(
        ("pairs with a US-align score", sum(1 for r in rows if r["tmscore_exp"] is not None))
    )

    # attrition.json goes out BEFORE the parquet. When the funnel empties, it is
    # the only record of where the pairs went -- and the parquet write is exactly
    # the step that cannot run in that case (an empty row list gives a frame with
    # no columns, and the join dies on a missing "query"). Writing it first means
    # the diagnostic survives the failure it is there to explain.
    with open(args.work_dir / "attrition.json", "w") as fh:
        json.dump(
            {
                "steps": attrition,
                "usalign_version": version,
                "resolution_cutoff": args.resolution_cutoff,
                "coverage_guard": args.coverage_guard,
                "sources": {
                    "pairs_parquet": _source_stamp(args.pairs_parquet),
                    "entry_sifts": _source_stamp(args.entry_sifts),
                    "chain_sifts": _source_stamp(Path(chain_sifts)),
                    "entries_idx": _source_stamp(Path(entries_idx)),
                },
                "candidate_entry_methods": dict(sorted(method_hist.items())),
                "checkpoint_pairs_rescored_as_stale": n_stale,
                "reject_reasons": dict(sorted(Counter(reject.values()).items())),
            },
            fh, indent=2,
        )

    logger.info("=" * 64)
    for name, n in attrition:
        logger.info("%-46s %8d", name, n)

    if not rows:
        # Exit NON-zero. The failure this module was rewritten to fix was a
        # silently empty result that logged "0 pairs" and exited 0, which reads
        # as success to every caller and to run_ivan_pipeline.sh's run_or_skip.
        logger.error(
            "No pair survived the funnel, so there is nothing to write. "
            "%s records where they went.",
            args.work_dir / "attrition.json",
        )
        raise SystemExit(1)

    # validate="1:1": rows has exactly one entry per row of `pairs`, so a
    # duplicated (query, target) on either side would multiply the output and
    # quietly inflate n. sets/test.parquet has none; fail loudly if that changes.
    out = pl.DataFrame(rows).join(pairs, on=["query", "target"], how="left", validate="1:1")
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.output_parquet)
    logger.info("Wrote %s", args.output_parquet)


if __name__ == "__main__":
    main()
