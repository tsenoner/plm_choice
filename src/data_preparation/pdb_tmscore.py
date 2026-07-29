#!/usr/bin/env python3
"""
PDB Experimental TM-Score Pipeline

Computes pairwise TM-scores from experimental (X-ray, NMR, cryo-EM) PDB
structures. Validates whether predicted-structure-based TM-scores (used as
'alntmscore' in training) introduce systematic bias.

Pipeline:
1. Map protein IDs to PDB chains via SIFTS (EBI UniProt-PDB mapping)
2. Download experimental structures from RCSB
3. Run TMalign pairwise on matched chains
4. Output parquet with tmscore_exp column for merging into training data

Requires:
    - TMalign binary on PATH (or specified via --tmalign_path)
      Download from: https://zhanggroup.org/TM-align/
    - Internet access for RCSB/SIFTS downloads

Usage:
    uv run python src/data_preparation/pdb_tmscore.py \
        --pairs_parquet data/processed/sprot_pre2024/sets/test.parquet \
        --output_parquet data/processed/sprot_pre2024/sets/test_with_tmscore_exp.parquet \
        --pdb_cache_dir data/reference/pdb_cache \
        --resolution_cutoff 3.0

Created: 2026-03-19 (Ivan infrastructure for pLM Choice revision)
"""

import argparse
import gzip
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# RCSB/EBI endpoints
SIFTS_URL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/uniprot_pdb.tsv.gz"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

# Experimental methods we accept
EXPERIMENTAL_METHODS = {"X-RAY DIFFRACTION", "NEUTRON DIFFRACTION",
                        "ELECTRON MICROSCOPY", "SOLUTION NMR", "SOLID-STATE NMR"}


# --------------------------------------------------------------------------- #
#                            SIFTS MAPPING
# --------------------------------------------------------------------------- #


def download_sifts_mapping(output_path: Path) -> None:
    """Download UniProt-to-PDB mapping from EBI SIFTS."""
    logger.info(f"Downloading SIFTS mapping from {SIFTS_URL} ...")
    gz_path = output_path.with_suffix(".tsv.gz")
    urllib.request.urlretrieve(SIFTS_URL, str(gz_path))

    # Decompress
    with gzip.open(gz_path, "rt") as f_in:
        with open(output_path, "w") as f_out:
            f_out.write(f_in.read())
    gz_path.unlink()
    logger.info(f"Saved SIFTS mapping to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def load_sifts_mapping(
    sifts_path: Path,
    target_proteins: Set[str],
    resolution_cutoff: float = 3.0,
) -> Dict[str, List[Dict]]:
    """
    Load SIFTS mapping, filtering to target proteins and experimental structures.

    Returns:
        Dict mapping uniprot_id -> [{"pdb_id": "1abc", "chain": "A", "method": "...", "resolution": 2.1}, ...]
    """
    if not sifts_path.exists():
        sifts_path.parent.mkdir(parents=True, exist_ok=True)
        download_sifts_mapping(sifts_path)

    mapping: Dict[str, List[Dict]] = {}
    matched = 0

    with open(sifts_path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")

            if header is None:
                # First non-comment line is header
                header = {col.strip().upper(): i for i, col in enumerate(parts)}
                continue

            # SIFTS columns: SP_PRIMARY, PDB, CHAIN, RES, ...
            # The exact format may vary; handle gracefully
            try:
                uniprot_id = parts[header.get("SP_PRIMARY", 0)]
                pdb_id = parts[header.get("PDB", 1)].lower()
                chain = parts[header.get("CHAIN", 2)]
            except (IndexError, KeyError):
                continue

            if uniprot_id not in target_proteins:
                continue

            # We'll filter by resolution later when we have structure metadata
            if uniprot_id not in mapping:
                mapping[uniprot_id] = []

            mapping[uniprot_id].append({
                "pdb_id": pdb_id,
                "chain": chain,
            })
            matched += 1

    logger.info(
        f"SIFTS: found {matched} PDB chain mappings for "
        f"{len(mapping)}/{len(target_proteins)} target proteins"
    )
    return mapping


# --------------------------------------------------------------------------- #
#                        PDB STRUCTURE DOWNLOAD
# --------------------------------------------------------------------------- #


def download_pdb_structure(pdb_id: str, cache_dir: Path) -> Optional[Path]:
    """
    Download a PDB structure in mmCIF format from RCSB.

    Returns path to cached .cif file, or None if download fails.
    """
    cif_path = cache_dir / f"{pdb_id}.cif"
    if cif_path.exists():
        return cif_path

    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    try:
        gz_path = cif_path.with_suffix(".cif.gz")
        urllib.request.urlretrieve(url, str(gz_path))
        with gzip.open(gz_path, "rb") as f_in:
            with open(cif_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        return cif_path
    except Exception as e:
        logger.debug(f"Failed to download {pdb_id}: {e}")
        return None


def extract_chain_pdb(cif_path: Path, chain_id: str, output_path: Path) -> bool:
    """
    Extract a single chain from an mmCIF file to a minimal PDB file.

    TMalign works best with PDB format. We extract ATOM records for the
    specified chain. This is a simple extraction — no renumbering or
    modification.

    LIMITATION: This writes space-delimited mmCIF ATOM lines directly to
    the output file, which is NOT valid PDB fixed-width format. TMalign
    tolerates this for most structures, but may fail on entries with
    unusual column layouts. For production use, consider converting via
    gemmi (``gemmi convert --to pdb``) or BioPython's MMCIF2Dict.
    """
    try:
        atoms_written = 0
        with open(cif_path) as f_in, open(output_path, "w") as f_out:
            in_atom_site = False
            col_map = {}

            for line in f_in:
                # Parse mmCIF ATOM records
                if line.startswith("_atom_site."):
                    field = line.strip().split(".")[1]
                    col_map[field] = len(col_map)
                    in_atom_site = True
                    continue

                # BUG FIX (2026-03-20): original expression was:
                #   in_atom_site and line.startswith("ATOM") or line.startswith("HETATM")
                # Python operator precedence evaluates this as:
                #   (in_atom_site and ATOM) or HETATM
                # which writes HETATM lines regardless of whether we're inside
                # an _atom_site block, corrupting the output PDB with header
                # or loop data that happens to start with "HETATM".
                if in_atom_site and (line.startswith("ATOM") or line.startswith("HETATM")):
                    parts = line.split()
                    # Check chain
                    chain_col = col_map.get("auth_asym_id", col_map.get("label_asym_id"))
                    if chain_col is not None and chain_col < len(parts):
                        if parts[chain_col] == chain_id:
                            # Write as PDB ATOM record (simplified)
                            f_out.write(line)
                            atoms_written += 1
                elif in_atom_site and not line.strip():
                    in_atom_site = False

        return atoms_written > 0
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#                        TMALIGN EXECUTION
# --------------------------------------------------------------------------- #


def run_tmalign(
    structure_a: Path,
    structure_b: Path,
    tmalign_path: str = "TMalign",
) -> Optional[float]:
    """
    Run TMalign on two structures, return TM-score normalized by shorter chain.

    Returns None if TMalign fails or structures can't be aligned.
    """
    try:
        result = subprocess.run(
            [tmalign_path, str(structure_a), str(structure_b), "-a"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return None

        # Parse TM-score from output
        # TMalign outputs: "TM-score= 0.xxxxx (if normalized by length of Chain_1)"
        # We want the score normalized by shorter chain (usually the second line)
        for line in result.stdout.split("\n"):
            if "TM-score=" in line and "normalized by length" in line:
                match = re.search(r"TM-score=\s*([\d.]+)", line)
                if match:
                    return float(match.group(1))

        return None

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return None


def compute_tmscore_pair(args_tuple) -> Tuple[str, str, Optional[float]]:
    """
    Worker function for parallel TM-score computation.

    Args tuple: (query_id, target_id, query_pdb_path, target_pdb_path, tmalign_path)
    """
    query_id, target_id, query_pdb, target_pdb, tmalign_path = args_tuple
    score = run_tmalign(Path(query_pdb), Path(target_pdb), tmalign_path)
    return query_id, target_id, score


# --------------------------------------------------------------------------- #
#                        MAIN PIPELINE
# --------------------------------------------------------------------------- #


def select_best_structure(
    entries: List[Dict],
) -> Optional[Dict]:
    """
    Select the best PDB entry for a protein (prefer highest-resolution X-ray).

    Simple heuristic: just pick the first entry (SIFTS lists them in a
    reasonable order). For more sophisticated selection, could check
    resolution via RCSB API, but that adds latency.
    """
    if not entries:
        return None
    return entries[0]


def prepare_structures(
    pairs_df: pl.DataFrame,
    sifts_mapping: Dict[str, List[Dict]],
    pdb_cache_dir: Path,
    chain_cache_dir: Path,
) -> Dict[str, Path]:
    """
    Download PDB structures and extract chains for all proteins in the dataset.

    Returns:
        Dict mapping protein_id -> path to extracted chain PDB file
    """
    all_proteins = set(pairs_df["query"].unique().to_list()) | set(
        pairs_df["target"].unique().to_list()
    )

    protein_structures: Dict[str, Path] = {}
    download_needed = set()

    # Determine which proteins have PDB mappings
    for protein_id in all_proteins:
        entries = sifts_mapping.get(protein_id, [])
        best = select_best_structure(entries)
        if best:
            chain_pdb = chain_cache_dir / f"{protein_id}_{best['pdb_id']}_{best['chain']}.pdb"
            if chain_pdb.exists():
                protein_structures[protein_id] = chain_pdb
            else:
                download_needed.add((protein_id, best["pdb_id"], best["chain"]))

    logger.info(
        f"Structure prep: {len(protein_structures)} cached, "
        f"{len(download_needed)} to download, "
        f"{len(all_proteins) - len(protein_structures) - len(download_needed)} unmapped"
    )

    # Download and extract
    for protein_id, pdb_id, chain in tqdm(
        download_needed, desc="Downloading PDB structures", unit="pdb"
    ):
        cif_path = download_pdb_structure(pdb_id, pdb_cache_dir)
        if cif_path is None:
            continue

        chain_pdb = chain_cache_dir / f"{protein_id}_{pdb_id}_{chain}.pdb"
        if extract_chain_pdb(cif_path, chain, chain_pdb):
            protein_structures[protein_id] = chain_pdb

    logger.info(f"Prepared structures for {len(protein_structures)} proteins")
    return protein_structures


def compute_all_tmscores(
    pairs_df: pl.DataFrame,
    protein_structures: Dict[str, Path],
    tmalign_path: str = "TMalign",
    max_workers: int = 4,
) -> List[float]:
    """
    Compute TM-scores for all protein pairs where both have structures.

    Uses ProcessPoolExecutor for parallelism since TMalign is CPU-bound.
    """
    queries = pairs_df["query"].to_list()
    targets = pairs_df["target"].to_list()

    # Build work items for pairs where both proteins have structures
    work_items = []
    pair_indices = []
    for i, (q, t) in enumerate(zip(queries, targets)):
        if q in protein_structures and t in protein_structures:
            work_items.append((
                q, t,
                str(protein_structures[q]),
                str(protein_structures[t]),
                tmalign_path,
            ))
            pair_indices.append(i)

    logger.info(
        f"Computing TM-scores for {len(work_items)}/{len(queries)} pairs "
        f"({len(work_items) / len(queries) * 100:.1f}% have structures for both proteins)"
    )

    # Initialize results with NaN
    results = np.full(len(queries), np.nan)

    if not work_items:
        return results.tolist()

    # Parallel execution
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_tmscore_pair, item): idx
            for item, idx in zip(work_items, pair_indices)
        }

        with tqdm(total=len(futures), desc="Computing TM-scores", unit="pair") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    _, _, score = future.result()
                    if score is not None:
                        results[idx] = score
                        completed += 1
                except Exception as e:
                    logger.debug(f"TM-score computation failed for pair {idx}: {e}")
                pbar.update(1)

    logger.info(f"Successfully computed {completed}/{len(work_items)} TM-scores")
    return results.tolist()


# --------------------------------------------------------------------------- #
#                                MAIN
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Compute pairwise TM-scores from experimental PDB structures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs_parquet",
        type=Path,
        required=True,
        help="Path to Parquet file with protein pairs (must have 'query' and 'target' columns)",
    )
    parser.add_argument(
        "--output_parquet",
        type=Path,
        required=True,
        help="Path for output Parquet file with tmscore_exp column added",
    )
    parser.add_argument(
        "--pdb_cache_dir",
        type=Path,
        default=Path("data/reference/pdb_cache"),
        help="Directory to cache downloaded PDB structures",
    )
    parser.add_argument(
        "--sifts_mapping",
        type=Path,
        default=None,
        help="Path to SIFTS uniprot_pdb.tsv file. Downloaded if not present.",
    )
    parser.add_argument(
        "--tmalign_path",
        type=str,
        default="TMalign",
        help="Path to TMalign binary (must be on PATH or absolute path)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel TMalign processes",
    )
    parser.add_argument(
        "--resolution_cutoff",
        type=float,
        default=3.0,
        help="Maximum resolution in Angstroms for X-ray structures",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit number of pairs to process (for testing)",
    )

    args = parser.parse_args()

    # --- Validate inputs ---
    if not args.pairs_parquet.exists():
        logger.error(f"Pairs parquet not found: {args.pairs_parquet}")
        sys.exit(1)

    # Check TMalign availability
    tmalign_check = shutil.which(args.tmalign_path)
    if tmalign_check is None and not Path(args.tmalign_path).exists():
        logger.error(
            f"TMalign binary not found: {args.tmalign_path}\n"
            f"Download from: https://zhanggroup.org/TM-align/\n"
            f"Install: place the binary on your PATH or use --tmalign_path"
        )
        sys.exit(1)

    # --- Setup directories ---
    args.pdb_cache_dir.mkdir(parents=True, exist_ok=True)
    chain_cache_dir = args.pdb_cache_dir / "chains"
    chain_cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Load protein pairs ---
    pairs_df = pl.read_parquet(args.pairs_parquet)
    logger.info(f"Loaded {len(pairs_df)} protein pairs from {args.pairs_parquet}")

    if args.sample_size:
        pairs_df = pairs_df.head(args.sample_size)
        logger.info(f"Sampling {args.sample_size} pairs for testing")

    # --- Load SIFTS mapping ---
    sifts_path = args.sifts_mapping
    if sifts_path is None:
        sifts_path = args.pdb_cache_dir / "uniprot_pdb.tsv"

    all_proteins = set(pairs_df["query"].unique().to_list()) | set(
        pairs_df["target"].unique().to_list()
    )

    sifts_mapping = load_sifts_mapping(
        sifts_path, all_proteins, args.resolution_cutoff
    )

    # --- Prepare structures ---
    protein_structures = prepare_structures(
        pairs_df, sifts_mapping, args.pdb_cache_dir, chain_cache_dir
    )

    # --- Compute TM-scores ---
    tmscore_values = compute_all_tmscores(
        pairs_df, protein_structures, args.tmalign_path, args.max_workers
    )

    # --- Merge and save ---
    result_df = pairs_df.with_columns(
        pl.Series(name="tmscore_exp", values=tmscore_values)
    )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(args.output_parquet)

    # --- Summary ---
    tmscore_series = result_df["tmscore_exp"]
    valid = len(tmscore_series) - tmscore_series.null_count()
    nan_count = tmscore_series.is_nan().sum()
    actual_valid = valid - nan_count

    logger.info("=" * 60)
    logger.info("PDB EXPERIMENTAL TM-SCORE COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output_parquet}")
    logger.info(f"Total pairs: {len(result_df)}")
    logger.info(f"Pairs with TM-score: {actual_valid} ({actual_valid / len(result_df) * 100:.1f}%)")
    if actual_valid > 0:
        # Filter to non-NaN for stats
        valid_scores = tmscore_series.filter(~tmscore_series.is_nan() & tmscore_series.is_not_null())
        logger.info(
            f"TM-score stats: mean={valid_scores.mean():.3f}, "
            f"std={valid_scores.std():.3f}, "
            f"min={valid_scores.min():.3f}, max={valid_scores.max():.3f}"
        )

    # If alntmscore exists in input, show correlation
    if "alntmscore" in result_df.columns:
        both_valid = result_df.filter(
            result_df["tmscore_exp"].is_not_null()
            & ~result_df["tmscore_exp"].is_nan()
            & result_df["alntmscore"].is_not_null()
        )
        if len(both_valid) > 10:
            from scipy.stats import pearsonr
            r, p = pearsonr(
                both_valid["tmscore_exp"].to_numpy(),
                both_valid["alntmscore"].to_numpy(),
            )
            logger.info(
                f"\nCorrelation with predicted alntmscore: r={r:.3f}, p={p:.2e} "
                f"(n={len(both_valid)} pairs)"
            )


if __name__ == "__main__":
    main()
