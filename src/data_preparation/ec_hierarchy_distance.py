# --- Ivan infrastructure (2026-03-19) ---
#!/usr/bin/env python3
"""
EC-Number Hierarchy Distance

Computes Enzyme Commission (EC) number distances between protein pairs. EC
numbers are a hierarchical classification system with 4 levels:

    Level 1: General class of enzyme (e.g. 3 = Hydrolases)
    Level 2: Subclass by bond type acted on (e.g. 3.4 = peptide bonds)
    Level 3: Sub-subclass by mechanism (e.g. 3.4.21 = serine endopeptidases)
    Level 4: Specific enzyme (e.g. 3.4.21.9 = enteropeptidase)

The EC distance between two enzymes is defined as:
    0  — identical EC numbers (same enzyme)
    1  — differ only at level 4 (same sub-subclass, different enzyme)
    2  — differ at level 3 (same subclass, different sub-subclass)
    3  — differ at level 2 (same general class, different subclass)
    4  — differ at level 1 (completely different enzyme classes)
    None — one or both EC numbers have wildcards ("-") at or before the
           divergence point, making the distance ambiguous.

This provides a clean ordinal metric for functional similarity that plugs
into the existing pLM Choice training pipeline alongside fident, hfsp,
and alntmscore.

Usage:
    uv run python src/data_preparation/ec_hierarchy_distance.py \
        --annotations data/processed/ec_annotations.tsv \
        --pairs_parquet data/processed/sprot_pre2024/sets/test.parquet \
        --output_parquet data/processed/sprot_pre2024/sets/test_with_ec.parquet

Created: 2026-03-19 (Ivan infrastructure for pLM Choice revision)
"""

import argparse
import logging
import sys
from collections import defaultdict
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


# --------------------------------------------------------------------------- #
#                        EC NUMBER PARSING & DISTANCE
# --------------------------------------------------------------------------- #


def parse_ec_number(ec_str: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Parse an EC number string into a 4-element tuple.

    Each level is an integer, or None if the level is a wildcard ("-")
    indicating an incomplete or provisional classification.

    Args:
        ec_str: EC number string, e.g. "3.4.21.9" or "3.4.21.-"

    Returns:
        Tuple of 4 elements, each int or None.

    Examples:
        >>> parse_ec_number("3.4.21.9")
        (3, 4, 21, 9)
        >>> parse_ec_number("3.4.21.-")
        (3, 4, 21, None)
        >>> parse_ec_number("3.-.-.-")
        (3, None, None, None)
    """
    parts = ec_str.strip().split(".")
    if len(parts) != 4:
        raise ValueError(f"Invalid EC number format (expected 4 levels): {ec_str!r}")

    result: List[Optional[int]] = []
    for part in parts:
        part = part.strip()
        if part == "-" or part == "n" or part == "":
            result.append(None)
        else:
            try:
                result.append(int(part))
            except ValueError:
                raise ValueError(
                    f"Invalid EC number component {part!r} in {ec_str!r}"
                )

    return (result[0], result[1], result[2], result[3])


def ec_distance(ec_a: str, ec_b: str) -> Optional[int]:
    """
    Compute the hierarchy distance between two EC numbers.

    Distance is 4 minus the number of leading levels that match:
        0  — identical (all 4 levels match)
        1  — differ at level 4 only
        2  — differ at level 3
        3  — differ at level 2
        4  — differ at level 1

    Returns None if either EC number has a wildcard ("-") at or before
    the level where the two numbers first diverge, since the true distance
    is ambiguous.

    Args:
        ec_a: First EC number string (e.g. "3.4.21.9")
        ec_b: Second EC number string (e.g. "3.4.21.4")

    Returns:
        Integer distance 0-4, or None if distance is ambiguous due to wildcards.
    """
    parsed_a = parse_ec_number(ec_a)
    parsed_b = parse_ec_number(ec_b)

    # Walk levels from top (level 1) to bottom (level 4)
    matching_levels = 0
    for i in range(4):
        a_val = parsed_a[i]
        b_val = parsed_b[i]

        # If either is a wildcard at this level, distance is ambiguous
        if a_val is None or b_val is None:
            return None

        if a_val != b_val:
            # First divergence at level i+1 → distance = 4 - matching_levels
            break

        matching_levels += 1

    return 4 - matching_levels


# --------------------------------------------------------------------------- #
#                        ANNOTATION LOADING
# --------------------------------------------------------------------------- #


def load_ec_annotations(
    path: Path,
    target_proteins: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Load protein-to-EC-number mappings from a TSV file.

    Supports two formats:

    1. Simple TSV (tab-separated, optional header):
           protein_id    ec_number
       e.g.:
           P00750    2.7.7.6
           P00750    3.1.3.48

    2. UniProt ID-mapping format (tab-separated with header):
           From    Entry    EC number
       where EC number column may contain semicolon-separated values.

    Proteins with no valid EC annotations are silently skipped.

    Args:
        path: Path to TSV file with EC annotations.
        target_proteins: If provided, only load annotations for these protein
            IDs. If None, load all.

    Returns:
        Dict mapping protein_id → set of EC number strings.
    """
    annotations: Dict[str, Set[str]] = defaultdict(set)

    skipped = 0
    loaded = 0
    header_seen = False

    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue

            parts = line.split("\t")

            # Detect and skip header line
            if not header_seen:
                header_seen = True
                # Check if first row looks like a header
                if any(
                    h in parts[0].lower()
                    for h in ("from", "entry", "protein", "accession", "id")
                ):
                    continue

            if len(parts) >= 3:
                # UniProt ID-mapping format: From | Entry | EC number [| ...]
                # Use Entry (column 1) as protein_id
                protein_id = parts[1].strip()
                ec_field = parts[2].strip()

                if not ec_field or ec_field == "-":
                    skipped += 1
                    continue

                # EC number column may contain semicolon-separated values
                ec_numbers = [e.strip() for e in ec_field.split(";") if e.strip()]

            elif len(parts) == 2:
                # Simple TSV: protein_id | ec_number
                protein_id = parts[0].strip()
                ec_raw = parts[1].strip()

                if not ec_raw or ec_raw == "-":
                    skipped += 1
                    continue

                ec_numbers = [e.strip() for e in ec_raw.split(";") if e.strip()]

            else:
                skipped += 1
                continue

            # Filter by target proteins if specified
            if target_proteins is not None and protein_id not in target_proteins:
                skipped += 1
                continue

            # Validate and add EC numbers
            for ec_num in ec_numbers:
                # Basic validation: should look like X.X.X.X
                if ec_num.count(".") == 3:
                    annotations[protein_id].add(ec_num)
                    loaded += 1
                else:
                    skipped += 1

    logger.info(
        f"Loaded {loaded} EC annotations for {len(annotations)} proteins "
        f"(skipped {skipped} entries)"
    )
    return dict(annotations)


# --------------------------------------------------------------------------- #
#                     PAIR DISTANCE COMPUTATION
# --------------------------------------------------------------------------- #


def compute_pair_ec_distances(
    pairs_df: pl.DataFrame,
    annotations: Dict[str, Set[str]],
) -> Dict[str, List[float]]:
    """
    Compute EC hierarchy distances for all protein pairs.

    For each pair where both proteins have EC annotations, computes all
    pairwise EC distances (Cartesian product of EC sets) and reports the
    minimum, maximum, and mean distance. This handles multi-functional
    enzymes (proteins with multiple EC numbers) by capturing the closest
    and furthest functional relationships.

    Args:
        pairs_df: DataFrame with 'query' and 'target' columns.
        annotations: Dict mapping protein_id → set of EC number strings.

    Returns:
        Dict mapping column name → list of float values, with keys:
        - 'ec_dist_min': minimum EC distance across all EC pairs
        - 'ec_dist_max': maximum EC distance across all EC pairs
        - 'ec_dist_mean': mean EC distance across all EC pairs
        Values are NaN where either protein lacks EC annotations or all
        pairwise distances are ambiguous (None).
    """
    results: Dict[str, List[float]] = {
        "ec_dist_min": [],
        "ec_dist_max": [],
        "ec_dist_mean": [],
    }

    queries = pairs_df["query"].to_list()
    targets = pairs_df["target"].to_list()

    annotated_count = 0
    total = len(queries)

    for i in tqdm(range(total), desc="Computing EC distances", unit="pair"):
        q_id = queries[i]
        t_id = targets[i]

        q_ecs = annotations.get(q_id, set())
        t_ecs = annotations.get(t_id, set())

        if not q_ecs or not t_ecs:
            # One or both proteins lack EC annotations
            results["ec_dist_min"].append(np.nan)
            results["ec_dist_max"].append(np.nan)
            results["ec_dist_mean"].append(np.nan)
            continue

        # Compute all pairwise distances
        distances: List[int] = []
        for ec_q in q_ecs:
            for ec_t in t_ecs:
                d = ec_distance(ec_q, ec_t)
                if d is not None:
                    distances.append(d)

        if distances:
            results["ec_dist_min"].append(float(min(distances)))
            results["ec_dist_max"].append(float(max(distances)))
            results["ec_dist_mean"].append(float(np.mean(distances)))
            annotated_count += 1
        else:
            # All pairwise distances were ambiguous (wildcards)
            results["ec_dist_min"].append(np.nan)
            results["ec_dist_max"].append(np.nan)
            results["ec_dist_mean"].append(np.nan)

    logger.info(
        f"Computed EC distances for {annotated_count}/{total} pairs "
        f"({annotated_count / max(total, 1) * 100:.1f}% had EC annotations in both proteins)"
    )
    return results


# --------------------------------------------------------------------------- #
#                                MAIN
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Compute EC-number hierarchy distances between protein pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to EC annotations file (TSV: protein_id, ec_number; or UniProt ID-mapping format)",
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
        help="Path for output Parquet file with EC distance columns added",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit number of pairs to process (for testing)",
    )

    args = parser.parse_args()

    # --- Validate inputs ---
    if not args.annotations.exists():
        logger.error(f"Annotations file not found: {args.annotations}")
        sys.exit(1)
    if not args.pairs_parquet.exists():
        logger.error(f"Pairs parquet not found: {args.pairs_parquet}")
        sys.exit(1)

    # --- Load annotations ---
    # First peek at the pairs to determine which proteins we need
    pairs_df = pl.read_parquet(args.pairs_parquet)
    logger.info(f"Loaded {len(pairs_df)} protein pairs from {args.pairs_parquet}")

    if args.sample_size:
        pairs_df = pairs_df.head(args.sample_size)
        logger.info(f"Sampling {args.sample_size} pairs for testing")

    all_proteins = set(pairs_df["query"].unique().to_list()) | set(
        pairs_df["target"].unique().to_list()
    )

    annotations = load_ec_annotations(args.annotations, target_proteins=all_proteins)

    # Check annotation coverage
    annotated_proteins = set(annotations.keys()) & all_proteins
    logger.info(
        f"EC annotation coverage: {len(annotated_proteins)}/{len(all_proteins)} proteins "
        f"({len(annotated_proteins) / max(len(all_proteins), 1) * 100:.1f}%)"
    )

    # --- Compute distances ---
    distance_columns = compute_pair_ec_distances(pairs_df, annotations)

    # --- Merge results ---
    result_df = pairs_df.clone()
    for col_name, values in distance_columns.items():
        result_df = result_df.with_columns(
            pl.Series(name=col_name, values=values)
        )

    # --- Save ---
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(args.output_parquet)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("EC HIERARCHY DISTANCE COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output_parquet}")
    for col_name in distance_columns:
        series = result_df[col_name]
        valid = len(series) - series.null_count()
        if valid > 0:
            logger.info(
                f"  {col_name}: {valid}/{len(series)} valid "
                f"({valid / len(series) * 100:.1f}%), "
                f"mean={series.mean():.3f}, std={series.std():.3f}"
            )
        else:
            logger.info(f"  {col_name}: no valid values")


if __name__ == "__main__":
    main()
