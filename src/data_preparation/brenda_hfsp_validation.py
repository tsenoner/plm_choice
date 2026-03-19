#!/usr/bin/env python3
"""
BRENDA/HFSP Validation Script

Validates HFSP (Homologous Function Similarity Prediction) scores against
curated enzyme functional classifications. Uses UniProt annotations to group
enzymes by their functional class (e.g., Ambler classes A/B/C/D for
beta-lactamases), then tests whether HFSP correctly separates within-class
pairs (high similarity) from between-class pairs (low similarity).

This is a sanity check for HFSP as a target parameter: if HFSP can't
separate well-characterized functional classes, it's not a reliable training
signal.

Usage:
    # Validate HFSP on beta-lactamases (EC 3.5.2.6)
    uv run python src/data_preparation/brenda_hfsp_validation.py \\
        --pairs_parquet data/processed/sprot_pre2024/sets/test.parquet \\
        --output_dir out/hfsp_validation \\
        --enzyme_ec 3.5.2.6

    # Validate on any enzyme class
    uv run python src/data_preparation/brenda_hfsp_validation.py \\
        --pairs_parquet data/processed/sprot_pre2024/sets/test.parquet \\
        --output_dir out/hfsp_validation \\
        --enzyme_ec 2.7.11.1 \\
        --class_column protein_families

Created: 2026-03-19 (Ivan infrastructure for pLM Choice revision)
"""
# --- Ivan infrastructure (2026-03-19) ---

import argparse
import json
import logging
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# UniProt REST API
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

# Ambler class patterns for beta-lactamases (EC 3.5.2.6)
# These are well-characterized and serve as the primary validation target
AMBLER_CLASS_PATTERNS = {
    "Class_A": ["class a", "class-a", "tem-", "shv-", "ctx-m", "kpc-"],
    "Class_B": ["class b", "class-b", "metallo-beta-lactamase", "ndm-", "vim-", "imp-"],
    "Class_C": ["class c", "class-c", "ampc", "cephalosporinase"],
    "Class_D": ["class d", "class-d", "oxa-", "oxacillinase"],
}


# --------------------------------------------------------------------------- #
#                        UNIPROT ANNOTATION FETCHING
# --------------------------------------------------------------------------- #


def fetch_enzyme_annotations(
    ec_number: str,
    fields: str = "accession,protein_name,ec,keyword,ft_domain",
    max_results: int = 5000,
) -> List[Dict]:
    """
    Fetch enzyme entries from UniProt REST API by EC number.

    Returns list of dicts with protein annotations.
    """
    params = {
        "query": f"(ec:{ec_number}) AND (reviewed:true)",
        "format": "tsv",
        "fields": fields,
        "size": str(min(max_results, 500)),
    }

    url = f"{UNIPROT_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    logger.info(f"Fetching UniProt entries for EC {ec_number} ...")

    entries = []
    try:
        while url:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "plm_choice/1.0 (validation script)")

            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode("utf-8")

                # Parse TSV
                lines = content.strip().split("\n")
                if not lines:
                    break

                if not entries:
                    # First page — extract header
                    header = lines[0].split("\t")
                    data_lines = lines[1:]
                else:
                    data_lines = lines

                for line in data_lines:
                    parts = line.split("\t")
                    if len(parts) >= len(header):
                        entry = dict(zip(header, parts))
                        entries.append(entry)

                # Check for pagination (Link header)
                link_header = response.headers.get("Link", "")
                if 'rel="next"' in link_header:
                    # Extract next URL
                    next_url = link_header.split(";")[0].strip("<>")
                    url = next_url
                else:
                    url = None

                if len(entries) >= max_results:
                    break

    except Exception as e:
        logger.error(f"Error fetching from UniProt: {e}")

    logger.info(f"Fetched {len(entries)} reviewed entries for EC {ec_number}")
    return entries


def classify_beta_lactamases(entries: List[Dict]) -> Dict[str, str]:
    """
    Classify beta-lactamase entries into Ambler classes (A/B/C/D).

    Uses protein name, keywords, and domain annotations to determine class.
    Returns dict mapping accession -> class label.
    """
    classifications: Dict[str, str] = {}

    for entry in entries:
        accession = entry.get("Entry", entry.get("accession", ""))
        if not accession:
            continue

        # Combine all text fields for pattern matching
        text = " ".join([
            entry.get("Protein names", ""),
            entry.get("Keywords", ""),
            entry.get("Domain [FT]", ""),
        ]).lower()

        assigned = False
        for class_name, patterns in AMBLER_CLASS_PATTERNS.items():
            if any(p in text for p in patterns):
                classifications[accession] = class_name
                assigned = True
                break

        if not assigned:
            classifications[accession] = "Unclassified"

    # Log distribution
    class_counts = defaultdict(int)
    for cls in classifications.values():
        class_counts[cls] += 1
    logger.info(f"Classification distribution: {dict(class_counts)}")

    return classifications


def classify_by_annotation(
    entries: List[Dict], class_column: str = "Keywords"
) -> Dict[str, str]:
    """
    Generic classification: group proteins by a UniProt annotation field.

    Uses the first keyword or domain as the class label.
    """
    classifications: Dict[str, str] = {}

    for entry in entries:
        accession = entry.get("Entry", entry.get("accession", ""))
        if not accession:
            continue

        # Use first entry in the specified column as class
        field_value = entry.get(class_column, "")
        if field_value:
            # Take first keyword/annotation as class
            first_class = field_value.split(";")[0].strip()
            classifications[accession] = first_class
        else:
            classifications[accession] = "Unknown"

    return classifications


# --------------------------------------------------------------------------- #
#                        HFSP VALIDATION
# --------------------------------------------------------------------------- #


def validate_hfsp(
    pairs_df: pl.DataFrame,
    classifications: Dict[str, str],
    hfsp_col: str = "hfsp",
) -> Dict:
    """
    Validate HFSP by comparing within-class vs between-class distributions.

    Returns dict with validation statistics.
    """
    queries = pairs_df["query"].to_list()
    targets = pairs_df["target"].to_list()

    # Check if hfsp column exists
    if hfsp_col not in pairs_df.columns:
        logger.error(f"Column '{hfsp_col}' not found. Available: {pairs_df.columns}")
        return {"error": f"Column '{hfsp_col}' not found"}

    hfsp_values = pairs_df[hfsp_col].to_numpy()

    within_class = []
    between_class = []
    annotated_pairs = 0

    for i in range(len(queries)):
        q_class = classifications.get(queries[i])
        t_class = classifications.get(targets[i])

        if q_class is None or t_class is None:
            continue
        if q_class == "Unclassified" or t_class == "Unclassified":
            continue
        if q_class == "Unknown" or t_class == "Unknown":
            continue

        hfsp = hfsp_values[i]
        if np.isnan(hfsp):
            continue

        annotated_pairs += 1
        if q_class == t_class:
            within_class.append(hfsp)
        else:
            between_class.append(hfsp)

    within_arr = np.array(within_class) if within_class else np.array([])
    between_arr = np.array(between_class) if between_class else np.array([])

    results = {
        "annotated_pairs": annotated_pairs,
        "within_class_pairs": len(within_arr),
        "between_class_pairs": len(between_arr),
        "within_class_mean": float(np.mean(within_arr)) if len(within_arr) > 0 else None,
        "within_class_std": float(np.std(within_arr)) if len(within_arr) > 0 else None,
        "within_class_median": float(np.median(within_arr)) if len(within_arr) > 0 else None,
        "between_class_mean": float(np.mean(between_arr)) if len(between_arr) > 0 else None,
        "between_class_std": float(np.std(between_arr)) if len(between_arr) > 0 else None,
        "between_class_median": float(np.median(between_arr)) if len(between_arr) > 0 else None,
    }

    # Statistical tests
    if len(within_arr) >= 5 and len(between_arr) >= 5:
        from scipy.stats import mannwhitneyu

        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = mannwhitneyu(
            within_arr, between_arr, alternative="greater"
        )
        results["mann_whitney_u"] = float(u_stat)
        results["mann_whitney_p"] = float(p_value)

        # Cohen's d (effect size)
        pooled_std = np.sqrt(
            (np.var(within_arr) * (len(within_arr) - 1)
             + np.var(between_arr) * (len(between_arr) - 1))
            / (len(within_arr) + len(between_arr) - 2)
        )
        if pooled_std > 0:
            cohens_d = (np.mean(within_arr) - np.mean(between_arr)) / pooled_std
            results["cohens_d"] = float(cohens_d)
        else:
            results["cohens_d"] = None

        # Separation quality assessment
        if p_value < 0.001 and results.get("cohens_d", 0) and results["cohens_d"] > 0.8:
            results["assessment"] = "GOOD: Strong separation between functional classes"
        elif p_value < 0.05:
            results["assessment"] = "MODERATE: Significant but weak separation"
        else:
            results["assessment"] = "POOR: HFSP does not separate these functional classes"
    else:
        results["assessment"] = "INSUFFICIENT DATA: Need >= 5 pairs in each group"

    return results


# --------------------------------------------------------------------------- #
#                                MAIN
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Validate HFSP scores against curated enzyme functional classifications.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs_parquet",
        type=Path,
        required=True,
        help="Parquet file with protein pairs (must have 'hfsp' column)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("out/hfsp_validation"),
        help="Output directory for validation report",
    )
    parser.add_argument(
        "--enzyme_ec",
        type=str,
        default="3.5.2.6",
        help="EC number to validate (default: 3.5.2.6 = beta-lactamases)",
    )
    parser.add_argument(
        "--hfsp_col",
        type=str,
        default="hfsp",
        help="Column name for HFSP values in the parquet",
    )
    parser.add_argument(
        "--class_column",
        type=str,
        default=None,
        help="UniProt field to use for classification (default: auto-detect for beta-lactamases)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit pairs for testing",
    )

    args = parser.parse_args()

    if not args.pairs_parquet.exists():
        logger.error(f"Parquet not found: {args.pairs_parquet}")
        sys.exit(1)

    # Load pairs
    pairs_df = pl.read_parquet(args.pairs_parquet)
    logger.info(f"Loaded {len(pairs_df)} pairs from {args.pairs_parquet}")

    if args.sample_size:
        pairs_df = pairs_df.head(args.sample_size)

    # Fetch enzyme annotations
    entries = fetch_enzyme_annotations(args.enzyme_ec)

    if not entries:
        logger.error("No entries fetched. Check EC number and internet connection.")
        sys.exit(1)

    # Classify enzymes
    if args.enzyme_ec == "3.5.2.6" and args.class_column is None:
        # Special handling for beta-lactamases: use Ambler classification
        classifications = classify_beta_lactamases(entries)
    elif args.class_column:
        classifications = classify_by_annotation(entries, args.class_column)
    else:
        # Default: use Keywords field
        classifications = classify_by_annotation(entries, "Keywords")

    # Validate HFSP
    results = validate_hfsp(pairs_df, classifications, args.hfsp_col)
    results["enzyme_ec"] = args.enzyme_ec
    results["total_entries_fetched"] = len(entries)

    # Save report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"hfsp_validation_{args.enzyme_ec.replace('.', '_')}.json"

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    logger.info("=" * 60)
    logger.info("HFSP VALIDATION REPORT")
    logger.info("=" * 60)
    logger.info(f"Enzyme: EC {args.enzyme_ec}")
    logger.info(f"UniProt entries: {len(entries)}")
    logger.info(f"Annotated pairs: {results['annotated_pairs']}")
    logger.info(f"Within-class pairs: {results['within_class_pairs']}")
    logger.info(f"Between-class pairs: {results['between_class_pairs']}")

    if results.get("within_class_mean") is not None:
        logger.info(
            f"Within-class HFSP:  mean={results['within_class_mean']:.3f}, "
            f"median={results['within_class_median']:.3f}"
        )
    if results.get("between_class_mean") is not None:
        logger.info(
            f"Between-class HFSP: mean={results['between_class_mean']:.3f}, "
            f"median={results['between_class_median']:.3f}"
        )
    if results.get("mann_whitney_p") is not None:
        logger.info(f"Mann-Whitney p={results['mann_whitney_p']:.2e}")
    if results.get("cohens_d") is not None:
        logger.info(f"Cohen's d={results['cohens_d']:.2f}")

    logger.info(f"\nAssessment: {results.get('assessment', 'N/A')}")
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
