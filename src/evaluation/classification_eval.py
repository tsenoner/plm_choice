#!/usr/bin/env python3
"""
Classification Evaluation at SCOP/ECOD Hierarchy Levels

Evaluates embedding-based similarity as a classifier: for each SCOP/ECOD
hierarchy level (Family, Superfamily, Fold), computes AUROC and
recall-at-first-false-positive.

The question we answer: "If I use embedding distance as a predictor of
structural/functional relatedness, how well does it work at each level?"

Supports:
- SCOP hierarchy: fold_id, sf_id (superfamily), fa_id (family)
- ECOD hierarchy: T_group (topology), H_group (homology), X_group, F_group
- Custom hierarchy: any column with categorical labels

Usage:
    uv run python src/evaluation/classification_eval.py \\
        --pairs_parquet data/processed/sprot_pre2024/sets/test_with_distances.parquet \\
        --classification_parquet data/processed/sprot_pre2024/scop_classifications.parquet \\
        --distance_columns dist_prott5 dist_esm2_650m dist_esm2_3b \\
        --hierarchy_columns fold_id sf_id fa_id \\
        --output_dir out/classification_eval

Created: 2026-03-19 (Ivan infrastructure for pLM Choice revision)
"""
# --- Ivan infrastructure (2026-03-19) ---

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#                     CORE CLASSIFICATION EVALUATION
# --------------------------------------------------------------------------- #


def recall_at_first_fp(
    distances: np.ndarray,
    labels: np.ndarray,
    lower_is_similar: bool = True,
) -> Dict[str, float]:
    """
    Recall at first false positive.

    Sort pairs by predicted distance, scan until first false positive,
    report fraction of true positives retrieved.

    Args:
        distances: predicted distances (lower = more similar by default)
        labels: boolean array (True = same class, False = different class)
        lower_is_similar: if True, sort ascending; if False, sort descending

    Returns:
        dict with recall_at_first_fp, n_retrieved, n_positives
    """
    n_positives = int(labels.sum())

    if n_positives == 0:
        return {"recall_at_first_fp": 0.0, "n_retrieved": 0, "n_positives": 0}

    if (~labels).sum() == 0:
        # No negatives — all pairs are positive, recall = 1
        return {"recall_at_first_fp": 1.0, "n_retrieved": n_positives, "n_positives": n_positives}

    # Sort by distance
    order = np.argsort(distances) if lower_is_similar else np.argsort(-distances)
    sorted_labels = labels[order]

    # Scan until first false positive
    n_retrieved = 0
    for label in sorted_labels:
        if not label:
            break
        n_retrieved += 1

    return {
        "recall_at_first_fp": n_retrieved / n_positives,
        "n_retrieved": n_retrieved,
        "n_positives": n_positives,
    }


def auroc_at_level(
    distances: np.ndarray,
    labels: np.ndarray,
    lower_is_similar: bool = True,
) -> float:
    """
    AUROC for binary classification: same class vs different class.

    Args:
        distances: predicted distances
        labels: boolean array (True = same class)
        lower_is_similar: if True, negate distances for sklearn (higher score = more similar)

    Returns:
        AUROC score, or NaN if only one class present
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return np.nan

    # sklearn expects higher scores for positive class
    scores = -distances if lower_is_similar else distances

    try:
        return float(roc_auc_score(labels.astype(int), scores))
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- #
#                     HIERARCHY-LEVEL EVALUATION
# --------------------------------------------------------------------------- #


def build_same_level_labels(
    pairs_df: pl.DataFrame,
    protein_classifications: Dict[str, str],
) -> np.ndarray:
    """
    Build boolean labels: True if both proteins share the same classification.

    Args:
        pairs_df: DataFrame with 'query' and 'target' columns
        protein_classifications: dict mapping protein_id -> class label

    Returns:
        boolean numpy array (True = same class, False = different or unknown)
    """
    queries = pairs_df["query"].to_list()
    targets = pairs_df["target"].to_list()

    labels = np.full(len(queries), False)

    for i, (q, t) in enumerate(zip(queries, targets)):
        q_class = protein_classifications.get(q)
        t_class = protein_classifications.get(t)

        if q_class is not None and t_class is not None:
            labels[i] = (q_class == t_class)

    return labels


def evaluate_at_hierarchy_levels(
    pairs_df: pl.DataFrame,
    distance_columns: List[str],
    classification_map: Dict[str, Dict[str, str]],
) -> pl.DataFrame:
    """
    Evaluate all distance columns at all hierarchy levels.

    Args:
        pairs_df: DataFrame with query, target, and distance columns
        distance_columns: list of distance column names (e.g. ["dist_prott5", "dist_esm2_650m"])
        classification_map: dict of {level_name: {protein_id: class_label}}

    Returns:
        Summary DataFrame with columns: embedding, level, auroc, recall_at_first_fp, n_positive_pairs, n_negative_pairs
    """
    results = []

    for level_name, protein_classes in classification_map.items():
        logger.info(f"Evaluating at level: {level_name}")

        # Build labels for this hierarchy level
        labels = build_same_level_labels(pairs_df, protein_classes)

        n_positive = int(labels.sum())
        n_negative = int((~labels).sum())

        # Need both classes for meaningful evaluation
        if n_positive < 10 or n_negative < 10:
            logger.warning(
                f"  Skipping {level_name}: insufficient pairs "
                f"(positive={n_positive}, negative={n_negative})"
            )
            continue

        logger.info(
            f"  {level_name}: {n_positive} same-class, {n_negative} different-class pairs"
        )

        for dist_col in distance_columns:
            if dist_col not in pairs_df.columns:
                logger.warning(f"  Column {dist_col} not found, skipping")
                continue

            distances = pairs_df[dist_col].to_numpy().astype(np.float64)

            # Filter NaN distances
            valid_mask = ~np.isnan(distances)
            if valid_mask.sum() < 20:
                logger.warning(f"  {dist_col}: too few valid distances, skipping")
                continue

            valid_distances = distances[valid_mask]
            valid_labels = labels[valid_mask]

            # Compute metrics
            auc = auroc_at_level(valid_distances, valid_labels)
            recall = recall_at_first_fp(valid_distances, valid_labels)

            results.append({
                "embedding": dist_col.replace("dist_", ""),
                "level": level_name,
                "auroc": auc,
                "recall_at_first_fp": recall["recall_at_first_fp"],
                "n_retrieved_before_fp": recall["n_retrieved"],
                "n_positive_pairs": int(valid_labels.sum()),
                "n_negative_pairs": int((~valid_labels).sum()),
                "n_valid_pairs": int(valid_mask.sum()),
            })

    if not results:
        logger.warning("No results computed. Check classification coverage.")
        return pl.DataFrame()

    return pl.DataFrame(results)


# --------------------------------------------------------------------------- #
#                     CLASSIFICATION LOADING
# --------------------------------------------------------------------------- #


def load_classifications_from_parquet(
    classification_path: Path,
    hierarchy_columns: List[str],
) -> Dict[str, Dict[str, str]]:
    """
    Load protein classifications from a parquet file.

    Expected format: protein_id column + one or more hierarchy columns
    (e.g., fold_id, sf_id, fa_id for SCOP; T_group, H_group for ECOD).

    Returns:
        Dict mapping level_name -> {protein_id: class_label}
    """
    df = pl.read_parquet(classification_path)

    # Auto-detect protein ID column
    id_col = None
    for candidate in ["protein_id", "query", "accession", "Entry", "id"]:
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is None:
        raise ValueError(
            f"No protein ID column found. Available: {df.columns}. "
            f"Expected one of: protein_id, query, accession, Entry, id"
        )

    classification_map = {}
    for col in hierarchy_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not in {classification_path.name}, skipping")
            continue

        # Build protein -> class mapping, dropping nulls
        valid = df.filter(pl.col(col).is_not_null()).select([id_col, col])
        protein_classes = dict(zip(
            valid[id_col].to_list(),
            valid[col].cast(pl.Utf8).to_list(),
        ))

        n_classes = len(set(protein_classes.values()))
        logger.info(f"  {col}: {len(protein_classes)} proteins, {n_classes} classes")
        classification_map[col] = protein_classes

    return classification_map


# --------------------------------------------------------------------------- #
#                                MAIN
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding distances as classifiers at SCOP/ECOD hierarchy levels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs_parquet",
        type=Path,
        required=True,
        help="Parquet with protein pairs and distance columns",
    )
    parser.add_argument(
        "--classification_parquet",
        type=Path,
        required=True,
        help="Parquet with protein classifications (protein_id + hierarchy columns)",
    )
    parser.add_argument(
        "--distance_columns",
        nargs="+",
        required=True,
        help="Distance column names to evaluate (e.g., dist_prott5 dist_esm2_650m)",
    )
    parser.add_argument(
        "--hierarchy_columns",
        nargs="+",
        required=True,
        help="Classification hierarchy columns (e.g., fold_id sf_id fa_id)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("out/classification_eval"),
        help="Output directory",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit pairs for testing",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.pairs_parquet.exists():
        logger.error(f"Pairs parquet not found: {args.pairs_parquet}")
        sys.exit(1)
    if not args.classification_parquet.exists():
        logger.error(f"Classification parquet not found: {args.classification_parquet}")
        sys.exit(1)

    # Load data
    pairs_df = pl.read_parquet(args.pairs_parquet)
    logger.info(f"Loaded {len(pairs_df)} pairs with columns: {pairs_df.columns}")

    if args.sample_size:
        pairs_df = pairs_df.head(args.sample_size)

    # Load classifications
    logger.info(f"Loading classifications from {args.classification_parquet}")
    classification_map = load_classifications_from_parquet(
        args.classification_parquet, args.hierarchy_columns
    )

    if not classification_map:
        logger.error("No valid classifications loaded.")
        sys.exit(1)

    # Evaluate
    results_df = evaluate_at_hierarchy_levels(
        pairs_df, args.distance_columns, classification_map
    )

    if results_df.is_empty():
        logger.error("No results. Check classification coverage vs dataset proteins.")
        sys.exit(1)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "classification_eval_results.parquet"
    csv_path = args.output_dir / "classification_eval_results.csv"

    results_df.write_parquet(parquet_path)
    results_df.write_csv(csv_path)

    # Print summary table
    logger.info("=" * 80)
    logger.info("CLASSIFICATION EVALUATION RESULTS")
    logger.info("=" * 80)
    print(results_df.to_pandas().to_string(index=False, float_format="%.4f"))
    logger.info(f"\nResults saved to: {parquet_path}")
    logger.info(f"CSV export: {csv_path}")


if __name__ == "__main__":
    main()
