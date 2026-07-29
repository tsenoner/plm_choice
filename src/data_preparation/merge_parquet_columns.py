#!/usr/bin/env python3
"""
Merge New Columns Into Training Parquet Files

Merges columns from analysis parquets (GO similarity, TM-scores, etc.) into
the main train/val/test split parquets used by the training pipeline.

The training pipeline (datasets.py) reads a single parquet per split with
columns: query, target, <param_name>. New target parameters must be merged
into these split files before they can be used for training.

Usage:
    # Merge GO similarity columns into all splits
    uv run python src/data_preparation/merge_parquet_columns.py \
        --source data/processed/sprot_pre2024/sets/test_with_go.parquet \
        --target_dir data/processed/sprot_pre2024/sets/ \
        --columns go_wang_mfo go_wang_bpo go_wang_cco

    # Merge TM-score column
    uv run python src/data_preparation/merge_parquet_columns.py \
        --source data/processed/sprot_pre2024/sets/test_with_tmscore_exp.parquet \
        --target_dir data/processed/sprot_pre2024/sets/ \
        --columns tmscore_exp

Created: 2026-03-19 (Ivan infrastructure for pLM Choice revision)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def merge_columns(
    source_path: Path,
    target_path: Path,
    columns: List[str],
) -> int:
    """
    Merge specific columns from source parquet into target parquet.

    Joins on (query, target) pair keys. Overwrites target file in place.

    Returns:
        Number of rows with non-null values in the merged columns
    """
    source_df = pl.read_parquet(source_path)
    target_df = pl.read_parquet(target_path)

    # Validate source has the requested columns
    missing = [c for c in columns if c not in source_df.columns]
    if missing:
        logger.error(f"Columns {missing} not found in {source_path}")
        logger.info(f"Available columns: {source_df.columns}")
        return 0

    # Drop existing columns in target that we're replacing
    existing = [c for c in columns if c in target_df.columns]
    if existing:
        logger.info(f"  Replacing existing columns in {target_path.name}: {existing}")
        target_df = target_df.drop(existing)

    # Select join keys + new columns from source
    source_subset = source_df.select(["query", "target"] + columns)

    # Left join: keep all target rows, add source columns where matching
    merged = target_df.join(source_subset, on=["query", "target"], how="left")

    # Count valid values
    valid_counts = {}
    for col in columns:
        valid = len(merged) - merged[col].null_count()
        nan_count = merged[col].is_nan().sum() if merged[col].dtype == pl.Float64 else 0
        valid_counts[col] = valid - nan_count

    # Write back
    merged.write_parquet(target_path)
    return sum(valid_counts.values())


def main():
    parser = argparse.ArgumentParser(
        description="Merge new columns from analysis parquets into train/val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source parquet with new columns (e.g., test_with_go.parquet)",
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        required=True,
        help="Directory containing train.parquet, val.parquet, test.parquet",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="Column names to merge (e.g., go_wang_mfo go_wang_bpo go_wang_cco)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which split files to merge into",
    )

    args = parser.parse_args()

    if not args.source.exists():
        logger.error(f"Source file not found: {args.source}")
        sys.exit(1)

    logger.info(f"Source: {args.source}")
    logger.info(f"Columns to merge: {args.columns}")
    logger.info(f"Target directory: {args.target_dir}")

    for split in args.splits:
        target_path = args.target_dir / f"{split}.parquet"
        if not target_path.exists():
            logger.warning(f"  {split}.parquet not found, skipping")
            continue

        valid = merge_columns(args.source, target_path, args.columns)
        logger.info(f"  {split}.parquet: merged {len(args.columns)} columns ({valid} total valid values)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
