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
from typing import Dict, List, Optional

import polars as pl

from shared.atomic_io import atomic_write

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_source_subset(source_path: Path, columns: List[str]) -> Optional[pl.DataFrame]:
    """
    Read only the join keys and the columns being merged from the source parquet.

    Read once and reused across every split: the source is the same file for all of
    them, and projecting at read time skips decoding the columns we never join on.

    Returns None if the source lacks a requested column.
    """
    source_columns = pl.read_parquet_schema(source_path).keys()
    missing = [c for c in columns if c not in source_columns]
    if missing:
        logger.error(f"Columns {missing} not found in {source_path}")
        logger.info(f"Available columns: {list(source_columns)}")
        return None

    return pl.read_parquet(source_path, columns=["query", "target"] + columns)


def merge_columns(
    source_subset: pl.DataFrame,
    target_path: Path,
    columns: List[str],
) -> Dict[str, int]:
    """
    Merge specific columns from a source frame into target parquet.

    Joins on (query, target) pair keys. Replaces the target file in place.

    Returns:
        Per-column count of rows carrying a real (non-null, non-NaN) value.
    """
    target_df = pl.read_parquet(target_path)

    # Drop existing columns in target that we're replacing
    existing = [c for c in columns if c in target_df.columns]
    if existing:
        logger.info(f"  Replacing existing columns in {target_path.name}: {existing}")
        target_df = target_df.drop(existing)

    # Left join: keep all target rows, add source columns where matching
    merged = target_df.join(source_subset, on=["query", "target"], how="left")

    # Count valid values
    valid_counts: Dict[str, int] = {}
    for col in columns:
        valid = len(merged) - merged[col].null_count()
        # is_nan() is only defined on floats; check the kind, not one exact dtype,
        # so a Float32 column is not silently counted as NaN-free.
        nan_count = (
            merged[col].is_nan().sum() if merged[col].dtype.is_float() else 0
        )
        valid_counts[col] = int(valid - nan_count)

    # Write back through a tmp file + os.replace. The targets are the canonical
    # train/val/test splits; a plain write_parquet() over them means a job killed
    # mid-write (OOM, walltime) leaves a truncated training set that still looks
    # like a parquet.
    atomic_write(target_path, merged.write_parquet, mode="replace")
    return valid_counts


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

    # Fail loudly rather than "merging" nothing into every split in turn.
    source_subset = load_source_subset(args.source, args.columns)
    if source_subset is None:
        sys.exit(1)

    for split in args.splits:
        target_path = args.target_dir / f"{split}.parquet"
        if not target_path.exists():
            logger.warning(f"  {split}.parquet not found, skipping")
            continue

        valid_counts = merge_columns(source_subset, target_path, args.columns)
        total = sum(valid_counts.values())
        logger.info(
            f"  {split}.parquet: merged {len(args.columns)} columns "
            f"({total} total valid values)"
        )

        # An all-null column is the silent failure this tool invites: the source
        # is typically computed on ONE split (e.g. test_with_go.parquet), so
        # merging it into train/val adds columns that are 100% null and would
        # only surface much later, as a training run with no usable targets.
        empty = [col for col, n in valid_counts.items() if n == 0]
        if empty:
            logger.warning(
                "  %s.parquet: %s came out entirely null — no (query, target) pair "
                "in %s matched this split. Run the producer on %s.parquet too, or "
                "restrict --splits.",
                split,
                ", ".join(empty),
                args.source.name,
                split,
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
