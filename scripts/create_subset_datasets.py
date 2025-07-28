#!/usr/bin/env python
"""
Create subset datasets by randomly sampling 1/10 of each set.
This script reads the original train/val/test parquet files and creates
smaller versions for faster experimentation.
"""

import polars as pl
from pathlib import Path
import argparse


def create_subset_datasets(
    input_dir: Path, output_dir: Path, sample_fraction: float = 0.1
):
    """
    Create subset datasets by sampling a fraction of each dataset.

    Args:
        input_dir: Path to directory containing train.parquet, val.parquet, test.parquet
        output_dir: Path to output directory for subset datasets
        sample_fraction: Fraction of data to sample (default: 0.1 for 10%)
    """

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset files to process
    dataset_files = ["train.parquet", "val.parquet", "test.parquet"]

    for filename in dataset_files:
        input_file = input_dir / filename
        output_file = output_dir / filename

        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue

        print(f"Processing {filename}...")

        # Read the parquet file
        df = pl.read_parquet(input_file)
        original_size = len(df)

        # Sample the data randomly
        subset_df = df.sample(fraction=sample_fraction, seed=42)
        subset_size = len(subset_df)

        # Save the subset
        subset_df.write_parquet(output_file)

        print(f"  Original size: {original_size:,} rows")
        print(
            f"  Subset size: {subset_size:,} rows ({subset_size / original_size:.1%})"
        )
        print(f"  Saved to: {output_file}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Create subset datasets for faster experimentation"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/processed/sprot_pre2024/sets"),
        help="Input directory containing the original datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/sprot_pre2024/sets_subset"),
        help="Output directory for subset datasets",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=0.1,
        help="Fraction of data to sample (default: 0.1 for 10%)",
    )

    args = parser.parse_args()

    print("Creating subset datasets...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample fraction: {args.sample_fraction} ({args.sample_fraction:.1%})")
    print("-" * 50)

    create_subset_datasets(args.input_dir, args.output_dir, args.sample_fraction)

    print("Subset creation completed!")


if __name__ == "__main__":
    main()
