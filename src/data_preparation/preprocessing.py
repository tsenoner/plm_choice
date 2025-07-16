"""
This script preprocesses the CSV files by setting values below thresholds to NaN.
It applies the thresholds to the specified columns and saves the processed files to the output directory.

Usage:
python preprocess_fident.py --input_dir data/raw/training --output_dir data/processed/training --files train.csv val.csv test.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from typing import Dict


def apply_thresholds(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Applies lower-bound thresholds to specified columns, setting values below to NaN."""
    df_processed = df.copy()
    for column, threshold in thresholds.items():
        if column not in df_processed.columns:
            print(f"Warning: Column '{column}' not found, skipping thresholding.")
            continue

        try:
            # Ensure column is numeric, coerce errors to NaN
            numeric_col = pd.to_numeric(df_processed[column], errors="coerce")
            original_nan_count = numeric_col.isna().sum()

            # Apply the threshold condition (only to non-NaN values)
            condition = numeric_col < threshold
            rows_to_change = condition.sum()  # This correctly ignores NaNs

            if rows_to_change > 0:
                df_processed.loc[condition, column] = np.nan
                print(
                    f"  Set {rows_to_change} values in '{column}' to NaN (threshold < {threshold})."
                )
            else:
                print(
                    f"  No values found below threshold {threshold} in column '{column}'."
                )

            # Check if original column had non-numeric values coerced to NaN
            if numeric_col.isna().sum() > original_nan_count:
                print(
                    f"  Note: Some non-numeric values in '{column}' were coerced to NaN during processing."
                )
                # Update the main dataframe if coercion happened
                df_processed[column] = numeric_col
                df_processed.loc[condition, column] = (
                    np.nan
                )  # Re-apply condition after coercion update

        except Exception as e:
            print(
                f"Error applying threshold to column '{column}': {e}", file=sys.stderr
            )

    return df_processed


def process_directory(
    input_dir: Path, output_dir: Path, thresholds: Dict[str, float], file_list: list
):
    """Processes specified CSV files in an input directory and saves to output directory."""
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Applying thresholds: {thresholds}")
    print("-" * 30)

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in file_list:
        input_path = input_dir / filename
        output_path = output_dir / filename

        if not input_path.is_file():
            print(f"Warning: Input file not found, skipping: {input_path}")
            print("-" * 30)
            continue

        try:
            print(f"Processing {input_path}...")
            df = pd.read_csv(input_path)
            df_processed = apply_thresholds(df, thresholds)

            df_processed.to_csv(output_path, index=False)
            print(f"Saved processed file to {output_path}")

        except Exception as e:
            print(f"Error processing file {input_path}: {e}", file=sys.stderr)

        print("-" * 30)

    print("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess CSV files by setting values below thresholds to NaN."
    )
    # Use arguments for directories for more flexibility
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing the raw CSV files (e.g., data/raw/training)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where processed CSV files will be saved (e.g., data/processed/training)",
    )
    # Specify which files within the directories to process
    parser.add_argument(
        "--files",
        nargs="+",
        default=["train.csv", "val.csv", "test.csv"],
        help="List of CSV filenames to process (default: train.csv val.csv test.csv)",
    )

    args = parser.parse_args()

    # Define the thresholds
    processing_thresholds = {
        "fident": 0.3,
        "alntmscore": 0.4,
        "hfsp": 0.0,
    }

    process_directory(
        args.input_dir, args.output_dir, processing_thresholds, args.files
    )
