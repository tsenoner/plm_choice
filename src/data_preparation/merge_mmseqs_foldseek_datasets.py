#!/usr/bin/env python3
"""
Script to merge MMseqs2 and Foldseek results with HFSP score computation.

This script:
1. Loads MMseqs2 results and fixes formatting issues
2. Computes HFSP scores using the provided formula
3. Loads Foldseek results (keeping only selected columns)
4. Merges the datasets on query-target pairs
5. Filters to keep only: query, target, fident, hfsp, alntmscore
6. Saves the filtered merged result
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def calc_hfsp(seq_id, ungapped_len):
    """HFSP score from seq id (0-1) and ungapped align len; returns NaN if <0."""
    seq_id = seq_id * 100
    hfsp = np.where(
        ungapped_len <= 11,
        seq_id - 100,
        np.where(
            ungapped_len <= 450,
            seq_id - 770 * ungapped_len ** (-0.33 * (1 + np.exp(ungapped_len / 1000))),
            seq_id - 28.4,
        ),
    )
    return np.where(hfsp < 0, np.nan, hfsp)


def compute_hfsp(df):
    """Compute HFSP using 'fident', 'nident', 'mismatch' columns."""
    df["ungapped_len"] = df["nident"] + df["mismatch"]
    df["hfsp"] = calc_hfsp(df["fident"].values, df["ungapped_len"].values)
    return df


def load_mmseqs_results(mmseqs_file):
    """Load MMseqs2 results, selecting only needed columns."""
    needed = ["query", "target", "fident", "nident", "mismatch"]
    df = pd.read_csv(mmseqs_file, sep="\t", usecols=needed)
    return df


def load_foldseek_results(foldseek_file, keep_columns=None):
    """Load Foldseek results, selecting only needed columns."""
    if keep_columns is None:
        keep_columns = ["query", "target", "lddt", "rmsd", "alntmscore"]
    df = pd.read_csv(foldseek_file, sep="\t", usecols=keep_columns)
    return df


def merge_datasets(mmseqs_df, foldseek_df):
    """Merge MMseqs2 and Foldseek datasets on query-target pairs."""
    merged_df = pd.merge(mmseqs_df, foldseek_df, on=["query", "target"], how="outer")
    return merged_df


def setup_argument_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Merge MMseqs2 and Foldseek results with HFSP computation"
    )
    parser.add_argument(
        "--mmseqs_file",
        type=Path,
        required=True,
        help="Path to MMseqs2 results TSV file",
    )
    parser.add_argument(
        "--foldseek_file",
        type=Path,
        required=True,
        help="Path to Foldseek results TSV file",
    )
    parser.add_argument(
        "--output_file", type=Path, required=True, help="Path to output merged TSV file"
    )

    args = parser.parse_args()

    # Validate input files
    if not args.mmseqs_file.exists():
        print(f"Error: MMseqs2 file not found: {args.mmseqs_file}")
        sys.exit(1)

    if not args.foldseek_file.exists():
        print(f"Error: Foldseek file not found: {args.foldseek_file}")
        sys.exit(1)

    # Create output directory
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    return args


def main():
    args = setup_argument_parser()

    try:
        # Load and filter MMseqs2 results
        mmseqs_df = load_mmseqs_results(args.mmseqs_file)
        mmseqs_df = compute_hfsp(mmseqs_df)
        initial_mmseqs = len(mmseqs_df)
        mmseqs_df = mmseqs_df[mmseqs_df["fident"] >= 0.3]
        print(
            f"MMseqs2: {initial_mmseqs:,} → {len(mmseqs_df):,} ({len(mmseqs_df) / initial_mmseqs * 100:.1f}%) - fident ≥ 0.3"
        )

        # Load and filter Foldseek results
        foldseek_df = load_foldseek_results(args.foldseek_file)
        initial_foldseek = len(foldseek_df)
        foldseek_df = foldseek_df[foldseek_df["alntmscore"] >= 0.4]
        print(
            f"Foldseek: {initial_foldseek:,} → {len(foldseek_df):,} ({len(foldseek_df) / initial_foldseek * 100:.1f}%) - alntmscore ≥ 0.4"
        )

        # Merge and clean
        final_df = merge_datasets(mmseqs_df, foldseek_df)
        merged_count = len(final_df)
        final_df = final_df[final_df["query"] != final_df["target"]]
        final_count = len(final_df)
        print(
            f"Merged: {merged_count:,} → {final_count:,} (removed {merged_count - final_count:,} self-comparisons)"
        )

        # Save results
        final_columns = ["query", "target", "fident", "hfsp", "alntmscore"]
        final_df = final_df[final_columns]
        final_df.to_csv(args.output_file, index=False)
        print(f"Saved: {final_count:,} entries → {args.output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
