#!/usr/bin/env python3

import argparse
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np


def load_data_file(file_path):
    """Load a data file (CSV, TSV, or Parquet) based on file extension"""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # Try to load the file based on extension
    try:
        if suffix == ".parquet":
            df = pl.read_parquet(file_path)
        elif suffix == ".tsv":
            df = pl.read_csv(file_path, separator="\t")
        elif suffix == ".csv":
            df = pl.read_csv(file_path, separator=",")
        else:
            print(f"Unsupported file format: {suffix}")
            return None

        print(f"Loaded {df.height} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compare_and_plot(df1, df2, output_dir, columns_to_compare):
    """Compare the two datasets and create plots for multiple columns"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate that required columns exist
    required_cols = ["query", "target"] + columns_to_compare
    for df, label in [(df1, "Dataset 1"), (df2, "Dataset 2")]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{label} is missing required columns: {missing_cols}")

    # Create normalized pair keys efficiently using Polars operations
    df1 = df1.with_columns(
        pl.concat_list([pl.col("query"), pl.col("target")])
        .list.sort()
        .list.join("|")
        .alias("pair_key")
    )
    df2 = df2.with_columns(
        pl.concat_list([pl.col("query"), pl.col("target")])
        .list.sort()
        .list.join("|")
        .alias("pair_key")
    )

    # Remove duplicates
    df1_dedup = df1.unique(subset=["pair_key"])
    df2_dedup = df2.unique(subset=["pair_key"])

    # Get sets of pairs for Venn diagram
    pairs1 = set(df1_dedup.get_column("pair_key").to_list())
    pairs2 = set(df2_dedup.get_column("pair_key").to_list())

    # Create Venn diagram of pairs (only once)
    plt.figure(figsize=(10, 10))
    venn2([pairs1, pairs2], set_labels=("Dataset 1", "Dataset 2"))
    plt.title("Comparison of Found Pairs Between Datasets")
    plt.savefig(output_dir / "pairs_venn.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Find common pairs efficiently using set intersection
    common_pairs = pairs1.intersection(pairs2)

    # Save pair statistics to file
    with open(output_dir / "comparison_stats.txt", "w") as f:
        f.write("Comparison Statistics:\n")
        f.write(f"Pairs in Dataset 1: {len(pairs1)}\n")
        f.write(f"Pairs in Dataset 2: {len(pairs2)}\n")
        f.write(f"Common pairs: {len(common_pairs)}\n")
        f.write(f"Pairs unique to Dataset 1: {len(pairs1 - pairs2)}\n")
        f.write(f"Pairs unique to Dataset 2: {len(pairs2 - pairs1)}\n\n")

        # Process each column
        for column_to_compare in columns_to_compare:
            f.write(f"=== {column_to_compare} ===\n")

            if len(common_pairs) > 0:
                # Get values for common pairs using Polars filtering
                df1_common = df1_dedup.filter(pl.col("pair_key").is_in(common_pairs))
                df2_common = df2_dedup.filter(pl.col("pair_key").is_in(common_pairs))

                # Join on pair_key to align values
                joined_df = df1_common.join(
                    df2_common.select(["pair_key", column_to_compare]),
                    on="pair_key",
                    how="inner",
                    suffix="_2",
                )

                # Get values for comparison
                values1 = joined_df.get_column(column_to_compare).to_numpy()
                values2 = joined_df.get_column(f"{column_to_compare}_2").to_numpy()

                # Ensure all values are numeric
                values1 = pl.Series(values1).cast(pl.Float64, strict=False).to_numpy()
                values2 = pl.Series(values2).cast(pl.Float64, strict=False).to_numpy()

                # Remove any NaN values
                valid_mask = ~(np.isnan(values1) | np.isnan(values2))
                values1 = values1[valid_mask]
                values2 = values2[valid_mask]

                if len(values1) > 0:
                    # Create hexbin plot
                    plt.figure(figsize=(10, 10))
                    hb = plt.hexbin(
                        values1, values2, gridsize=50, cmap="viridis", mincnt=1
                    )
                    plt.colorbar(hb, label="Count")
                    plt.xlabel(f"Dataset 1 {column_to_compare}")
                    plt.ylabel(f"Dataset 2 {column_to_compare}")
                    plt.title(f"Comparison of {column_to_compare} Values")

                    # Add diagonal line
                    min_val = min(values1.min(), values2.min())
                    max_val = max(values1.max(), values2.max())
                    plt.plot(
                        [min_val, max_val],
                        [min_val, max_val],
                        "r--",
                        label="Perfect Agreement",
                    )

                    plt.legend()
                    plt.savefig(
                        output_dir / f"{column_to_compare}_comparison_hexbin.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

                    # Calculate correlation for common pairs
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    f.write(f"Correlation coefficient: {correlation:.4f}\n")
                    f.write(f"Valid numeric pairs: {len(values1)}\n")
                else:
                    f.write("No valid numeric pairs found\n")
            else:
                f.write("No common pairs found\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two datasets with sequence similarity results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare fident values between two files
  python compare_mmseqs_runs.py file1.csv file2.tsv

  # Compare multiple columns
  python compare_mmseqs_runs.py file1.csv file2.tsv -c fident hfsp alntmscore

  # Use custom output directory
  python compare_mmseqs_runs.py file1.parquet file2.tsv -o custom_output/
        """,
    )

    parser.add_argument("file1", help="First dataset file (CSV/TSV/Parquet)")
    parser.add_argument("file2", help="Second dataset file (CSV/TSV/Parquet)")
    parser.add_argument(
        "-c",
        "--column",
        nargs="+",
        default=["fident"],
        help="Columns to compare (default: fident)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="out/tmp",
        help="Output directory for results (default: out/tmp)",
    )

    args = parser.parse_args()

    # Load datasets
    df1 = load_data_file(args.file1)
    if df1 is None:
        return 1

    df2 = load_data_file(args.file2)
    if df2 is None:
        return 1

    # Perform comparison
    try:
        print(f"Comparing {', '.join(args.column)} between datasets...")
        compare_and_plot(df1, df2, args.output, args.column)
        print(f"Results saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error during comparison: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
