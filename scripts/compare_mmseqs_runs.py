#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np


def load_data_file(file_path, sep=None):
    """Load a data file (CSV or TSV) and auto-detect separator if not provided"""
    file_path = Path(file_path)

    # Auto-detect separator if not provided
    if sep is None:
        if file_path.suffix.lower() == ".tsv":
            sep = "\t"
        else:
            sep = ","

    # Try to load the file
    try:
        df = pd.read_csv(file_path, sep=sep, low_memory=False)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def create_pair_key(row):
    """Create a unique key for each query-target pair (order-independent)"""
    return tuple(sorted([row["query"], row["target"]]))


def compare_and_plot(
    df1, df2, output_dir, column_to_compare, label1="Dataset 1", label2="Dataset 2"
):
    """Compare the two datasets and create plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate that required columns exist
    required_cols = ["query", "target", column_to_compare]
    for df, label in [(df1, label1), (df2, label2)]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{label} is missing required columns: {missing_cols}")

    # Create normalized pair keys efficiently using vectorized operations
    df1 = df1.copy()
    df2 = df2.copy()

    # Create normalized pair keys (lexicographically sorted)
    df1["pair_key"] = df1[["query", "target"]].apply(
        lambda x: tuple(sorted([x["query"], x["target"]])), axis=1
    )
    df2["pair_key"] = df2[["query", "target"]].apply(
        lambda x: tuple(sorted([x["query"], x["target"]])), axis=1
    )

    # Remove duplicates and set pair_key as index for O(1) lookup instead of O(n) filtering
    print(f"Before deduplication: {len(df1)} vs {len(df2)} pairs")
    df1_indexed = df1.drop_duplicates("pair_key").set_index("pair_key")
    df2_indexed = df2.drop_duplicates("pair_key").set_index("pair_key")
    print(f"After deduplication: {len(df1_indexed)} vs {len(df2_indexed)} pairs")
    print(
        f"Duplicates removed: {len(df1) - len(df1_indexed)} vs {len(df2) - len(df2_indexed)}"
    )

    # Get sets of pairs for Venn diagram
    pairs1 = set(df1_indexed.index)
    pairs2 = set(df2_indexed.index)

    # Create Venn diagram of pairs
    plt.figure(figsize=(10, 10))
    venn2([pairs1, pairs2], set_labels=(label1, label2))
    plt.title("Comparison of Found Pairs Between Datasets")
    plt.savefig(output_dir / "pairs_venn.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Find common pairs efficiently using index intersection
    common_pairs = df1_indexed.index.intersection(df2_indexed.index)

    if len(common_pairs) > 0:
        # Get values for common pairs using vectorized indexing
        values1 = df1_indexed.loc[common_pairs, column_to_compare].values
        values2 = df2_indexed.loc[common_pairs, column_to_compare].values

        # Ensure all values are numeric
        values1 = pd.to_numeric(values1, errors="coerce")
        values2 = pd.to_numeric(values2, errors="coerce")

        # Remove any NaN values
        valid_mask = ~(np.isnan(values1) | np.isnan(values2))
        values1 = values1[valid_mask]
        values2 = values2[valid_mask]

        print(f"Valid numeric pairs for correlation: {len(values1)}")

        if len(values1) > 0:
            # Create hexbin plot
            plt.figure(figsize=(10, 10))
            # Set mincnt=1 to avoid plotting empty bins (purple areas)
            hb = plt.hexbin(values1, values2, gridsize=50, cmap="viridis", mincnt=1)
            plt.colorbar(hb, label="Count")
            plt.xlabel(f"{label1} {column_to_compare}")
            plt.ylabel(f"{label2} {column_to_compare}")
            plt.title(f"Comparison of {column_to_compare} Values")

            # Add diagonal line
            min_val = min(values1.min(), values2.min())
            max_val = max(values1.max(), values2.max())
            plt.plot(
                [min_val, max_val], [min_val, max_val], "r--", label="Perfect Agreement"
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
        else:
            correlation = np.nan
            print(
                f"Warning: No valid numeric pairs found for {column_to_compare} comparison"
            )
    else:
        correlation = np.nan
        print("Warning: No common pairs found for correlation calculation")

    # Print statistics
    print("\nComparison Statistics:")
    print(f"Pairs in {label1}: {len(pairs1)}")
    print(f"Pairs in {label2}: {len(pairs2)}")
    print(f"Common pairs: {len(common_pairs)}")
    print(f"Pairs unique to {label1}: {len(pairs1 - pairs2)}")
    print(f"Pairs unique to {label2}: {len(pairs2 - pairs1)}")
    print(f"Correlation coefficient for {column_to_compare}: {correlation:.4f}")

    # Save pair statistics to file
    with open(output_dir / "comparison_stats.txt", "w") as f:
        f.write("Comparison Statistics:\n")
        f.write(f"Pairs in {label1}: {len(pairs1)}\n")
        f.write(f"Pairs in {label2}: {len(pairs2)}\n")
        f.write(f"Common pairs: {len(common_pairs)}\n")
        f.write(f"Pairs unique to {label1}: {len(pairs1 - pairs2)}\n")
        f.write(f"Pairs unique to {label2}: {len(pairs2 - pairs1)}\n")
        f.write(f"Correlation coefficient for {column_to_compare}: {correlation:.4f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two datasets with sequence similarity results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare fident values between two files
  python compare_mmseqs_runs.py file1.csv file2.tsv -o results/

  # Compare different column (e.g., hfsp)
  python compare_mmseqs_runs.py file1.csv file2.tsv -c hfsp -o results/

  # Specify custom labels and separators
  python compare_mmseqs_runs.py file1.csv file2.tsv -c fident \\
    --labels "Original" "Updated" --sep1 "," --sep2 "\t" -o results/
        """,
    )

    parser.add_argument("file1", help="First dataset file (CSV/TSV)")
    parser.add_argument("file2", help="Second dataset file (CSV/TSV)")
    parser.add_argument(
        "-c", "--column", default="fident", help="Column to compare (default: fident)"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=["Dataset 1", "Dataset 2"],
        help='Labels for the two datasets (default: "Dataset 1" "Dataset 2")',
    )
    parser.add_argument(
        "--sep1", help="Separator for file1 (auto-detect if not provided)"
    )
    parser.add_argument(
        "--sep2", help="Separator for file2 (auto-detect if not provided)"
    )

    args = parser.parse_args()

    # Load datasets
    print(f"Loading {args.file1}...")
    df1 = load_data_file(args.file1, args.sep1)
    if df1 is None:
        return 1

    print(f"Loading {args.file2}...")
    df2 = load_data_file(args.file2, args.sep2)
    if df2 is None:
        return 1

    # Show available columns
    print(f"\nColumns in {args.file1}: {list(df1.columns)}")
    print(f"Columns in {args.file2}: {list(df2.columns)}")

    # Perform comparison
    try:
        print(f"\nComparing {args.column} between datasets...")
        compare_and_plot(
            df1, df2, args.output, args.column, args.labels[0], args.labels[1]
        )
        print(f"\nResults saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error during comparison: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
