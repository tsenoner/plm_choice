#!/usr/bin/env python3
"""
Rank PLMs by their performance across parameters based on Spearman correlation.

This script:
1. Loads parsed metrics data
2. Filters for FNN models
3. Ranks PLMs by Spearman performance for each parameter
4. Averages rankings across parameters
5. Saves the ranking table
"""

import pandas as pd
import argparse
from pathlib import Path


def rank_plms_by_performance(csv_path: Path, output_path: Path = None):
    """
    Rank PLMs based on FNN model Spearman performance across parameters.

    Args:
        csv_path: Path to the parsed_metrics_all.csv file
        output_path: Optional output path. If None, saves to parent dir of csv_path
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Filter for FNN models only
    fnn_df = df[df["Model Type"] == "fnn"].copy()

    # Parameters to analyze
    parameters = ["fident", "hfsp", "alntmscore"]

    # Store rankings for each parameter
    rankings = {}

    for param in parameters:
        # Filter for this parameter
        param_df = fnn_df[fnn_df["Parameter"] == param].copy()

        # Sort by absolute Spearman (higher is better)
        param_df["Abs_Spearman"] = param_df["Spearman"].abs()
        param_df = param_df.sort_values("Abs_Spearman", ascending=False)

        # Create rank (1 = best)
        param_df["Rank"] = range(1, len(param_df) + 1)

        # Store ranking for this parameter (only keep absolute Spearman)
        rankings[param] = param_df[["Embedding", "Abs_Spearman", "Rank"]].copy()
        rankings[param].columns = [
            "Embedding",
            f"Abs_Spearman_{param}",
            f"Rank_{param}",
        ]

    # Merge rankings from all parameters
    result = rankings["fident"]
    for param in ["hfsp", "alntmscore"]:
        result = result.merge(rankings[param], on="Embedding", how="outer")

    # Calculate average rank
    rank_cols = [f"Rank_{param}" for param in parameters]
    result["Average_Rank"] = result[rank_cols].mean(axis=1)

    # Sort by average rank (lower is better)
    result = result.sort_values("Average_Rank")

    # Add final ranking
    result["Final_Rank"] = range(1, len(result) + 1)

    # Reorder columns for clarity (only include absolute Spearman)
    cols = ["Final_Rank", "Embedding", "Average_Rank"]
    for param in parameters:
        cols.extend([f"Rank_{param}", f"Abs_Spearman_{param}"])

    result = result[cols]

    # Round all float columns to 2 decimal places
    float_cols = result.select_dtypes(include=["float64"]).columns
    result[float_cols] = result[float_cols].round(2)

    # Determine output path
    if output_path is None:
        output_path = csv_path.parent / "plm_ranking_by_spearman.csv"

    # Save to CSV
    result.to_csv(output_path, index=False)
    print(f"Ranking saved to: {output_path}")
    print(f"\nTop 5 PLMs:")
    print(
        result[["Final_Rank", "Embedding", "Average_Rank"]]
        .head(5)
        .to_string(index=False)
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank PLMs by FNN model Spearman performance across parameters"
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("out/sprot_pre2024_subset/parsed_metrics_all.csv"),
        help="Path to parsed_metrics_all.csv file",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output path for ranking CSV (default: same dir as input)",
    )

    args = parser.parse_args()
    rank_plms_by_performance(args.csv_path, args.output_path)
