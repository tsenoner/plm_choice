#!/usr/bin/env python3
"""
Analyze correlation between PLM performance and distance distribution quartiles.

This script explores whether there's a relationship between model performance
(Spearman correlation) and the characteristics of their distance distributions
(Q25, median, Q75).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import argparse


def analyze_performance_quartile_correlation(
    ranking_csv: Path, quartile_csv: Path, output_dir: Path = None
):
    """
    Analyze and visualize correlation between performance and quartiles.

    Args:
        ranking_csv: Path to PLM ranking CSV
        quartile_csv: Path to ridge plot statistics CSV
        output_dir: Output directory for plots
    """
    # Load data
    print(f"Loading ranking data from: {ranking_csv}")
    ranking_df = pd.read_csv(ranking_csv)

    print(f"Loading quartile data from: {quartile_csv}")
    quartile_df = pd.read_csv(quartile_csv)

    # Create mapping from display names to embedding names
    display_to_embedding = {
        "Ankh Base": "ankh_base",
        "Ankh Large": "ankh_large",
        "CLEAN": "clean",
        "ESM1b": "esm1b",
        "ESM2-8M": "esm2_8m",
        "ESM2-35M": "esm2_35m",
        "ESM2-150M": "esm2_150m",
        "ESM2-650M": "esm2_650m",
        "ESM2-3B": "esm2_3b",
        "ESM3": "esm3_open",
        "ESM C-300M": "esmc_300m",
        "ESM C-600M": "esmc_600m",
        "ProstT5": "prostt5",
        "ProtT5": "prott5",
        "ProtTucker": "prottucker",
        "Random": "random_1024",
    }

    # Convert display names to embedding names in quartile_df
    quartile_df["embedding_name"] = quartile_df["plm_name"].map(display_to_embedding)

    # Merge on PLM name
    merged_df = ranking_df.merge(
        quartile_df, left_on="Embedding", right_on="embedding_name", how="inner"
    )

    print(f"\nMerged {len(merged_df)} PLMs with both performance and quartile data")

    # Calculate IQR (Interquartile Range) and distribution width
    merged_df["IQR"] = merged_df["q75"] - merged_df["q25"]
    merged_df["Range"] = merged_df["q75"] - merged_df["q25"]

    # Calculate average Spearman performance and variance across tasks
    merged_df["Avg_Spearman"] = merged_df[
        ["Abs_Spearman_fident", "Abs_Spearman_hfsp", "Abs_Spearman_alntmscore"]
    ].mean(axis=1)

    merged_df["Spearman_Std"] = merged_df[
        ["Abs_Spearman_fident", "Abs_Spearman_hfsp", "Abs_Spearman_alntmscore"]
    ].std(axis=1)

    merged_df["Spearman_Range"] = merged_df[
        ["Abs_Spearman_fident", "Abs_Spearman_hfsp", "Abs_Spearman_alntmscore"]
    ].max(axis=1) - merged_df[
        ["Abs_Spearman_fident", "Abs_Spearman_hfsp", "Abs_Spearman_alntmscore"]
    ].min(axis=1)

    # Compute correlations
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    metrics = {
        "Average_Rank": "Average Rank (lower is better)",
        "Abs_Spearman_fident": "Spearman - PIDE",
        "Abs_Spearman_hfsp": "Spearman - HFSP",
        "Abs_Spearman_alntmscore": "Spearman - TM-score",
    }

    quartile_metrics = {
        "q25": "Q25",
        "median": "Median",
        "q75": "Q75",
        "IQR": "IQR (Q75-Q25)",
    }

    # Create correlation matrix
    correlation_results = []

    for perf_key, perf_label in metrics.items():
        print(f"\n{perf_label}:")
        for quart_key, quart_label in quartile_metrics.items():
            # Calculate Pearson and Spearman correlations
            pearson_r, pearson_p = stats.pearsonr(
                merged_df[perf_key], merged_df[quart_key]
            )
            spearman_r, spearman_p = stats.spearmanr(
                merged_df[perf_key], merged_df[quart_key]
            )

            # For Average_Rank, negative correlation means better rank = lower quartile
            # For Spearman scores, positive correlation means higher score = higher quartile
            # We want to report in intuitive terms

            print(
                f"  vs {quart_label:15s}: Pearson r={pearson_r:+.3f} (p={pearson_p:.4f}), "
                f"Spearman ρ={spearman_r:+.3f} (p={spearman_p:.4f})"
            )

            correlation_results.append(
                {
                    "Performance_Metric": perf_label,
                    "Quartile_Metric": quart_label,
                    "Pearson_r": pearson_r,
                    "Pearson_p": pearson_p,
                    "Spearman_rho": spearman_r,
                    "Spearman_p": spearman_p,
                }
            )

    # Save correlation results
    corr_df = pd.DataFrame(correlation_results)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        corr_path = output_dir / "performance_quartile_correlations.csv"
        corr_df.to_csv(corr_path, index=False)
        print(f"\nCorrelation results saved to: {corr_path}")

    # Create visualizations
    if output_dir:
        create_visualizations(merged_df, metrics, quartile_metrics, output_dir)

    # Summary insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    # Find strongest correlations
    significant = corr_df[corr_df["Pearson_p"] < 0.05].sort_values(
        "Pearson_r", key=abs, ascending=False
    )

    if len(significant) > 0:
        print("\nStrongest significant correlations (p < 0.05):")
        for _, row in significant.head(5).iterrows():
            print(
                f"  {row['Performance_Metric']:25s} vs {row['Quartile_Metric']:15s}: "
                f"r={row['Pearson_r']:+.3f}"
            )

    # Summary statistics
    print(f"\nBest performing models (Rank 1-3):")
    top3 = merged_df.nsmallest(3, "Average_Rank")
    for _, row in top3.iterrows():
        print(
            f"  {row['Embedding']:15s}: median={row['median']:.3f}, IQR={row['IQR']:.3f}"
        )

    print(f"\nWorst performing models (Bottom 3, excluding random):")
    bottom3 = merged_df[merged_df["Embedding"] != "random_1024"].nlargest(
        3, "Average_Rank"
    )
    for _, row in bottom3.iterrows():
        print(
            f"  {row['Embedding']:15s}: median={row['median']:.3f}, IQR={row['IQR']:.3f}"
        )

    return merged_df, corr_df


def create_visualizations(df, metrics, quartile_metrics, output_dir):
    """Create scatter plots and heatmaps of correlations."""

    # Set style
    sns.set_theme(style="whitegrid")

    # Display name mapping
    display_name_map = {
        "ankh_base": "Ankh Base",
        "ankh_large": "Ankh Large",
        "clean": "CLEAN",
        "esm1b": "ESM1b",
        "esm2_150m": "ESM2-150M",
        "esm2_3b": "ESM2-3B",
        "esm2_650m": "ESM2-650M",
        "esm2_35m": "ESM2-35M",
        "esm2_8m": "ESM2-8M",
        "esm3_open": "ESM3",
        "esmc_300m": "ESMC-300M",
        "esmc_600m": "ESMC-600M",
        "prostt5": "ProstT5",
        "prott5": "ProtT5",
        "prottucker": "ProtTucker",
        "random_1024": "Random",
    }

    # --- GRID PLOT FOR AVERAGE RANK ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    quartile_cols = ["q25", "median", "q75", "IQR"]
    titles = ["Q25", "Median", "Q75", "IQR"]

    for idx, (col, title) in enumerate(zip(quartile_cols, titles)):
        ax = axes[idx]

        ax.scatter(
            df[col],
            df["Average_Rank"],
            s=200,
            alpha=0.7,
            edgecolors="black",
            linewidth=2,
            zorder=3,
        )

        # Add model labels
        for _, row in df.iterrows():
            display_name = display_name_map.get(row["Embedding"], row["Embedding"])
            ax.annotate(
                display_name,
                (row[col], row["Average_Rank"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
                fontweight="bold",
            )

        # Add trend line
        z = np.polyfit(df[col], df["Average_Rank"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.6, linewidth=2)

        # Correlation - moved to top right
        r, p_val = stats.pearsonr(df[col], df["Average_Rank"])
        ax.text(
            0.97,
            0.97,
            f"r = {r:.3f}\np = {p_val:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round", facecolor="wheat", edgecolor="black", alpha=0.9
            ),
            fontweight="bold",
        )

        ax.set_xlabel(f"{title} Normalized Distance", fontsize=13, fontweight="bold")
        ax.set_ylabel("Average Rank (lower is better)", fontsize=13, fontweight="bold")
        ax.set_title(f"Performance vs {title}", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Average Rank vs Distribution Quartiles",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "performance_vs_quartiles_grid.png", dpi=300, bbox_inches="tight"
    )
    print(f"Saved: {output_dir / 'performance_vs_quartiles_grid.png'}")
    plt.close()

    # --- GRID PLOT FOR AVERAGE SPEARMAN ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(zip(quartile_cols, titles)):
        ax = axes[idx]

        ax.scatter(
            df[col],
            df["Avg_Spearman"],
            s=200,
            alpha=0.7,
            edgecolors="black",
            linewidth=2,
            zorder=3,
            c="#E63946",
        )

        # Add model labels
        for _, row in df.iterrows():
            display_name = display_name_map.get(row["Embedding"], row["Embedding"])
            ax.annotate(
                display_name,
                (row[col], row["Avg_Spearman"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
                fontweight="bold",
            )

        # Add trend line
        z = np.polyfit(df[col], df["Avg_Spearman"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.6, linewidth=2)

        # Correlation - moved to top right
        r, p_val = stats.pearsonr(df[col], df["Avg_Spearman"])
        ax.text(
            0.97,
            0.97,
            f"r = {r:.3f}\np = {p_val:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round", facecolor="wheat", edgecolor="black", alpha=0.9
            ),
            fontweight="bold",
        )

        ax.set_xlabel(f"{title} Normalized Distance", fontsize=13, fontweight="bold")
        ax.set_ylabel(
            "Average Spearman (higher is better)", fontsize=13, fontweight="bold"
        )
        ax.set_title(f"Avg Performance vs {title}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Average Spearman vs Distribution Quartiles",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "avg_spearman_vs_quartiles_grid.png", dpi=300, bbox_inches="tight"
    )
    print(f"Saved: {output_dir / 'avg_spearman_vs_quartiles_grid.png'}")
    plt.close()

    # --- HEATMAP FOR AVERAGE RANK ---
    perf_cols = [
        "Average_Rank",
        "Abs_Spearman_fident",
        "Abs_Spearman_hfsp",
        "Abs_Spearman_alntmscore",
    ]
    quart_cols = ["q25", "median", "q75", "IQR"]

    corr_matrix = np.zeros((len(perf_cols), len(quart_cols)))

    for i, perf in enumerate(perf_cols):
        for j, quart in enumerate(quart_cols):
            corr_matrix[i, j], _ = stats.pearsonr(df[perf], df[quart])

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=["Q25", "Median", "Q75", "IQR"],
        yticklabels=["Avg Rank", "Spearman-PIDE", "Spearman-HFSP", "Spearman-TM"],
        cbar_kws={"label": "Pearson Correlation"},
        linewidths=0.5,
        ax=ax,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )

    ax.set_title(
        "Performance vs Quartile Correlations", fontsize=15, fontweight="bold", pad=20
    )
    ax.set_xlabel("Distribution Quartiles", fontsize=13, fontweight="bold")
    ax.set_ylabel("Performance Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'correlation_heatmap.png'}")
    plt.close()

    # --- HEATMAP FOR AVERAGE SPEARMAN ---
    perf_cols_spearman = [
        "Avg_Spearman",
        "Abs_Spearman_fident",
        "Abs_Spearman_hfsp",
        "Abs_Spearman_alntmscore",
    ]

    corr_matrix_spearman = np.zeros((len(perf_cols_spearman), len(quart_cols)))

    for i, perf in enumerate(perf_cols_spearman):
        for j, quart in enumerate(quart_cols):
            corr_matrix_spearman[i, j], _ = stats.pearsonr(df[perf], df[quart])

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix_spearman,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=["Q25", "Median", "Q75", "IQR"],
        yticklabels=["Avg Spearman", "Spearman-PIDE", "Spearman-HFSP", "Spearman-TM"],
        cbar_kws={"label": "Pearson Correlation"},
        linewidths=0.5,
        ax=ax,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )

    ax.set_title(
        "Average Spearman vs Quartile Correlations",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Distribution Quartiles", fontsize=13, fontweight="bold")
    ax.set_ylabel("Performance Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        output_dir / "avg_spearman_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {output_dir / 'avg_spearman_correlation_heatmap.png'}")
    plt.close()


def create_iqr_performance_plot(df, output_dir):
    """Create a dedicated plot for IQR vs Average Rank with model labels."""

    # Set style
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Scatter plot
    ax.scatter(
        df["IQR"],
        df["Average_Rank"],
        s=300,
        alpha=0.7,
        c="#2E86AB",
        edgecolors="black",
        linewidth=2,
        zorder=3,
    )

    # Add model names next to each point
    for _, row in df.iterrows():
        # Convert embedding names to display names
        display_name_map = {
            "esmc_600m": "ESM C-600M",
            "esmc_300m": "ESM C-300M",
            "prott5": "ProtT5",
            "esm3_open": "ESM3",
            "ankh_large": "Ankh Large",
            "esm1b": "ESM1b",
            "prostt5": "ProstT5",
            "esm2_650m": "ESM2-650M",
            "esm2_150m": "ESM2-150M",
            "esm2_3b": "ESM2-3B",
            "prottucker": "ProtTucker",
            "ankh_base": "Ankh Base",
            "clean": "CLEAN",
            "esm2_35m": "ESM2-35M",
            "esm2_8m": "ESM2-8M",
            "random_1024": "Random",
        }

        display_name = display_name_map.get(row["Embedding"], row["Embedding"])

        ax.annotate(
            display_name,
            (row["IQR"], row["Average_Rank"]),
            xytext=(8, 0),  # Offset to the right
            textcoords="offset points",
            fontsize=11,
            ha="left",
            va="center",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
            ),
        )

    # Calculate correlation
    r, p_val = stats.pearsonr(df["IQR"], df["Average_Rank"])

    # Add correlation text in top right
    ax.text(
        0.97,
        0.97,
        f"Pearson r = {r:+.3f}\np-value = {p_val:.4f}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="wheat",
            edgecolor="black",
            alpha=0.9,
            linewidth=2,
        ),
        fontweight="bold",
    )

    # Styling
    ax.set_xlabel("IQR (Q75 - Q25)", fontsize=16, fontweight="bold")
    ax.set_ylabel(
        "Average Performance Rank (lower is better)", fontsize=16, fontweight="bold"
    )
    ax.set_title(
        "Distribution Spread vs Model Performance",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.invert_yaxis()  # Lower rank at top

    # Add subtle background shading for regions
    ax.axhspan(1, 5, alpha=0.05, color="green", zorder=0)  # Best performers
    ax.axhspan(11, 16, alpha=0.05, color="red", zorder=0)  # Worst performers

    plt.tight_layout()
    plt.savefig(output_dir / "iqr_vs_performance.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'iqr_vs_performance.png'}")
    plt.close()


def create_avg_spearman_analysis(df, output_dir):
    """Create plots analyzing average Spearman vs quartiles and performance variance."""

    # Set style
    sns.set_theme(style="whitegrid")

    # Create a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # Plot 1: Average Spearman vs Median
    ax = axes[0]
    ax.scatter(
        df["median"],
        df["Avg_Spearman"],
        s=250,
        alpha=0.7,
        c="#E63946",
        edgecolors="black",
        linewidth=2,
        zorder=3,
    )

    r, p_val = stats.pearsonr(df["median"], df["Avg_Spearman"])
    ax.text(
        0.97,
        0.97,
        f"r = {r:.3f}\np = {p_val:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="wheat", edgecolor="black", alpha=0.9
        ),
    )

    ax.set_xlabel("Median Normalized Distance", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Spearman (across tasks)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Average Performance vs Median Distance", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # Plot 2: Average Spearman vs IQR
    ax = axes[1]
    ax.scatter(
        df["IQR"],
        df["Avg_Spearman"],
        s=250,
        alpha=0.7,
        c="#F77F00",
        edgecolors="black",
        linewidth=2,
        zorder=3,
    )

    r, p_val = stats.pearsonr(df["IQR"], df["Avg_Spearman"])
    ax.text(
        0.97,
        0.97,
        f"r = {r:.3f}\np = {p_val:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="wheat", edgecolor="black", alpha=0.9
        ),
    )

    ax.set_xlabel("IQR (Q75 - Q25)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Spearman (across tasks)", fontsize=13, fontweight="bold")
    ax.set_title("Average Performance vs IQR", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Plot 3: Performance Std vs Median
    ax = axes[2]
    ax.scatter(
        df["median"],
        df["Spearman_Std"],
        s=250,
        alpha=0.7,
        c="#06AED5",
        edgecolors="black",
        linewidth=2,
        zorder=3,
    )

    r, p_val = stats.pearsonr(df["median"], df["Spearman_Std"])
    ax.text(
        0.97,
        0.97,
        f"r = {r:.3f}\np = {p_val:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="wheat", edgecolor="black", alpha=0.9
        ),
    )

    ax.set_xlabel("Median Normalized Distance", fontsize=13, fontweight="bold")
    ax.set_ylabel("Std of Spearman (across tasks)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Performance Variability vs Median Distance", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # Plot 4: Performance Std vs IQR
    ax = axes[3]
    ax.scatter(
        df["IQR"],
        df["Spearman_Std"],
        s=250,
        alpha=0.7,
        c="#9D4EDD",
        edgecolors="black",
        linewidth=2,
        zorder=3,
    )

    r, p_val = stats.pearsonr(df["IQR"], df["Spearman_Std"])
    ax.text(
        0.97,
        0.97,
        f"r = {r:.3f}\np = {p_val:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="wheat", edgecolor="black", alpha=0.9
        ),
    )

    ax.set_xlabel("IQR (Q75 - Q25)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Std of Spearman (across tasks)", fontsize=13, fontweight="bold")
    ax.set_title("Performance Variability vs IQR", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Average Performance Metrics vs Distribution Characteristics",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "avg_spearman_vs_quartiles.png", dpi=300, bbox_inches="tight"
    )
    print(f"Saved: {output_dir / 'avg_spearman_vs_quartiles.png'}")
    plt.close()

    # Print correlation summary
    print("\n" + "=" * 80)
    print("AVERAGE SPEARMAN CORRELATIONS")
    print("=" * 80)

    print("\nAverage Spearman vs Quartiles:")
    for quart in ["q25", "median", "q75", "IQR"]:
        r, p = stats.pearsonr(df[quart], df["Avg_Spearman"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  vs {quart:8s}: r = {r:+.3f}, p = {p:.4f} {sig}")

    print("\nPerformance Std (variability) vs Quartiles:")
    for quart in ["q25", "median", "q75", "IQR"]:
        r, p = stats.pearsonr(df[quart], df["Spearman_Std"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  vs {quart:8s}: r = {r:+.3f}, p = {p:.4f} {sig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze correlation between PLM performance and quartiles"
    )
    parser.add_argument(
        "--ranking_csv",
        type=Path,
        default=Path("out/sprot_pre2024_subset/plm_ranking_by_spearman.csv"),
        help="Path to PLM ranking CSV",
    )
    parser.add_argument(
        "--quartile_csv",
        type=Path,
        default=Path(
            "out/pairwise/sprot_pre2024_train/distribution_comparison_normalized_ridge_statistics.csv"
        ),
        help="Path to ridge plot statistics CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("out/analysis/performance_vs_quartiles"),
        help="Output directory for results",
    )

    args = parser.parse_args()

    analyze_performance_quartile_correlation(
        args.ranking_csv, args.quartile_csv, args.output_dir
    )
