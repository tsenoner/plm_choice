#!/usr/bin/env python3
# --- Ivan infrastructure (2026-03-19) ---
"""Visualization of retrieval/classification evaluation metrics.

Reads the classification evaluation results parquet produced by
``src/evaluation/classification_eval.py`` and creates publication-quality
figures comparing pLM embeddings across SCOP/ECOD hierarchy levels.

Figures produced:
    1. AUROC bar chart (grouped by hierarchy level)
    2. Recall-at-first-FP bar chart (grouped by hierarchy level)
    3. Combined AUROC heatmap (embeddings x levels)
    4. Summary scatter small-multiples (AUROC vs recall per level)

Usage:
    uv run python src/visualization/create_retrieval_plots.py \\
        --results_parquet out/classification_eval/classification_eval_results.parquet \\
        --output_dir out/classification_eval/figures \\
        --format png --dpi 300
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Sizes, families, colours and labels are shared with every other figure; a divergence
# here draws the same model in two colours across two panels of the same paper.
from shared.hierarchies import LEVEL_LABELS
from visualization.plm_constants import (
    EMBEDDING_COLOR_MAP,
    EMBEDDING_DISPLAY_NAMES,
    PLM_SIZES,
    human_readable_number,
)

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _sort_embeddings_by_size(embeddings: List[str]) -> List[str]:
    """Return embedding names sorted by model parameter count (ascending)."""
    return sorted(
        embeddings,
        key=lambda e: PLM_SIZES.get(e.lower(), -1),
    )


def _display_level(level: str) -> str:
    """Map a raw level column name to a human-readable label."""
    return LEVEL_LABELS.get(level, level)


def _get_bar_color(embedding: str) -> str:
    """Return the family color for an embedding name."""
    key = embedding.lower()
    return EMBEDDING_COLOR_MAP.get(key, "#808080")


def _embedding_tick_labels(embeddings: List[str]) -> List[str]:
    """Label each pLM the way the other figures do, annotated with its size."""
    labels = []
    for emb in embeddings:
        key = emb.lower()
        name = EMBEDDING_DISPLAY_NAMES.get(key, emb).replace("\n", " ")
        size = PLM_SIZES.get(key)
        labels.append(f"{name}\n({human_readable_number(size)})" if size is not None else name)
    return labels


# ---------------------------------------------------------------------------
#  Plots 1 & 2: grouped metric bar charts
# ---------------------------------------------------------------------------


def plot_metric_bars(
    df: pd.DataFrame,
    output_path: Path,
    fmt: str,
    dpi: int,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    filename_stem: str,
    legend_loc: str,
    baseline: Optional[float] = None,
) -> None:
    """Grouped bar chart: x = pLM (sorted by size), y = ``value_col``, hue = level.

    AUROC and recall@1FP are the same figure with a different value column, so they
    share one implementation — styling changes land on both panels at once.
    """
    levels = df["level"].unique().tolist()
    embeddings = _sort_embeddings_by_size(df["embedding"].unique().tolist())
    n_levels = len(levels)
    n_embeddings = len(embeddings)

    fig, ax = plt.subplots(figsize=(max(12, n_embeddings * 0.9), 7))

    bar_width = 0.8 / n_levels
    x = np.arange(n_embeddings)

    # Use a qualitative palette for levels
    level_colors = sns.color_palette("Set2", n_colors=n_levels)

    for i, level in enumerate(levels):
        level_data = df[df["level"] == level]
        # reindex fills a missing (embedding, level) combination with NaN, which is
        # exactly what the bar chart should leave blank. drop_duplicates keeps the
        # first row per embedding, matching the previous `.values[0]`.
        values = (
            level_data.drop_duplicates("embedding")
            .set_index("embedding")[value_col]
            .reindex(embeddings)
            .to_numpy()
        )

        offset = (i - n_levels / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=_display_level(level),
            color=level_colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

    if baseline is not None:
        ax.axhline(
            y=baseline, color="grey", linestyle="--", linewidth=1.0, alpha=0.7,
            label=f"Random ({baseline})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(_embedding_tick_labels(embeddings), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(title="Level", fontsize=10, title_fontsize=11, loc=legend_loc)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_file = output_path / f"{filename_stem}.{fmt}"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {ylabel} bar chart: {out_file}")


# ---------------------------------------------------------------------------
#  Plot 3: Combined AUROC heatmap
# ---------------------------------------------------------------------------


def plot_auroc_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Heatmap: rows = embeddings (sorted by size), columns = levels, values = AUROC."""
    embeddings = _sort_embeddings_by_size(df["embedding"].unique().tolist())
    levels = df["level"].unique().tolist()

    # Build matrix
    matrix = pd.DataFrame(index=embeddings, columns=levels, dtype=float)
    for _, row in df.iterrows():
        matrix.loc[row["embedding"], row["level"]] = row["auroc"]

    # Rename columns for display
    matrix.columns = [_display_level(c) for c in matrix.columns]

    fig, ax = plt.subplots(figsize=(max(6, len(levels) * 1.8), max(8, len(embeddings) * 0.5)))

    # Diverging colormap: 0.5 = red, 1.0 = green
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)

    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=".3f",
        cmap=cmap,
        center=0.75,
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "AUROC", "shrink": 0.8},
        ax=ax,
    )

    ax.set_title("AUROC: Embedding vs Hierarchy Level", fontsize=16, pad=12)
    ax.set_ylabel("Embedding (sorted by model size)", fontsize=12)
    ax.set_xlabel("Hierarchy Level", fontsize=12)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    out_file = output_path / f"auroc_heatmap.{fmt}"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved AUROC heatmap: {out_file}")


# ---------------------------------------------------------------------------
#  Plot 4: Summary scatter — small multiples
# ---------------------------------------------------------------------------


def plot_summary_scatter(
    df: pd.DataFrame,
    output_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Small multiples: one panel per hierarchy level, AUROC vs recall, labeled points."""
    levels = df["level"].unique().tolist()
    n_levels = len(levels)
    ncols = min(n_levels, 3)
    nrows = int(np.ceil(n_levels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5.5 * nrows),
        squeeze=False,
    )

    for idx, level in enumerate(levels):
        row_idx = idx // ncols
        col_idx = idx % ncols
        ax = axes[row_idx][col_idx]

        level_data = df[df["level"] == level].copy()

        # Plot each embedding as a colored point
        for _, data_row in level_data.iterrows():
            emb = data_row["embedding"]
            color = _get_bar_color(emb)
            ax.scatter(
                data_row["auroc"],
                data_row["recall_at_first_fp"],
                c=color,
                s=100,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            # Label: offset text to avoid overlap
            ax.annotate(
                emb,
                (data_row["auroc"], data_row["recall_at_first_fp"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                alpha=0.85,
            )

        ax.set_xlabel("AUROC", fontsize=11)
        ax.set_ylabel("Recall at First FP", fontsize=11)
        ax.set_title(_display_level(level), fontsize=14)
        ax.set_xlim(0.4, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axvline(x=0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(alpha=0.3)

    # Hide unused panels
    for idx in range(n_levels, nrows * ncols):
        row_idx = idx // ncols
        col_idx = idx % ncols
        axes[row_idx][col_idx].set_visible(False)

    fig.suptitle(
        "AUROC vs Recall at First FP by Hierarchy Level",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    out_file = output_path / f"summary_scatter.{fmt}"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved summary scatter: {out_file}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create retrieval/classification evaluation plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_parquet",
        type=Path,
        required=True,
        help="Parquet file from classification_eval.py "
             "(columns: embedding, level, auroc, recall_at_first_fp, ...)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("out/classification_eval/figures"),
        help="Directory for output figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (for raster formats)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.results_parquet.exists():
        log.error(f"Results parquet not found: {args.results_parquet}")
        sys.exit(1)

    # Read data — support both polars-written and pandas-written parquets
    log.info(f"Reading results from {args.results_parquet}")
    df = pd.read_parquet(args.results_parquet)
    log.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    required_cols = {"embedding", "level", "auroc", "recall_at_first_fp"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error(f"Missing required columns: {missing}")
        sys.exit(1)

    if df.empty:
        log.error("Results dataframe is empty. Nothing to plot.")
        sys.exit(1)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {args.output_dir}")

    # Set seaborn style to match existing plots
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # Generate all plots
    log.info("Generating AUROC bar chart...")
    plot_metric_bars(
        df, args.output_dir, args.format, args.dpi,
        value_col="auroc",
        ylabel="AUROC",
        title="AUROC by Embedding and Hierarchy Level",
        filename_stem="auroc_bars",
        legend_loc="lower right",
        baseline=0.5,
    )

    log.info("Generating recall-at-first-FP bar chart...")
    plot_metric_bars(
        df, args.output_dir, args.format, args.dpi,
        value_col="recall_at_first_fp",
        ylabel="Recall at First FP",
        title="Recall at First False Positive by Embedding and Hierarchy Level",
        filename_stem="recall_at_first_fp_bars",
        legend_loc="upper left",
    )

    log.info("Generating AUROC heatmap...")
    plot_auroc_heatmap(df, args.output_dir, args.format, args.dpi)

    log.info("Generating summary scatter plot...")
    plot_summary_scatter(df, args.output_dir, args.format, args.dpi)

    log.info("All figures saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
