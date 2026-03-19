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
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants — reused from create_performance_summary_plots.py
# ---------------------------------------------------------------------------

PLM_SIZES: Dict[str, int] = {
    "prott5": 1_500_000_000,
    "prottucker": 1_500_000_000,
    "prostt5": 1_500_000_000,
    "clean": 650_000_000,
    "esm1b": 650_000_000,
    "esm2_8m": 8_000_000,
    "esm2_35m": 35_000_000,
    "esm2_150m": 150_000_000,
    "esm2_650m": 650_000_000,
    "esm2_3b": 3_000_000_000,
    "esmc_300m": 300_000_000,
    "esmc_600m": 600_000_000,
    "esm3_open": 1_400_000_000,
    "ankh_base": 450_000_000,
    "ankh_large": 1_150_000_000,
    "random_1024": 0,
}

EMBEDDING_FAMILY_MAP: Dict[str, str] = {
    "prott5": "ProtT5",
    "prottucker": "ProtT5",
    "prostt5": "ProtT5",
    "clean": "ESM-1",
    "esm1b": "ESM-1",
    "esm2_8m": "ESM-2",
    "esm2_35m": "ESM-2",
    "esm2_150m": "ESM-2",
    "esm2_650m": "ESM-2",
    "esm2_3b": "ESM-2",
    "esmc_300m": "ESM-C",
    "esmc_600m": "ESM-C",
    "esm3_open": "ESM-3",
    "ankh_base": "Ankh",
    "ankh_large": "Ankh",
    "random_1024": "Random",
}

EMBEDDING_FAMILY_COLOR_MAP: Dict[str, str] = {
    "ProtT5": "#ff1493",
    "ESM-1": "#4daf4a",
    "ESM-2": "#ff7f00",
    "ESM-C": "#1f77b4",
    "ESM-3": "#984ea3",
    "Ankh": "#ffd700",
    "Random": "#808080",
}

EMBEDDING_COLOR_MAP: Dict[str, str] = {
    embedding: EMBEDDING_FAMILY_COLOR_MAP.get(family, "#808080")
    for embedding, family in EMBEDDING_FAMILY_MAP.items()
}

# Friendly display names for hierarchy level columns
LEVEL_DISPLAY_NAMES: Dict[str, str] = {
    "fa_id": "Family",
    "sf_id": "Superfamily",
    "fold_id": "Fold",
    "F_group": "F-group",
    "H_group": "Homology",
    "X_group": "X-group",
    "T_group": "Topology",
}

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
    return LEVEL_DISPLAY_NAMES.get(level, level)


def _get_bar_color(embedding: str) -> str:
    """Return the family color for an embedding name."""
    key = embedding.lower()
    return EMBEDDING_COLOR_MAP.get(key, "#808080")


def _format_size(n_params: int) -> str:
    """Format parameter count to compact string (e.g. '650M', '3B')."""
    if n_params == 0:
        return "0"
    if n_params >= 1_000_000_000:
        return f"{n_params / 1_000_000_000:.1f}B".replace(".0B", "B")
    return f"{n_params / 1_000_000:.0f}M"


# ---------------------------------------------------------------------------
#  Plot 1: AUROC bar chart
# ---------------------------------------------------------------------------


def plot_auroc_bars(
    df: pd.DataFrame,
    output_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Grouped bar chart: x = pLM (sorted by size), y = AUROC, hue = level."""
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
        values = []
        for emb in embeddings:
            row = level_data[level_data["embedding"] == emb]
            values.append(row["auroc"].values[0] if len(row) > 0 else np.nan)

        offset = (i - n_levels / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=_display_level(level),
            color=level_colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

    # Random baseline
    ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=1.0, alpha=0.7,
               label="Random (0.5)")

    # X-axis: embedding names with size annotation
    x_labels = []
    for emb in embeddings:
        size = PLM_SIZES.get(emb.lower(), None)
        if size is not None:
            x_labels.append(f"{emb}\n({_format_size(size)})")
        else:
            x_labels.append(emb)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("AUROC", fontsize=14)
    ax.set_title("AUROC by Embedding and Hierarchy Level", fontsize=16)
    ax.legend(title="Level", fontsize=10, title_fontsize=11, loc="lower right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_file = output_path / f"auroc_bars.{fmt}"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved AUROC bar chart: {out_file}")


# ---------------------------------------------------------------------------
#  Plot 2: Recall-at-first-FP bar chart
# ---------------------------------------------------------------------------


def plot_recall_bars(
    df: pd.DataFrame,
    output_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Grouped bar chart: x = pLM (sorted by size), y = recall@1FP, hue = level."""
    levels = df["level"].unique().tolist()
    embeddings = _sort_embeddings_by_size(df["embedding"].unique().tolist())
    n_levels = len(levels)
    n_embeddings = len(embeddings)

    fig, ax = plt.subplots(figsize=(max(12, n_embeddings * 0.9), 7))

    bar_width = 0.8 / n_levels
    x = np.arange(n_embeddings)

    level_colors = sns.color_palette("Set2", n_colors=n_levels)

    for i, level in enumerate(levels):
        level_data = df[df["level"] == level]
        values = []
        for emb in embeddings:
            row = level_data[level_data["embedding"] == emb]
            values.append(
                row["recall_at_first_fp"].values[0] if len(row) > 0 else np.nan
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

    # X-axis: embedding names with size annotation
    x_labels = []
    for emb in embeddings:
        size = PLM_SIZES.get(emb.lower(), None)
        if size is not None:
            x_labels.append(f"{emb}\n({_format_size(size)})")
        else:
            x_labels.append(emb)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Recall at First FP", fontsize=14)
    ax.set_title("Recall at First False Positive by Embedding and Hierarchy Level",
                 fontsize=16)
    ax.legend(title="Level", fontsize=10, title_fontsize=11, loc="upper left")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_file = output_path / f"recall_at_first_fp_bars.{fmt}"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved recall bar chart: {out_file}")


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
    plot_auroc_bars(df, args.output_dir, args.format, args.dpi)

    log.info("Generating recall-at-first-FP bar chart...")
    plot_recall_bars(df, args.output_dir, args.format, args.dpi)

    log.info("Generating AUROC heatmap...")
    plot_auroc_heatmap(df, args.output_dir, args.format, args.dpi)

    log.info("Generating summary scatter plot...")
    plot_summary_scatter(df, args.output_dir, args.format, args.dpi)

    log.info("All figures saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
