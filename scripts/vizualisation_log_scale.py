import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
import matplotlib.lines as mlines
from typing import Dict, List
import matplotlib.ticker as mticker

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# Suppress matplotlib DEBUG messages
matplotlib_general_logger = logging.getLogger("matplotlib")
matplotlib_general_logger.setLevel(logging.INFO)

# --- Project Constants & Configuration ---

# PLM Parameter Sizes (Update as needed)
PLM_SIZES: Dict[str, int] = {
    "prott5": 1_500_000_000,
    "prottucker": 1_500_000_000,
    "prostt5": 1_500_000_000,
    "esm1b": 650_000_000,
    "clean": 650_000_000,
    "esm2_8m": 8_000_000,
    "esm2_35m": 35_000_000,
    "esm2_150m": 150_000_000,
    "esm2_650m": 650_000_000,
    "esm2_3b": 3_000_000_000,
    "ankh_base": 450_000_000,
    "ankh_large": 1_150_000_000,
    "random_1024": 0,
    # Add other embeddings here
}

# Mapping from embedding name (lowercase stem) to family
EMBEDDING_FAMILY_MAP: Dict[str, str] = {
    "prott5": "ProtT5",
    "prottucker": "ProtT5",
    "prostt5": "ProtT5",
    "esm1b": "ESM-1",
    "clean": "ESM-1",
    "esm2_8m": "ESM-2",
    "esm2_35m": "ESM-2",
    "esm2_150m": "ESM-2",
    "esm2_650m": "ESM-2",
    "esm2_3b": "ESM-2",
    "ankh_base": "Ankh",
    "ankh_large": "Ankh",
    "random_1024": "Random",
    # Add other embeddings here
}

# Color map for individual embeddings (lowercase stem)
EMBEDDING_COLOR_MAP: Dict[str, str] = {
    "prott5": "#377eb8",
    "prottucker": "#373bb7",
    "prostt5": "#1217b5",
    "esm1b": "#5fd35b",
    "clean": "#4daf4a",
    "esm2_8m": "#fdae61",
    "esm2_35m": "#ff7f00",
    "esm2_150m": "#f46d43",
    "esm2_650m": "#d73027",
    "esm2_3b": "#a50026",
    "ankh_base": "#ffd700",
    "ankh_large": "#a88c01",
    "random_1024": "#808080",  # Grey for random
    # Add other embeddings here
}

# Marker map for model types
MODEL_MARKER_MAP: Dict[str, str] = {
    "fnn": "o",  # Circle
    "linear": "s",  # Square
    "linear_distance": "^",  # Triangle up
    "euclidean": "X",  # X
    # Add other model types here
}

# Families for which connecting lines should be drawn in the plot
FAMILIES_TO_CONNECT: List[str] = ["ProtT5", "ESM-2", "Ankh"]


# --- Data Parsing ---
def parse_metrics_file(filepath: Path) -> Dict[str, float]:
    """Parses a metrics file to extract Pearson R2 and its standard error."""
    float_pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)"
    metrics = {"Pearson R2": None, "Pearson R2 SE": None}

    with filepath.open("r") as f:
        for line in f:
            for metric_key, file_prefix in [
                ("Pearson R2", "Pearson_r2:"),
                ("Pearson R2 SE", "Pearson_r2_SE:"),
            ]:
                if line.startswith(file_prefix):
                    match = re.search(f"{file_prefix}\\s*{float_pattern}", line)
                    metrics[metric_key] = float(match.group(1))

    # Check if any metrics are missing
    missing = [k for k, v in metrics.items() if v is None]
    if missing:
        raise ValueError(f"Metrics not found in {filepath}: {', '.join(missing)}")

    return metrics


def load_results_data(base_dir: Path) -> pd.DataFrame:
    """Parses metrics files from model directories and returns results as a DataFrame."""
    results = []
    log.info(f"Searching recursively for '*_metrics.txt' files under: {base_dir}")

    # Use rglob to find all _metrics.txt files recursively
    metric_files = list(base_dir.rglob("*_metrics.txt"))
    log.info(f"Found {len(metric_files)} potential '*_metrics.txt' files via rglob:")

    if not metric_files:
        raise ValueError("No '*_metrics.txt' files found anywhere under {base_dir}.")

    # Define expected number of path parts relative to base_dir
    # models/train_sub/<model_type>/<param_name>/<embedding_name>/<timestamp>/evaluation_results/*_metrics.txt
    #                 | 1          | 2          | 3              | 4         | 5                | 6
    EXPECTED_DEPTH = 6

    for metric_file in metric_files:
        relative_path = metric_file.relative_to(base_dir)
        relative_parts = relative_path.parts

        # Skip files with unexpected path structure
        if len(relative_parts) != EXPECTED_DEPTH:
            log.warning(
                f"Skipping file with unexpected path structure: {relative_path}\n"
                f"Expected path format: <model_type>/<param_name>/<embedding_name>/<timestamp>/evaluation_results/*_metrics.txt\n"
                f"Found {len(relative_parts)} path components instead of expected {EXPECTED_DEPTH}"
            )
            continue

        # Extract metadata from path
        model_type, param_name, embedding_name = relative_parts[0:3]
        embedding_key = embedding_name.lower()
        metrics = parse_metrics_file(metric_file)

        # Add result to collection
        results.append(
            {
                "Model Type": model_type,
                "Parameter": param_name,
                "Embedding": embedding_name,
                "Embedding Key": embedding_key,
                "Embedding Family": EMBEDDING_FAMILY_MAP.get(embedding_key, "Unknown"),
                "PLM Size": PLM_SIZES.get(embedding_key),
                "Pearson R2": metrics["Pearson R2"],
                "Pearson R2 SE": metrics["Pearson R2 SE"],
            }
        )

    # Create DataFrame and convert columns to numeric types
    results_df = pd.DataFrame(results)
    for col in ["Pearson R2", "Pearson R2 SE", "PLM Size"]:
        results_df[col] = pd.to_numeric(results_df[col])

    log.info(f"Successfully parsed {len(results_df)} valid result entries.")
    return results_df


# --- Plotting Helpers ---
def _add_error_bars(ax: plt.Axes, data: pd.DataFrame):
    """Adds error bars for Pearson R2 SE to the plot axes."""
    for _, row in data.iterrows():
        color = EMBEDDING_COLOR_MAP.get(row["Embedding Key"], "grey")
        ax.errorbar(
            x=row["PLM Size"],
            y=row["Pearson R2"],
            yerr=row["Pearson R2 SE"],
            fmt="none",
            ecolor=color,
            capsize=3,
            alpha=0.6,
            zorder=1,  # Ensure error bars are behind points
        )


def _add_connecting_lines(ax: plt.Axes, data: pd.DataFrame):
    """Adds connecting lines for specified embedding families."""
    # Group by family and model type, then sort by PLM size for line plotting
    for (family, model_type), group in data.groupby(["Embedding Family", "Model Type"]):
        if family in FAMILIES_TO_CONNECT:
            sorted_group = group.sort_values("PLM Size")
            if len(sorted_group) > 1:
                # Use the color of the first embedding in the sorted group for the line
                # This assumes embeddings within a family share a base color scheme
                line_color_key = sorted_group.iloc[0]["Embedding Key"]
                line_color = EMBEDDING_COLOR_MAP.get(line_color_key, "grey")
                ax.plot(
                    sorted_group["PLM Size"],
                    sorted_group["Pearson R2"],
                    marker="",
                    linestyle="-",
                    color=line_color,
                    alpha=0.7,
                    zorder=2,  # Ensure lines are behind points but above error bars
                )


def _create_embedding_legend(fig: plt.Figure, embeddings: List[str]) -> plt.legend:
    """Creates and returns the embedding legend (colors)."""
    handles = []
    labels = []
    log.debug(f"Creating embedding legend for: {sorted(embeddings)}")
    for embedding_name in sorted(embeddings):
        key = embedding_name.lower()
        color = EMBEDDING_COLOR_MAP.get(key, "grey")
        # Use a square marker for the color legend instead of a patch
        handles.append(
            mlines.Line2D(
                [], [], color=color, marker="s", linestyle="None", markersize=7
            )
        )
        labels.append(key)

    return fig.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.65),
        frameon=False,
        title="Embedding",
    )


def _create_model_type_legend(fig: plt.Figure, model_types: List[str]) -> plt.legend:
    """Creates and returns the model type legend (markers)."""
    handles = []
    labels = []
    # Sort model types alphabetically for the legend
    for model_type in sorted(model_types):
        marker = MODEL_MARKER_MAP.get(model_type, "?")
        handles.append(
            mlines.Line2D(
                [], [], color="black", marker=marker, linestyle="None", markersize=7
            )
        )
        labels.append(model_type)

    return fig.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.30),
        frameon=False,
        title="Model Type",
    )


def _human_readable_formatter(x, pos):
    """Custom formatter for large numbers (B, M, K)"""
    if x >= 1e9:
        return f"{x * 1e-9:.1f}B"
    elif x >= 1e6:
        return f"{x * 1e-6:.0f}M"
    elif x >= 1e3:
        return f"{x * 1e-3:.0f}K"
    else:
        return f"{x:.0f}"


# --- Main Plotting Function ---
def plot_summary_results(df: pd.DataFrame, output_file: Path):
    """Generates the summary faceted scatter plot with custom legends."""
    sns.set_theme(style="whitegrid")

    # --- Prepare Tick Locations ---
    # Get unique non-zero PLM sizes from the data
    unique_plm_sizes = df[df["PLM Size"] > 0]["PLM Size"].unique()
    # Create the list of tick locations: 0 plus all unique non-zero sizes, sorted.
    major_tick_locs = sorted([0] + unique_plm_sizes.tolist())
    log.debug(f"Using dynamic major tick locations: {major_tick_locs}")

    # Create the base FacetGrid using relplot (scatter plot)
    g = sns.relplot(
        data=df,
        x="PLM Size",
        y="Pearson R2",
        col="Parameter",  # Facet by parameter
        hue="Embedding",  # Color by embedding (original name)
        style="Model Type",  # Marker style by model type
        palette=EMBEDDING_COLOR_MAP,  # Use the defined color map (expects keys matching hue)
        hue_order=sorted(df["Embedding"].unique()),
        style_order=sorted(df["Model Type"].unique()),
        markers=MODEL_MARKER_MAP,  # Use the defined marker map
        kind="scatter",
        s=100,  # Point size
        facet_kws={"sharey": True, "sharex": True},
        legend=False,  # Disable automatic legend
        zorder=5,  # Ensure scatter points are on top
    )

    # Configure overall plot appearance (set symlog scale type)
    g.set(xscale="symlog")

    # Add manual elements (error bars, lines, baseline) to each facet
    for param, ax in g.axes_dict.items():
        param_df = df[df["Parameter"] == param]

        _add_error_bars(ax, param_df)
        _add_connecting_lines(ax, param_df)

        # --- Axis Formatting ---
        ax.set_xlabel("pLM Parameter Count")
        ax.set_ylabel("Pearson r²")
        ax.set_title(f"Parameter: {param}")
        # Apply symlog scale, threshold, and adjust linscale
        # Increase linthresh to push log compression further out
        # Adjust linscale to balance visual weight
        ax.set_xscale("symlog", linthresh=8_000_000, linscale=0.1)
        # Set explicit major ticks and apply custom formatter
        ax.xaxis.set_major_locator(mticker.FixedLocator(major_tick_locs))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_human_readable_formatter))
        ax.minorticks_off()  # Remove minor ticks
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # g.figure.suptitle("Model Performance Summary: Pearson R² vs PLM Size", y=1.03)

    # Create and place custom legends
    all_embeddings = df["Embedding"].unique().tolist()
    all_model_types = df["Model Type"].unique().tolist()
    _create_embedding_legend(g.figure, all_embeddings)
    _create_model_type_legend(g.figure, all_model_types)
    # plt.tight_layout(rect=[0, 0, 1.02, 0.97])

    # Save the figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        log.info(f"Plot saved successfully to {output_file}")
    except Exception as e:
        log.error(f"Failed to save plot to {output_file}: {e}", exc_info=True)
    finally:
        plt.close(g.figure)  # Ensure figure is closed


# --- Main Execution ---
def main():
    """Main function to parse arguments, load data, and generate the plot."""
    parser = argparse.ArgumentParser(
        description="Parse model evaluation results and generate a summary plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Base directory containing the model results (e.g., 'models/train_sub').",
    )
    parser.add_argument(
        "--output_plot",
        type=Path,
        default=Path("summary_results_plot.png"),
        help="Path to save the output plot PNG file.",
    )
    args = parser.parse_args()

    # --- Load Data ---
    results_df = load_results_data(args.results_dir)
    # print(results_df.head())

    # --- Generate Plot ---
    plot_summary_results(results_df, args.output_plot)


if __name__ == "__main__":
    main()
