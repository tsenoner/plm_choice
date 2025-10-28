import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
import matplotlib.lines as mlines
from typing import Dict, List, Optional
import numpy as np
from scipy import stats

# --- Logging Configuration ---
# Default level will be set in main() based on verbose flag
log = logging.getLogger(__name__)

# --- Plot Configuration ---
# Font sizes and visual settings - easily adjustable
PLOT_CONFIG = {
    "font_scale": 1.5,  # Base font scale multiplier
    "title_fontsize": 26,  # Facet titles
    "label_fontsize": 22,  # Axis labels
    "tick_fontsize": 18,  # Tick labels
    "panel_label_fontsize": 32,  # A, B, C labels
    "legend_title_fontsize": 26,  # Legend titles
    "legend_fontsize": 22,  # Legend text
    "marker_size": 200,  # Scatter plot marker size
    "plot_height": 7,  # Plot height in inches
    "plot_aspect": 1.1,  # Plot aspect ratio
    "dpi": 300,  # Output DPI
    "subplot_spacing": 0.1,  # Space between subplots
}

# --- Project Constants & Configuration ---

# PLM Parameter Sizes (Update as needed)
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

# Mapping from embedding name (lowercase stem) to family
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

# Color map for embedding families
EMBEDDING_FAMILY_COLOR_MAP: Dict[str, str] = {
    "ProtT5": "#ff1493",
    "ESM-1": "#4daf4a",
    "ESM-2": "#ff7f00",
    "ESM-C": "#1f77b4",
    "ESM-3": "#984ea3",
    "Ankh": "#ffd700",
    "Random": "#808080",
}

# Assign family color to each embedding
EMBEDDING_COLOR_MAP: Dict[str, str] = {
    embedding: EMBEDDING_FAMILY_COLOR_MAP.get(family, "#808080")
    for embedding, family in EMBEDDING_FAMILY_MAP.items()
}

# Marker map for model types
MODEL_MARKER_MAP: Dict[str, str] = {
    "fnn": "o",  # Circle
    "linear": "s",  # Square
    "linear_distance": "^",  # Triangle up
    "euclidean": "X",  # X
}

# Titles for the plot facets
PARAMETER_TITLES: Dict[str, str] = {
    "fident": "Sequence - PIDE",
    "alntmscore": "Structure - alignment TM-score",
    "hfsp": "Function - HFSP",
}


# --- Data Parsing ---
def parse_metrics_file(filepath: Path) -> Dict[str, float]:
    """Parses a metrics file to extract key performance metrics.

    Args:
        filepath: Path to the metrics text file

    Returns:
        Dictionary mapping metric names to their values (NaN if not found)
    """
    # Regex to match floats/scientific notation or 'nan'
    float_pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)"
    # Define the metrics we want to extract and their prefixes in the file
    metrics_to_extract = {
        "Pearson R2": "Pearson_r2:",
        "Pearson R2 SE": "Pearson_r2_SE:",
        "MAE": "MAE:",
        "Spearman": "Spearman:",
        "Spearman SE": "Spearman_SE:",
        "R2": "R2:",
    }
    metrics = {key: None for key in metrics_to_extract}

    try:
        with filepath.open("r") as f:
            for line in f:
                for metric_key, file_prefix in metrics_to_extract.items():
                    if line.startswith(file_prefix):
                        match = re.search(f"{file_prefix}\\s*{float_pattern}", line)
                        if match:
                            val_str = match.group(1)
                            # Store NaN as NaN, otherwise convert to float
                            metrics[metric_key] = (
                                float("nan") if val_str == "nan" else float(val_str)
                            )
                            break  # Move to next line once metric is found

        # Warn if any metrics are missing
        missing = [k for k, v in metrics.items() if v is None]
        if missing:
            log.warning(f"Metrics not found in {filepath}: {', '.join(missing)}")

    except FileNotFoundError:
        log.error(f"Metrics file not found: {filepath}")
        # Return dict with NaNs if file not found
        metrics = {key: float("nan") for key in metrics_to_extract}
    except Exception as e:
        log.error(f"Error parsing file {filepath}: {e}", exc_info=True)
        metrics = {key: float("nan") for key in metrics_to_extract}

    return metrics


def load_results_data(base_dir: Path) -> pd.DataFrame:
    """Parses metrics files from model directories and returns results as a DataFrame.

    Args:
        base_dir: Base directory containing model evaluation results

    Returns:
        DataFrame with parsed metrics and metadata for each model/embedding combination
    """
    results = []
    log.info(f"Searching recursively for '*_metrics.txt' files under: {base_dir}")

    metric_files = list(base_dir.rglob("*_metrics.txt"))
    log.info(f"Found {len(metric_files)} potential '*_metrics.txt' files via rglob:")

    if not metric_files:
        raise ValueError(f"No '*_metrics.txt' files found anywhere under {base_dir}.")

    for metric_file in metric_files:
        metrics = {}  # Initialize metrics for each file
        try:
            relative_path = metric_file.relative_to(base_dir)

            try:
                eval_results_index = relative_path.parts.index("evaluation_results")
            except ValueError:
                log.warning(
                    f"Skipping file with unexpected structure (missing 'evaluation_results'): {relative_path}"
                )
                continue

            # Expected structure: <...>/<model_type>/<param_name>/<embedding_name>/evaluation_results/...
            if eval_results_index < 3:
                log.warning(
                    f"Skipping file due to unexpected path depth relative to 'evaluation_results': {relative_path}"
                )
                continue

            embedding_name = relative_path.parts[eval_results_index - 1]
            param_name = relative_path.parts[eval_results_index - 2]
            model_type = relative_path.parts[eval_results_index - 3]

            embedding_key = embedding_name.lower()
            metrics = parse_metrics_file(metric_file)

            results.append(
                {
                    "Model Type": model_type,
                    "Parameter": param_name,
                    "Embedding": embedding_name,
                    "Embedding Key": embedding_key,
                    "Embedding Family": EMBEDDING_FAMILY_MAP.get(
                        embedding_key, "Unknown"
                    ),
                    "PLM Size": PLM_SIZES.get(embedding_key),
                    **metrics,
                }
            )
        except Exception as e:
            log.error(f"Failed to process file {metric_file}: {e}", exc_info=True)

    if not results:
        log.error(
            "No valid metric files were parsed. Please check the directory structure and file contents."
        )
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Convert metric columns to numeric types
    if results:
        numeric_cols = metrics.keys()
        for col in numeric_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col])
            else:
                log.warning(
                    f"Column '{col}' not found in parsed data, skipping numeric conversion."
                )

    log.info(f"Successfully parsed {len(results_df)} valid result entries.")
    return results_df


# --- Plotting Helpers ---
def _add_error_bars(
    ax: plt.Axes, data: pd.DataFrame, y_metric: str, se_metric: Optional[str]
):
    """Adds error bars for a given metric to the plot axes.

    Args:
        ax: Matplotlib axes to add error bars to
        data: DataFrame containing the data
        y_metric: Name of the metric column for y-values
        se_metric: Name of the standard error column (optional)
    """
    if not se_metric or se_metric not in data.columns:
        return  # No SE column provided or found

    for _, row in data.iterrows():
        se_value = row[se_metric]
        # Check if SE value is valid (not NaN and > 0)
        if pd.notna(se_value) and se_value > 0:
            color = EMBEDDING_COLOR_MAP.get(row["Embedding Key"], "grey")
            ax.errorbar(
                x=row["Embedding"],  # Use categorical x
                y=row[y_metric],  # Use specified y metric
                yerr=se_value,  # Use specified SE value
                fmt="none",
                ecolor=color,
                capsize=3,
                alpha=0.6,
                zorder=1,
            )


def _add_connecting_lines(ax: plt.Axes, data: pd.DataFrame, y_metric: str):
    """Adds vertical connecting lines between model types for each embedding.

    Args:
        ax: Matplotlib axes to add lines to
        data: DataFrame containing the data
        y_metric: Name of the metric column for y-values
    """
    # Group by embedding to draw vertical lines
    for embedding, group in data.groupby("Embedding"):
        if len(group) > 1:
            # Sort by model type to ensure lines are drawn in a consistent order
            group = group.sort_values(
                "Model Type",
                key=lambda s: s.map(
                    {
                        "fnn": 0,
                        "linear": 1,
                        "linear_distance": 2,
                        "euclidean": 3,
                    }
                ),
            )
            ax.plot(
                group["Embedding"],
                group[y_metric],
                marker="",
                linestyle=":",
                color=EMBEDDING_COLOR_MAP.get(embedding, "grey"),
                alpha=0.3,
                zorder=2,
            )


def _add_trendlines(
    ax: plt.Axes,
    data: pd.DataFrame,
    y_metric: str,
    category_order: List[str],
    parameter: str,
):
    """Adds straight trendlines for each model type based on PLM size vs metric.

    Args:
        ax: The matplotlib axes to plot on
        data: DataFrame containing the data for this facet
        y_metric: The metric column name to use for y-axis
        category_order: The ordered list of embedding names for x-axis positioning
        parameter: The parameter name (e.g., 'fident', 'alntmscore', 'hfsp')

    Returns:
        List of dicts containing trend statistics for each model type
    """
    # Define line styles for different model types
    model_type_linestyles = {
        "fnn": "--",  # Dashed
        "linear": "-",  # Solid
        "linear_distance": ":",  # Dotted
        "euclidean": "-.",  # Dash-dot
    }

    trend_stats = []  # Collect statistics for CSV export

    # Group by model type and fit trendlines
    for model_type, group in data.groupby("Model Type"):
        # Filter out rows with NaN values in PLM Size or y_metric
        valid_group = group.dropna(subset=["PLM Size", y_metric])

        # Exclude esm2_3b from FNN trendline
        if model_type == "fnn":
            valid_group = valid_group[valid_group["Embedding Key"] != "esm2_3b"]
            log.debug(f"Excluding esm2_3b from {model_type} trendline")

        if len(valid_group) < 2:
            log.debug(f"Skipping trendline for {model_type}: insufficient data points")
            continue

        # Get numeric x (PLM sizes) and y (metric) values
        x_numeric = valid_group["PLM Size"].values
        y_values = valid_group[y_metric].values

        # Skip if we have zero or negative PLM sizes (like random_1024)
        if np.any(x_numeric <= 0):
            log.debug(
                f"Skipping trendline for {model_type}: contains zero or negative PLM sizes"
            )
            continue

        # Fit linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x_numeric, y_values
            )

            # Calculate y values at the min and max PLM sizes
            x_min = x_numeric.min()
            x_max = x_numeric.max()
            y_min = slope * x_min + intercept
            y_max = slope * x_max + intercept

            # Map these to categorical positions
            # Find positions of min and max PLM sizes in the category order
            size_to_pos = {}
            for i, emb_name in enumerate(category_order):
                emb_key = emb_name.lower()
                if emb_key in PLM_SIZES:
                    size_to_pos[PLM_SIZES[emb_key]] = i

            # Get the positions for min and max sizes
            if x_min in size_to_pos and x_max in size_to_pos:
                pos_min = size_to_pos[x_min]
                pos_max = size_to_pos[x_max]

                # Draw straight line from leftmost to rightmost point
                linestyle = model_type_linestyles.get(model_type, "-")
                ax.plot(
                    [pos_min, pos_max],
                    [y_min, y_max],
                    linestyle=linestyle,
                    color="black",
                    linewidth=2.5,
                    alpha=0.7,
                    zorder=3,
                )

                # Store statistics for CSV export
                # Scale slope to "per 1B" with 2 significant figures
                slope_per_1b = slope * 1e9
                trend_stats.append(
                    {
                        "parameter": parameter,
                        "model_type": model_type,
                        "slope_per_1b": slope_per_1b,
                        "r2": r_value**2,
                        "p_value": p_value,
                    }
                )

                log.debug(
                    f"Added trendline for {model_type}: slope={slope_per_1b:.2g} per 1B, R²={r_value**2:.3f}, p={p_value:.3e}"
                )

        except Exception as e:
            log.warning(f"Failed to fit trendline for {model_type}: {e}")

    return trend_stats


def _create_embedding_legend(fig: plt.Figure, df: pd.DataFrame) -> plt.legend:
    """Creates and returns the embedding family legend (colors).

    Args:
        fig: Matplotlib figure to add legend to
        df: DataFrame containing the data

    Returns:
        The created legend object
    """
    handles = []
    labels = []

    # Get unique families and their colors
    unique_families = df[["Embedding Family", "Embedding Key"]].drop_duplicates()

    # Sort families for a consistent legend order
    sorted_families = sorted(unique_families["Embedding Family"].unique())

    for family in sorted_families:
        # Find the first embedding key for this family to get the color
        embedding_key = unique_families[
            unique_families["Embedding Family"] == family
        ].iloc[0]["Embedding Key"]
        color = EMBEDDING_COLOR_MAP.get(embedding_key, "grey")

        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                marker="s",
                linestyle="None",
                markersize=PLOT_CONFIG["legend_fontsize"],
            )
        )
        labels.append(family)

    return fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.35, 0.1),
        frameon=False,
        title="pLM Family",
        title_fontsize=PLOT_CONFIG["legend_title_fontsize"],
        fontsize=PLOT_CONFIG["legend_fontsize"],
        ncol=6,  # Adjust number of columns if needed
    )


def _create_model_type_legend(fig: plt.Figure, model_types: List[str]) -> plt.legend:
    """Creates and returns the model type legend (markers).

    Args:
        fig: Matplotlib figure to add legend to
        model_types: List of model type names

    Returns:
        The created legend object
    """
    handles = []
    labels = []
    # Sort model types alphabetically for the legend
    for model_type in sorted(model_types):
        marker = MODEL_MARKER_MAP.get(model_type, "?")
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                marker=marker,
                linestyle="None",
                markersize=PLOT_CONFIG["legend_fontsize"],
            )
        )
        labels.append(model_type)

    return fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.85, 0.1),
        frameon=False,
        title="Model Type",
        title_fontsize=PLOT_CONFIG["legend_title_fontsize"],
        fontsize=PLOT_CONFIG["legend_fontsize"],
        ncol=4,  # Display in multiple columns
    )


def _human_readable_formatter(x, pos=None):
    """Formatter for large numbers with SI suffixes (K, M, B, T, etc).

    Args:
        x: Number to format
        pos: Position (unused, for matplotlib compatibility)

    Returns:
        Formatted string with appropriate SI suffix
    """
    abs_x = abs(x)
    units = ["", "K", "M", "B", "T", "P", "E", "Z", "Y"]
    magnitude = 0
    while abs_x >= 1000 and magnitude < len(units) - 1:
        abs_x /= 1000.0
        magnitude += 1
    if magnitude == 0:
        return str(int(x))
    # Show up to 3 significant digits
    value_str = f"{abs_x:.3g}"
    sign = "-" if x < 0 else ""
    return f"{sign}{value_str}{units[magnitude]}"


# --- Main Plotting Function ---
def generate_metric_plot(
    df: pd.DataFrame, y_metric: str, se_metric: Optional[str], output_file: Path
):
    """Generates a summary faceted scatter plot for a specific metric.

    Args:
        df: DataFrame containing the data
        y_metric: Name of the metric column to plot
        se_metric: Name of the standard error column (optional)
        output_file: Path to save the output plot

    Returns:
        List of dicts containing trend statistics for all facets
    """
    sns.set_theme(style="whitegrid", font_scale=PLOT_CONFIG["font_scale"])

    # Collect trend statistics across all facets
    all_trend_stats = []

    # --- Reorder and Prepare DataFrame ---
    param_order = ["fident", "alntmscore", "hfsp"]
    df["Parameter"] = pd.Categorical(
        df["Parameter"], categories=param_order, ordered=True
    )
    df_sorted = df.sort_values(by=["Parameter", "PLM Size", "Model Type"])
    category_order = df_sorted["Embedding"].unique().tolist()

    # Create the base FacetGrid using relplot (scatter plot)
    g = sns.relplot(
        data=df_sorted,
        x="Embedding",
        y=y_metric,
        col="Parameter",
        hue="Embedding Family",
        style="Model Type",
        palette=EMBEDDING_FAMILY_COLOR_MAP,
        hue_order=sorted(df_sorted["Embedding Family"].unique()),
        style_order=sorted(df_sorted["Model Type"].unique()),
        markers=MODEL_MARKER_MAP,
        kind="scatter",
        s=PLOT_CONFIG["marker_size"],
        height=PLOT_CONFIG["plot_height"],
        aspect=PLOT_CONFIG["plot_aspect"],
        facet_kws={"sharey": True, "sharex": False},
        legend=False,
        col_order=param_order,
        zorder=5,
    )

    # Add manual elements and format each facet
    panel_labels = ["A", "B", "C"]
    for i, (param, ax) in enumerate(g.axes_dict.items()):
        param_df = df_sorted[df_sorted["Parameter"] == param]
        _add_error_bars(ax, param_df, y_metric, se_metric)
        _add_connecting_lines(ax, param_df, y_metric)
        trend_stats = _add_trendlines(ax, param_df, y_metric, category_order, param)
        all_trend_stats.extend(trend_stats)

        ax.set_xlabel("pLM Parameter Count", fontsize=PLOT_CONFIG["label_fontsize"])
        ax.set_ylabel(y_metric, fontsize=PLOT_CONFIG["label_fontsize"])
        ax.set_title(
            PARAMETER_TITLES.get(param, param), fontsize=PLOT_CONFIG["title_fontsize"]
        )
        ax.grid(True, which="major", axis="y", linestyle="-", linewidth=1, alpha=0.15)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", rotation=45, labelsize=PLOT_CONFIG["tick_fontsize"])
        ax.tick_params(axis="y", labelsize=PLOT_CONFIG["tick_fontsize"])

        # Set y-axis to start from 0
        ax.set_ylim(0, 1)

        # Add panel labels (A, B, C)
        if i == 0:  # Panel A (leftmost) - position further left
            x_pos = -0.15
        else:  # Panels B and C - can be positioned closer
            x_pos = -0.08

        ax.text(
            x_pos,
            1.05,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=PLOT_CONFIG["panel_label_fontsize"],
            fontweight="bold",
            va="top",
        )

        # Set custom x-axis labels
        size_labels = [
            _human_readable_formatter(PLM_SIZES.get(emb.lower()))
            if emb.lower() in PLM_SIZES
            else emb
            for emb in category_order
        ]
        ax.set_xticks(range(len(category_order)))
        ax.set_xticklabels(size_labels)

    # Create and place custom legends
    _create_embedding_legend(g.figure, df_sorted)
    _create_model_type_legend(g.figure, df_sorted["Model Type"].unique().tolist())

    # Adjust spacing - use subplots_adjust instead of tight_layout for better control
    g.figure.subplots_adjust(
        wspace=PLOT_CONFIG["subplot_spacing"],
        bottom=0.25,  # Make room for legends
        top=1,
        left=0.1,
        right=1,
    )

    # Save the figure
    try:
        plt.savefig(output_file, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
        log.info(f"Plot saved successfully to {output_file}")
    except Exception as e:
        log.error(f"Failed to save plot to {output_file}: {e}", exc_info=True)
    finally:
        plt.close(g.figure)

    return all_trend_stats


# --- Main Execution ---
def main():
    """Main function to parse arguments, load data, and generate plots."""
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
        "--output",
        type=Path,
        default=Path("plots"),  # Default to a 'plots' directory
        help="Directory to save the output plot PNG files.",
    )
    parser.add_argument(
        "--ignore-random",
        action="store_true",
        help="If set, exclude results from 'Random' embeddings from the plots.",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",  # Allows multiple model types to be specified
        default=None,  # Default to None, meaning all model types
        help="Space-separated list of model types to include in the plots (fnn, linear, linear_distance, euclidean). If not specified, all are included.",
    )
    parser.add_argument(
        "--exclude_plms",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of pLM names (embedding names) to exclude from the plots.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output.",
    )
    args = parser.parse_args()

    # Configure logging based on verbose flag
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    # Suppress matplotlib DEBUG messages
    matplotlib_general_logger = logging.getLogger("matplotlib")
    matplotlib_general_logger.setLevel(logging.INFO)

    # --- Load Data ---
    results_df = load_results_data(args.results_dir)
    if results_df.empty:
        log.error("No data loaded, exiting.")
        return

    # --- Save Full Dataframe ---
    output_dir = args.output
    csv_filename = "parsed_metrics_all.csv"
    csv_output_path = output_dir / csv_filename
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_output_path, index=False)
        log.info(f"Full parsed data saved to {csv_output_path}")
    except Exception as e:
        log.error(
            f"Failed to save data to CSV at {csv_output_path}: {e}", exc_info=True
        )

    # --- Optional Filtering ---
    if args.ignore_random:
        initial_count = len(results_df)
        results_df = results_df[results_df["Embedding Family"] != "Random"].copy()
        log.info(
            f"--ignore-random flag set. Filtered out {initial_count - len(results_df)} 'Random' entries."
        )
        if results_df.empty:
            log.error(
                "DataFrame is empty after filtering 'Random' entries. Cannot generate plots."
            )
            return

    # --- Optional Filtering for PLMs ---
    if args.exclude_plms:
        initial_count = len(results_df)
        results_df = results_df[~results_df["Embedding"].isin(args.exclude_plms)].copy()
        log.info(
            f"--exclude-plms flag set. Filtered out {initial_count - len(results_df)} entries for: {args.exclude_plms}"
        )
        if results_df.empty:
            log.error(
                "DataFrame is empty after filtering excluded pLMs. Cannot generate plots."
            )
            return

    # --- Optional Filtering for Model Types ---
    if args.model_types:
        initial_count = len(results_df)
        results_df = results_df[results_df["Model Type"].isin(args.model_types)].copy()
        log.info(
            f"--model_types specified. Filtered to {len(results_df)} entries from {initial_count}. Keeping: {args.model_types}"
        )
        if results_df.empty:
            log.error(
                f"DataFrame is empty after filtering for model types: {args.model_types}. Cannot generate plots."
            )
            return

    # --- Add absolute Spearman correlation ---
    if "Spearman" in results_df.columns:
        results_df["Absolute Spearman"] = results_df["Spearman"].abs()
        log.info("Added 'Absolute Spearman' column to the dataset")

    # --- Define plots to generate ---
    plots_to_generate = [
        {"y": "Pearson R2", "se": "Pearson R2 SE", "suffix": "pearson_r2"},
        {"y": "Spearman", "se": "Spearman SE", "suffix": "spearman_rho"},
        {
            "y": "Absolute Spearman",
            "se": "Spearman SE",
            "suffix": "absolute_spearman_rho",
        },
        {"y": "MAE", "se": None, "suffix": "mae"},
        {"y": "R2", "se": None, "suffix": "r2"},
    ]

    # --- Generate Plots ---
    all_trendline_stats = []  # Collect all trendline stats across all plots

    for plot_config in plots_to_generate:
        y_metric = plot_config["y"]
        se_metric = plot_config["se"]
        suffix = plot_config["suffix"]

        # Check if metric column exists
        if y_metric not in results_df.columns:
            log.warning(f"Metric column '{y_metric}' not found in data. Skipping plot.")
            continue
        if se_metric and se_metric not in results_df.columns:
            log.warning(
                f"SE column '{se_metric}' for metric '{y_metric}' not found. Plotting without error bars."
            )
            se_metric = None  # Don't attempt to plot missing SE

        # Construct output path
        output_filename = f"{suffix}.png"
        output_path = output_dir / output_filename
        log.info(f"--- Generating plot for {y_metric} -> {output_path} ---")

        trend_stats = generate_metric_plot(results_df, y_metric, se_metric, output_path)

        # Add metric name to each stat and collect
        for stat in trend_stats:
            stat["metric"] = y_metric
            all_trendline_stats.append(stat)

    # --- Save Trendline Statistics to CSV ---
    if all_trendline_stats:
        trendline_csv_path = output_dir / "trendline_statistics.csv"
        trendline_df = pd.DataFrame(all_trendline_stats)

        # Reorder columns for better readability
        column_order = [
            "metric",
            "parameter",
            "model_type",
            "slope_per_1b",
            "r2",
            "p_value",
        ]
        trendline_df = trendline_df[column_order]

        # Format slope_per_1b to 2 significant figures
        trendline_df["slope_per_1b"] = trendline_df["slope_per_1b"].apply(
            lambda x: f"{x:.2g}"
        )

        try:
            trendline_df.to_csv(trendline_csv_path, index=False)
            log.info(f"Trendline statistics saved to {trendline_csv_path}")
        except Exception as e:
            log.error(
                f"Failed to save trendline statistics to {trendline_csv_path}: {e}",
                exc_info=True,
            )
    else:
        log.warning("No trendline statistics collected")

    log.info("--- Plot generation complete ---")


if __name__ == "__main__":
    main()
