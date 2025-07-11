import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
import matplotlib.lines as mlines
from typing import Dict, List, Optional

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
    # Add other embeddings here
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
    # Add other embeddings here
}

# Color map for individual embeddings (lowercase stem)
EMBEDDING_COLOR_MAP: Dict[str, str] = {
    "prott5": "#ff1493",
    "prottucker": "#7342e5",
    "prostt5": "#1217b5",
    "clean": "#4daf4a",
    "esm1b": "#5fd35b",
    "esm2_8m": "#fdae61",
    "esm2_35m": "#ff7f00",
    "esm2_150m": "#f46d43",
    "esm2_650m": "#d73027",
    "esm2_3b": "#a50026",
    "esmc_300m": "#17becf",
    "esmc_600m": "#1f77b4",
    "esm3_open": "#984ea3",
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
FAMILIES_TO_CONNECT: List[str] = ["ProtT5", "ESM-2", "ESM-C", "ESM-3", "Ankh", "ESM-1"]


# --- Data Parsing ---
def parse_metrics_file(filepath: Path) -> Dict[str, float]:
    """Parses a metrics file to extract key performance metrics."""
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

        # Check if any essential metrics are missing (optional, depending on requirements)
        # For now, we allow missing metrics, they will be NaN in the DataFrame
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

        # Add result to collection, merging dicts
        results.append(
            {
                "Model Type": model_type,
                "Parameter": param_name,
                "Embedding": embedding_name,
                "Embedding Key": embedding_key,
                "Embedding Family": EMBEDDING_FAMILY_MAP.get(embedding_key, "Unknown"),
                "PLM Size": PLM_SIZES.get(embedding_key),
                **metrics,  # Unpack the parsed metrics dictionary here
            }
        )

    # Create DataFrame and convert columns to numeric types
    results_df = pd.DataFrame(results)
    # Define all expected numeric columns
    # numeric_cols = [
    #     "Pearson R2",
    #     "Pearson R2 SE",
    #     "MAE",
    #     "Spearman",
    #     "Spearman SE",
    #     "R2",
    #     "PLM Size",
    # ]
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
    """Adds error bars for a given metric to the plot axes."""
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
    """Adds connecting lines for specified embedding families for a given metric."""
    # Group by family and model type. Order is determined by main df sort.
    for (family, model_type), group in data.groupby(["Embedding Family", "Model Type"]):
        if family in FAMILIES_TO_CONNECT:
            # No need to sort here if main df is sorted
            if len(group) > 1:
                line_color_key = group.iloc[0]["Embedding Key"]
                line_color = EMBEDDING_COLOR_MAP.get(line_color_key, "grey")
                ax.plot(
                    group["Embedding"],  # Use categorical x
                    group[y_metric],  # Use specified y metric
                    marker="",
                    linestyle="-",
                    color=line_color,
                    alpha=0.2,
                    zorder=2,
                )


def _create_embedding_legend(fig: plt.Figure, embeddings: List[str]) -> plt.legend:
    """Creates and returns the embedding legend (colors)."""
    handles = []
    labels = []
    log.debug(f"Creating embedding legend for: {sorted(embeddings)}")

    # Group embeddings by family, then sort within each family by size
    family_groups = {}
    for embedding_name in embeddings:
        key = embedding_name.lower()
        family = EMBEDDING_FAMILY_MAP.get(key, "Unknown")
        plm_size = PLM_SIZES.get(key, 0)  # Default to 0 if not found

        if family not in family_groups:
            family_groups[family] = []
        family_groups[family].append((embedding_name, plm_size))

    # Sort families alphabetically, and within each family sort by size
    for family in sorted(family_groups.keys()):
        # Sort embeddings within family by size, then by name for consistency
        family_embeddings = sorted(family_groups[family], key=lambda x: (x[1], x[0]))

        for embedding_name, _ in family_embeddings:
            key = embedding_name.lower()
            color = EMBEDDING_COLOR_MAP.get(key, "grey")
            # Use a square marker for the color legend instead of a patch
            handles.append(
                mlines.Line2D(
                    [], [], color=color, marker="s", linestyle="None", markersize=7
                )
            )
            # Use "Random" as label if key is "random_1024", else use original name
            label = "Random" if key == "random_1024" else embedding_name
            labels.append(label)

    return fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.35, 0.2),
        frameon=False,
        title="Embedding",
        ncol=6,  # Display in multiple columns for better space usage
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
        loc="upper center",
        bbox_to_anchor=(0.8, 0.2),
        frameon=False,
        title="Model Type",
        ncol=4,  # Display in multiple columns
    )


def _human_readable_formatter(x, pos=None):
    """Custom formatter for large numbers (B, M, K)"""
    # Input x is expected to be the numerical PLM size
    if x >= 1e9:
        return f"{x * 1e-9:.1f}B"
    elif x >= 1e6:
        return f"{x * 1e-6:.0f}M"
    elif x >= 1e3:
        return f"{x * 1e-3:.0f}K"
    else:
        # Handle 0 specifically for the baseline
        return f"{x:.0f}" if x != 0 else "0"


# --- Main Plotting Function ---
def generate_metric_plot(
    df: pd.DataFrame, y_metric: str, se_metric: Optional[str], output_file: Path
):
    """Generates a summary faceted scatter plot for a specific metric."""
    sns.set_theme(style="whitegrid")

    # --- Sort DataFrame by PLM Size for correct category order ---
    df_sorted = df.sort_values(by=["PLM Size", "Parameter", "Model Type"])
    category_order = df_sorted["Embedding"].unique().tolist()
    log.debug(f"Plotting '{y_metric}'. Using category order: {category_order}")

    # Convert Embedding column to categorical with the desired order
    df_sorted["Embedding"] = pd.Categorical(
        df_sorted["Embedding"], categories=category_order, ordered=True
    )

    # Create the base FacetGrid using relplot (scatter plot)
    g = sns.relplot(
        data=df_sorted,  # Use sorted data
        x="Embedding",  # Use Embedding name as x-axis category
        y=y_metric,  # Use the specified metric for y-axis
        col="Parameter",  # Facet by parameter
        hue="Embedding",  # Color by embedding (original name)
        style="Model Type",
        palette=EMBEDDING_COLOR_MAP,
        hue_order=sorted(df_sorted["Embedding"].unique()),
        style_order=sorted(df_sorted["Model Type"].unique()),
        markers=MODEL_MARKER_MAP,
        kind="scatter",
        s=100,
        height=6,  # Make plots taller/more rectangular
        aspect=0.8,  # Adjust aspect ratio for more rectangular shape
        facet_kws={"sharey": True, "sharex": False},
        legend=False,
        zorder=5,
    )

    # Add manual elements (error bars, lines) to each facet
    for param, ax in g.axes_dict.items():
        # Get data for the current facet, respecting the main sort order
        param_df = df_sorted[df_sorted["Parameter"] == param]

        _add_error_bars(ax, param_df, y_metric, se_metric)
        _add_connecting_lines(ax, param_df, y_metric)

        # --- Axis Formatting ---
        ax.set_xlabel("pLM Parameter Count")
        ax.set_ylabel(y_metric)  # Set label dynamically
        ax.set_title(f"Parameter: {param}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.tick_params(axis="x", rotation=45)

        # Set custom labels based on the category order and PLM sizes
        # category_order holds the embedding names in the correct plot order
        size_labels = []
        tick_positions = range(len(category_order))
        for emb_name in category_order:
            plm_size = PLM_SIZES.get(emb_name.lower())
            if plm_size is not None:
                size_labels.append(_human_readable_formatter(plm_size))
            else:
                size_labels.append(emb_name)  # Fallback to name if size missing
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(size_labels)

    # Create and place custom legends
    all_embeddings = df["Embedding"].unique().tolist()
    all_model_types = df["Model Type"].unique().tolist()
    _create_embedding_legend(g.figure, all_embeddings)
    _create_model_type_legend(g.figure, all_model_types)
    plt.tight_layout(rect=[0, 0.2, 1.0, 0.97])

    # Save the figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        log.info(f"Plot saved successfully to {output_file}")
    except Exception as e:
        log.error(f"Failed to save plot to {output_file}: {e}", exc_info=True)
    finally:
        plt.close(g.figure)


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
    args = parser.parse_args()

    # --- Load Data ---
    results_df = load_results_data(args.results_dir)
    if results_df.empty:
        log.error("No data loaded, exiting.")
        return

    # --- Save Full Dataframe ---
    output_dir = args.output  # Get output directory from args
    csv_filename = "parsed_metrics_all.csv"
    csv_output_path = output_dir / csv_filename
    try:
        # Ensure directory exists (redundant if plot generation runs, but safe)
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

    # --- Define plots to generate ---
    plots_to_generate = [
        {"y": "Pearson R2", "se": "Pearson R2 SE", "suffix": "pearson_r2"},
        {"y": "Spearman", "se": "Spearman SE", "suffix": "spearman_rho"},
        {"y": "MAE", "se": None, "suffix": "mae"},
        {"y": "R2", "se": None, "suffix": "r2"},
    ]

    # --- Generate Plots ---
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

        # Construct output filename within the specified directory
        output_filename = f"{suffix}.png"  # Filename based only on suffix
        output_path = output_dir / output_filename  # Join with output directory
        log.info(f"--- Generating plot for {y_metric} -> {output_path} ---")

        generate_metric_plot(results_df, y_metric, se_metric, output_path)

    log.info("--- Plot generation complete ---")


if __name__ == "__main__":
    main()
