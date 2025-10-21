#!/usr/bin/env python3
"""
Pairwise Embedding Comparison Visualization Script

This script generates comprehensive visualizations comparing different protein language model (PLM)
embeddings through various analytical approaches:

1. Hexagonal heatmap for distance comparisons across embedding pairs
2. Correlation heatmap with confidence intervals
3. Wasserstein distance heatmap between normalized distributions
4. Distribution comparison plots (both raw and normalized)
5. Violin plots for PLM distance differences

The script follows the project structure and uses consistent color schemes and styling
conventions from the project's visualization framework.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress matplotlib DEBUG messages
logging.getLogger("matplotlib").setLevel(logging.INFO)

# --- Project Constants & Configuration ---
# These match the constants from create_performance_summary_plots.py for consistency

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

# Color map for embedding families (consistent with create_performance_summary_plots.py)
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

EMBEDDING_DISPLAY_NAMES: Dict[str, str] = {
    "ankh_base": "Ankh Base",
    "ankh_large": "Ankh Large",
    "clean": "CLEAN",
    "esm1b": "ESM1b",
    "esm2_8m": "ESM2-8M",
    "esm2_35m": "ESM2-35M",
    "esm2_150m": "ESM2-150M",
    "esm2_650m": "ESM2-650M",
    "esm2_3b": "ESM2-3B",
    "esm3_open": "ESM3",
    "esmc_300m": "ESM C-300M",
    "esmc_600m": "ESM C-600M",
    "prostt5": "ProstT5",
    "prott5": "ProtT5",
    "prottucker": "ProtTucker",
    "random_1024": "Random",
}

DEFAULT_STYLE = {
    "figure_size": (15, 12),
    "dpi": 300,
    "font_scale": 1.0,
    "title_size": 16,
    "label_size": 12,
    "tick_size": 10,
    "legend_size": 10,
}


class EmbeddingComparisonVisualizer:
    """
    A comprehensive visualizer for comparing protein language model embeddings.

    Provides methods for various types of embedding comparison visualizations,
    including distance heatmaps, correlation analyses, and distribution comparisons.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        output_dir: Union[str, Path],
        sample_size: Optional[int] = None,
        font_scale: float = 1.0,
    ):
        """
        Initialize the visualizer.

        Args:
            data_path: Path to CSV file or pandas DataFrame containing the data
            output_dir: Directory where output files will be saved
            sample_size: Optional limit on number of rows to process
            font_scale: Scaling factor for all font sizes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.font_scale = font_scale

        # Load and process data
        self.df = self._load_data(data_path)
        self.dist_cols = self._identify_distance_columns()

        # Set up matplotlib styling
        self._setup_plotting_style()

    def _load_data(self, data_path: Union[str, Path]) -> pl.DataFrame:
        """Load data from various sources, returning polars DataFrame."""
        data_path = Path(data_path)
        logger.info(f"Loading data from {data_path}")

        # Support both CSV and Parquet files
        if data_path.suffix.lower() == ".parquet":
            df = pl.read_parquet(data_path)
            logger.info(f"Loaded Parquet file with {len(df)} rows")
        else:
            df = pl.read_csv(data_path)
            logger.info(f"Loaded CSV file with {len(df)} rows")

        if self.sample_size:
            df = df.head(self.sample_size)
            logger.info(f"Limited dataset to {len(df)} rows")

        return df

    def _identify_distance_columns(self) -> List[str]:
        """Identify and validate distance columns in the dataset, excluding random embeddings."""
        # Get all distance columns
        all_dist_cols = [
            col
            for col in self.df.columns
            if col.startswith("dist_") and not col.startswith("pca_")
        ]

        # Filter out random embeddings
        dist_cols = [
            col
            for col in all_dist_cols
            if not col.replace("dist_", "").lower().startswith("random")
        ]

        # Sort by PLM family, then by size within family (same as create_performance_summary_plots.py)
        def get_sort_key(col: str) -> tuple:
            embedding_name = col.replace("dist_", "").lower()
            family = EMBEDDING_FAMILY_MAP.get(embedding_name, "Unknown")
            plm_size = PLM_SIZES.get(embedding_name, 0)
            return (
                family,
                plm_size,
                embedding_name,
            )  # Sort by family, then size, then name

        dist_cols = sorted(dist_cols, key=get_sort_key)

        if not dist_cols:
            raise ValueError(
                "No valid distance columns found. Columns should start with 'dist_' and not be random embeddings"
            )

        filtered_count = len(all_dist_cols) - len(dist_cols)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} random embedding columns")

        logger.info(
            f"Found {len(dist_cols)} distance columns (sorted by PLM family, then size): {dist_cols}"
        )
        return dist_cols

    def _setup_plotting_style(self):
        """Configure matplotlib plotting parameters."""
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.size": DEFAULT_STYLE["font_scale"] * self.font_scale,
                "axes.titlesize": DEFAULT_STYLE["title_size"] * self.font_scale,
                "axes.labelsize": DEFAULT_STYLE["label_size"] * self.font_scale,
                "xtick.labelsize": DEFAULT_STYLE["tick_size"] * self.font_scale,
                "ytick.labelsize": DEFAULT_STYLE["tick_size"] * self.font_scale,
                "legend.fontsize": DEFAULT_STYLE["legend_size"] * self.font_scale,
                "figure.dpi": DEFAULT_STYLE["dpi"],
            }
        )

    def _get_embedding_color(self, dist_col: str) -> str:
        """Get color for a distance column based on embedding name."""
        embedding_name = dist_col.replace("dist_", "").lower()
        return EMBEDDING_COLOR_MAP.get(embedding_name, "#808080")

    def _save_json_data(self, data: Dict, save_path: Path, description: str):
        """Helper method to save JSON data with consistent logging."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(data, f)
        logger.info(f"{description} saved to {save_path}")

    def _load_cached_data(
        self, cache_path: Path, force_recompute: bool
    ) -> Optional[Dict]:
        """Helper method to load cached data if available."""
        if not force_recompute and cache_path.exists():
            with open(cache_path, "r") as f:
                return json.load(f)
        return None

    # --- Hexagonal Distance Comparison ---

    def compute_hexbin_data(
        self, gridsize: int = 50, save_path: Optional[Path] = None
    ) -> Dict:
        """Pre-compute hexbin data for distance comparisons."""
        logger.info("Computing hexbin data for distance comparisons...")

        hexbin_data = {
            "metadata": {
                "dist_cols": self.dist_cols,
                "gridsize": gridsize,
                "max_count": 0,
            }
        }

        n = len(self.dist_cols)
        total_pairs = n * (n - 1)

        with tqdm(total=total_pairs, desc="Computing hexbin data") as pbar:
            for i, col1 in enumerate(self.dist_cols):
                for j, col2 in enumerate(self.dist_cols):
                    if i == j:
                        continue

                    mask = ~(self.df[col1].is_nan() | self.df[col2].is_nan())
                    if mask.sum() < 10:
                        pbar.update(1)
                        continue

                    filtered_df = self.df.filter(mask)
                    x_data = filtered_df[col1].to_numpy()
                    y_data = filtered_df[col2].to_numpy()

                    counts, xedges, yedges = np.histogram2d(
                        x_data, y_data, bins=gridsize
                    )

                    max_count = counts.max()
                    if max_count > hexbin_data["metadata"]["max_count"]:
                        hexbin_data["metadata"]["max_count"] = max_count

                    hexbin_data[f"{col1}_vs_{col2}"] = {
                        "counts": counts.tolist(),
                        "xedges": xedges.tolist(),
                        "yedges": yedges.tolist(),
                    }
                    pbar.update(1)

        if save_path:
            self._save_json_data(hexbin_data, save_path, "Hexbin data")

        return hexbin_data

    def plot_hexagonal_distance_comparison(
        self,
        hexbin_data: Optional[Dict] = None,
        gridsize: int = 50,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Create a grid of hexbin plots showing distance comparisons between embeddings."""
        if hexbin_data is None:
            hexbin_data = self.compute_hexbin_data(gridsize)

        logger.info("Creating hexagonal distance comparison plot...")

        dist_cols = hexbin_data["metadata"]["dist_cols"]
        n = len(dist_cols)
        vmax = hexbin_data["metadata"]["max_count"]

        fig, axes = plt.subplots(
            n, n, figsize=(20 * self.font_scale, 18 * self.font_scale)
        )
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        with tqdm(total=n * n, desc="Creating hexagonal plots") as pbar:
            for i, col1 in enumerate(dist_cols):
                for j, col2 in enumerate(dist_cols):
                    ax = axes[i, j]

                    if i == j:
                        # Diagonal: show embedding name
                        embedding_name = col1.replace("dist_", "")
                        embedding_name = embedding_name.replace("_", "\n")
                        if embedding_name == "prottucker":
                            embedding_name = "prot-\ntucker"
                        ax.text(
                            0.5,
                            0.5,
                            embedding_name,
                            ha="center",
                            va="center",
                            # rotation=45,
                            fontsize=16 * self.font_scale,
                            weight="bold",
                            transform=ax.transAxes,
                        )
                        ax.axis("off")
                    elif i < j:
                        # Upper triangle: turn off (show only lower triangle)
                        ax.axis("off")
                    else:
                        # Lower triangle: show hexbin plots
                        key = f"{col1}_vs_{col2}"
                        if key in hexbin_data:
                            self._plot_hexbin_pair(ax, hexbin_data[key], vmax)
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                "No Data",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.axis("off")

                    # Add labels for edge plots
                    if i == n - 1:
                        ax.set_xlabel(
                            col2.replace("dist_", ""),
                            rotation=45,
                            ha="right",
                            fontsize=16 * self.font_scale,
                        )
                    if j == 0:
                        ax.set_ylabel(
                            col1.replace("dist_", ""),
                            rotation=0,
                            ha="right",
                            fontsize=16 * self.font_scale,
                        )

                    pbar.update(1)

        # Add title and colorbar
        fig.suptitle(
            "PairwiseDistance Comparison",
            x=0.5,
            y=0.91,
            fontsize=22 * self.font_scale,
            weight="bold",
        )

        # Create a dummy mappable for colorbar
        im = plt.cm.ScalarMappable(cmap="pink", norm=plt.Normalize(vmin=1, vmax=vmax))
        cbar = fig.colorbar(
            im, ax=axes.ravel().tolist(), label="Count", shrink=1.0, aspect=30, pad=0.02
        )
        cbar.ax.tick_params(labelsize=14 * self.font_scale)
        cbar.set_label("Count", fontsize=16 * self.font_scale)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Hexagonal comparison plot saved to {save_path}")

        return fig, axes

    def _plot_hexbin_pair(self, ax: plt.Axes, data: Dict, vmax: float):
        """Plot a single hexbin pair on the given axes."""
        counts = np.array(data["counts"])
        xedges = np.array(data["xedges"])
        yedges = np.array(data["yedges"])

        X, Y = np.meshgrid(
            xedges[:-1] + np.diff(xedges) / 2, yedges[:-1] + np.diff(yedges) / 2
        )

        masked_counts = np.ma.masked_where(counts == 0, counts)
        ax.pcolormesh(X, Y, masked_counts.T, cmap="pink", vmin=1, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Correlation Analysis ---

    def compute_correlation_data(self, save_path: Optional[Path] = None) -> Dict:
        """Pre-compute Spearman correlations and confidence intervals for distance columns."""
        logger.info("Computing correlation data...")

        n = len(self.dist_cols)
        correlations = np.full((n, n), np.nan)
        ci_lower = np.full((n, n), np.nan)
        ci_upper = np.full((n, n), np.nan)

        with tqdm(total=(n * (n + 1)) // 2, desc="Calculating correlations") as pbar:
            for i in range(n):
                for j in range(i, n):
                    mask = ~(
                        self.df[self.dist_cols[i]].is_nan()
                        | self.df[self.dist_cols[j]].is_nan()
                    )
                    if mask.sum() > 3:
                        filtered_df = self.df.filter(mask)
                        correlation, _ = stats.spearmanr(
                            filtered_df[self.dist_cols[i]],
                            filtered_df[self.dist_cols[j]],
                        )
                        correlations[i, j] = correlations[j, i] = correlation

                        # Compute confidence intervals
                        if abs(correlation) > 0.9999:
                            ci_lower[i, j] = ci_lower[j, i] = correlation
                            ci_upper[i, j] = ci_upper[j, i] = correlation
                        else:
                            z = np.arctanh(correlation)
                            sigma = 1.0 / np.sqrt(mask.sum() - 3)
                            z_ci = stats.norm.interval(0.95, loc=z, scale=sigma)
                            ci = np.tanh(z_ci)
                            ci_lower[i, j] = ci_lower[j, i] = ci[0]
                            ci_upper[i, j] = ci_upper[j, i] = ci[1]

                    pbar.update(1)

        data = {
            "correlations": correlations.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
            "columns": [col.replace("dist_", "") for col in self.dist_cols],
        }

        if save_path:
            self._save_json_data(data, save_path, "Correlation data")

        return data

    def plot_correlation_heatmap(
        self,
        correlation_data: Optional[Dict] = None,
        show_ci: bool = False,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a correlation heatmap from pre-computed correlation data."""
        if correlation_data is None:
            correlation_data = self.compute_correlation_data()

        logger.info("Creating correlation heatmap...")

        correlations = np.array(correlation_data["correlations"])
        columns = correlation_data["columns"]
        n = len(columns)

        fig, ax = plt.subplots(figsize=(15 * self.font_scale, 12 * self.font_scale))

        masked_correlations = np.ma.array(
            correlations, mask=np.isnan(correlations) | np.eye(n, dtype=bool)
        )
        im = ax.imshow(masked_correlations, cmap="OrRd", vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, pad=0.02, aspect=50)
        cbar.ax.tick_params(labelsize=12 * self.font_scale)
        cbar.set_label("Spearman Correlation", fontsize=14 * self.font_scale)

        # Add annotations
        self._add_correlation_annotations(ax, correlation_data, show_ci, n)

        # Customize ticks and labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(
            columns, rotation=45, ha="right", fontsize=14 * self.font_scale
        )
        ax.set_yticklabels(columns, fontsize=14 * self.font_scale)

        title = "Spearman Correlations"
        if show_ci:
            title += " with 95% CI"
        plt.title(title, pad=20, fontsize=16 * self.font_scale, weight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Correlation heatmap saved to {save_path}")

        return fig, ax

    def _add_correlation_annotations(
        self, ax: plt.Axes, data: Dict, show_ci: bool, n: int
    ):
        """Add correlation annotations to the heatmap."""
        correlations = np.array(data["correlations"])
        columns = data["columns"]

        if show_ci:
            ci_lower = np.array(data["ci_lower"])
            ci_upper = np.array(data["ci_upper"])

        with tqdm(total=n * n, desc="Adding correlation annotations") as pbar:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        # Diagonal: show embedding name
                        embedding_name = columns[i].replace("_", "\n")
                        if embedding_name == "prottucker":
                            embedding_name = "prot-\ntucker"
                        elif embedding_name == "prostt5":
                            embedding_name = "prost-\nt5"
                        elif embedding_name == "prott5":
                            embedding_name = "prot-\nt5"
                        ax.text(
                            j,
                            i,
                            embedding_name,
                            ha="center",
                            va="center",
                            fontsize=14 * self.font_scale,
                            weight="bold",
                            color=self._get_embedding_color(columns[i]),
                        )
                    elif not np.isnan(correlations[i, j]):
                        if show_ci:
                            text = f"{correlations[i, j]:.2f}\n[{ci_lower[i, j]:.2f}, {ci_upper[i, j]:.2f}]"
                        else:
                            text = f"{correlations[i, j]:.2f}"

                        color = "white" if abs(correlations[i, j]) > 0.5 else "black"
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=16 * self.font_scale,
                        )
                    else:
                        ax.text(
                            j,
                            i,
                            "NA",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=16 * self.font_scale,
                        )
                    pbar.update(1)

    # --- Wasserstein Distance Analysis ---

    @staticmethod
    def normalize_distribution(x: pl.Series) -> np.ndarray:
        """Normalize a distribution to [0,1] range using MinMax scaling."""
        # Drop nulls and convert to numpy for compatibility with existing code
        x_clean = x.drop_nulls().to_numpy()
        if len(x_clean) == 0:
            return x_clean

        # Additional check for infinite values
        x_clean = x_clean[np.isfinite(x_clean)]
        if len(x_clean) == 0:
            return x_clean

        # Check if all values are the same (would cause division by zero)
        if np.all(x_clean == x_clean[0]):
            # Return array of 0.5s (middle of [0,1] range)
            return np.full_like(x_clean, 0.5)

        scaler = MinMaxScaler()
        return scaler.fit_transform(x_clean.reshape(-1, 1)).ravel()

    def _compute_wasserstein_pair(self, col1: str, col2: str) -> Tuple[float, int]:
        """Compute Wasserstein distance between two columns."""
        # Create a mask for rows where both columns have valid values
        mask = ~(self.df[col1].is_nan() | self.df[col2].is_nan())
        valid_df = self.df.filter(mask)

        if len(valid_df) < 10:
            return np.nan, len(valid_df)

        # Normalize the distributions using only the valid data
        dist1_normalized = self.normalize_distribution(valid_df[col1])
        dist2_normalized = self.normalize_distribution(valid_df[col2])

        if len(dist1_normalized) == 0 or len(dist2_normalized) == 0:
            return np.nan, len(valid_df)

        try:
            dist = wasserstein_distance(dist1_normalized, dist2_normalized)
            logger.debug(
                f"{col1} vs {col2}: {len(valid_df)} samples, distance = {dist:.4f}"
            )
            return dist, len(valid_df)
        except Exception as e:
            logger.warning(
                f"Error computing Wasserstein distance for {col1} vs {col2}: {e}"
            )
            return np.nan, len(valid_df)

    def compute_wasserstein_data(self, save_path: Optional[Path] = None) -> Dict:
        """Compute Wasserstein distances between all pairs of normalized distance distributions."""
        logger.info("Computing Wasserstein distances...")

        n = len(self.dist_cols)
        distances = np.zeros((n, n))

        # Compute pairwise distances
        with tqdm(total=(n * (n + 1)) // 2, desc="Computing distances") as pbar:
            for i in range(n):
                for j in range(i, n):
                    col1 = self.dist_cols[i]
                    col2 = self.dist_cols[j]

                    dist, sample_count = self._compute_wasserstein_pair(col1, col2)
                    distances[i, j] = distances[j, i] = dist

                    if np.isnan(dist) and sample_count < 10:
                        logger.warning(
                            f"Insufficient valid samples for {col1} vs {col2}: {sample_count} samples"
                        )

                    pbar.update(1)

        data = {
            "distances": distances.tolist(),
            "columns": [col.replace("dist_", "") for col in self.dist_cols],
        }

        if save_path:
            self._save_json_data(data, save_path, "Wasserstein data")

        return data

    def plot_wasserstein_heatmap(
        self,
        wasserstein_data: Optional[Dict] = None,
        show_values: bool = True,
        cmap: str = "Blues",
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a heatmap from pre-computed Wasserstein distance data."""
        if wasserstein_data is None:
            wasserstein_data = self.compute_wasserstein_data()

        logger.info("Creating Wasserstein distance heatmap...")

        distances = np.array(wasserstein_data["distances"])
        columns = wasserstein_data["columns"]
        n = len(columns)

        fig, ax = plt.subplots(figsize=(15 * self.font_scale, 12 * self.font_scale))

        masked_distances = np.ma.array(distances, mask=np.isnan(distances))
        im = ax.imshow(masked_distances, cmap=cmap)

        # Add colorbar
        cbar = plt.colorbar(im, pad=0.02, aspect=50)
        cbar.ax.tick_params(labelsize=10 * self.font_scale)
        cbar.set_label("Normalized Wasserstein Distance", fontsize=12 * self.font_scale)

        # Add annotations
        if show_values:
            self._add_wasserstein_annotations(ax, distances, columns, cmap, n)

        # Customize ticks and labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(
            columns, rotation=45, ha="right", fontsize=10 * self.font_scale
        )
        ax.set_yticklabels(columns, fontsize=10 * self.font_scale)

        plt.title(
            "Normalized Wasserstein Distances\nBetween Distance Embeddings",
            pad=20,
            fontsize=14 * self.font_scale,
            weight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Wasserstein heatmap saved to {save_path}")

        return fig, ax

    def _add_wasserstein_annotations(
        self, ax: plt.Axes, distances: np.ndarray, columns: List[str], cmap: str, n: int
    ):
        """Add annotations to Wasserstein heatmap with optimized text colors."""
        colormap = plt.colormaps[cmap]

        with tqdm(total=n * n, desc="Adding Wasserstein annotations") as pbar:
            for i in range(n):
                for j in range(n):
                    if not np.isnan(distances[i, j]):
                        if i == j:
                            text = columns[i]
                            weight = "bold"
                        else:
                            text = f"{distances[i, j]:.2f}"
                            weight = "normal"

                        # Optimize text color based on background
                        normalized_value = (distances[i, j] - np.nanmin(distances)) / (
                            np.nanmax(distances) - np.nanmin(distances)
                        )
                        rgba = colormap(normalized_value)
                        brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        color = "white" if brightness < 0.6 else "black"

                        rotation = 45 if i == j else 0
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            rotation=rotation,
                            color=color,
                            fontsize=8 * self.font_scale,
                            weight=weight,
                        )
                    else:
                        ax.text(
                            j,
                            i,
                            "NA",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=8 * self.font_scale,
                        )

    # --- Distribution Analysis ---

    def plot_ridge_distributions(
        self,
        distribution_data: Optional[Dict] = None,
        show_median: bool = True,
        alpha: float = 0.8,
        save_path: Optional[Path] = None,
        ranking_csv: Optional[Path] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a ridge plot using pre-computed distribution data for much faster rendering."""
        if distribution_data is None:
            distribution_data = self.compute_distribution_data(normalize=True)

        logger.info(
            "Creating optimized ridge plot using pre-computed distribution data..."
        )

        # Extract PLM names and optionally sort by ranking
        plm_names = [col.replace("dist_", "") for col in self.dist_cols]

        # Sort by performance ranking if provided
        if ranking_csv is not None and ranking_csv.exists():
            logger.info(f"Loading PLM ranking from {ranking_csv}")
            ranking_df = pl.read_csv(ranking_csv)
            # Create mapping from embedding name to rank
            rank_map = {
                row["Embedding"]: row["Final_Rank"]
                for row in ranking_df.iter_rows(named=True)
            }
            # Sort plm_names by rank (lower rank = better, so put at top)
            plm_names = sorted(
                plm_names,
                key=lambda x: rank_map.get(x, 999),  # Unranked goes to bottom
                reverse=False,  # Best (rank 1) is at top
            )
            logger.info("Sorted PLMs by performance ranking (best at top)")
        else:
            logger.info("No ranking CSV provided, using default order")

        # Set style for ridge plot
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Extract or estimate percentile values from pre-computed distribution data
        percentile_data = {}
        if show_median:
            logger.info(
                "Extracting/estimating percentiles from precomputed distribution data..."
            )
            for col in self.dist_cols:
                plm_name = col.replace("dist_", "")
                if col in distribution_data["distributions"]:
                    dist_data = distribution_data["distributions"][col]

                    # First try to get precomputed median
                    median_val = dist_data.get("median")

                    # Estimate percentiles from KDE
                    x_range = np.array(dist_data["x_range"])
                    density = np.array(dist_data["density"])

                    if len(x_range) > 1 and len(density) > 1:
                        # Normalize density to integrate to 1
                        dx = x_range[1] - x_range[0]
                        density_normalized = density / (np.sum(density) * dx)

                        # Compute cumulative distribution
                        cumulative = np.cumsum(density_normalized) * dx

                        # Estimate percentiles
                        percentiles = {}
                        for p_name, p_val in [
                            ("q25", 0.25),
                            ("median", 0.5),
                            ("q75", 0.75),
                        ]:
                            # Use precomputed median if available
                            if (
                                p_name == "median"
                                and median_val is not None
                                and not np.isnan(median_val)
                            ):
                                percentiles[p_name] = median_val
                            else:
                                p_idx = np.searchsorted(cumulative, p_val)
                                if p_idx < len(x_range):
                                    percentiles[p_name] = float(x_range[p_idx])
                                else:
                                    percentiles[p_name] = None

                        if all(v is not None for v in percentiles.values()):
                            percentile_data[plm_name] = percentiles
                            logger.debug(
                                f"Estimated percentiles for {col}: "
                                f"Q25={percentiles['q25']:.4f}, "
                                f"Median={percentiles['median']:.4f}, "
                                f"Q75={percentiles['q75']:.4f}"
                            )
                        else:
                            logger.debug(
                                f"Could not estimate all percentiles for {col}"
                            )
                    else:
                        logger.warning(
                            f"Insufficient data to estimate percentiles for {col}"
                        )
                else:
                    logger.warning(f"No distribution data found for {col}")

        # Set up the figure - use a single large figure and manually position subplots
        fig = plt.figure(
            figsize=(15 * self.font_scale, len(plm_names) * self.font_scale)
        )

        # Create subplots with controlled spacing
        gs = fig.add_gridspec(len(plm_names), 1, hspace=-0.25)  # Small positive spacing

        axes = []
        for i, plm_name in enumerate(plm_names):
            # Create subplot
            ax = fig.add_subplot(gs[i])
            axes.append(ax)

            col = f"dist_{plm_name}"

            if col not in distribution_data["distributions"]:
                logger.warning(f"No distribution data found for {col}")
                ax.axis("off")
                continue

            dist_data = distribution_data["distributions"][col]
            x_range = np.array(dist_data["x_range"])
            density = np.array(dist_data["density"])
            color = self._get_embedding_color(col)

            # Plot the distribution - key insight: plot normally, let overlap create ridge effect
            ax.fill_between(x_range, density, alpha=alpha, color=color)
            ax.plot(x_range, density, color="white", linewidth=2, zorder=2)

            # Draw baseline at y=0
            ax.axhline(y=0, color="black", linewidth=2, clip_on=False)

            # Add percentile lines if requested
            if show_median and plm_name in percentile_data:
                percentiles = percentile_data[plm_name]

                # Define line styles for each percentile
                percentile_styles = {
                    "q25": {
                        "color": "0.3",
                        "linestyle": ":",
                        "linewidth": 3,
                        "alpha": 0.7,
                    },
                    "median": {
                        "color": "black",
                        "linestyle": "--",
                        "linewidth": 3,
                        "alpha": 0.8,
                    },
                    "q75": {
                        "color": "0.3",
                        "linestyle": ":",
                        "linewidth": 3,
                        "alpha": 0.7,
                    },
                }

                for p_name, p_val in percentiles.items():
                    if p_val is not None and not np.isnan(p_val) and 0 <= p_val <= 1:
                        # Interpolate to find the height at this percentile
                        p_y = np.interp(p_val, x_range, density)
                        # Only draw if interpolated y is valid
                        if not np.isnan(p_y) and p_y >= 0:
                            style = percentile_styles[p_name]
                            ax.plot(
                                [p_val, p_val],
                                [0, p_y],
                                color=style["color"],
                                linestyle=style["linestyle"],
                                linewidth=style["linewidth"],
                                alpha=style["alpha"],
                                clip_on=False,
                                zorder=1,
                            )

            # Add PLM label on the left
            plm_display_name = EMBEDDING_DISPLAY_NAMES.get(plm_name, plm_name)
            ax.text(
                -0.02,
                0.1,
                plm_display_name,
                fontweight="bold",
                color=color,
                ha="right",
                va="center",
                transform=ax.transAxes,
                fontsize=int(26 * self.font_scale),
            )

            # Style the subplot
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)  # Start y-axis at 0

            # Remove y-axis elements
            ax.set_yticks([])
            ax.set_ylabel("")

            # Remove unnecessary spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # X-axis: only show on bottom subplot
            if i == len(plm_names) - 1:
                # Bottom subplot: full x-axis
                ax.spines["bottom"].set_visible(True)
                ax.set_xticks(np.arange(0, 1.1, 0.2))
                ax.set_xlabel(
                    "Min-Max Normalized Distances",
                    fontsize=int(28 * self.font_scale),
                    labelpad=15,
                )
                ax.tick_params(
                    axis="x", which="major", labelsize=int(20 * self.font_scale)
                )
            else:
                # Other subplots: no x-axis
                ax.spines["bottom"].set_visible(False)
                ax.set_xticks([])
                ax.tick_params(axis="x", which="both", length=0, labelbottom=False)

        # Add y-axis label on the left side (figure-level)
        fig.text(
            -0.075,  # x position - far left
            0.5,  # y position - middle
            "pLM Models",
            fontsize=int(28 * self.font_scale),
            va="center",
            ha="left",
            rotation=90,
        )

        # Add overall title
        fig.suptitle(
            "Normalized Pairwise All-vs-All Distance Distributions",
            fontsize=int(32 * self.font_scale),
            fontweight="bold",
            y=0.92,
        )

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Optimized ridge plot saved to {save_path}")

            # Save percentile statistics to CSV
            if percentile_data:
                stats_path = save_path.parent / f"{save_path.stem}_statistics.csv"

                # Convert nested dict to flat structure for CSV
                rows = []
                for plm_name, percentiles in percentile_data.items():
                    plm_display_name = EMBEDDING_DISPLAY_NAMES.get(plm_name, plm_name)
                    rows.append(
                        {
                            "plm_name": plm_display_name,
                            "q25": percentiles["q25"],
                            "median": percentiles["median"],
                            "q75": percentiles["q75"],
                        }
                    )

                # Create DataFrame and save to CSV
                stats_df = pl.DataFrame(rows)
                stats_df.write_csv(stats_path)
                logger.info(f"Ridge plot percentile statistics saved to {stats_path}")

        return fig, axes

    def compute_distribution_data(
        self, normalize: bool = False, save_path: Optional[Path] = None
    ) -> Dict:
        """Pre-compute distribution data for plotting."""
        logger.info(f"Computing distribution data (normalize={normalize})...")

        distribution_data = {"metadata": {"normalized": normalize}, "distributions": {}}

        for col in tqdm(self.dist_cols, desc="Processing distributions"):
            # Get data using polars operations
            col_series = self.df.select(col).drop_nulls().get_column(col)
            data = col_series.to_numpy()

            if normalize:
                # Use normalized data for KDE
                data_normalized = self.normalize_distribution(col_series)
                data_clean = (
                    data_normalized[np.isfinite(data_normalized)]
                    if len(data_normalized) > 0
                    else np.array([])
                )
                x_range = np.linspace(0, 1, 500)
            else:
                # Use original data for KDE
                data_clean = data[np.isfinite(data)] if len(data) > 0 else np.array([])
                if len(data_clean) > 0:
                    x_range = np.linspace(data_clean.min(), data_clean.max(), 500)
                else:
                    x_range = np.linspace(0, 1, 500)

            if len(data_clean) > 1:  # Need at least 2 points for KDE
                kernel = stats.gaussian_kde(data_clean)
                density = kernel(x_range)
                peak_idx = np.argmax(density)
                peak_x = float(x_range[peak_idx])
                peak_y = float(density[peak_idx])
            else:
                density = np.zeros_like(x_range)
                peak_x = peak_y = 0.0

            # Compute median for normalized data
            if normalize and len(data_clean) > 0:
                median_val = float(np.median(data_clean))
            elif not normalize and len(data_clean) > 0:
                # For non-normalized, we still compute it but it won't be used in ridge plots
                median_val = float(np.median(data_clean))
            else:
                median_val = None

            distribution_data["distributions"][col] = {
                "x_range": x_range.tolist(),
                "density": density.tolist(),
                "peak_x": peak_x,
                "peak_y": peak_y,
                "min": float(data_clean.min()) if len(data_clean) > 0 else 0.0,
                "max": float(data_clean.max()) if len(data_clean) > 0 else 0.0,
                "median": median_val,
            }

        if save_path:
            self._save_json_data(distribution_data, save_path, "Distribution data")

        return distribution_data

    def plot_distributions(
        self,
        distribution_data: Optional[Dict] = None,
        normalize: bool = False,
        alpha: float = 0.05,
        linewidth: float = 2.5,
        show_peaks: bool = True,
        y_break: Optional[Tuple[float, float]] = None,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]]:
        """Create distribution plots from pre-computed data."""
        if distribution_data is None:
            distribution_data = self.compute_distribution_data(normalize)

        logger.info("Creating distribution comparison plot...")

        color_dict = {col: self._get_embedding_color(col) for col in self.dist_cols}

        # Font sizes
        font_sizes = {
            "title": int(24 * self.font_scale),
            "axis": int(22 * self.font_scale),
            "legend": int(16 * self.font_scale),
            "tick": int(14 * self.font_scale),
        }

        figsize = (12 * self.font_scale, 10 * self.font_scale)

        if y_break is None:
            fig, ax = self._create_single_distribution_plot(
                distribution_data,
                color_dict,
                alpha,
                linewidth,
                show_peaks,
                font_sizes,
                figsize,
            )
            axes = ax
        else:
            fig, (ax1, ax2) = self._create_broken_distribution_plot(
                distribution_data,
                color_dict,
                alpha,
                linewidth,
                show_peaks,
                y_break,
                font_sizes,
                figsize,
            )
            axes = (ax1, ax2)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Distribution plot saved to {save_path}")

        return fig, axes

    def _create_single_distribution_plot(
        self,
        distribution_data: Dict,
        color_dict: Dict,
        alpha: float,
        linewidth: float,
        show_peaks: bool,
        font_sizes: Dict,
        figsize: Tuple,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a single distribution plot."""
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Collect peak data for smart positioning
        peak_data = []

        for dist_name, color in tqdm(color_dict.items(), desc="Plotting distributions"):
            if dist_name not in distribution_data["distributions"]:
                continue

            dist_data = distribution_data["distributions"][dist_name]
            x_range = np.array(dist_data["x_range"])
            density = np.array(dist_data["density"])

            ax.fill_between(x_range, density, alpha=alpha, color=color)
            ax.plot(
                x_range,
                density,
                color=color,
                linewidth=linewidth,
                label=dist_name.replace("dist_", ""),
            )

            if show_peaks:
                peak_data.append(
                    (
                        dist_data["peak_x"],
                        dist_data["peak_y"],
                        dist_name.replace("dist_", ""),
                        color,
                    )
                )

        # Add peak labels with smart positioning
        if show_peaks:
            self._add_peak_labels_smart(ax, peak_data)

        self._customize_distribution_plot(ax, distribution_data, font_sizes)

        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fontsize=font_sizes["legend"],
            markerscale=2.0,  # Make legend markers 2x larger
        )
        fig.subplots_adjust(right=0.85)

        return fig, ax

    def _create_broken_distribution_plot(
        self,
        distribution_data: Dict,
        color_dict: Dict,
        alpha: float,
        linewidth: float,
        show_peaks: bool,
        y_break: Tuple[float, float],
        font_sizes: Dict,
        figsize: Tuple,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """Create a distribution plot with broken y-axis."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.08)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        fig.patch.set_facecolor("white")
        ax1.set_facecolor("white")
        ax2.set_facecolor("white")

        # Collect peak data for smart positioning
        peak_data_ax1 = []
        peak_data_ax2 = []

        # Plot on both axes
        for dist_name, color in tqdm(color_dict.items(), desc="Plotting distributions"):
            if dist_name not in distribution_data["distributions"]:
                continue

            dist_data = distribution_data["distributions"][dist_name]
            x_range = np.array(dist_data["x_range"])
            density = np.array(dist_data["density"])

            for ax in [ax1, ax2]:
                ax.fill_between(x_range, density, alpha=alpha, color=color)
                ax.plot(
                    x_range,
                    density,
                    color=color,
                    linewidth=linewidth,
                    label=dist_name.replace("dist_", ""),
                )

            if show_peaks:
                if dist_data["peak_y"] > y_break[1]:
                    peak_data_ax1.append(
                        (
                            dist_data["peak_x"],
                            dist_data["peak_y"],
                            dist_name.replace("dist_", ""),
                            color,
                        )
                    )
                elif dist_data["peak_y"] < y_break[0]:
                    peak_data_ax2.append(
                        (
                            dist_data["peak_x"],
                            dist_data["peak_y"],
                            dist_name.replace("dist_", ""),
                            color,
                        )
                    )

        # Add peak labels with smart positioning
        if show_peaks:
            self._add_peak_labels_smart(ax1, peak_data_ax1)
            self._add_peak_labels_smart(ax2, peak_data_ax2)

        # Set y-limits and add broken axis marks
        ax1.set_ylim(y_break[1], None)
        ax2.set_ylim(0, y_break[0])
        self._add_broken_axis_marks(ax1, ax2)

        # Customize plots
        ax1.set_xticklabels([])
        ax1.tick_params(axis="x", which="both", length=0)

        is_normalized = distribution_data["metadata"]["normalized"]
        xlabel = "Normalized Distance" if is_normalized else "Distance"
        title = (
            "Distribution Comparison of\n"
            + ("Normalized " if is_normalized else "")
            + "Distance Embeddings"
        )

        ax2.set_xlabel(xlabel, fontsize=font_sizes["axis"], labelpad=15)
        ax1.set_title(title, fontsize=font_sizes["title"], pad=20, weight="bold")
        ax2.set_ylabel("Density", fontsize=font_sizes["axis"], labelpad=15)

        for ax in [ax1, ax2]:
            ax.tick_params(axis="both", which="major", labelsize=font_sizes["tick"])
            ax.grid(True, alpha=0.2)

        ax1.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fontsize=font_sizes["legend"],
            markerscale=2.0,  # Make legend markers 2x larger
        )
        fig.subplots_adjust(right=0.85)

        return fig, (ax1, ax2)

    def _add_peak_labels_smart(
        self, ax: plt.Axes, peak_data: List[Tuple[float, float, str, str]]
    ):
        """Add peak labels with smart positioning to avoid overlaps."""
        if not peak_data:
            return

        # Sort by x position
        peak_data = sorted(peak_data, key=lambda x: x[0])

        # Calculate positions with overlap avoidance
        positions = []
        for i, (x, y, label, color) in enumerate(peak_data):
            base_text_y = y + 0.05 * y

            # Check for overlaps with previous labels
            final_text_y = base_text_y
            for prev_x, prev_y in positions:
                # If x positions are close, adjust y position
                if abs(x - prev_x) < 0.15 * (ax.get_xlim()[1] - ax.get_xlim()[0]):
                    final_text_y = max(final_text_y, prev_y + 0.05 * y)

            positions.append((x, final_text_y))

            # Add the label
            ax.text(
                x,
                final_text_y,
                label,
                color=color,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8 * self.font_scale,  # Smaller font for better fit
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
                rotation=15,  # Slight rotation
            )
            ax.plot(
                [x, x],
                [y, final_text_y],
                color=color,
                linestyle=":",
                linewidth=1,
                alpha=0.5,
            )

    def _add_peak_label(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        label: str,
        color: str,
        offset: float = 0.05,
    ):
        """Add a label above the peak with a connecting line."""
        text_y = y + offset * y
        ax.text(
            x,
            text_y,
            label,
            color=color,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8 * self.font_scale,  # Smaller font to reduce overlap
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
            rotation=15,  # Reduced rotation for better readability
        )
        ax.plot([x, x], [y, text_y], color=color, linestyle=":", linewidth=1, alpha=0.5)

    def _customize_distribution_plot(
        self, ax: plt.Axes, distribution_data: Dict, font_sizes: Dict
    ):
        """Apply common customizations to distribution plots."""
        is_normalized = distribution_data["metadata"]["normalized"]
        xlabel = "Normalized Distance" if is_normalized else "Distance"
        title = (
            "Distribution Comparison of\n"
            + ("Normalized " if is_normalized else "")
            + "Distance Embeddings"
        )

        ax.set_xlabel(xlabel, fontsize=font_sizes["axis"], labelpad=15)
        ax.set_ylabel("Density", fontsize=font_sizes["axis"], labelpad=15)
        ax.set_title(title, fontsize=font_sizes["title"], pad=20, weight="bold")
        ax.tick_params(axis="both", which="major", labelsize=font_sizes["tick"])
        ax.grid(True, alpha=0.2)

    def _add_broken_axis_marks(self, ax1: plt.Axes, ax2: plt.Axes):
        """Add diagonal marks to indicate broken axis."""
        d = 0.015
        kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax1.spines["bottom"].set_color("#D3D3D3")
        ax2.spines["top"].set_color("#D3D3D3")

    # --- Violin Plot Analysis ---

    def create_violin_plot_comparison(
        self, sample_size: int = 10_000, save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Create violin plots comparing PLM distance differences."""
        logger.info("Creating violin plot comparison...")

        df_sample = self.df.head(sample_size) if len(self.df) > sample_size else self.df

        # Normalize data
        # Build normalized data using with_columns
        normalized_data = df_sample
        for col in tqdm(self.dist_cols, desc="Normalizing for violin plots"):
            if not df_sample[col].is_nan().all():
                normalized_col = (df_sample[col] - df_sample[col].min()) / (
                    df_sample[col].max() - df_sample[col].min()
                )
                normalized_data = normalized_data.with_columns(
                    normalized_col.alias(col)
                )

        n_models = len(self.dist_cols)
        fig, axes = plt.subplots(
            n_models, n_models, figsize=(20 * self.font_scale, 20 * self.font_scale)
        )
        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        # Compute differences and medians
        differences, row_ylims, min_median, max_median = self._compute_violin_data(
            normalized_data, n_models
        )

        # Create plots
        self._create_violin_plots(
            axes, differences, row_ylims, min_median, max_median, n_models
        )

        # Add title and colorbar
        self._add_violin_plot_decorations(fig, min_median, max_median)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Violin plot comparison saved to {save_path}")

        return fig, axes

    def _compute_violin_data(
        self, normalized_data: pl.DataFrame, n_models: int
    ) -> Tuple[Dict, Dict, float, float]:
        """Compute differences and statistics for violin plots."""
        all_medians = []
        differences = {}
        row_ylims = {}

        with tqdm(
            total=(n_models * (n_models - 1)) // 2, desc="Computing differences"
        ) as pbar:
            for i in range(n_models):
                row_diffs = []
                for j in range(n_models):
                    if i != j:
                        diff = abs(
                            normalized_data[self.dist_cols[i]]
                            - normalized_data[self.dist_cols[j]]
                        )
                        row_diffs.extend(diff.drop_nulls().to_numpy())
                        if i < j:
                            median = diff.median()
                            all_medians.append(median)
                            differences[(i, j)] = (diff, median)
                            pbar.update(1)

                if row_diffs:
                    row_ylims[i] = (0, max(row_diffs) * 1.1)

        min_median = min(all_medians) if all_medians else 0
        max_median = max(all_medians) if all_medians else 1

        return differences, row_ylims, min_median, max_median

    def _create_violin_plots(
        self,
        axes: np.ndarray,
        differences: Dict,
        row_ylims: Dict,
        min_median: float,
        max_median: float,
        n_models: int,
    ):
        """Create the violin plot grid."""
        total_plots = ((n_models * (n_models - 1)) // 2) + n_models

        with tqdm(total=total_plots, desc="Creating violin plots") as pbar:
            # Diagonal plots
            for i in range(n_models):
                ax = axes[i, i]
                ax.text(
                    0.5,
                    0.5,
                    self.dist_cols[i].replace("dist_", ""),
                    ha="center",
                    va="center",
                    rotation=45,
                    fontsize=12 * self.font_scale,
                    weight="bold",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                pbar.update(1)

            # Off-diagonal plots
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    diff, median = differences[(i, j)]
                    gray_val = 0.9 - 0.8 * (median - min_median) / (
                        max_median - min_median
                    )

                    self._create_single_violin_plot(
                        axes[i, j], diff, median, gray_val, row_ylims[i]
                    )
                    self._create_single_violin_plot(
                        axes[j, i], diff, median, gray_val, row_ylims[j]
                    )
                    pbar.update(1)

            # Format labels
            for i in range(n_models):
                for j in range(n_models):
                    if j == 0:
                        axes[i, j].set_ylabel(self.dist_cols[i].replace("dist_", ""))
                    else:
                        axes[i, j].set_ylabel("")
                        axes[i, j].set_yticklabels([])

                    if i == n_models - 1:
                        axes[i, j].set_xlabel(
                            self.dist_cols[j].replace("dist_", ""), rotation=45
                        )
                    else:
                        axes[i, j].set_xlabel("")

    def _create_single_violin_plot(
        self,
        ax: plt.Axes,
        diff: pl.Series,
        median: float,
        gray_val: float,
        ylim: Optional[Tuple[float, float]],
    ):
        """Create a single violin plot with consistent styling."""
        sns.violinplot(y=diff.drop_nulls(), ax=ax, inner="box", color=str(gray_val))
        if ylim:
            ax.set_ylim(ylim)
        ax.text(
            0,
            ax.get_ylim()[1],
            f"{median:.3f}",
            ha="center",
            va="bottom",
            fontsize=8 * self.font_scale,
        )
        ax.set_xticks([])
        ax.tick_params(axis="y", length=0)

    def _add_violin_plot_decorations(
        self, fig: plt.Figure, min_median: float, max_median: float
    ):
        """Add title and colorbar to violin plot."""
        plt.suptitle(
            "All-vs-All PLM Distance Differences\n(darker = larger median difference)",
            fontsize=16 * self.font_scale,
            y=0.925,
            weight="bold",
        )

        ax_legend = fig.add_axes([0.94, 0.1, 0.02, 0.8])
        gradient = np.linspace(0, 1, 256).reshape(256, 1)
        ax_legend.imshow(gradient, aspect="auto", cmap="gray")
        ax_legend.set_xticks([])

        n_ticks = 5
        tick_positions = np.linspace(0, 255, n_ticks)
        tick_values = np.linspace(max_median, min_median, n_ticks)
        ax_legend.set_yticks(tick_positions)
        ax_legend.set_yticklabels([f"{val:.3f}" for val in tick_values])
        ax_legend.set_title("Median\nDifference", fontsize=10 * self.font_scale)

    # --- Main Generation Method ---

    def generate_all_visualizations(
        self, force_recompute: bool = False
    ) -> Dict[str, Path]:
        """Generate all visualization types and save them to the output directory."""
        logger.info("Generating all embedding comparison visualizations...")

        output_paths = {}
        cache_dir = self.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        # Define visualization configurations
        viz_configs = [
            {
                "name": "hexagonal_comparison",
                "description": "hexagonal distance comparison",
                "cache_file": "hexbin_data.json",
                "output_file": "hexagonal_distance_comparison.png",
                "compute_func": self.compute_hexbin_data,
                "plot_func": lambda data: self.plot_hexagonal_distance_comparison(
                    data, save_path=output_paths.get("temp_path")
                ),
            },
            {
                "name": "correlation_heatmap",
                "description": "correlation heatmap",
                "cache_file": "correlation_data.json",
                "output_file": "correlation_heatmap.png",
                "compute_func": self.compute_correlation_data,
                "plot_func": lambda data: self.plot_correlation_heatmap(
                    data, show_ci=False, save_path=output_paths.get("temp_path")
                ),
            },
            {
                "name": "wasserstein_heatmap",
                "description": "Wasserstein distance heatmap",
                "cache_file": "wasserstein_data.json",
                "output_file": "wasserstein_heatmap.png",
                "compute_func": self.compute_wasserstein_data,
                "plot_func": lambda data: self.plot_wasserstein_heatmap(
                    data, cmap="Blues", save_path=output_paths.get("temp_path")
                ),
            },
            {
                "name": "distribution_comparison",
                "description": "raw distribution comparison",
                "cache_file": "distribution_data.json",
                "output_file": "distribution_comparison.png",
                "compute_func": lambda: self.compute_distribution_data(normalize=False),
                "plot_func": lambda data: self.plot_distributions(
                    data, save_path=output_paths.get("temp_path")
                ),
            },
            {
                "name": "distribution_comparison_normalized",
                "description": "normalized distribution comparison",
                "cache_file": "distribution_normalized_data.json",
                "output_file": "distribution_comparison_normalized.png",
                "compute_func": lambda: self.compute_distribution_data(normalize=True),
                "plot_func": lambda data: self.plot_distributions(
                    data, save_path=output_paths.get("temp_path")
                ),
            },
            {
                "name": "distribution_comparison_normalized_ridge",
                "description": "normalized distribution ridge plot",
                "cache_file": "distribution_normalized_data.json",
                "output_file": "distribution_comparison_normalized_ridge.png",
                "compute_func": lambda: self.compute_distribution_data(normalize=True),
                "plot_func": lambda data: self.plot_ridge_distributions(
                    data, alpha=0.8, save_path=output_paths.get("temp_path")
                ),
            },
        ]

        # Generate cacheable visualizations
        for config in viz_configs:
            logger.info(f"=== Generating {config['description']} ===")

            cache_path = cache_dir / config["cache_file"]
            output_path = self.output_dir / config["output_file"]
            output_paths["temp_path"] = output_path

            # Load or compute data
            data = self._load_cached_data(cache_path, force_recompute)
            if data is None:
                if "save_path" in config["compute_func"].__code__.co_varnames:
                    data = config["compute_func"](save_path=cache_path)
                else:
                    data = config["compute_func"]()
                    self._save_json_data(
                        data, cache_path, config["description"].title() + " data"
                    )

            # Generate plot
            config["plot_func"](data)
            output_paths[config["name"]] = output_path

        # Generate violin plot (no caching)
        logger.info("=== Generating violin plot comparison ===")
        violin_output = self.output_dir / "violin_plot_comparison.png"
        self.create_violin_plot_comparison(save_path=violin_output)
        output_paths["violin_plot_comparison"] = violin_output

        # Clean up temp key
        output_paths.pop("temp_path", None)

        logger.info("=== All visualizations complete ===")
        for viz_type, path in output_paths.items():
            logger.info(f"{viz_type}: {path}")

        return output_paths

    def plot_combined_wasserstein_correlation(
        self,
        wasserstein_data: Optional[Dict] = None,
        correlation_data: Optional[Dict] = None,
        gridsize: int = 50,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Create a combined plot with Wasserstein distance (upper triangle),
        model names (diagonal), and correlation values (lower triangle)."""

        # Data should be provided by the caller (from viz_map)
        if wasserstein_data is None or correlation_data is None:
            raise ValueError(
                "Both wasserstein_data and correlation_data must be provided"
            )

        logger.info("Creating combined Wasserstein-correlation plot...")

        dist_cols = [f"dist_{col}" for col in wasserstein_data["columns"]]
        correlations = np.array(correlation_data["correlations"])
        wasserstein_distances = np.array(wasserstein_data["distances"])
        n = len(dist_cols)

        # Calculate figure size to ensure square cells
        # Base size per cell to ensure readability
        cell_size = 1.5 * self.font_scale
        grid_size = n * cell_size
        cbar_width = 0.8 * self.font_scale  # Reduced colorbar width
        spacing = 1.0 * self.font_scale  # Extra spacing between grid and colorbars

        # Total figure dimensions
        fig_width = grid_size + spacing + cbar_width
        fig_height = grid_size

        # Create figure
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Calculate margins to keep grid square
        left_margin = 0.08
        right_margin = 0.02
        top_margin = 0.08
        bottom_margin = 0.08

        # Available space for grid + spacing + colorbar
        available_width = 1 - left_margin - right_margin
        available_height = 1 - top_margin - bottom_margin

        # Fraction of figure width for grid, spacing, and colorbar
        grid_frac = grid_size / fig_width
        spacing_frac = spacing / fig_width
        cbar_frac = cbar_width / fig_width

        # Create grid with space for colorbars on the right
        # Add a spacing column between grid and colorbars
        gs = fig.add_gridspec(
            n,
            n + 2,  # n for grid, 1 for spacing, 1 for colorbar
            width_ratios=[1] * n
            + [spacing_frac / grid_frac * n, cbar_frac / grid_frac * n],
            wspace=0,
            hspace=0,
            left=left_margin,
            right=left_margin + grid_frac + spacing_frac + cbar_frac,
            top=1 - top_margin,
            bottom=bottom_margin,
        )

        # Create axes for the main plot
        axes = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                axes[i, j] = fig.add_subplot(gs[i, j])

        with tqdm(total=n * n, desc="Creating combined plot") as pbar:
            for i, col1 in enumerate(dist_cols):
                for j, col2 in enumerate(dist_cols):
                    ax = axes[i, j]

                    # Remove all ticks and spines for cleaner look
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    if i == j:
                        # Diagonal: show embedding name
                        embedding_name = col1.replace("dist_", "").replace("_", "\n")
                        if embedding_name == "prottucker":
                            embedding_name = "prot-\ntucker"
                        elif embedding_name == "prostt5":
                            embedding_name = "prost-\nT5"
                        elif embedding_name == "prott5":
                            embedding_name = "prot-\nT5"
                        ax.text(
                            0.5,
                            0.5,
                            embedding_name,
                            ha="center",
                            va="center",
                            fontsize=24 * self.font_scale,
                            weight="bold",
                            color=self._get_embedding_color(col1),
                            transform=ax.transAxes,
                        )
                    elif i < j:
                        # Upper triangle: Wasserstein distance
                        if not np.isnan(wasserstein_distances[i, j]):
                            wasserstein_val = wasserstein_distances[i, j]
                            # Create a heatmap cell with proper extent
                            im = ax.imshow(
                                [[wasserstein_val]],
                                cmap="Blues",
                                vmin=0,
                                vmax=np.nanmax(wasserstein_distances),
                                aspect="auto",
                                extent=[0, 1, 0, 1],
                            )
                            # Determine text color based on background
                            normalized_val = wasserstein_val / np.nanmax(
                                wasserstein_distances
                            )
                            text_color = "white" if normalized_val > 0.5 else "black"
                            ax.text(
                                0.5,
                                0.5,
                                f"{wasserstein_val:.2f}",
                                ha="center",
                                va="center",
                                color=text_color,
                                fontsize=24 * self.font_scale,
                                weight="bold",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.set_facecolor("lightgray")
                            ax.text(
                                0.5,
                                0.5,
                                "NA",
                                ha="center",
                                va="center",
                                color="black",
                                fontsize=24 * self.font_scale,
                                transform=ax.transAxes,
                            )
                    else:
                        # Lower triangle: correlation values
                        if not np.isnan(correlations[i, j]):
                            correlation_val = correlations[i, j]
                            # Create a heatmap cell with proper extent
                            im = ax.imshow(
                                [[correlation_val]],
                                cmap="OrRd",
                                vmin=0,
                                vmax=1,
                                aspect="auto",
                                extent=[0, 1, 0, 1],
                            )
                            text_color = "white" if correlation_val > 0.5 else "black"
                            ax.text(
                                0.5,
                                0.5,
                                f"{correlation_val:.2f}",
                                ha="center",
                                va="center",
                                color=text_color,
                                fontsize=24 * self.font_scale,
                                weight="bold",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.set_facecolor("lightgray")
                            ax.text(
                                0.5,
                                0.5,
                                "NA",
                                ha="center",
                                va="center",
                                color="black",
                                fontsize=24 * self.font_scale,
                                transform=ax.transAxes,
                            )

                    pbar.update(1)

        # Add row and column labels outside the grid
        for i, col in enumerate(dist_cols):
            label = col.replace("dist_", "")
            # Y-axis labels (left side)
            axes[i, 0].set_ylabel(
                label,
                rotation=0,
                ha="right",
                va="center",
                fontsize=26 * self.font_scale,
                labelpad=10,
            )
            # X-axis labels (bottom)
            axes[-1, i].set_xlabel(
                label,
                rotation=45,
                ha="right",
                va="top",
                fontsize=26 * self.font_scale,
                labelpad=5,
            )

        # Add title
        fig.suptitle(
            "Combined Pairwise Comparison\n(Upper: Wasserstein Distance, Diagonal: Models, Lower: Correlations)",
            fontsize=32 * self.font_scale,
            weight="bold",
            y=0.98,
        )

        # Create a sub-grid for the colorbar column (rightmost) to split it into two equal parts
        gs_right = gs[:, -1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.2)

        # Add colorbar for Wasserstein distances (top half)
        cbar_wass_ax = fig.add_subplot(gs_right[0])
        im_wass = plt.cm.ScalarMappable(
            cmap="Blues",
            norm=plt.Normalize(vmin=0, vmax=np.nanmax(wasserstein_distances)),
        )
        cbar_wass = plt.colorbar(im_wass, cax=cbar_wass_ax, orientation="vertical")
        cbar_wass.ax.tick_params(labelsize=24 * self.font_scale)
        cbar_wass.set_label("Wasserstein Distance", fontsize=26 * self.font_scale)

        # Add colorbar for correlations (bottom half)
        cbar_corr_ax = fig.add_subplot(gs_right[1])
        im_corr = plt.cm.ScalarMappable(cmap="OrRd", norm=plt.Normalize(vmin=0, vmax=1))
        cbar_corr = plt.colorbar(im_corr, cax=cbar_corr_ax, orientation="vertical")
        cbar_corr.ax.tick_params(labelsize=24 * self.font_scale)
        cbar_corr.set_label("Spearman Correlation", fontsize=26 * self.font_scale)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Combined Wasserstein-correlation plot saved to {save_path}")

        return fig, axes


def main():
    """Main function to parse arguments and run the visualization script."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive pairwise embedding comparison visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add arguments
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to CSV file containing the embedding distance data.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("out/embedding_comparison"),
        help="Directory to save output visualizations and cache files.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit the number of rows to process (for testing or memory constraints).",
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=1.0,
        help="Scaling factor for all font sizes in visualizations.",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of all cached intermediate data.",
    )
    parser.add_argument(
        "--visualizations",
        nargs="+",
        choices=[
            "hexagonal",
            "correlation",
            "wasserstein",
            "distribution",
            "distribution_normalized",
            "distribution_normalized_ridge",
            "violin",
            "combined",
            "all",
        ],
        default=["all"],
        help="Specific visualizations to generate.",
    )
    parser.add_argument(
        "--ranking_csv",
        type=Path,
        default=None,
        help="Path to PLM ranking CSV to sort ridge plot by performance (e.g., plm_ranking_by_spearman.csv).",
    )

    args = parser.parse_args()

    # Create visualizer and generate visualizations
    visualizer = EmbeddingComparisonVisualizer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        font_scale=args.font_scale,
    )

    if "all" in args.visualizations:
        output_paths = visualizer.generate_all_visualizations(
            force_recompute=args.force_recompute
        )
    else:
        # Generate individual visualizations
        output_paths = {}
        cache_dir = args.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        viz_map = {
            "hexagonal": (
                visualizer.compute_hexbin_data,
                visualizer.plot_hexagonal_distance_comparison,
                "hexbin_data.json",
                "hexagonal_distance_comparison.png",
            ),
            "correlation": (
                visualizer.compute_correlation_data,
                visualizer.plot_correlation_heatmap,
                "correlation_data.json",
                "correlation_heatmap.png",
            ),
            "wasserstein": (
                visualizer.compute_wasserstein_data,
                visualizer.plot_wasserstein_heatmap,
                "wasserstein_data.json",
                "wasserstein_heatmap.png",
            ),
            "distribution": (
                lambda: visualizer.compute_distribution_data(normalize=False),
                visualizer.plot_distributions,
                "distribution_data.json",
                "distribution_comparison.png",
            ),
            "distribution_normalized": (
                lambda: visualizer.compute_distribution_data(normalize=True),
                visualizer.plot_distributions,
                "distribution_normalized_data.json",
                "distribution_comparison_normalized.png",
            ),
            "distribution_normalized_ridge": (
                lambda: visualizer.compute_distribution_data(normalize=True),
                visualizer.plot_ridge_distributions,
                "distribution_normalized_data.json",
                "distribution_comparison_normalized_ridge.png",
            ),
            "violin": (
                None,
                visualizer.create_violin_plot_comparison,
                None,
                "violin_plot_comparison.png",
            ),
            "combined": (
                [
                    visualizer.compute_correlation_data,
                    visualizer.compute_wasserstein_data,
                ],
                visualizer.plot_combined_wasserstein_correlation,
                ["correlation_data.json", "wasserstein_data.json"],
                "combined_wasserstein_correlation.png",
            ),
        }

        for viz_type in args.visualizations:
            if viz_type in viz_map:
                compute_func, plot_func, cache_file, output_file = viz_map[viz_type]
                output_path = args.output_dir / output_file

                logger.info(f"Generating {viz_type}...")

                if isinstance(compute_func, list):
                    # Handle combined plot with multiple compute functions
                    data_list = []
                    for i, (func, cache_file) in enumerate(
                        zip(compute_func, cache_file)
                    ):
                        cache_path = cache_dir / cache_file
                        data = visualizer._load_cached_data(
                            cache_path, args.force_recompute
                        )
                        if data is None:
                            data = func()
                            visualizer._save_json_data(
                                data, cache_path, f"{viz_type.title()} data {i + 1}"
                            )
                        data_list.append(data)

                    # Pass data to plot function
                    plot_func(
                        wasserstein_data=data_list[1],
                        correlation_data=data_list[0],
                        save_path=output_path,
                    )
                elif cache_file and compute_func:
                    cache_path = cache_dir / cache_file
                    data = visualizer._load_cached_data(
                        cache_path, args.force_recompute
                    )
                    if data is None:
                        data = compute_func()
                        visualizer._save_json_data(
                            data, cache_path, f"{viz_type.title()} data"
                        )
                    # Special handling for ridge plot to pass ranking_csv
                    if viz_type == "distribution_normalized_ridge":
                        plot_func(
                            data, save_path=output_path, ranking_csv=args.ranking_csv
                        )
                    else:
                        plot_func(data, save_path=output_path)
                else:
                    plot_func(save_path=output_path)

                output_paths[viz_type] = output_path

    logger.info("=== Visualization generation complete ===")
    for viz_type, path in output_paths.items():
        logger.info(f"{viz_type}: {path}")


if __name__ == "__main__":
    main()
