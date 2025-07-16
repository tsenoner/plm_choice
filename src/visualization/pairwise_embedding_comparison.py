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
import pandas as pd
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

EMBEDDING_COLOR_MAP: Dict[str, str] = {
    "prott5": "#ff75be",
    "prottucker": "#ff69b4",
    "prostt5": "#ffc1e2",
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
    "random_1024": "#808080",
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
        data_path: Union[str, Path, pd.DataFrame],
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

    def _load_data(self, data_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load data from various sources."""
        if isinstance(data_path, pd.DataFrame):
            df = data_path.copy()
            logger.info("Using provided DataFrame")
        else:
            data_path = Path(data_path)
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)

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

                    mask = ~(self.df[col1].isna() | self.df[col2].isna())
                    if mask.sum() < 10:
                        pbar.update(1)
                        continue

                    x_data = self.df[col1][mask].values
                    y_data = self.df[col2][mask].values

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
                    else:
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
                        self.df[self.dist_cols[i]].isna()
                        | self.df[self.dist_cols[j]].isna()
                    )
                    if mask.sum() > 3:
                        correlation, _ = stats.spearmanr(
                            self.df[self.dist_cols[i]][mask],
                            self.df[self.dist_cols[j]][mask],
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

        masked_correlations = np.ma.array(correlations, mask=np.isnan(correlations))
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

        if show_ci:
            ci_lower = np.array(data["ci_lower"])
            ci_upper = np.array(data["ci_upper"])

        with tqdm(total=n * n, desc="Adding correlation annotations") as pbar:
            for i in range(n):
                for j in range(n):
                    if not np.isnan(correlations[i, j]):
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
    def normalize_distribution(x: pd.Series) -> np.ndarray:
        """Normalize a distribution to [0,1] range using MinMax scaling."""
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return x

        scaler = MinMaxScaler()
        return scaler.fit_transform(x.values.reshape(-1, 1)).ravel()

    def compute_wasserstein_data(self, save_path: Optional[Path] = None) -> Dict:
        """Compute Wasserstein distances between all pairs of normalized distance distributions."""
        logger.info("Computing Wasserstein distances...")

        n = len(self.dist_cols)
        distances = np.zeros((n, n))

        # Pre-compute normalized distributions
        logger.info("Normalizing distributions...")
        normalized_dists = {}
        for col in tqdm(self.dist_cols, desc="Normalizing"):
            normalized_dists[col] = self.normalize_distribution(self.df[col])

        # Compute pairwise distances
        with tqdm(total=(n * (n + 1)) // 2, desc="Computing distances") as pbar:
            for i in range(n):
                for j in range(i, n):
                    dist1 = normalized_dists[self.dist_cols[i]]
                    dist2 = normalized_dists[self.dist_cols[j]]

                    if len(dist1) > 0 and len(dist2) > 0:
                        dist = wasserstein_distance(dist1, dist2)
                        distances[i, j] = distances[j, i] = dist
                    else:
                        distances[i, j] = distances[j, i] = np.nan

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
                    pbar.update(1)

    # --- Distribution Analysis ---

    def compute_distribution_data(
        self, normalize: bool = False, save_path: Optional[Path] = None
    ) -> Dict:
        """Pre-compute distribution data for plotting."""
        logger.info(f"Computing distribution data (normalize={normalize})...")

        distribution_data = {"metadata": {"normalized": normalize}, "distributions": {}}

        for col in tqdm(self.dist_cols, desc="Processing distributions"):
            data = self.df[col].dropna().values

            if normalize:
                data = self.normalize_distribution(self.df[col])
                x_range = np.linspace(0, 1, 200)
            else:
                x_range = np.linspace(data.min(), data.max(), 200)

            # Calculate KDE
            if len(data) > 0:
                kernel = stats.gaussian_kde(data)
                density = kernel(x_range)
                peak_idx = np.argmax(density)
                peak_x = float(x_range[peak_idx])
                peak_y = float(density[peak_idx])
            else:
                density = np.zeros_like(x_range)
                peak_x = peak_y = 0.0

            distribution_data["distributions"][col] = {
                "x_range": x_range.tolist(),
                "density": density.tolist(),
                "peak_x": peak_x,
                "peak_y": peak_y,
                "min": float(data.min()) if len(data) > 0 else 0.0,
                "max": float(data.max()) if len(data) > 0 else 0.0,
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
        normalized_data = pd.DataFrame()
        for col in tqdm(self.dist_cols, desc="Normalizing for violin plots"):
            if not df_sample[col].isna().all():
                normalized_data[col] = (df_sample[col] - df_sample[col].min()) / (
                    df_sample[col].max() - df_sample[col].min()
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
        self, normalized_data: pd.DataFrame, n_models: int
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
                        row_diffs.extend(diff.dropna().values)
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
        diff: pd.Series,
        median: float,
        gray_val: float,
        ylim: Optional[Tuple[float, float]],
    ):
        """Create a single violin plot with consistent styling."""
        sns.violinplot(y=diff.dropna(), ax=ax, inner="box", color=str(gray_val))
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
            "violin",
            "all",
        ],
        default=["all"],
        help="Specific visualizations to generate.",
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
            "violin": (
                None,
                visualizer.create_violin_plot_comparison,
                None,
                "violin_plot_comparison.png",
            ),
        }

        for viz_type in args.visualizations:
            if viz_type in viz_map:
                compute_func, plot_func, cache_file, output_file = viz_map[viz_type]
                output_path = args.output_dir / output_file

                logger.info(f"Generating {viz_type}...")

                if cache_file and compute_func:
                    cache_path = cache_dir / cache_file
                    data = visualizer._load_cached_data(
                        cache_path, args.force_recompute
                    )
                    if data is None:
                        data = compute_func()
                        visualizer._save_json_data(
                            data, cache_path, f"{viz_type.title()} data"
                        )
                    plot_func(data, save_path=output_path)
                else:
                    plot_func(save_path=output_path)

                output_paths[viz_type] = output_path

    logger.info("=== Visualization generation complete ===")
    for viz_type, path in output_paths.items():
        logger.info(f"{viz_type}: {path}")


if __name__ == "__main__":
    main()
