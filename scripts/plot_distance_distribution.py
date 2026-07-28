"""
Plot distance distribution from all-vs-all distance computation.

This script generates density plots and histograms from the output of all_vs_all.py.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# Constants
TARGET_BINS_FOR_VISUALIZATION = 1000
DEFAULT_GAUSSIAN_SIGMA = 1.0
DEFAULT_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
PLOT_DPI = 300


class HistogramData:
    """Container for histogram data and metadata."""

    def __init__(self, histogram: np.ndarray, bin_edges: np.ndarray, metadata: Dict):
        self.histogram = histogram
        self.bin_edges = bin_edges
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 0
        self.metadata = metadata

    @property
    def total_comparisons(self) -> int:
        return int(self.histogram.sum())

    @property
    def distance_range(self) -> Tuple[float, float]:
        return float(self.bin_edges[0]), float(self.bin_edges[-1])


def load_from_npz(file_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load histogram data from NPZ file."""
    data = np.load(file_path)
    histogram = data["histogram"]
    bin_edges = data["bin_edges"]

    metadata = {
        "n_embeddings": int(data.get("n_embeddings", 0)),
        "embedding_dim": int(data.get("embedding_dim", 0)),
        "n_bins": int(data.get("n_bins", len(histogram))),
        "min_distance": float(data.get("min_distance", bin_edges[0])),
        "max_distance": float(data.get("max_distance", bin_edges[-1])),
    }

    return histogram, bin_edges, metadata


def load_from_json(file_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load histogram data from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)

    histogram = np.array(data["histogram"])
    metadata = data.get("metadata", {})

    min_val = metadata.get("min_distance", 0.0)
    max_val = metadata.get("max_distance", 1.0)
    n_bins = metadata.get("n_bins", len(histogram))

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    return histogram, bin_edges, metadata


def load_histogram_data(input_path: str) -> HistogramData:
    """Load histogram data from JSON or NPZ file."""
    path = Path(input_path)

    loaders = {
        ".npz": load_from_npz,
        ".json": load_from_json,
    }

    loader = loaders.get(path.suffix)
    if loader is None:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Use .json or .npz"
        )

    print(f"Loading data from {path.suffix.upper()} file: {path}")
    histogram, bin_edges, metadata = loader(path)

    data = HistogramData(histogram, bin_edges, metadata)

    print(f"Loaded {len(data.histogram):,} bins with {data.total_comparisons:,} total comparisons")
    dist_min, dist_max = data.distance_range
    print(f"Distance range: [{dist_min:.6f}, {dist_max:.6f}]")

    return data


def trim_zero_bins(data: HistogramData) -> HistogramData:
    """Remove leading and trailing bins with zero counts."""
    nonzero_indices = np.nonzero(data.histogram)[0]

    if len(nonzero_indices) == 0:
        return data

    first_idx = nonzero_indices[0]
    last_idx = nonzero_indices[-1]

    histogram_trimmed = data.histogram[first_idx:last_idx + 1]
    bin_edges_trimmed = data.bin_edges[first_idx:last_idx + 2]

    n_removed_start = first_idx
    n_removed_end = len(data.histogram) - last_idx - 1

    if n_removed_start > 0 or n_removed_end > 0:
        print(f"Trimmed {n_removed_start} leading and {n_removed_end} trailing zero bins")
        trimmed_data = HistogramData(histogram_trimmed, bin_edges_trimmed, data.metadata)
        dist_min, dist_max = trimmed_data.distance_range
        print(f"Distance range after trimming: [{dist_min:.6f}, {dist_max:.6f}]")
        return trimmed_data

    return data


def downsample_bins(data: HistogramData, target_bins: int = TARGET_BINS_FOR_VISUALIZATION) -> HistogramData:
    """Downsample bins by combining adjacent bins."""
    n_bins = len(data.histogram)

    if n_bins <= target_bins:
        return data

    print(f"Downsampling from {n_bins:,} to {target_bins:,} bins for visualization...")

    rebin_factor = n_bins // target_bins
    n_new_bins = n_bins // rebin_factor

    new_histogram = np.add.reduceat(
        data.histogram,
        np.arange(0, n_new_bins * rebin_factor, rebin_factor)
    )
    new_bin_edges = data.bin_edges[::rebin_factor]

    return HistogramData(new_histogram, new_bin_edges, data.metadata)


def compute_statistics(data: HistogramData) -> Dict:
    """Compute statistical measures from histogram."""
    total = data.total_comparisons

    # Mean and standard deviation
    mean = np.average(data.bin_centers, weights=data.histogram)
    variance = np.average((data.bin_centers - mean) ** 2, weights=data.histogram)
    std = np.sqrt(variance)

    # Mode (most frequent bin)
    mode_idx = np.argmax(data.histogram)
    mode = data.bin_centers[mode_idx]

    # Percentiles
    cumsum = np.cumsum(data.histogram)
    percentiles = {}
    for p in DEFAULT_PERCENTILES:
        idx = np.searchsorted(cumsum, p / 100 * total)
        idx = min(idx, len(data.bin_centers) - 1)
        percentiles[p] = data.bin_centers[idx]

    return {
        "total_comparisons": total,
        "mean": float(mean),
        "std": float(std),
        "mode": float(mode),
        "percentiles": {k: float(v) for k, v in percentiles.items()},
    }


def normalize_data(data: HistogramData, stats: Dict) -> Tuple[HistogramData, Dict]:
    """Normalize distances to [0, 1] range."""
    dist_min, dist_max = data.distance_range

    if dist_max <= dist_min:
        return data, stats

    range_width = dist_max - dist_min
    normalized_edges = (data.bin_edges - dist_min) / range_width

    normalized_stats = {
        "mean": (stats["mean"] - dist_min) / range_width,
        "mode": (stats["mode"] - dist_min) / range_width,
        "percentiles": {
            k: (v - dist_min) / range_width
            for k, v in stats["percentiles"].items()
        },
    }

    return HistogramData(data.histogram, normalized_edges, data.metadata), normalized_stats


def add_vertical_reference_lines(ax: plt.Axes, stats: Dict):
    """Add mean and median reference lines to plot."""
    ax.axvline(
        stats["mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {stats['mean']:.4f}",
        alpha=0.8,
    )

    ax.axvline(
        stats["percentiles"][50],
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {stats['percentiles'][50]:.4f}",
        alpha=0.8,
    )


def configure_plot_aesthetics(ax: plt.Axes, xlabel: str, ylabel: str,
                              title: str, metadata: Dict, log_scale: bool):
    """Configure plot labels, title, grid, and legend."""
    # Axis labels
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ylabel_text = f"{ylabel} (log scale)" if log_scale else ylabel
    ax.set_ylabel(ylabel_text, fontsize=12, fontweight="bold")

    # Title with metadata
    if metadata.get("n_embeddings"):
        n_emb = metadata['n_embeddings']
        dim = metadata.get('embedding_dim', '?')
        title += f"\n({n_emb:,} embeddings, {dim} dimensions)"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Scale and grid
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    # Legend
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)


def plot_histogram(data: HistogramData, stats: Dict, output_path: Path,
                   log_scale: bool = False, normalize: bool = True):
    """Generate and save histogram plot."""
    # Prepare data
    plot_data = trim_zero_bins(data) if normalize else data
    plot_data = downsample_bins(plot_data)

    plot_stats = stats
    xlabel = "Euclidean Distance"

    if normalize:
        plot_data, plot_stats = normalize_data(plot_data, stats)
        xlabel = "Normalized Euclidean Distance (0-1)"

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(
        plot_data.bin_centers,
        plot_data.histogram,
        width=plot_data.bin_width * 0.9,
        color="steelblue",
        alpha=0.7,
        edgecolor="darkblue",
        linewidth=0.5,
    )

    add_vertical_reference_lines(ax, plot_stats)
    configure_plot_aesthetics(
        ax, xlabel, "Count",
        "Histogram of All-vs-All Euclidean Distances",
        data.metadata, log_scale
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved histogram plot to: {output_path}")
    plt.close()


def plot_density_smooth(data: HistogramData, stats: Dict, output_path: Path,
                       log_scale: bool = False, normalize: bool = True,
                       sigma: float = DEFAULT_GAUSSIAN_SIGMA):
    """Generate and save smoothed density plot."""
    # Prepare data (no downsampling for density)
    plot_data = trim_zero_bins(data) if normalize else data

    # Compute density
    total = plot_data.total_comparisons
    density = plot_data.histogram / total / plot_data.bin_width
    density_smooth = gaussian_filter1d(density, sigma=sigma)

    plot_stats = stats
    xlabel = "Euclidean Distance"

    if normalize:
        plot_data, plot_stats = normalize_data(plot_data, stats)
        xlabel = "Normalized Euclidean Distance (0-1)"

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        plot_data.bin_centers,
        density_smooth,
        color="darkblue",
        linewidth=2,
        alpha=0.8,
        label=f"Smoothed Density (Gaussian filter, σ={sigma})"
    )

    add_vertical_reference_lines(ax, plot_stats)
    configure_plot_aesthetics(
        ax, xlabel, "Density",
        "Smoothed Density Distribution of All-vs-All Euclidean Distances",
        data.metadata, log_scale
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved smooth density plot to: {output_path}")
    plt.close()


def print_statistics(stats: Dict):
    """Print computed statistics to console."""
    print("\nStatistics:")
    print(f"  Mean distance: {stats['mean']:.6f}")
    print(f"  Std distance: {stats['std']:.6f}")
    print(f"  Mode distance: {stats['mode']:.6f}")
    print(f"  Median distance: {stats['percentiles'][50]:.6f}")
    print(f"  25th percentile: {stats['percentiles'][25]:.6f}")
    print(f"  75th percentile: {stats['percentiles'][75]:.6f}")
    print(f"  Total comparisons: {stats['total_comparisons']:,}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot histogram of distance distribution from all-vs-all analysis"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input file (.json or .npz from all_vs_all.py)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for y-axis",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize distances to [0, 1] range (default: True)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Disable distance normalization",
    )
    parser.add_argument(
        "--plot-types",
        type=str,
        nargs="+",
        default=["histogram", "density"],
        choices=["histogram", "density"],
        help="Types of plots to generate (default: both)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_GAUSSIAN_SIGMA,
        help=f"Gaussian filter sigma for smoothing (default: {DEFAULT_GAUSSIAN_SIGMA})",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Load data
    data = load_histogram_data(args.input)

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(data)
    print_statistics(stats)

    # Determine output directory
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    base_name = input_path.stem

    if "histogram" in args.plot_types:
        output_path = output_dir / f"{base_name}_histogram.png"
        plot_histogram(data, stats, output_path, args.log_scale, args.normalize)

    if "density" in args.plot_types:
        output_path = output_dir / f"{base_name}_density.png"
        plot_density_smooth(data, stats, output_path, args.log_scale,
                          args.normalize, args.sigma)

    print("\nDone!")


if __name__ == "__main__":
    main()