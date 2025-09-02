import argparse
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
from pathlib import Path
from scipy import stats
from scipy.stats import wasserstein_distance
from tqdm import tqdm

warnings.filterwarnings("ignore")


class DistributionConvergenceAnalyzer:
    def __init__(self, hdf5_filepath):
        """Initialize with HDF5 file containing embeddings"""
        self.filepath = hdf5_filepath
        self.embeddings = None
        self.sample_names = None
        self.n_samples = None
        self.embedding_dim = None

    def load_embeddings(self):
        """Load embeddings from HDF5 file using proper HDF5 handling"""
        print("Loading embeddings...")

        with h5py.File(self.filepath, "r", swmr=True) as f:
            # Get all dataset names (sample identifiers)
            self.sample_names = list(f.keys())
            self.n_samples = len(self.sample_names)

            # Get embedding dimension from first sample (flatten to ensure 1D)
            first_embedding = f[self.sample_names[0]][:].flatten().astype(np.float32)
            self.embedding_dim = len(first_embedding)

            # Load all embeddings into numpy array
            self.embeddings = np.zeros(
                (self.n_samples, self.embedding_dim), dtype=np.float32
            )

            for i, sample_name in enumerate(tqdm(self.sample_names, desc="Loading")):
                # Flatten embedding to ensure 1D and convert to float32
                embedding = f[sample_name][:].flatten().astype(np.float32)
                self.embeddings[i] = embedding

        print(f"Loaded {self.n_samples:,} embeddings of dimension {self.embedding_dim}")
        print(f"Memory usage: ~{self.embeddings.nbytes / 1e9:.2f} GB")
        return self.embeddings

    def sample_pairwise_distances(self, n_pairs, random_state=None):
        """Sample random pairs and compute their Euclidean distances using vectorized approach"""
        if random_state is not None:
            np.random.seed(random_state)

        # Total number of unique pairs (upper triangle, excluding diagonal)
        total_pairs = self.n_samples * (self.n_samples - 1) // 2

        # Ensure we don't sample more pairs than exist
        n_pairs = min(n_pairs, total_pairs)

        # More efficient sampling for large populations
        if total_pairs > 1e9:  # Use efficient method for very large populations
            # Generate random floats and scale to range
            random_floats = np.random.random(n_pairs)
            pair_indices = (random_floats * total_pairs).astype(np.int64)
            # Remove duplicates if any (rare for large populations)
            pair_indices = np.unique(pair_indices)
            # If we lost some due to duplicates, generate a few more
            while len(pair_indices) < n_pairs:
                additional_needed = n_pairs - len(pair_indices)
                additional_floats = np.random.random(additional_needed)
                additional_indices = (additional_floats * total_pairs).astype(np.int64)
                pair_indices = np.unique(
                    np.concatenate([pair_indices, additional_indices])
                )
            pair_indices = pair_indices[:n_pairs]  # Take exactly n_pairs
        else:
            # Use standard method for smaller populations
            pair_indices = np.random.choice(total_pairs, size=n_pairs, replace=False)

        # Vectorized conversion to (i,j) coordinates
        # Find row i: largest i such that i*(2n-1-i)/2 <= idx
        # Using quadratic formula: i = ((2n-1) - sqrt((2n-1)² - 8*idx)) / 2
        i = np.floor(
            (
                (2 * self.n_samples - 1)
                - np.sqrt((2 * self.n_samples - 1) ** 2 - 8 * pair_indices)
            )
            / 2
        ).astype(int)

        # Calculate j using the offset within row i
        # Elements before row i: i*(2n-1-i)/2
        elements_before_row_i = i * (2 * self.n_samples - 1 - i) // 2
        j = i + 1 + (pair_indices - elements_before_row_i)

        # Check bounds
        if (
            i.max() >= self.n_samples
            or j.max() >= self.n_samples
            or i.min() < 0
            or j.min() < 0
        ):
            raise ValueError(
                f"Index out of bounds! i_range=[{i.min()}, {i.max()}], j_range=[{j.min()}, {j.max()}], n_samples={self.n_samples}"
            )

        # Compute distances in batches to avoid memory issues
        batch_size = 50000  # Process in smaller batches
        distances = []

        for start_idx in range(0, len(i), batch_size):
            end_idx = min(start_idx + batch_size, len(i))
            i_batch = i[start_idx:end_idx]
            j_batch = j[start_idx:end_idx]

            batch_distances = np.linalg.norm(
                self.embeddings[i_batch] - self.embeddings[j_batch], axis=1
            )
            distances.append(batch_distances)

        distances = np.concatenate(distances)

        return distances

    def compute_distribution_metrics(self, distances):
        """Compute key distribution statistics"""
        return {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "median": float(np.median(distances)),
            "q25": float(np.percentile(distances, 25)),
            "q75": float(np.percentile(distances, 75)),
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
            "skewness": float(stats.skew(distances)),
            "kurtosis": float(stats.kurtosis(distances)),
        }

    def analyze_convergence(
        self, sample_sizes=None, n_repetitions=5, reference_size=1000000
    ):
        """
        Analyze how distribution metrics converge with increasing sample size
        """
        if sample_sizes is None:
            # Logarithmic scale from 1k to 1M samples
            sample_sizes = np.logspace(3, 6, 20).astype(int)

        print(f"Computing reference distribution ({reference_size:,} samples)...")

        # Compute reference distribution
        reference_distances = self.sample_pairwise_distances(
            reference_size, random_state=42
        )
        reference_metrics = self.compute_distribution_metrics(reference_distances)

        # Storage for results
        convergence_results = {
            "sample_sizes": sample_sizes.tolist(),
            "mean_errors": [],
            "std_errors": [],
            "wasserstein_distances": [],
            "ks_statistics": [],
            "mean_stds": [],  # Standard deviation across repetitions
            "std_stds": [],
        }

        # Main convergence analysis with progress bar
        print(
            f"Testing {len(sample_sizes)} sample sizes with {n_repetitions} repetitions each..."
        )

        # Progress bar for sample sizes
        sample_progress = tqdm(sample_sizes, desc="Sample sizes", position=0)

        for i, sample_size in enumerate(sample_progress):
            # Update outer progress description
            sample_progress.set_description(
                f"Size {sample_size:,} ({i + 1}/{len(sample_sizes)})"
            )

            # Repeat sampling to get error bars
            metrics_list = []
            wasserstein_dists = []
            ks_stats = []

            # Inner progress for repetitions
            rep_progress = tqdm(
                range(n_repetitions), desc="Repetitions", position=1, leave=False
            )

            for rep in rep_progress:
                rep_progress.set_description(f"Rep {rep + 1}/{n_repetitions}")

                # Sample distances
                distances = self.sample_pairwise_distances(
                    sample_size, random_state=42 + rep
                )

                # Compute metrics
                metrics = self.compute_distribution_metrics(distances)
                metrics_list.append(metrics)

                # Compute Wasserstein distance to reference
                wd = wasserstein_distance(reference_distances, distances)
                wasserstein_dists.append(wd)

                # Compute KS statistic
                ks_stat, _ = stats.ks_2samp(reference_distances, distances)
                ks_stats.append(ks_stat)

            rep_progress.close()

            # Aggregate results
            means = [m["mean"] for m in metrics_list]
            stds = [m["std"] for m in metrics_list]

            convergence_results["mean_errors"].append(
                abs(np.mean(means) - reference_metrics["mean"])
            )
            convergence_results["std_errors"].append(
                abs(np.mean(stds) - reference_metrics["std"])
            )
            convergence_results["wasserstein_distances"].append(
                np.mean(wasserstein_dists)
            )
            convergence_results["ks_statistics"].append(np.mean(ks_stats))
            convergence_results["mean_stds"].append(np.std(means))
            convergence_results["std_stds"].append(np.std(stds))

        sample_progress.close()

        return convergence_results, reference_distances, reference_metrics

    def plot_convergence_analysis(
        self,
        convergence_results,
        reference_distances,
        reference_metrics,
        save_path=None,
    ):
        """Create comprehensive convergence visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        sample_sizes = np.array(convergence_results["sample_sizes"])

        # 1. Mean convergence
        axes[0, 0].semilogx(
            sample_sizes, convergence_results["mean_errors"], "b-o", markersize=4
        )
        axes[0, 0].fill_between(
            sample_sizes,
            np.array(convergence_results["mean_errors"])
            - np.array(convergence_results["mean_stds"]),
            np.array(convergence_results["mean_errors"])
            + np.array(convergence_results["mean_stds"]),
            alpha=0.3,
        )
        axes[0, 0].set_xlabel("Number of Sampled Pairs")
        axes[0, 0].set_ylabel("Absolute Error in Mean")
        axes[0, 0].set_title("Mean Distance Convergence")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Standard deviation convergence
        axes[0, 1].semilogx(
            sample_sizes, convergence_results["std_errors"], "r-o", markersize=4
        )
        axes[0, 1].fill_between(
            sample_sizes,
            np.array(convergence_results["std_errors"])
            - np.array(convergence_results["std_stds"]),
            np.array(convergence_results["std_errors"])
            + np.array(convergence_results["std_stds"]),
            alpha=0.3,
        )
        axes[0, 1].set_xlabel("Number of Sampled Pairs")
        axes[0, 1].set_ylabel("Absolute Error in Std Dev")
        axes[0, 1].set_title("Standard Deviation Convergence")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Wasserstein distance (distribution similarity)
        axes[0, 2].semilogx(
            sample_sizes,
            convergence_results["wasserstein_distances"],
            "g-o",
            markersize=4,
        )
        axes[0, 2].set_xlabel("Number of Sampled Pairs")
        axes[0, 2].set_ylabel("Wasserstein Distance to Reference")
        axes[0, 2].set_title("Overall Distribution Convergence")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. KS statistic
        axes[1, 0].semilogx(
            sample_sizes,
            convergence_results["ks_statistics"],
            "purple",
            marker="o",
            markersize=4,
        )
        axes[1, 0].set_xlabel("Number of Sampled Pairs")
        axes[1, 0].set_ylabel("KS Statistic")
        axes[1, 0].set_title("Kolmogorov-Smirnov Test Statistic")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Reference distribution
        axes[1, 1].hist(reference_distances, bins=50, alpha=0.7, density=True)
        axes[1, 1].set_xlabel("Euclidean Distance")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title(
            f"Reference Distribution\n({len(reference_distances):,} samples)"
        )

        # 6. Summary statistics
        stats_text = f"""Reference Distribution Statistics:

Mean: {reference_metrics["mean"]:.4f}
Std:  {reference_metrics["std"]:.4f}
Median: {reference_metrics["median"]:.4f}
Min: {reference_metrics["min"]:.4f}
Max: {reference_metrics["max"]:.4f}
Skewness: {reference_metrics["skewness"]:.4f}
Kurtosis: {reference_metrics["kurtosis"]:.4f}

Dataset Info:
Samples: {self.n_samples:,}
Dimensions: {self.embedding_dim}
Total Possible Pairs: {self.n_samples * (self.n_samples - 1) // 2:,}
        """
        axes[1, 2].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[1, 2].transAxes,
            fontfamily="monospace",
            fontsize=10,
            verticalalignment="top",
        )
        axes[1, 2].set_title("Summary Statistics")
        axes[1, 2].axis("off")

        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved: {Path(save_path).name}")

        # Save individual plots for cumulative analysis
        if save_path:
            output_dir = Path(save_path).parent
            self._save_individual_plots(
                convergence_results, reference_distances, output_dir
            )

        plt.close()  # Close figure to free memory

        return fig

    def _save_individual_plots(
        self, convergence_results, reference_distances, output_dir
    ):
        """Save individual plots for cumulative analysis across pLMs"""
        sample_sizes = np.array(convergence_results["sample_sizes"])

        # Mean convergence plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogx(
            sample_sizes, convergence_results["mean_errors"], "b-o", markersize=4
        )
        ax.fill_between(
            sample_sizes,
            np.array(convergence_results["mean_errors"])
            - np.array(convergence_results["mean_stds"]),
            np.array(convergence_results["mean_errors"])
            + np.array(convergence_results["mean_stds"]),
            alpha=0.3,
        )
        ax.set_xlabel("Number of Sampled Pairs")
        ax.set_ylabel("Absolute Error in Mean")
        ax.set_title("Mean Distance Convergence")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "mean_convergence.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Standard deviation convergence plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogx(
            sample_sizes, convergence_results["std_errors"], "r-o", markersize=4
        )
        ax.fill_between(
            sample_sizes,
            np.array(convergence_results["std_errors"])
            - np.array(convergence_results["std_stds"]),
            np.array(convergence_results["std_errors"])
            + np.array(convergence_results["std_stds"]),
            alpha=0.3,
        )
        ax.set_xlabel("Number of Sampled Pairs")
        ax.set_ylabel("Absolute Error in Std Dev")
        ax.set_title("Standard Deviation Convergence")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "std_convergence.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Wasserstein distance plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogx(
            sample_sizes,
            convergence_results["wasserstein_distances"],
            "g-o",
            markersize=4,
        )
        ax.set_xlabel("Number of Sampled Pairs")
        ax.set_ylabel("Wasserstein Distance to Reference")
        ax.set_title("Overall Distribution Convergence")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "wasserstein_convergence.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # KS statistic plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogx(
            sample_sizes,
            convergence_results["ks_statistics"],
            "purple",
            marker="o",
            markersize=4,
        )
        ax.set_xlabel("Number of Sampled Pairs")
        ax.set_ylabel("KS Statistic")
        ax.set_title("Kolmogorov-Smirnov Test Statistic")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "ks_statistic.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Reference distribution plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(reference_distances, bins=50, alpha=0.7, density=True)
        ax.set_xlabel("Euclidean Distance")
        ax.set_ylabel("Density")
        ax.set_title(f"Reference Distribution ({len(reference_distances):,} samples)")
        plt.tight_layout()
        plt.savefig(
            output_dir / "reference_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("Individual plots saved for cumulative analysis")

    def recommend_sample_size(
        self, convergence_results, tolerance_mean=0.01, tolerance_wasserstein=0.05
    ):
        """
        This method has been completely removed - no sample size recommendations
        """
        pass

    def save_results(self, convergence_results, reference_metrics, output_dir):
        """Save analysis results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save convergence results as JSON
        results_path = output_path / "convergence_results.json"
        with open(results_path, "w") as f:
            json.dump(convergence_results, f, indent=2)

        # Save reference metrics as JSON
        metrics_path = output_path / "reference_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(reference_metrics, f, indent=2)

        # Save summary report
        summary_path = output_path / "analysis_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Distance Distribution Convergence Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {Path(self.filepath).name}\n")
            f.write(f"Analysis Date: {np.datetime64('today')}\n\n")

            f.write("Dataset Properties:\n")
            f.write(f"  Samples: {self.n_samples:,}\n")
            f.write(f"  Dimensions: {self.embedding_dim}\n")
            f.write(
                f"  Total Possible Pairs: {self.n_samples * (self.n_samples - 1) // 2:,}\n\n"
            )

            f.write("Analysis Parameters:\n")
            f.write(
                f"  Sample sizes tested: {len(convergence_results['sample_sizes'])}\n"
            )
            f.write(
                f"  Range: {convergence_results['sample_sizes'][0]:,} to {convergence_results['sample_sizes'][-1]:,}\n"
            )
            f.write("  Reference distribution size: 1,000,000 pairs\n\n")

            f.write("Results:\n")
            f.write(
                f"  Mean distance: {reference_metrics['mean']:.4f} ± {reference_metrics['std']:.4f}\n"
            )

        print(f"Results saved to: {output_dir}")


def create_cumulative_plots(output_dir, results):
    """
    Create cumulative plots comparing convergence across all successfully processed pLMs
    """
    output_path = Path(output_dir)

    # Include both successful and skipped results (skipped means results already exist)
    available_files = [
        filename
        for filename, result in results.items()
        if result["status"] in ["success", "skipped"]
    ]

    if len(available_files) < 2:
        print(
            f"Need at least 2 analyses for cumulative plots, found {len(available_files)}"
        )
        return

    print(f"Creating cumulative plots for {len(available_files)} pLMs...")

    # Load convergence data from all available analyses
    all_data = {}
    sample_sizes = None

    for filename in available_files:
        plm_name = Path(filename).stem
        results_path = output_path / plm_name / "convergence_results.json"

        try:
            with open(results_path, "r") as f:
                data = json.load(f)
                all_data[plm_name] = data

                # Set sample sizes from first file (should be consistent across all)
                if sample_sizes is None:
                    sample_sizes = np.array(data["sample_sizes"])

        except Exception as e:
            print(f"Warning: Could not load data for {plm_name}: {e}")
            continue

    if len(all_data) < 2:
        print("Not enough valid data files found for cumulative plots")
        return

    # Create cumulative plots directory
    cumulative_dir = output_path / "cumulative_plots"
    cumulative_dir.mkdir(exist_ok=True)

    # Define colors for different pLMs
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_data)))

    # 1. Mean Convergence Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (plm_name, data) in enumerate(all_data.items()):
        ax.semilogx(
            sample_sizes,
            data["mean_errors"],
            marker="o",
            markersize=3,
            label=plm_name,
            color=colors[i],
            linewidth=2,
        )

        # Add error bars
        ax.fill_between(
            sample_sizes,
            np.array(data["mean_errors"]) - np.array(data["mean_stds"]),
            np.array(data["mean_errors"]) + np.array(data["mean_stds"]),
            alpha=0.2,
            color=colors[i],
        )

    ax.set_xlabel("Number of Sampled Pairs", fontsize=12)
    ax.set_ylabel("Absolute Error in Mean Distance", fontsize=12)
    ax.set_title(
        "Mean Distance Convergence Comparison Across pLMs",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(
        cumulative_dir / "mean_convergence_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Standard Deviation Convergence Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (plm_name, data) in enumerate(all_data.items()):
        ax.semilogx(
            sample_sizes,
            data["std_errors"],
            marker="s",
            markersize=3,
            label=plm_name,
            color=colors[i],
            linewidth=2,
        )

        # Add error bars
        ax.fill_between(
            sample_sizes,
            np.array(data["std_errors"]) - np.array(data["std_stds"]),
            np.array(data["std_errors"]) + np.array(data["std_stds"]),
            alpha=0.2,
            color=colors[i],
        )

    ax.set_xlabel("Number of Sampled Pairs", fontsize=12)
    ax.set_ylabel("Absolute Error in Standard Deviation", fontsize=12)
    ax.set_title(
        "Standard Deviation Convergence Comparison Across pLMs",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(
        cumulative_dir / "std_convergence_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Wasserstein Distance Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (plm_name, data) in enumerate(all_data.items()):
        ax.semilogx(
            sample_sizes,
            data["wasserstein_distances"],
            marker="^",
            markersize=3,
            label=plm_name,
            color=colors[i],
            linewidth=2,
        )

    ax.set_xlabel("Number of Sampled Pairs", fontsize=12)
    ax.set_ylabel("Wasserstein Distance to Reference", fontsize=12)
    ax.set_title(
        "Distribution Similarity Convergence Comparison Across pLMs",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(
        cumulative_dir / "wasserstein_convergence_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. KS Statistic Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (plm_name, data) in enumerate(all_data.items()):
        ax.semilogx(
            sample_sizes,
            data["ks_statistics"],
            marker="d",
            markersize=3,
            label=plm_name,
            color=colors[i],
            linewidth=2,
        )

    ax.set_xlabel("Number of Sampled Pairs", fontsize=12)
    ax.set_ylabel("Kolmogorov-Smirnov Statistic", fontsize=12)
    ax.set_title(
        "KS Test Statistic Convergence Comparison Across pLMs",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(
        cumulative_dir / "ks_statistic_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 5. Reference Distribution Comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    # Load reference metrics for distribution comparison
    for i, filename in enumerate(available_files):
        plm_name = Path(filename).stem
        metrics_path = output_path / plm_name / "reference_metrics.json"

        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            # Create a simple bar chart of key statistics
            stats_to_plot = ["mean", "std", "median"]
            x_pos = np.arange(len(stats_to_plot)) + i * 0.1
            values = [metrics[stat] for stat in stats_to_plot]

            ax.bar(
                x_pos, values, width=0.08, label=plm_name, color=colors[i], alpha=0.7
            )

        except Exception as e:
            print(f"Warning: Could not load metrics for {plm_name}: {e}")
            continue

    ax.set_xlabel("Distance Statistics", fontsize=12)
    ax.set_ylabel("Distance Value", fontsize=12)
    ax.set_title(
        "Reference Distribution Statistics Comparison Across pLMs",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(np.arange(len(stats_to_plot)) + (len(available_files) - 1) * 0.05)
    ax.set_xticklabels(["Mean", "Standard Deviation", "Median"])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        cumulative_dir / "reference_distribution_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 6. Reference Distribution Grid (NEW!)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()  # Make it easy to iterate

    print("Generating reference distribution grid...")

    # Load and plot each pLM's reference distribution
    for i, filename in enumerate(sorted(available_files)):
        if i >= 16:  # Safety check for grid size
            break

        plm_name = Path(filename).stem
        metrics_path = output_path / plm_name / "reference_metrics.json"

        # Try to load the actual reference distances if available
        ref_distances = None
        try:
            # First try to load the raw reference distances (if saved)
            ref_data_path = output_path / plm_name / "convergence_results.json"
            with open(ref_data_path, "r") as f:
                conv_data = json.load(f)
            # Reference distances aren't saved in JSON, so we'll use metrics to create approximate distribution
        except:
            pass

        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            # Create a representative distribution using the statistics
            # This is an approximation since we don't save the full reference distances
            mean_dist = metrics["mean"]
            std_dist = metrics["std"]

            # Generate approximate distribution based on statistics
            # Using a gamma distribution which often fits distance distributions well
            from scipy.stats import gamma

            # Estimate gamma parameters from mean and std
            # For gamma distribution: mean = a*scale, variance = a*scale^2
            # So: scale = variance/mean, a = mean/scale = mean^2/variance
            if std_dist > 0:
                variance = std_dist**2
                scale = variance / mean_dist
                a = mean_dist / scale

                # Generate x values for plotting
                x_min = max(0, mean_dist - 4 * std_dist)
                x_max = mean_dist + 4 * std_dist
                x = np.linspace(x_min, x_max, 1000)
                y = gamma.pdf(x, a, scale=scale)

                axes[i].plot(x, y, color=colors[i % len(colors)], linewidth=2)
                axes[i].fill_between(x, y, alpha=0.3, color=colors[i % len(colors)])
            else:
                # Fallback for zero std
                axes[i].axvline(mean_dist, color=colors[i % len(colors)], linewidth=2)

            # Formatting
            axes[i].set_title(plm_name, fontsize=10, fontweight="bold")
            axes[i].set_xlabel("Euclidean Distance", fontsize=8)
            axes[i].set_ylabel("Density", fontsize=8)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(labelsize=7)

            # Add key statistics as text
            stats_text = f"μ={mean_dist:.3f}\nσ={std_dist:.3f}"
            axes[i].text(
                0.05,
                0.95,
                stats_text,
                transform=axes[i].transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        except Exception as e:
            # Handle missing data
            axes[i].text(
                0.5,
                0.5,
                f"{plm_name}\n(No data)",
                transform=axes[i].transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="red",
            )
            axes[i].set_title(plm_name, fontsize=10, fontweight="bold", color="red")

    # Hide unused subplots if less than 16 pLMs
    for i in range(len(available_files), 16):
        axes[i].set_visible(False)

    plt.suptitle(
        "Reference Distance Distributions Across pLMs", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        cumulative_dir / "reference_distribution_grid.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("Reference distribution grid generated!")

    # Create summary report for cumulative analysis
    summary_path = cumulative_dir / "cumulative_analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Cumulative Distance Distribution Analysis Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Analysis Date: {np.datetime64('today')}\n")
        f.write(f"Number of pLMs analyzed: {len(all_data)}\n\n")

        f.write("Protein Language Models included:\n")
        for plm_name in sorted(all_data.keys()):
            f.write(f"  - {plm_name}\n")
        f.write("\n")

        f.write("Generated cumulative plots:\n")
        f.write("  - mean_convergence_comparison.png\n")
        f.write("  - std_convergence_comparison.png\n")
        f.write("  - wasserstein_convergence_comparison.png\n")
        f.write("  - ks_statistic_comparison.png\n")
        f.write("  - reference_distribution_comparison.png\n")
        f.write("  - reference_distribution_grid.png\n\n")

        f.write("Analysis Parameters:\n")
        f.write(f"  Sample sizes tested: {len(sample_sizes)}\n")
        f.write(f"  Range: {sample_sizes[0]:,} to {sample_sizes[-1]:,}\n")
        f.write("  Reference distribution size: 1,000,000 pairs\n\n")

        # Add summary statistics comparison
        f.write("Reference Distribution Statistics Summary:\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'pLM':<15} {'Mean':<10} {'Std':<10} {'Median':<10}\n")
        f.write("-" * 45 + "\n")

        for filename in sorted(available_files):
            plm_name = Path(filename).stem
            metrics_path = output_path / plm_name / "reference_metrics.json"

            try:
                with open(metrics_path, "r") as mf:
                    metrics = json.load(mf)
                f.write(
                    f"{plm_name:<15} {metrics['mean']:<10.4f} {metrics['std']:<10.4f} {metrics['median']:<10.4f}\n"
                )
            except:
                f.write(f"{plm_name:<15} {'Error':<10} {'Error':<10} {'Error':<10}\n")

    print(f"Cumulative plots saved to: {cumulative_dir}")
    print("Generated files:")
    print("  - mean_convergence_comparison.png")
    print("  - std_convergence_comparison.png")
    print("  - wasserstein_convergence_comparison.png")
    print("  - ks_statistic_comparison.png")
    print("  - reference_distribution_comparison.png")
    print("  - reference_distribution_grid.png")
    print("  - cumulative_analysis_summary.txt")


def run_analysis_batch(
    input_path, output_dir, sample_sizes=None, n_repetitions=3, reference_size=1000000
):
    """
    Run analysis on either a single HDF5 file or all HDF5 files in a directory
    """

    input_path = Path(input_path)

    if input_path.is_file():
        # Single file processing
        if not input_path.suffix == ".h5":
            raise ValueError(
                f"Input file must be an HDF5 file (.h5), got: {input_path.suffix}"
            )

        hdf5_files = [input_path]
        print(f"Processing single file: {input_path}")

    elif input_path.is_dir():
        # Directory processing
        hdf5_files = list(input_path.glob("*.h5"))
        if not hdf5_files:
            raise ValueError(f"No HDF5 files found in directory: {input_path}")

        hdf5_files.sort()  # Process in consistent order
        print(f"Found {len(hdf5_files)} HDF5 files in directory: {input_path}")
        print("Files to process:")
        for file_path in hdf5_files:
            print(f"  - {file_path.name}")
        print()

    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    # Process each file
    results = {}

    for i, hdf5_file in enumerate(hdf5_files, 1):
        print(f"[{i}/{len(hdf5_files)}] Processing: {hdf5_file.name}")
        print("-" * 60)

        try:
            result = run_analysis(
                data_source=str(hdf5_file),
                output_dir=output_dir,
                sample_sizes=sample_sizes,
                n_repetitions=n_repetitions,
                reference_size=reference_size,
            )

            # Handle skipped files (when results already exist)
            if result[0] is None:  # analyzer is None when skipped
                results[hdf5_file.name] = {
                    "status": "skipped",
                    "reason": "Results already exist",
                }
            else:
                analyzer, convergence_results = result
                results[hdf5_file.name] = {"status": "success"}

        except Exception as e:
            print(f"Error processing {hdf5_file.name}: {str(e)}")
            results[hdf5_file.name] = {"error": str(e), "status": "failed"}

        print()

    # Summary report
    print("=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results.values() if r["status"] == "success")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    failed = len(results) - successful - skipped

    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print()

    if successful > 0:
        print("Successfully processed files:")
        for filename, result in results.items():
            if result["status"] == "success":
                print(f"  - {filename}")

    if skipped > 0:
        print("\nSkipped files (results already exist):")
        for filename, result in results.items():
            if result["status"] == "skipped":
                print(f"  - {filename}")

    if failed > 0:
        print("\nFailed files:")
        for filename, result in results.items():
            if result["status"] == "failed":
                print(f"  - {filename}: {result['error']}")

    print(f"\nAll results saved to: {output_dir}")

    # Generate cumulative plots if we have multiple analyses (including skipped ones)
    if successful + skipped >= 2:  # Include skipped files as they have existing results
        print("\n" + "=" * 60)
        print("GENERATING CUMULATIVE ANALYSIS")
        print("=" * 60)
        create_cumulative_plots(output_dir, results)
    else:
        print(
            f"\nSkipping cumulative plots (need ≥2 analyses, found {successful + skipped})"
        )

    return results


def run_analysis(
    data_source, output_dir, sample_sizes=None, n_repetitions=3, reference_size=1000000
):
    """
    Complete analysis workflow for a single HDF5 file
    """

    # Create subdirectory based on input filename
    input_path = Path(data_source)
    input_filename = input_path.stem  # Filename without extension
    final_output_dir = Path(output_dir) / input_filename

    # Check if results already exist
    if final_output_dir.exists():
        print(f"Results already exist for {input_filename}, skipping...")
        return None, None

    final_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {final_output_dir}")

    # Initialize analyzer
    analyzer = DistributionConvergenceAnalyzer(data_source)

    # Load data
    analyzer.load_embeddings()

    # Set default sample sizes if not provided
    if sample_sizes is None:
        sample_sizes = np.logspace(3, 6, 25).astype(int)  # 1K to 1M samples

    # Run convergence analysis
    print("Starting convergence analysis...")
    convergence_results, reference_distances, reference_metrics = (
        analyzer.analyze_convergence(
            sample_sizes=sample_sizes,
            n_repetitions=n_repetitions,
            reference_size=reference_size,
        )
    )

    # Plot results
    plot_path = final_output_dir / "convergence_analysis.png"
    print("Generating plots...")
    analyzer.plot_convergence_analysis(
        convergence_results, reference_distances, reference_metrics, save_path=plot_path
    )

    # Save all results
    analyzer.save_results(convergence_results, reference_metrics, final_output_dir)

    print("Analysis complete!")

    return analyzer, convergence_results


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze convergence of distance distribution sampling for high-dimensional embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/processed/sprot_pre2024/embeddings/prott5.h5",
        help="Path to input HDF5 file or directory containing HDF5 files",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/distance_analysis",
        help="Output directory for all analysis results and plots",
    )

    parser.add_argument(
        "--reference-size",
        type=int,
        default=1000000,
        help="Size of reference distribution for comparison",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per sample size for error estimation",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=1000,
        help="Minimum number of samples to test",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000000,
        help="Maximum number of samples to test",
    )

    parser.add_argument(
        "--num-points",
        type=int,
        default=25,
        help="Number of sample sizes to test (logarithmically spaced)",
    )

    return parser.parse_args()


def main():
    """Main function to run the complete analysis"""

    # Parse command line arguments
    args = parse_arguments()

    # Check if input path exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

    # Generate sample sizes
    sample_sizes = np.logspace(
        np.log10(args.min_samples), np.log10(args.max_samples), args.num_points
    ).astype(int)

    # Determine input type
    if input_path.is_file():
        input_type = "Single HDF5 file"
    elif input_path.is_dir():
        hdf5_count = len(list(input_path.glob("*.h5")))
        input_type = f"Directory with {hdf5_count} HDF5 files"
    else:
        print(f"Error: Input must be either an HDF5 file or directory: {args.input}")
        sys.exit(1)

    # Print configuration
    print("Distance Distribution Convergence Analysis")
    print("=" * 50)
    print(f"Input: {args.input} ({input_type})")
    print(f"Output: {args.output}")
    print(
        f"Sample range: {args.min_samples:,} to {args.max_samples:,} ({args.num_points} points)"
    )
    print(f"Reference size: {args.reference_size:,}")
    print()

    try:
        # Run the batch analysis
        run_analysis_batch(
            input_path=args.input,
            output_dir=args.output,
            sample_sizes=sample_sizes,
            n_repetitions=args.repetitions,
            reference_size=args.reference_size,
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
