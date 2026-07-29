import h5py
import numpy as np
from tqdm import tqdm
import argparse
import json
import warnings

# Suppress specific numpy warnings for known numerical edge cases
# We handle invalid values explicitly in the code
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")


def load_embeddings(h5_path, max_embeddings=None):
    """Load all embeddings from H5 file efficiently.

    ``max_embeddings`` truncates to the first N keys. It exists only for quick
    smoke runs and MUST stay opt-in: an earlier version of this function had a
    hardcoded ``[:500]`` slice with no way to disable it, which silently
    produced a 500-protein histogram that looked like a full-cohort result
    (see out/tmp/prottucker.npz, n_embeddings=500).
    """
    embeddings = []
    ids = []

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if max_embeddings is not None:
            keys = keys[:max_embeddings]
            print(
                f"WARNING: --max-embeddings={max_embeddings} — using {len(keys)} of "
                f"{len(f.keys())} embeddings. This is NOT a full-cohort result."
            )
        else:
            print(f"Loading {len(keys)} embeddings...")
        for key in tqdm(keys, desc="Loading", unit="emb"):
            embeddings.append(f[key][:])
            ids.append(key)

    # Stack into single array and convert to float32 for efficiency
    embeddings = np.vstack(embeddings).astype(np.float32)
    print(f"Loaded embeddings shape: {embeddings.shape}")

    # Check for data quality issues
    n_nan = np.isnan(embeddings).any(axis=1).sum()
    n_inf = np.isinf(embeddings).any(axis=1).sum()

    if n_nan > 0 or n_inf > 0:
        print(f"WARNING: Found {n_nan} embeddings with NaN and {n_inf} with Inf values")
        print("Cleaning data by replacing NaN with 0 and clipping extreme values...")

        # Replace NaN with 0
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip extreme values to prevent overflow
        embeddings = np.clip(embeddings, -1e10, 1e10)

    # Check for extremely large values that might cause overflow
    max_val = np.abs(embeddings).max()
    mean_val = np.abs(embeddings).mean()
    std_val = np.abs(embeddings).std()

    print(f"Data statistics: mean={mean_val:.2f}, std={std_val:.2f}, max={max_val:.2f}")

    if max_val > 1e6:
        print(f"WARNING: Found very large values (max: {max_val:.2e})")
        print(
            "This may cause numerical overflow. Consider normalizing your embeddings."
        )

    # Check for zero vectors (can cause divide by zero)
    norms = np.linalg.norm(embeddings, axis=1)
    n_zero = (norms == 0).sum()
    if n_zero > 0:
        print(
            f"WARNING: Found {n_zero} zero vectors - replacing with small random values"
        )
        zero_mask = norms == 0
        embeddings[zero_mask] = np.random.randn(n_zero, embeddings.shape[1]) * 1e-6

    return embeddings, ids


def compute_distance_range(embeddings):
    """
    Compute theoretical min/max Euclidean distance from embedding norms.

    For Euclidean distance:
    - Minimum distance = 0 (vector to itself)
    - Maximum distance = ||a|| + ||b|| when vectors point in opposite directions
    - Worst case: two vectors with maximum norm pointing opposite directions
    - Therefore: max_distance = 2 * max_norm
    """
    print("Computing distance range from embedding norms...")

    # Compute L2 norms of all embeddings
    norms = np.linalg.norm(embeddings, axis=1)

    # Find statistics
    max_norm = float(norms.max())
    mean_norm = float(norms.mean())
    min_norm = float(norms.min())

    print(
        f"Embedding norms - min: {min_norm:.6f}, mean: {mean_norm:.6f}, max: {max_norm:.6f}"
    )

    # Theoretical bounds
    min_distance = 0.0  # Distance to itself
    max_distance = 2.0 * max_norm  # Two max-norm vectors in opposite directions

    print(f"Theoretical distance range: [0.000000, {max_distance:.6f}]")

    return min_distance, max_distance


def compute_histogram_batched(embeddings, n_bins, min_val, max_val, batch_size=1_000):
    """
    Compute histogram of all-vs-all Euclidean distances without storing full matrix.

    Args:
        embeddings: Embeddings (N x D)
        n_bins: Number of histogram bins
        min_val: Minimum distance value
        max_val: Maximum distance value
        batch_size: Number of rows to process at once

    Returns:
        histogram: Array of counts for each bin
        bin_edges: Edges of the bins
    """
    n = len(embeddings)

    # Initialize histogram
    histogram = np.zeros(n_bins, dtype=np.int64)

    # Create bin edges and precompute inverse width for faster binning
    bin_edges = np.linspace(min_val, max_val, n_bins + 1, dtype=np.float32)
    bin_width = (max_val - min_val) / n_bins

    # Precompute squared norms for all embeddings with float64 for stability
    print("Precomputing norms...")
    norms_sq = np.sum(embeddings**2, axis=1, dtype=np.float64)

    # Total number of comparisons for progress bar (excluding diagonal)
    total_comparisons = n * (n - 1) // 2  # Upper triangle excluding diagonal

    print(f"Computing histogram with {n_bins:,} bins...")
    print(f"Total pairwise comparisons (excluding self-comparisons): {total_comparisons:,}")

    # Process in batches with tqdm progress bar
    with tqdm(total=n, desc="Computing distances", unit="emb") as pbar:
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch_size_actual = end_i - i

            # Get batch data and use float64 for numerical stability
            batch = embeddings[i:end_i].astype(np.float64)
            batch_norms_sq = norms_sq[i:end_i]

            # Compute distances for this batch against all embeddings from i onwards
            target_embeddings = embeddings[i:].astype(np.float64)
            target_norms_sq = norms_sq[i:]

            # Compute dot products with better numerical stability
            dot_products = batch @ target_embeddings.T

            # Compute squared distances using broadcasting
            distances_sq = (
                batch_norms_sq[:, np.newaxis]
                + target_norms_sq[np.newaxis, :]
                - 2 * dot_products
            )

            # Handle numerical errors (negative values due to floating point)
            distances_sq = np.maximum(distances_sq, 0)

            # Compute actual distances
            distances = np.sqrt(distances_sq)

            # Keep each unordered pair EXACTLY once.
            #
            # Batch row j is global index i+j; target column k is global index
            # i+k. Masking only the diagonal (k == j) left every intra-batch pair
            # counted twice — once as (i+j, i+k) and once as (i+k, i+j) — while
            # cross-batch pairs were counted once. The `histogram * 2` below then
            # assumed a clean upper triangle, so the totals came out
            # batch-size-dependent: on a 40-protein set the histogram summed to
            # 1720 / 1920 / 3120 for batch sizes 5 / 10 / 40 against a true 1560.
            # Requiring k > j restores the strict upper triangle and makes the
            # result independent of --batch-size.
            cols = np.arange(distances.shape[1])
            upper_mask = cols[np.newaxis, :] > np.arange(batch_size_actual)[:, np.newaxis]

            valid_mask = np.isfinite(distances) & upper_mask
            distances_valid = distances[valid_mask]

            # Manual binning (faster than np.digitize for large data)
            if len(distances_valid) > 0:
                bin_indices = ((distances_valid - min_val) / bin_width).astype(np.int32)
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                # Update histogram using bincount (very fast)
                counts = np.bincount(bin_indices.ravel(), minlength=n_bins)
                histogram += counts[:n_bins].astype(np.int64)

            # Warn if we found invalid values
            n_invalid = (~valid_mask).sum()
            if n_invalid > 0:
                tqdm.write(
                    f"Warning: Found {n_invalid} invalid distance values in batch {i}-{end_i}"
                )

            pbar.update(batch_size_actual)

    # We counted upper triangle (excluding diagonal), so multiply by 2 for full matrix
    # No need to subtract diagonal since we already excluded it
    histogram_full = histogram * 2

    return histogram_full, bin_edges


def save_histogram(output_path, histogram, bin_edges, metadata):
    """Save histogram results to multiple formats."""

    # Save as NumPy format (efficient)
    npz_path = output_path.replace(".json", ".npz")
    np.savez_compressed(
        npz_path,
        histogram=histogram,
        bin_edges=bin_edges,
        **metadata,
    )

    # Save as JSON (human readable) - only store essential data
    results = {
        "metadata": metadata,
        "histogram": histogram.tolist(),
        "statistics": {
            "total_comparisons": int(histogram.sum()),
            "mean_distance": float(
                np.average((bin_edges[:-1] + bin_edges[1:]) / 2, weights=histogram)
            ),
            "most_common_bin_index": int(np.argmax(histogram)),
            "most_common_bin_range": [
                float(bin_edges[np.argmax(histogram)]),
                float(bin_edges[np.argmax(histogram) + 1]),
            ],
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to:")
    print(f"  - {output_path} (JSON)")
    print(f"  - {npz_path} (NumPy compressed)")


def print_histogram_stats(histogram, bin_edges):
    """Print summary statistics of the histogram."""
    total = histogram.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mean = np.average(bin_centers, weights=histogram)

    # Find percentiles
    cumsum = np.cumsum(histogram)
    p25_idx = np.searchsorted(cumsum, 0.25 * total)
    p50_idx = np.searchsorted(cumsum, 0.50 * total)
    p75_idx = np.searchsorted(cumsum, 0.75 * total)

    print("\n" + "=" * 60)
    print("HISTOGRAM STATISTICS")
    print("=" * 60)
    print(f"Total comparisons: {total:,}")
    print(f"Number of bins: {len(histogram):,}")
    print(f"Distance range: [{bin_edges[0]:.6f}, {bin_edges[-1]:.6f}]")
    print(f"Mean distance: {mean:.6f}")
    print(f"25th percentile: ~{bin_centers[p25_idx]:.6f}")
    print(f"50th percentile: ~{bin_centers[p50_idx]:.6f}")
    print(f"75th percentile: ~{bin_centers[p75_idx]:.6f}")
    print(
        f"\nMost common bin: [{bin_edges[np.argmax(histogram)]:.6f}, "
        f"{bin_edges[np.argmax(histogram) + 1]:.6f}] "
        f"with {histogram.max():,} comparisons"
    )
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute all-vs-all embedding Euclidean distance histogram"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input H5 file with embeddings",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="distance_histogram.json",
        help="Output file path (JSON)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=1_000_000,
        help="Number of histogram bins (default: 1M)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000,
        help="Batch size for processing (adjust based on memory)",
    )
    parser.add_argument(
        "--max-embeddings",
        type=int,
        default=None,
        help=(
            "Smoke-test escape hatch: use only the first N embeddings. "
            "Default is no limit — never set this for a reported result."
        ),
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=None,
        help="Minimum distance (auto-compute if not specified)",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=None,
        help="Maximum distance (auto-compute if not specified)",
    )

    args = parser.parse_args()

    print(f"Loading embeddings from {args.input}")
    embeddings, ids = load_embeddings(args.input, args.max_embeddings)
    n_embeddings = len(embeddings)

    # Find distance range
    if args.min_dist is None or args.max_dist is None:
        min_dist, max_dist = compute_distance_range(embeddings)
        # Allow manual override
        if args.min_dist is not None:
            min_dist = args.min_dist
        if args.max_dist is not None:
            max_dist = args.max_dist
    else:
        min_dist, max_dist = args.min_dist, args.max_dist
        print(f"Using manual distance range: [{min_dist:.6f}, {max_dist:.6f}]")

    # Compute histogram
    histogram, bin_edges = compute_histogram_batched(
        embeddings, args.n_bins, min_dist, max_dist, args.batch_size
    )

    # Print statistics
    print_histogram_stats(histogram, bin_edges)

    # Save results
    metadata = {
        "n_embeddings": n_embeddings,
        "embedding_dim": int(embeddings.shape[1]),
        "n_bins": args.n_bins,
        "min_distance": float(min_dist),
        "max_distance": float(max_dist),
        "total_comparisons": int(n_embeddings * n_embeddings),
    }

    save_histogram(args.output, histogram, bin_edges, metadata)

    print("Done!")


if __name__ == "__main__":
    main()
