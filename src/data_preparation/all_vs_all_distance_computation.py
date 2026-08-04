#!/usr/bin/env python3
"""
All-vs-All Embedding Distance Computation Script

This script computes euclidean distances between ALL proteins for all available
protein language model (PLM) embeddings. Instead of computing distances only for
specific protein pairs, this script discovers all proteins from the embedding files
and computes an all-vs-all distance matrix.

The script also generates all cache files needed for the pairwise embedding
comparison visualization, making it a one-stop solution for comprehensive
protein embedding analysis.

Features:
- Discovers all proteins from H5 embedding files
- Computes all-vs-all distance matrices for each embedding type
- Generates visualization cache files (hexbin, correlation, wasserstein, distribution)
- Handles memory efficiently with chunked processing
- Supports resuming interrupted computations
- Optional sampling for testing/memory constraints

Usage:
    # Basic usage - compute all distances and cache files
    uv run python src/data_preparation/all_vs_all_distance_computation.py \
        --embeddings_dir data/processed/sprot_embs \
        --output_dir out/all_vs_all_analysis \
        --max_proteins 1000

    # With custom settings
    uv run python src/data_preparation/all_vs_all_distance_computation.py \
        --embeddings_dir data/processed/sprot_embs \
        --output_dir out/all_vs_all_analysis \
        --max_proteins 5000 \
        --chunk_size 100 \
        --precision 3

    # Resume computation
    uv run python src/data_preparation/all_vs_all_distance_computation.py \
        --embeddings_dir data/processed/sprot_embs \
        --output_dir out/all_vs_all_analysis \
        --max_proteins 1000 \
        --resume
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import polars as pl
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from shared.embedding_names import is_iid_random_baseline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AllVsAllEmbeddingAnalyzer:
    """
    Comprehensive analyzer for computing all-vs-all protein embedding distances
    and generating visualization cache files.

    This class handles:
    1. Discovery of all proteins from embedding files
    2. All-vs-all distance computation for multiple embedding types
    3. Generation of visualization cache files for downstream analysis
    4. Memory-efficient processing with chunked computation
    """

    def __init__(
        self,
        embeddings_dir: Path,
        output_dir: Path,
        max_proteins: Optional[int] = None,
        chunk_size: int = 100,
        precision: int = 4,
    ):
        """
        Initialize the analyzer.

        Args:
            embeddings_dir: Directory containing H5 embedding files
            output_dir: Directory to save outputs and cache files
            max_proteins: Maximum number of proteins to analyze (None for all)
            chunk_size: Number of proteins to process in each chunk
            precision: Number of decimal places for distance values
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.max_proteins = max_proteins
        self.chunk_size = chunk_size
        self.precision = precision

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize embedding discovery
        self.embedding_files = self._discover_embedding_files()
        self.embedding_info = self._get_embedding_info()
        self.protein_universe = self._discover_protein_universe()

    def _discover_embedding_files(self) -> List[Path]:
        """Find all H5 embedding files in the directory."""
        embedding_files = list(self.embeddings_dir.glob("*.h5"))
        if not embedding_files:
            raise FileNotFoundError(f"No H5 files found in {self.embeddings_dir}")

        logger.info(f"Found {len(embedding_files)} embedding files:")
        for file in sorted(embedding_files):
            logger.info(f"  - {file.name}")

        return sorted(embedding_files)

    def _get_embedding_info(self) -> Dict[str, Dict]:
        """Get information about each embedding file (dimensions, protein count)."""
        embedding_info = {}

        for emb_file in self.embedding_files:
            embedding_name = emb_file.stem  # Remove .h5 extension

            try:
                with h5py.File(emb_file, "r") as f:
                    # Get sample embedding dataset
                    first_key = next(iter(f))
                    sample_dataset = f[first_key]

                    # Get shape from dataset metadata
                    sample_shape = sample_dataset.shape
                    dimensions = sample_shape[-1]  # Last dimension is embedding size

                    # Get protein count
                    protein_count = len(f.keys())
                    protein_ids = list(f.keys())

                    embedding_info[embedding_name] = {
                        "file_path": emb_file,
                        "dimensions": dimensions,
                        "protein_count": protein_count,
                        "sample_shape": sample_shape,
                        "protein_ids": protein_ids,
                    }

                    logger.info(
                        f"{embedding_name}: {protein_count} proteins, "
                        f"dim={dimensions}, shape={sample_shape}"
                    )

            except Exception as e:
                logger.error(f"Error reading {emb_file}: {e}")
                continue

        return embedding_info

    def _discover_protein_universe(self) -> List[str]:
        """
        Discover the universe of proteins across all embedding files.

        Returns:
            List of protein IDs present in all embedding files (intersection)
        """
        if not self.embedding_info:
            return []

        # Get intersection of all protein sets
        protein_sets = [
            set(info["protein_ids"]) for info in self.embedding_info.values()
        ]
        protein_universe = set.intersection(*protein_sets)

        logger.info(f"Found {len(protein_universe)} proteins common to all embeddings")

        # Sort for consistent ordering
        protein_list = sorted(list(protein_universe))

        # Apply max_proteins limit if specified
        if self.max_proteins and len(protein_list) > self.max_proteins:
            protein_list = protein_list[: self.max_proteins]
            logger.info(f"Limited analysis to {len(protein_list)} proteins")

        logger.info(f"Final protein universe: {len(protein_list)} proteins")

        # Warn about very large datasets
        if len(protein_list) > 100000:
            total_pairs = len(protein_list) ** 2
            logger.warning("=" * 60)
            logger.warning("LARGE DATASET WARNING")
            logger.warning("=" * 60)
            logger.warning(
                f"Processing {len(protein_list):,} proteins will generate {total_pairs:,} pairs"
            )
            logger.warning("This will require significant time and storage space.")
            logger.warning(
                "Consider using --max_proteins to limit the analysis for testing."
            )
            logger.warning(
                "Estimated storage: ~"
                + f"{total_pairs * len(self.embedding_info) * 8 / (1024**3):.1f}"
                + " GB"
            )
            logger.warning("=" * 60)

        return protein_list

    def _load_embedding_for_proteins(
        self, embedding_file: Path, protein_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Load embeddings for specific proteins from an H5 file.

        Args:
            embedding_file: Path to H5 embedding file
            protein_ids: List of protein IDs to load

        Returns:
            Dictionary mapping protein ID to embedding vector
        """
        embeddings = {}

        with h5py.File(embedding_file, "r") as f:
            for protein_id in protein_ids:
                if protein_id in f:
                    embedding = f[protein_id][:]
                    # If embedding is 2D (sequence-level), take mean to get protein-level
                    if embedding.ndim > 1:
                        embedding = np.mean(embedding, axis=0)
                    embeddings[protein_id] = embedding

        return embeddings

    def _compute_distance_chunk(
        self, protein_chunk: List[str], all_embeddings: Dict[str, np.ndarray]
    ) -> List[Tuple[str, str, float]]:
        """
        Compute distances for a chunk of proteins against all other proteins using vectorized operations.

        Args:
            protein_chunk: List of proteins to process as queries
            all_embeddings: Dictionary of all loaded embeddings

        Returns:
            List of (query_id, target_id, distance) tuples
        """
        # Filter to only valid proteins that have embeddings
        valid_chunk_proteins = [p for p in protein_chunk if p in all_embeddings]
        valid_target_proteins = [
            p for p in self.protein_universe if p in all_embeddings
        ]

        if not valid_chunk_proteins or not valid_target_proteins:
            return []

        # Create query embeddings matrix
        query_embeddings = np.stack([all_embeddings[p] for p in valid_chunk_proteins])

        distances = []

        # For very large target sets, process targets in sub-chunks to manage memory
        target_chunk_size = min(10_000, len(valid_target_proteins))

        for target_start in range(0, len(valid_target_proteins), target_chunk_size):
            target_end = min(
                target_start + target_chunk_size, len(valid_target_proteins)
            )
            target_chunk = valid_target_proteins[target_start:target_end]

            # Create target embeddings matrix for this sub-chunk
            target_embeddings = np.stack([all_embeddings[p] for p in target_chunk])

            # Vectorized distance computation
            distances_matrix = cdist(
                query_embeddings, target_embeddings, metric="euclidean"
            )

            # Convert chunk results to list of tuples
            for i, query_id in enumerate(valid_chunk_proteins):
                for j, target_id in enumerate(target_chunk):
                    distance = float(distances_matrix[i, j])
                    distances.append((query_id, target_id, distance))

        return distances

    def compute_all_vs_all_distances(self, resume: bool = False) -> pl.DataFrame:
        """
        Compute all-vs-all distances for all embeddings and create a consolidated DataFrame.

        Args:
            resume: Whether to resume from existing partial results

        Returns:
            DataFrame with columns: query, target, dist_<embedding1>, dist_<embedding2>, ...
        """
        logger.info("Starting all-vs-all distance computation...")

        output_file = self.output_dir / "all_vs_all_distances.parquet"

        # Initialize or load existing results
        if resume and output_file.exists():
            logger.info(f"Resuming from existing file: {output_file}")
            result_df = pl.read_parquet(output_file)
        else:
            # Calculate total pairs without creating them in memory
            total_pairs = len(self.protein_universe) ** 2
            logger.info(f"Total pairs to compute: {total_pairs:,}")

            # Create empty DataFrame with correct structure
            result_df = pl.DataFrame(
                {
                    "query": [],
                    "target": [],
                },
                schema={"query": pl.Utf8, "target": pl.Utf8},
            )

        # Process each embedding
        for embedding_name, embedding_info in self.embedding_info.items():
            dist_col = f"dist_{embedding_name}"

            # Check if already computed
            if dist_col in result_df.columns:
                existing_valid = result_df[dist_col].drop_nulls().len()
                logger.info(
                    f"  {embedding_name}: Already computed ({existing_valid} valid distances), skipping..."
                )
                continue

            logger.info(f"Computing distances for {embedding_name}...")

            try:
                # Load all embeddings for this model
                logger.info(
                    f"  Loading embeddings from {embedding_info['file_path'].name}"
                )
                all_embeddings = self._load_embedding_for_proteins(
                    embedding_info["file_path"], self.protein_universe
                )

                logger.info(f"  Loaded {len(all_embeddings)} embeddings")

                # Compute distances in chunks and build DataFrame incrementally
                n_chunks = (
                    len(self.protein_universe) + self.chunk_size - 1
                ) // self.chunk_size

                # If this is the first embedding, we need to build the DataFrame structure
                if len(result_df) == 0:
                    logger.info("  Building DataFrame structure...")

                with tqdm(
                    total=n_chunks, desc=f"Computing {embedding_name} distances"
                ) as pbar:
                    chunk_dataframes = []

                    for i in range(0, len(self.protein_universe), self.chunk_size):
                        chunk = self.protein_universe[i : i + self.chunk_size]
                        chunk_distances = self._compute_distance_chunk(
                            chunk, all_embeddings
                        )

                        # Convert chunk distances to DataFrame
                        if chunk_distances:
                            chunk_df = pl.DataFrame(
                                {
                                    "query": [d[0] for d in chunk_distances],
                                    "target": [d[1] for d in chunk_distances],
                                    dist_col: [
                                        round(d[2], self.precision)
                                        for d in chunk_distances
                                    ],
                                }
                            )
                            chunk_dataframes.append(chunk_df)

                        pbar.update(1)

                # Combine all chunks
                if chunk_dataframes:
                    chunk_combined = pl.concat(chunk_dataframes)

                    if len(result_df) == 0:
                        # First embedding - create the base structure
                        result_df = chunk_combined
                    else:
                        # Subsequent embeddings - join on query/target
                        result_df = result_df.join(
                            chunk_combined.select(["query", "target", dist_col]),
                            on=["query", "target"],
                            how="left",
                        )

                # Save intermediate results
                logger.info("  Saving intermediate results...")
                result_df.write_parquet(output_file)

                # Log statistics
                if dist_col in result_df.columns:
                    valid_distances = result_df[dist_col].drop_nulls()
                    total_distances = len(result_df)
                    if len(valid_distances) > 0:
                        logger.info(
                            f"  {embedding_name}: {len(valid_distances):,}/{total_distances:,} "
                            f"valid distances, mean={valid_distances.mean():.3f}, "
                            f"std={valid_distances.std():.3f}"
                        )
                    else:
                        logger.warning(
                            f"  {embedding_name}: No valid distances computed!"
                        )
                else:
                    logger.warning(
                        f"  {embedding_name}: Distance column not found in result!"
                    )

                # Clean up memory
                del all_embeddings, chunk_dataframes
                if "chunk_combined" in locals():
                    del chunk_combined
                gc.collect()

            except Exception as e:
                logger.error(f"Error computing distances for {embedding_name}: {e}")
                # Add column of NaNs to maintain structure if result_df has data
                if len(result_df) > 0 and dist_col not in result_df.columns:
                    null_series = pl.Series(
                        name=dist_col, values=[None] * len(result_df)
                    )
                    result_df = result_df.with_columns(null_series)
                    result_df.write_parquet(output_file)

        logger.info(f"All-vs-all distance computation complete: {output_file}")
        return result_df

    def generate_visualization_cache_files(self, df: pl.DataFrame) -> Dict[str, Path]:
        """
        Generate all cache files needed for pairwise embedding comparison visualization.

        Args:
            df: DataFrame with distance columns

        Returns:
            Dictionary mapping cache type to file path
        """
        logger.info("Generating visualization cache files...")

        # Identify distance columns, excluding the i.i.d. random noise floor.
        # Untrained-architecture baselines (random_init_*) are NOT excluded — they
        # are the R1.9 control and must reach the figures; see shared.embedding_names.
        all_dist_cols = [col for col in df.columns if col.startswith("dist_")]
        dist_cols = [
            col
            for col in all_dist_cols
            if not is_iid_random_baseline(col.replace("dist_", ""))
        ]

        if not dist_cols:
            raise ValueError("No valid distance columns found in DataFrame")

        dropped = sorted(set(all_dist_cols) - set(dist_cols))
        if dropped:
            logger.warning(
                "EXCLUDED %d i.i.d. random baseline column(s) from the visualization "
                "cache: %s. random_init_* untrained architectures are NOT excluded.",
                len(dropped),
                ", ".join(c.replace("dist_", "") for c in dropped),
            )

        logger.info(f"Found {len(dist_cols)} distance columns: {dist_cols}")

        cache_files = {}

        # 1. Generate hexbin data
        logger.info("Generating hexbin cache data...")
        hexbin_cache = self._generate_hexbin_cache(df, dist_cols)
        hexbin_path = self.cache_dir / "hexbin_data.json"
        self._save_json_data(hexbin_cache, hexbin_path, "Hexbin data")
        cache_files["hexbin"] = hexbin_path

        # 2. Generate correlation data
        logger.info("Generating correlation cache data...")
        correlation_cache = self._generate_correlation_cache(df, dist_cols)
        correlation_path = self.cache_dir / "correlation_data.json"
        self._save_json_data(correlation_cache, correlation_path, "Correlation data")
        cache_files["correlation"] = correlation_path

        # 3. Generate Wasserstein distance data
        logger.info("Generating Wasserstein cache data...")
        wasserstein_cache = self._generate_wasserstein_cache(df, dist_cols)
        wasserstein_path = self.cache_dir / "wasserstein_data.json"
        self._save_json_data(wasserstein_cache, wasserstein_path, "Wasserstein data")
        cache_files["wasserstein"] = wasserstein_path

        # 4. Generate distribution data (raw)
        logger.info("Generating raw distribution cache data...")
        distribution_cache = self._generate_distribution_cache(
            df, dist_cols, normalize=False
        )
        distribution_path = self.cache_dir / "distribution_data.json"
        self._save_json_data(distribution_cache, distribution_path, "Distribution data")
        cache_files["distribution"] = distribution_path

        # 5. Generate distribution data (normalized)
        logger.info("Generating normalized distribution cache data...")
        distribution_norm_cache = self._generate_distribution_cache(
            df, dist_cols, normalize=True
        )
        distribution_norm_path = self.cache_dir / "distribution_normalized_data.json"
        self._save_json_data(
            distribution_norm_cache,
            distribution_norm_path,
            "Normalized distribution data",
        )
        cache_files["distribution_normalized"] = distribution_norm_path

        logger.info("All visualization cache files generated successfully!")
        return cache_files

    def _generate_hexbin_cache(
        self, df: pl.DataFrame, dist_cols: List[str], gridsize: int = 50
    ) -> Dict:
        """Generate hexbin data cache for distance comparisons."""
        hexbin_data = {
            "metadata": {
                "dist_cols": dist_cols,
                "gridsize": gridsize,
                "max_count": 0,
            }
        }

        n = len(dist_cols)
        total_pairs = n * (n - 1)

        with tqdm(total=total_pairs, desc="Computing hexbin data") as pbar:
            for i, col1 in enumerate(dist_cols):
                for j, col2 in enumerate(dist_cols):
                    if i == j:
                        continue

                    mask = ~(df[col1].is_nan() | df[col2].is_nan())
                    if mask.sum() < 10:
                        pbar.update(1)
                        continue

                    filtered_df = df.filter(mask)
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

        return hexbin_data

    def _generate_correlation_cache(
        self, df: pl.DataFrame, dist_cols: List[str]
    ) -> Dict:
        """Generate correlation data cache with confidence intervals."""
        n = len(dist_cols)
        correlations = np.full((n, n), np.nan)
        ci_lower = np.full((n, n), np.nan)
        ci_upper = np.full((n, n), np.nan)

        with tqdm(total=(n * (n + 1)) // 2, desc="Calculating correlations") as pbar:
            for i in range(n):
                for j in range(i, n):
                    mask = ~(df[dist_cols[i]].is_nan() | df[dist_cols[j]].is_nan())
                    if mask.sum() > 3:
                        filtered_df = df.filter(mask)
                        correlation, _ = stats.spearmanr(
                            filtered_df[dist_cols[i]],
                            filtered_df[dist_cols[j]],
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

        return {
            "correlations": correlations.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
            "columns": [col.replace("dist_", "") for col in dist_cols],
        }

    def _generate_wasserstein_cache(
        self, df: pl.DataFrame, dist_cols: List[str]
    ) -> Dict:
        """Generate Wasserstein distance data cache."""
        n = len(dist_cols)
        distances = np.zeros((n, n))

        with tqdm(
            total=(n * (n + 1)) // 2, desc="Computing Wasserstein distances"
        ) as pbar:
            for i in range(n):
                for j in range(i, n):
                    col1 = dist_cols[i]
                    col2 = dist_cols[j]

                    dist, sample_count = self._compute_wasserstein_pair(df, col1, col2)
                    distances[i, j] = distances[j, i] = dist

                    if np.isnan(dist) and sample_count < 10:
                        logger.warning(
                            f"Insufficient valid samples for {col1} vs {col2}: {sample_count} samples"
                        )

                    pbar.update(1)

        return {
            "distances": distances.tolist(),
            "columns": [col.replace("dist_", "") for col in dist_cols],
        }

    def _compute_wasserstein_pair(
        self, df: pl.DataFrame, col1: str, col2: str
    ) -> Tuple[float, int]:
        """Compute Wasserstein distance between two columns."""
        mask = ~(df[col1].is_nan() | df[col2].is_nan())
        valid_df = df.filter(mask)

        if len(valid_df) < 10:
            return np.nan, len(valid_df)

        # Normalize the distributions
        dist1_normalized = self._normalize_distribution(valid_df[col1])
        dist2_normalized = self._normalize_distribution(valid_df[col2])

        if len(dist1_normalized) == 0 or len(dist2_normalized) == 0:
            return np.nan, len(valid_df)

        try:
            dist = wasserstein_distance(dist1_normalized, dist2_normalized)
            return dist, len(valid_df)
        except Exception as e:
            logger.warning(
                f"Error computing Wasserstein distance for {col1} vs {col2}: {e}"
            )
            return np.nan, len(valid_df)

    def _normalize_distribution(self, x: pl.Series) -> np.ndarray:
        """Normalize a distribution to [0,1] range using MinMax scaling."""
        x_clean = x.drop_nulls().to_numpy()
        if len(x_clean) == 0:
            return x_clean

        x_clean = x_clean[np.isfinite(x_clean)]
        if len(x_clean) == 0:
            return x_clean

        if np.all(x_clean == x_clean[0]):
            return np.full_like(x_clean, 0.5)

        scaler = MinMaxScaler()
        return scaler.fit_transform(x_clean.reshape(-1, 1)).ravel()

    def _generate_distribution_cache(
        self, df: pl.DataFrame, dist_cols: List[str], normalize: bool = False
    ) -> Dict:
        """Generate distribution data cache for plotting."""
        distribution_data = {"metadata": {"normalized": normalize}, "distributions": {}}

        for col in tqdm(dist_cols, desc="Processing distributions"):
            col_series = df.select(col).drop_nulls().get_column(col)
            data = col_series.to_numpy()

            if normalize:
                data_normalized = self._normalize_distribution(col_series)
                data_clean = (
                    data_normalized[np.isfinite(data_normalized)]
                    if len(data_normalized) > 0
                    else np.array([])
                )
                x_range = np.linspace(0, 1, 500)
            else:
                data_clean = data[np.isfinite(data)] if len(data) > 0 else np.array([])
                if len(data_clean) > 0:
                    x_range = np.linspace(data_clean.min(), data_clean.max(), 500)
                else:
                    x_range = np.linspace(0, 1, 500)

            if len(data_clean) > 1:
                kernel = stats.gaussian_kde(data_clean)
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
                "min": float(data_clean.min()) if len(data_clean) > 0 else 0.0,
                "max": float(data_clean.max()) if len(data_clean) > 0 else 0.0,
            }

        return distribution_data

    def _save_json_data(self, data: Dict, save_path: Path, description: str):
        """Helper method to save JSON data with consistent logging."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(data, f)
        logger.info(f"{description} saved to {save_path}")

    def run_complete_analysis(
        self,
        resume: bool = False,
        generate_cache: bool = True,
    ) -> Dict[str, Path]:
        """
        Run the complete all-vs-all analysis pipeline.

        Args:
            resume: Whether to resume from existing results
            generate_cache: Whether to generate visualization cache files

        Returns:
            Dictionary mapping output type to file path
        """
        logger.info("=" * 60)
        logger.info("ALL-VS-ALL EMBEDDING ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Embeddings directory: {self.embeddings_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of proteins: {len(self.protein_universe)}")
        logger.info(f"Number of embeddings: {len(self.embedding_info)}")
        logger.info(
            f"Total distance computations: {len(self.protein_universe) ** 2 * len(self.embedding_info):,}"
        )

        output_files = {}

        try:
            # 1. Compute all-vs-all distances
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 1: COMPUTING ALL-VS-ALL DISTANCES")
            logger.info("=" * 60)

            df = self.compute_all_vs_all_distances(resume=resume)

            distance_file = self.output_dir / "all_vs_all_distances.parquet"
            output_files["distances"] = distance_file

            # 2. Generate visualization cache files
            if generate_cache:
                logger.info("\n" + "=" * 60)
                logger.info("PHASE 2: GENERATING VISUALIZATION CACHE")
                logger.info("=" * 60)

                cache_files = self.generate_visualization_cache_files(df)
                output_files.update(cache_files)

            # 3. Summary statistics
            logger.info("\n" + "=" * 60)
            logger.info("ANALYSIS COMPLETE")
            logger.info("=" * 60)

            distance_cols = [col for col in df.columns if col.startswith("dist_")]
            logger.info(f"Generated distance file: {distance_file}")
            logger.info(f"Total protein pairs: {len(df):,}")
            logger.info(f"Distance columns: {len(distance_cols)}")

            # Coverage statistics
            for col in distance_cols:
                valid_count = df[col].drop_nulls().len()
                coverage = valid_count / len(df) * 100
                if valid_count > 0:
                    mean_dist = df[col].mean()
                    std_dist = df[col].std()
                    logger.info(
                        f"  {col}: {coverage:.1f}% coverage, mean={mean_dist:.3f}, std={std_dist:.3f}"
                    )
                else:
                    logger.info(
                        f"  {col}: {coverage:.1f}% coverage (no valid distances)"
                    )

            if generate_cache:
                logger.info("\nVisualization cache files:")
                for cache_type, cache_path in cache_files.items():
                    logger.info(f"  {cache_type}: {cache_path}")

                logger.info("\nNext steps:")
                logger.info("  Use visualization script with generated data:")
                logger.info(
                    "  uv run python src/visualization/pairwise_embedding_comparison.py \\"
                )
                logger.info(f"    --data_path {distance_file} \\")
                logger.info(f"    --output_dir {self.output_dir / 'visualizations'}")

            return output_files

        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            raise


def main():
    """Main function to parse arguments and run the all-vs-all analysis."""
    parser = argparse.ArgumentParser(
        description="Compute all-vs-all embedding distances and generate visualization cache files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--embeddings_dir",
        type=Path,
        required=True,
        help="Path to directory containing H5 embedding files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save output files and cache",
    )
    parser.add_argument(
        "--max_proteins",
        type=int,
        default=None,
        help="Maximum number of proteins to analyze (None for all available)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of proteins to process in each chunk (for memory management)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places for distance values",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume computation from existing partial results",
    )
    parser.add_argument(
        "--skip_cache",
        action="store_true",
        help="Skip generation of visualization cache files",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {args.embeddings_dir}")
        sys.exit(1)

    if not args.embeddings_dir.is_dir():
        logger.error(f"Embeddings path is not a directory: {args.embeddings_dir}")
        sys.exit(1)

    try:
        # Initialize analyzer
        analyzer = AllVsAllEmbeddingAnalyzer(
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            max_proteins=args.max_proteins,
            chunk_size=args.chunk_size,
            precision=args.precision,
        )

        # Run complete analysis
        analyzer.run_complete_analysis(
            resume=args.resume,
            generate_cache=not args.skip_cache,
        )

        logger.info("All-vs-all analysis completed successfully!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
