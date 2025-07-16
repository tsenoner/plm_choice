#!/usr/bin/env python3
"""
Compute Embedding Distances Script

This script computes euclidean distances between protein pairs for all available
protein language model (PLM) embeddings. It takes a CSV file containing protein
pairs and a directory of H5 embedding files, then outputs a CSV with distance
columns for use with pairwise embedding comparison visualizations.

Usage:
    uv run python scripts/compute_embedding_distances.py \
        --input_csv data/processed/sprot_train/test.csv \
        --embeddings_dir data/processed/sprot_embs \
        --output_csv data/processed/sprot_train/test_with_distances.csv

    # With options
    uv run python scripts/compute_embedding_distances.py \
        --input_csv data/processed/sprot_train/test.csv \
        --embeddings_dir data/processed/sprot_embs \
        --output_csv data/processed/sprot_train/test_with_distances.csv \
        --sample_size 10000 \
        --batch_size 1000
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EmbeddingDistanceComputer:
    """
    Computes euclidean distances between protein pairs for multiple PLM embeddings.

    This class handles loading embeddings from H5 files, computing pairwise distances,
    and managing memory efficiently for large datasets.
    """

    def __init__(self, embeddings_dir: Path, batch_size: int = 1000):
        """
        Initialize the distance computer.

        Args:
            embeddings_dir: Directory containing H5 embedding files
            batch_size: Number of protein pairs to process in each batch
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.batch_size = batch_size
        self.embedding_files = self._discover_embedding_files()
        self.embedding_info = self._get_embedding_info()

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
                    # Get sample embedding dataset (don't load data into memory)
                    first_key = next(iter(f))
                    sample_dataset = f[first_key]

                    # Get shape from dataset metadata (fast - no data loading)
                    sample_shape = sample_dataset.shape
                    dimensions = sample_shape[-1]  # Last dimension is embedding size

                    # Get protein count (this is still slow but needed for logging)
                    protein_count = len(f.keys())

                    embedding_info[embedding_name] = {
                        "file_path": emb_file,
                        "dimensions": dimensions,
                        "protein_count": protein_count,
                        "sample_shape": sample_shape,
                    }

                    logger.info(
                        f"{embedding_name}: {protein_count} proteins, "
                        f"dim={dimensions}, shape={sample_shape}"
                    )

            except Exception as e:
                logger.error(f"Error reading {emb_file}: {e}")
                continue

        return embedding_info

    def _load_embedding_for_proteins(
        self, embedding_file: Path, protein_ids: Set[str]
    ) -> Dict[str, np.ndarray]:
        """
        Load embeddings for specific proteins from an H5 file.

        Args:
            embedding_file: Path to H5 embedding file
            protein_ids: Set of protein IDs to load

        Returns:
            Dictionary mapping protein ID to embedding vector
        """
        embeddings = {}

        with h5py.File(embedding_file, "r") as f:
            available_proteins = set(f.keys())
            valid_proteins = protein_ids.intersection(available_proteins)

            for protein_id in valid_proteins:
                embedding = f[protein_id][:]
                # If embedding is 2D (sequence-level), take mean to get protein-level
                if embedding.ndim > 1:
                    embedding = np.mean(embedding, axis=0)
                embeddings[protein_id] = embedding

        return embeddings

    def _compute_distance_batch(
        self,
        pairs_batch: pd.DataFrame,
        embedding_name: str,
        embeddings: Dict[str, np.ndarray],
    ) -> List[float]:
        """
        Compute euclidean distances for a batch of protein pairs.

        Args:
            pairs_batch: DataFrame with query and target columns
            embedding_name: Name of the embedding being processed
            embeddings: Dictionary of protein embeddings

        Returns:
            List of distances (NaN for missing proteins)
        """
        distances = []

        for _, row in pairs_batch.iterrows():
            query_id = row["query"]
            target_id = row["target"]

            if query_id in embeddings and target_id in embeddings:
                query_emb = embeddings[query_id]
                target_emb = embeddings[target_id]

                # Compute euclidean distance
                distance = np.linalg.norm(query_emb - target_emb)
                distances.append(float(distance))
            else:
                distances.append(np.nan)

        return distances

    def compute_distances_for_embedding(
        self, df: pd.DataFrame, embedding_name: str
    ) -> pd.Series:
        """
        Compute all distances for one embedding type.

        Args:
            df: DataFrame with protein pairs
            embedding_name: Name of embedding to process

        Returns:
            Series of distances for all pairs
        """
        logger.info(f"Computing distances for {embedding_name}...")

        # Get all unique protein IDs
        all_proteins = set(df["query"].unique()) | set(df["target"].unique())
        logger.info(f"  Found {len(all_proteins)} unique proteins in dataset")

        # Load embeddings for required proteins
        embedding_file = self.embedding_info[embedding_name]["file_path"]
        embeddings = self._load_embedding_for_proteins(embedding_file, all_proteins)

        available_proteins = len(embeddings)
        logger.info(
            f"  Loaded embeddings for {available_proteins}/{len(all_proteins)} proteins"
        )

        # Process in batches
        all_distances = []
        n_batches = (len(df) + self.batch_size - 1) // self.batch_size

        with tqdm(total=len(df), desc=f"Computing {embedding_name} distances") as pbar:
            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i : i + self.batch_size]
                batch_distances = self._compute_distance_batch(
                    batch, embedding_name, embeddings
                )
                all_distances.extend(batch_distances)
                pbar.update(len(batch))

        # Clean up memory
        del embeddings
        gc.collect()

        return pd.Series(all_distances, name=f"dist_{embedding_name}")

    def compute_all_distances(self, df: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
        """
        Compute distances for all embeddings, saving intermediate results to CSV.

        This method processes one embedding at a time and saves results incrementally
        to minimize memory usage and allow resuming interrupted computations.

        Args:
            df: Input DataFrame with protein pairs
            output_csv: Path to output CSV file for incremental saving

        Returns:
            DataFrame with all distance columns
        """
        logger.info(f"Computing distances for {len(self.embedding_info)} embeddings...")
        logger.info(f"Processing {len(df)} protein pairs")
        logger.info(f"Results will be saved incrementally to: {output_csv}")

        # Start with input DataFrame
        result_df = df.copy()

        # Load existing results if file exists
        if output_csv.exists():
            logger.info(f"Loading existing results from {output_csv}")
            result_df = pd.read_csv(output_csv)
            logger.info(f"Loaded existing file with {len(result_df.columns)} columns")

        for embedding_name in self.embedding_info.keys():
            dist_col = f"dist_{embedding_name}"

            # Check if this embedding is already computed
            if dist_col in result_df.columns:
                existing_valid = result_df[dist_col].notna().sum()
                logger.info(
                    f"  {embedding_name}: Already computed ({existing_valid} valid distances), skipping..."
                )
                continue

            logger.info(f"  Computing distances for {embedding_name}...")

            try:
                distances = self.compute_distances_for_embedding(df, embedding_name)
                result_df[dist_col] = distances

                # Save intermediate results immediately
                logger.info(f"  Saving intermediate results to {output_csv}")
                result_df.to_csv(output_csv, index=False)

                # Log statistics
                valid_distances = distances.dropna()
                if len(valid_distances) > 0:
                    logger.info(
                        f"  {embedding_name}: {len(valid_distances)}/{len(distances)} "
                        f"valid distances, mean={valid_distances.mean():.3f}, "
                        f"std={valid_distances.std():.3f}"
                    )
                else:
                    logger.warning(f"  {embedding_name}: No valid distances computed!")

            except Exception as e:
                logger.error(f"Error computing distances for {embedding_name}: {e}")
                # Add column of NaNs to maintain structure
                result_df[dist_col] = np.nan
                # Save even the failed result to maintain consistency
                result_df.to_csv(output_csv, index=False)

        return result_df


def validate_inputs(input_csv: Path, embeddings_dir: Path) -> Tuple[pd.DataFrame, Path]:
    """
    Validate input files and directories.

    Args:
        input_csv: Path to input CSV file
        embeddings_dir: Path to embeddings directory

    Returns:
        Tuple of (loaded DataFrame, validated embeddings directory)
    """
    # Check input CSV
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Load and validate CSV structure
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Check for required columns
    required_cols = ["query", "target"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Check embeddings directory
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    if not embeddings_dir.is_dir():
        raise NotADirectoryError(
            f"Embeddings path is not a directory: {embeddings_dir}"
        )

    return df, embeddings_dir


def main():
    """Main function to compute embedding distances."""
    parser = argparse.ArgumentParser(
        description="Compute euclidean distances between protein pairs for all PLM embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Path to CSV file containing protein pairs (must have 'query' and 'target' columns)",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=Path,
        required=True,
        help="Path to directory containing H5 embedding files",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path for output CSV file with distance columns",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit number of rows to process (for testing or memory constraints)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of protein pairs to process in each batch",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )

    args = parser.parse_args()

    # Validate inputs
    try:
        df, embeddings_dir = validate_inputs(args.input_csv, args.embeddings_dir)
    except (FileNotFoundError, ValueError, NotADirectoryError) as e:
        logger.error(f"Input validation failed: {e}")
        sys.exit(1)

    # Check output file - now we support resuming, so only error if overwrite is explicitly disabled
    if args.output_csv.exists() and not args.overwrite:
        logger.info(
            f"Output file exists: {args.output_csv}. Will resume computation from existing results."
        )
    elif args.output_csv.exists() and args.overwrite:
        logger.info(
            f"Output file exists: {args.output_csv}. Will overwrite as requested."
        )
        args.output_csv.unlink()  # Remove the file to start fresh

    # Sample data if requested
    if args.sample_size:
        original_size = len(df)
        df = df.head(args.sample_size)
        logger.info(f"Limited dataset to {len(df)} rows (from {original_size})")

    # Create output directory
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EMBEDDING DISTANCE COMPUTATION")
    logger.info("=" * 60)
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Embeddings directory: {args.embeddings_dir}")
    logger.info(f"Output CSV: {args.output_csv}")
    logger.info(f"Protein pairs: {len(df)}")
    logger.info(f"Batch size: {args.batch_size}")

    try:
        # Initialize distance computer
        computer = EmbeddingDistanceComputer(
            embeddings_dir=args.embeddings_dir, batch_size=args.batch_size
        )

        # Compute distances (results are saved incrementally)
        result_df = computer.compute_all_distances(df, args.output_csv)

        # Summary statistics
        distance_cols = [col for col in result_df.columns if col.startswith("dist_")]
        logger.info("=" * 60)
        logger.info("COMPUTATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output file: {args.output_csv}")
        logger.info(f"Total columns: {len(result_df.columns)}")
        logger.info(f"Distance columns: {len(distance_cols)}")
        logger.info(f"Distance columns: {distance_cols}")

        # Coverage statistics
        for col in distance_cols:
            valid_count = result_df[col].notna().sum()
            coverage = valid_count / len(result_df) * 100
            if valid_count > 0:
                mean_dist = result_df[col].mean()
                std_dist = result_df[col].std()
                logger.info(
                    f"  {col}: {coverage:.1f}% coverage, mean={mean_dist:.3f}, std={std_dist:.3f}"
                )
            else:
                logger.info(f"  {col}: {coverage:.1f}% coverage (no valid distances)")

        logger.info("\nNext steps:")
        logger.info("  Use this file with pairwise embedding comparison:")
        logger.info(
            "  uv run python scripts/create_pairwise_embedding_visualizations.py \\"
        )
        logger.info(f"    --data_path {args.output_csv} \\")
        logger.info("    --output_dir out/embedding_analysis")

    except Exception as e:
        logger.error(f"Error during computation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
