#!/usr/bin/env python3
# --- Ivan infrastructure (2026-03-20) ---
"""
Organism Landscape Analysis — Embedding Distance Distributions by Organism Group

Compares embedding distance distributions between organism groups (human-human,
human-other, other-other, model_organism-model_organism) to detect organism-specific
biases in protein language models. If a pLM's latent space clusters proteins by
organism rather than by function, within-organism distances will be systematically
lower than between-organism distances, inflating apparent performance on organism-
biased benchmarks.

For each partition x embedding distance column, computes:
  - Distribution statistics (mean, std, median, quartiles)
  - Two-sample Kolmogorov-Smirnov test between partitions
  - Overlaid density plots colored by partition

Outputs:
  - summary_stats.csv       — per-partition per-embedding statistics + KS results
  - density_<embedding>.png — overlaid density plot per embedding
  - ks_results.csv          — pairwise KS test results between all partition pairs

Usage:
    uv run python src/data_preparation/organism_landscape.py \\
        --pairs_parquet data/processed/sprot_pre2024/sets/test_with_distances.parquet \\
        --organism_mapping data/reference/organism_annotations.tsv \\
        --distance_columns dist_prott5 dist_esm2_650m \\
        --output_dir out/organism_landscape

    # Quick test with 50k pairs
    uv run python src/data_preparation/organism_landscape.py \\
        --pairs_parquet data/processed/sprot_pre2024/sets/test_with_distances.parquet \\
        --organism_mapping data/reference/organism_annotations.tsv \\
        --output_dir out/organism_landscape \\
        --sample_size 50000
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde, ks_2samp

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

HUMAN_TAX_ID = 9606

MODEL_ORGANISMS: Dict[int, str] = {
    9606: "Human",
    10090: "Mouse",
    7227: "Drosophila",
    6239: "C. elegans",
    559292: "S. cerevisiae",
    83333: "E. coli K-12",
}

PARTITION_COLORS: Dict[str, str] = {
    "human_human": "#1f77b4",
    "human_other": "#ff7f0e",
    "other_other": "#2ca02c",
    "model_model": "#d62728",
}

PARTITION_LABELS: Dict[str, str] = {
    "human_human": "Human-Human",
    "human_other": "Human-Other",
    "other_other": "Other-Other",
    "model_model": "Model Org.-Model Org.",
}

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------


def load_organism_mapping(
    path: Path, target_proteins: Optional[Set[str]] = None
) -> Dict[str, int]:
    """
    Load protein-to-organism mapping from a TSV file.

    Expected format (tab-separated, with header):
        protein_id	organism_id
        P12345	9606
        Q67890	10090

    Args:
        path: Path to TSV file.
        target_proteins: If provided, only load mappings for these protein IDs
            to reduce memory usage on large reference files.

    Returns:
        Dict mapping protein_id -> organism taxonomy ID (int).
    """
    logger.info(f"Loading organism mapping from {path}")

    mapping_df = pl.read_csv(path, separator="\t")

    # Normalize column names — accept common variants
    col_map: Dict[str, str] = {}
    for col in mapping_df.columns:
        lower = col.lower().strip()
        if lower in ("protein_id", "accession", "entry", "id"):
            col_map[col] = "protein_id"
        elif lower in ("organism_id", "taxonomy_id", "taxid", "tax_id", "organism"):
            col_map[col] = "organism_id"

    if "protein_id" not in col_map.values() or "organism_id" not in col_map.values():
        raise ValueError(
            f"Could not identify protein_id and organism_id columns. "
            f"Found columns: {mapping_df.columns}. "
            f"Expected one of: protein_id/accession/entry/id and "
            f"organism_id/taxonomy_id/taxid/tax_id/organism."
        )

    mapping_df = mapping_df.rename(col_map)

    # Cast organism_id to integer (handle string representation)
    mapping_df = mapping_df.with_columns(
        pl.col("organism_id").cast(pl.Int64)
    )

    # Filter to target proteins if provided
    if target_proteins is not None:
        mapping_df = mapping_df.filter(pl.col("protein_id").is_in(target_proteins))

    mapping: Dict[str, int] = dict(
        zip(
            mapping_df["protein_id"].to_list(),
            mapping_df["organism_id"].to_list(),
        )
    )

    logger.info(f"Loaded organism mapping for {len(mapping)} proteins")

    # Log organism distribution (top 10)
    org_counts = mapping_df.group_by("organism_id").len().sort("len", descending=True)
    logger.info("Top organisms in mapping:")
    for row in org_counts.head(10).iter_rows(named=True):
        org_id = row["organism_id"]
        count = row["len"]
        label = MODEL_ORGANISMS.get(org_id, "")
        suffix = f" ({label})" if label else ""
        logger.info(f"  taxid {org_id}{suffix}: {count} proteins")

    return mapping


# ---------------------------------------------------------------------------
#  Partitioning
# ---------------------------------------------------------------------------


def partition_pairs(
    pairs_df: pl.DataFrame,
    organism_map: Dict[str, int],
    include_model_organisms: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Partition protein pairs into organism groups.

    Partitions:
      - human_human:  both query and target are Human (taxid 9606)
      - human_other:  exactly one of query/target is Human
      - other_other:  neither query nor target is Human
      - model_model:  both are model organisms (optional, includes human)

    Args:
        pairs_df: DataFrame with 'query' and 'target' columns.
        organism_map: Dict mapping protein_id -> taxonomy_id.
        include_model_organisms: Whether to compute the model_model partition.

    Returns:
        Dict mapping partition name -> boolean numpy array (mask over pairs_df rows).
    """
    n = len(pairs_df)

    model_org_ids = set(MODEL_ORGANISMS.keys())

    # Organism lookup as a polars hash join rather than a Python comprehension over
    # `.to_list()`, which would materialize two Python strings per pair row.
    # -1 marks "no organism annotation", as before.
    if organism_map:
        orgs = pairs_df.select(
            pl.col(side)
            .replace_strict(organism_map, default=-1, return_dtype=pl.Int64)
            .alias(side)
            for side in ("query", "target")
        )
        query_org = orgs["query"].to_numpy()
        target_org = orgs["target"].to_numpy()
    else:
        query_org = np.full(n, -1, dtype=np.int64)
        target_org = np.full(n, -1, dtype=np.int64)

    # Only consider pairs where both proteins have organism annotation
    both_annotated = (query_org != -1) & (target_org != -1)

    q_is_human = query_org == HUMAN_TAX_ID
    t_is_human = target_org == HUMAN_TAX_ID

    masks: Dict[str, np.ndarray] = {
        "human_human": both_annotated & q_is_human & t_is_human,
        "human_other": both_annotated & (q_is_human ^ t_is_human),  # XOR
        "other_other": both_annotated & ~q_is_human & ~t_is_human,
    }

    if include_model_organisms:
        q_is_model = np.isin(query_org, list(model_org_ids))
        t_is_model = np.isin(target_org, list(model_org_ids))
        masks["model_model"] = both_annotated & q_is_model & t_is_model

    # Log partition sizes
    annotated_count = int(both_annotated.sum())
    logger.info(
        f"Organism annotation coverage: {annotated_count}/{n} pairs "
        f"({annotated_count / n * 100:.1f}%)"
    )
    for name, mask in masks.items():
        count = int(mask.sum())
        logger.info(f"  {PARTITION_LABELS.get(name, name)}: {count} pairs")

    return masks


# ---------------------------------------------------------------------------
#  Statistics
# ---------------------------------------------------------------------------


def compute_partition_stats(
    pairs_df: pl.DataFrame,
    distance_columns: List[str],
    masks: Dict[str, np.ndarray],
) -> pl.DataFrame:
    """
    Compute per-partition distribution statistics for each embedding distance.

    For each (partition, distance_column) combination, computes:
      - count, mean, std, median, q25, q75, min, max

    Args:
        pairs_df: DataFrame containing the distance columns.
        distance_columns: List of column names (e.g., ["dist_prott5", "dist_esm2_650m"]).
        masks: Dict from partition_pairs().

    Returns:
        Polars DataFrame with columns:
            partition, embedding, count, mean, std, median, q25, q75, min, max
    """
    rows: List[Dict] = []

    for dist_col in distance_columns:
        if dist_col not in pairs_df.columns:
            logger.warning(f"Column {dist_col} not found in pairs, skipping")
            continue

        all_distances = np.asarray(pairs_df[dist_col].to_numpy(), dtype=np.float64)

        for partition_name, mask in masks.items():
            partition_distances = all_distances[mask]
            # Drop NaN values
            valid = partition_distances[~np.isnan(partition_distances)]

            if len(valid) == 0:
                logger.warning(
                    f"No valid distances for {dist_col} / {partition_name}"
                )
                rows.append({
                    "partition": partition_name,
                    "embedding": dist_col.replace("dist_", ""),
                    "distance_column": dist_col,
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "median": None,
                    "q25": None,
                    "q75": None,
                    "min": None,
                    "max": None,
                })
                continue

            rows.append({
                "partition": partition_name,
                "embedding": dist_col.replace("dist_", ""),
                "distance_column": dist_col,
                "count": len(valid),
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "median": float(np.median(valid)),
                "q25": float(np.percentile(valid, 25)),
                "q75": float(np.percentile(valid, 75)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
            })

    return pl.DataFrame(rows)


def compute_ks_tests(
    pairs_df: pl.DataFrame,
    distance_columns: List[str],
    masks: Dict[str, np.ndarray],
) -> pl.DataFrame:
    """
    Run two-sample Kolmogorov-Smirnov tests between all partition pairs.

    For each embedding distance, tests every unique pair of partitions to
    quantify how different the distance distributions are.

    Args:
        pairs_df: DataFrame containing distance columns.
        distance_columns: List of distance column names.
        masks: Dict from partition_pairs().

    Returns:
        Polars DataFrame with columns:
            embedding, partition_a, partition_b, ks_statistic, p_value, n_a, n_b
    """
    partition_names = list(masks.keys())
    rows: List[Dict] = []

    for dist_col in distance_columns:
        if dist_col not in pairs_df.columns:
            continue

        all_distances = np.asarray(pairs_df[dist_col].to_numpy(), dtype=np.float64)

        # Extract valid distances per partition
        partition_values: Dict[str, np.ndarray] = {}
        for name, mask in masks.items():
            vals = all_distances[mask]
            partition_values[name] = vals[~np.isnan(vals)]

        # Pairwise KS tests
        for i in range(len(partition_names)):
            for j in range(i + 1, len(partition_names)):
                name_a = partition_names[i]
                name_b = partition_names[j]
                vals_a = partition_values[name_a]
                vals_b = partition_values[name_b]

                if len(vals_a) < 2 or len(vals_b) < 2:
                    logger.warning(
                        f"Insufficient data for KS test: {dist_col} "
                        f"{name_a}({len(vals_a)}) vs {name_b}({len(vals_b)})"
                    )
                    rows.append({
                        "embedding": dist_col.replace("dist_", ""),
                        "distance_column": dist_col,
                        "partition_a": name_a,
                        "partition_b": name_b,
                        "ks_statistic": None,
                        "p_value": None,
                        "n_a": len(vals_a),
                        "n_b": len(vals_b),
                    })
                    continue

                ks_stat, p_val = ks_2samp(vals_a, vals_b)

                rows.append({
                    "embedding": dist_col.replace("dist_", ""),
                    "distance_column": dist_col,
                    "partition_a": name_a,
                    "partition_b": name_b,
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_val),
                    "n_a": len(vals_a),
                    "n_b": len(vals_b),
                })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Visualization
# ---------------------------------------------------------------------------


def plot_partition_densities(
    pairs_df: pl.DataFrame,
    dist_col: str,
    masks: Dict[str, np.ndarray],
    output_path: Path,
    max_kde_points: int = 500,
) -> None:
    """
    Create overlaid kernel density plots for each organism partition.

    One figure per embedding distance column, with one density curve per
    partition. Includes vertical dashed lines at partition medians.

    Args:
        pairs_df: DataFrame with the distance column.
        dist_col: Name of the distance column to plot.
        masks: Dict from partition_pairs().
        output_path: Path to save the PNG figure.
        max_kde_points: Number of points for the KDE evaluation grid.
    """
    all_distances = np.asarray(pairs_df[dist_col].to_numpy(), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 6))

    x_min_global = np.inf
    x_max_global = -np.inf

    # Collect valid data for all partitions first (for x-axis range)
    partition_data: Dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        vals = all_distances[mask]
        valid = vals[~np.isnan(vals)]
        if len(valid) >= 10:
            partition_data[name] = valid
            x_min_global = min(x_min_global, float(np.percentile(valid, 0.5)))
            x_max_global = max(x_max_global, float(np.percentile(valid, 99.5)))

    if not partition_data:
        logger.warning(f"No partitions have enough data to plot for {dist_col}")
        plt.close(fig)
        return

    x_grid = np.linspace(x_min_global, x_max_global, max_kde_points)

    for name, valid in partition_data.items():
        color = PARTITION_COLORS.get(name, "#888888")
        label = PARTITION_LABELS.get(name, name)

        try:
            kde = gaussian_kde(valid, bw_method="scott")
            density = kde(x_grid)
            ax.plot(x_grid, density, color=color, linewidth=2, label=label)
            ax.fill_between(x_grid, density, alpha=0.15, color=color)
        except np.linalg.LinAlgError:
            # KDE can fail with degenerate data
            logger.warning(f"KDE failed for {name} / {dist_col}, using histogram")
            ax.hist(
                valid,
                bins=50,
                density=True,
                alpha=0.3,
                color=color,
                label=label,
            )

        # Median line
        median_val = float(np.median(valid))
        ax.axvline(
            median_val,
            color=color,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

    embedding_name = dist_col.replace("dist_", "")
    ax.set_xlabel(f"Euclidean Distance ({embedding_name})", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Embedding Distance by Organism Group — {embedding_name}",
        fontsize=14,
    )
    ax.legend(fontsize=11, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved density plot: {output_path}")


# ---------------------------------------------------------------------------
#  Auto-detection helpers
# ---------------------------------------------------------------------------


def detect_distance_columns(pairs_df: pl.DataFrame) -> List[str]:
    """Find all dist_* columns in the DataFrame."""
    return [col for col in pairs_df.columns if col.startswith("dist_")]


# ---------------------------------------------------------------------------
#  Orchestrator
# ---------------------------------------------------------------------------


def run_organism_landscape(
    pairs_parquet: Path,
    organism_mapping_path: Path,
    distance_columns: Optional[List[str]],
    output_dir: Path,
    sample_size: Optional[int] = None,
    include_model_organisms: bool = True,
) -> None:
    """
    Main analysis orchestrator.

    Loads data, partitions pairs by organism group, computes statistics,
    runs KS tests, and generates density plots + summary tables.

    Args:
        pairs_parquet: Path to parquet with protein pairs + distance columns.
        organism_mapping_path: Path to TSV with protein_id -> organism_id.
        distance_columns: Explicit list of distance columns, or None to auto-detect.
        output_dir: Directory for all outputs.
        sample_size: If set, subsample pairs for quick testing.
        include_model_organisms: Whether to include model_model partition.
    """
    # --- Load pairs ---
    logger.info(f"Loading pairs from {pairs_parquet}")
    pairs_df = pl.read_parquet(pairs_parquet)
    logger.info(f"Loaded {len(pairs_df)} pairs, columns: {pairs_df.columns}")

    if sample_size is not None:
        pairs_df = pairs_df.head(sample_size)
        logger.info(f"Subsampled to {len(pairs_df)} pairs")

    # --- Resolve distance columns ---
    if distance_columns is None or len(distance_columns) == 0:
        distance_columns = detect_distance_columns(pairs_df)
        if not distance_columns:
            logger.error(
                "No dist_* columns found and none specified via --distance_columns"
            )
            sys.exit(1)
        logger.info(f"Auto-detected distance columns: {distance_columns}")
    else:
        missing = [c for c in distance_columns if c not in pairs_df.columns]
        if missing:
            logger.error(f"Missing distance columns: {missing}")
            sys.exit(1)

    # --- Load organism mapping ---
    all_proteins = set(pairs_df["query"].unique().to_list()) | set(
        pairs_df["target"].unique().to_list()
    )
    organism_map = load_organism_mapping(organism_mapping_path, target_proteins=all_proteins)

    mapped_proteins = all_proteins & set(organism_map.keys())
    logger.info(
        f"Organism mapping covers {len(mapped_proteins)}/{len(all_proteins)} "
        f"proteins ({len(mapped_proteins) / len(all_proteins) * 100:.1f}%)"
    )

    # --- Partition pairs ---
    masks = partition_pairs(
        pairs_df, organism_map, include_model_organisms=include_model_organisms
    )

    # --- Compute statistics ---
    logger.info("Computing partition statistics ...")
    stats_df = compute_partition_stats(pairs_df, distance_columns, masks)

    # --- KS tests ---
    logger.info("Running KS tests between partitions ...")
    ks_df = compute_ks_tests(pairs_df, distance_columns, masks)

    # --- Save tables ---
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "summary_stats.csv"
    stats_df.write_csv(stats_path)
    logger.info(f"Saved summary statistics: {stats_path}")

    ks_path = output_dir / "ks_results.csv"
    ks_df.write_csv(ks_path)
    logger.info(f"Saved KS test results: {ks_path}")

    # --- Density plots ---
    for dist_col in distance_columns:
        embedding_name = dist_col.replace("dist_", "")
        plot_path = output_dir / f"density_{embedding_name}.png"
        plot_partition_densities(pairs_df, dist_col, masks, plot_path)

    # --- Print summary ---
    logger.info("=" * 70)
    logger.info("ORGANISM LANDSCAPE ANALYSIS COMPLETE")
    logger.info("=" * 70)

    for dist_col in distance_columns:
        embedding_name = dist_col.replace("dist_", "")
        logger.info(f"\n--- {embedding_name} ---")

        col_stats = stats_df.filter(pl.col("distance_column") == dist_col)
        for row in col_stats.iter_rows(named=True):
            if row["count"] == 0:
                continue
            logger.info(
                f"  {PARTITION_LABELS.get(row['partition'], row['partition']):>22s}: "
                f"n={row['count']:>8,}  "
                f"mean={row['mean']:.4f}  "
                f"std={row['std']:.4f}  "
                f"median={row['median']:.4f}"
            )

        col_ks = ks_df.filter(pl.col("distance_column") == dist_col)
        for row in col_ks.iter_rows(named=True):
            if row["ks_statistic"] is None:
                continue
            a_label = PARTITION_LABELS.get(row["partition_a"], row["partition_a"])
            b_label = PARTITION_LABELS.get(row["partition_b"], row["partition_b"])
            sig = "***" if row["p_value"] < 0.001 else (
                "**" if row["p_value"] < 0.01 else (
                    "*" if row["p_value"] < 0.05 else "ns"
                )
            )
            logger.info(
                f"  KS {a_label} vs {b_label}: "
                f"D={row['ks_statistic']:.4f}  p={row['p_value']:.2e} {sig}"
            )

    logger.info(f"\nOutputs saved to: {output_dir}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare embedding distance distributions between organism groups "
            "to detect organism-specific biases in protein language models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs_parquet",
        type=Path,
        required=True,
        help=(
            "Parquet file with protein pairs and precomputed distance columns "
            "(must have 'query' and 'target' columns plus dist_* columns)"
        ),
    )
    parser.add_argument(
        "--organism_mapping",
        type=Path,
        required=True,
        help=(
            "TSV file mapping protein_id to organism taxonomy ID "
            "(columns: protein_id, organism_id)"
        ),
    )
    parser.add_argument(
        "--distance_columns",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Distance column names to analyze (e.g., dist_prott5 dist_esm2_650m). "
            "If omitted, all dist_* columns are auto-detected."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("out/organism_landscape"),
        help="Output directory for tables and plots",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit number of pairs for quick testing",
    )
    parser.add_argument(
        "--no_model_organisms",
        action="store_true",
        help="Skip the model_organism-model_organism partition",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.pairs_parquet.exists():
        logger.error(f"Pairs parquet not found: {args.pairs_parquet}")
        sys.exit(1)

    if not args.organism_mapping.exists():
        logger.error(f"Organism mapping not found: {args.organism_mapping}")
        sys.exit(1)

    run_organism_landscape(
        pairs_parquet=args.pairs_parquet,
        organism_mapping_path=args.organism_mapping,
        distance_columns=args.distance_columns,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        include_model_organisms=not args.no_model_organisms,
    )


if __name__ == "__main__":
    main()
