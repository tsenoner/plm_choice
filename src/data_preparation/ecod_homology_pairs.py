#!/usr/bin/env python3
# --- Ivan infrastructure (2026-03-20) ---
"""
ECOD Homology Pair Density Distributions

Filters protein pairs to ECOD structural classification groups and computes
per-group embedding distance distributions. Produces KDE density plots showing
how well each pLM separates structurally related from unrelated protein pairs
at each level of the ECOD hierarchy.

ECOD (Evolutionary Classification of protein Domains) organizes domains into
a four-level hierarchy:

    Architecture (A) — gross structural features (e.g. "a+b bundle")
    X-group (X)      — possible homology, shared topology
    H-group (H)      — probable homology, detectable sequence/structure similarity
    T-group (T)      — definite homology (same topology, confirmed evolutionary link)
    F-group (F)      — family-level, close homologs

A good pLM embedding should produce smaller distances for same-group pairs and
larger distances for different-group pairs, with the separation increasing as
we move from coarse (Architecture) to fine (F-group) levels. The density plots
make this visible: well-separated curves = informative embedding.

This addresses the reviewer request for density distributions restricted to
ECOD homology pairs (cf. "Contrastive learning unites sequence and structure
for accurate and efficient protein representation learning", Lu et al. 2024).

The ECOD domain file is from https://prodata.swmed.edu/ecod/ and maps PDB
chains to UniProt accessions with hierarchical structural classifications.

Usage:
    uv run python src/data_preparation/ecod_homology_pairs.py \
        --pairs_parquet data/processed/sprot_pre2024/sets/test_with_distances.parquet \
        --ecod_domains data/reference/ecod/ecod.latest.domains.txt \
        --distance_columns dist_prott5 dist_esm2_650m dist_esm2_3b \
        --output_dir out/ecod_density

    # Quick test with 10k pairs
    uv run python src/data_preparation/ecod_homology_pairs.py \
        --pairs_parquet data/processed/sprot_pre2024/sets/test_with_distances.parquet \
        --ecod_domains data/reference/ecod/ecod.latest.domains.txt \
        --distance_columns dist_prott5 \
        --output_dir out/ecod_density \
        --sample_size 10000

Created: 2026-03-20 (Ivan infrastructure for pLM Choice revision)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

# The level vocabulary is shared with the figures that plot these columns; see
# shared/hierarchies.py for why it does not live here.
from evaluation.stats import cohens_d
from shared.hierarchies import ECOD_LEVEL_LABELS, ECOD_LEVELS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Colors for same-group vs different-group curves (colorblind-safe)
SAME_GROUP_COLOR = "#2166ac"  # blue
DIFF_GROUP_COLOR = "#b2182b"  # red

# Colors per ECOD level for the overlay plot
LEVEL_COLORS = {
    "arch": "#66c2a5",       # teal
    "x_group": "#fc8d62",    # orange
    "h_group": "#8da0cb",    # periwinkle
    "t_group": "#e78ac3",    # pink
    "f_group": "#a6d854",    # lime
}


# --------------------------------------------------------------------------- #
#                        ECOD DOMAIN FILE PARSING
# --------------------------------------------------------------------------- #


def parse_ecod_domains(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse the ECOD domain definitions file and build a UniProt-to-classification map.

    The ECOD file is tab-separated with comment lines starting with '#' and a
    header line. Columns (0-indexed):

        0: uid              — unique domain identifier
        1: ecod_domain_id   — ECOD domain ID (e.g. e1a0aA1)
        2: manual_rep       — manual representative flag
        3: f_id             — family numeric ID
        4: pdb              — PDB code
        5: chain            — chain identifier
        6: pdb_range        — residue range in PDB
        7: seqid_range      — residue range by sequence index
        8: unp_acc          — UniProt accession (what we join on)
        9: arch_name        — Architecture name
       10: x_name           — X-group name
       11: h_name           — H-group name
       12: t_name           — T-group name
       13: f_name           — F-group name
       14: asm_status       — assembly status
       15: ligand           — bound ligand

    When a protein maps to multiple ECOD domains, we keep the domain with the
    most specific classification (i.e., the one with the fewest "UNCLASSIFIED"
    or empty levels). This avoids over-counting proteins that span multiple
    domains but gives preference to the most informative annotation.

    Args:
        path: Path to ECOD domain definitions file (e.g. ecod.latest.domains.txt).

    Returns:
        Dict mapping UniProt accession -> dict with keys:
        {arch, x_group, h_group, t_group, f_group}, each a string label.
    """
    # Accumulate all domains per UniProt ID, then pick the best
    domains_by_uniprot: Dict[str, List[Dict[str, str]]] = {}

    skipped = 0
    loaded = 0

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")

            # Skip comment and empty lines
            if line.startswith("#") or not line.strip():
                continue

            parts = line.split("\t")

            # Need at least 14 columns (through f_name)
            if len(parts) < 14:
                skipped += 1
                continue

            unp_acc = parts[8].strip()

            # Skip entries without UniProt mapping
            if not unp_acc or unp_acc == "NO_UNP" or unp_acc == "-":
                skipped += 1
                continue

            # Some ECOD entries have isoform suffixes (e.g. P12345-2); strip them
            # to match the canonical accession used in our pair datasets
            if "-" in unp_acc:
                unp_acc = unp_acc.split("-")[0]

            # Columns 9..13 are guaranteed present: rows with < 14 fields were
            # already skipped above.
            domain = {
                "arch": parts[9].strip(),
                "x_group": parts[10].strip(),
                "h_group": parts[11].strip(),
                "t_group": parts[12].strip(),
                "f_group": parts[13].strip(),
            }

            if unp_acc not in domains_by_uniprot:
                domains_by_uniprot[unp_acc] = []
            domains_by_uniprot[unp_acc].append(domain)
            loaded += 1

    # For each protein, pick the domain with the most specific classification.
    # Specificity = number of non-empty, non-"UNCLASSIFIED" hierarchy levels.
    ecod_map: Dict[str, Dict[str, str]] = {}

    for unp_acc, domain_list in domains_by_uniprot.items():
        best_domain = max(domain_list, key=_domain_specificity)
        ecod_map[unp_acc] = best_domain

    logger.info(
        f"Parsed {loaded} ECOD domain entries for {len(ecod_map)} unique UniProt IDs "
        f"(skipped {skipped} entries without valid UniProt mapping)"
    )

    # Report level coverage
    for level in ECOD_LEVELS:
        n_classified = sum(
            1 for d in ecod_map.values() if _is_classified(d[level])
        )
        logger.info(
            f"  {ECOD_LEVEL_LABELS[level]}: {n_classified}/{len(ecod_map)} "
            f"proteins classified ({n_classified / max(len(ecod_map), 1) * 100:.1f}%)"
        )

    return ecod_map


def _domain_specificity(domain: Dict[str, str]) -> int:
    """Score a domain by how many hierarchy levels are meaningfully classified."""
    return sum(1 for level in ECOD_LEVELS if _is_classified(domain[level]))


def _is_classified(label: str) -> bool:
    """Check whether an ECOD hierarchy label is a real classification."""
    if not label:
        return False
    label_lower = label.lower().strip()
    return label_lower not in ("", "unclassified", "no_x_name", "no_h_name",
                                "no_t_name", "no_f_name", "-", "n/a")


# --------------------------------------------------------------------------- #
#                     PAIR FILTERING BY ECOD LEVEL
# --------------------------------------------------------------------------- #


def masks_from_annotation(
    annotated_df: pl.DataFrame,
    level: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Read the same-group / different-group masks for one ECOD level off the
    annotated pairs frame.

    A pair is "same-group" if both proteins have a classified label at the given
    level AND those labels are identical; "different-group" if both are classified
    but the labels differ. Pairs where either protein lacks a classified label at
    this level are excluded from both masks — :func:`build_annotated_parquet`
    already encodes that as a null in ``ecod_{level}_same``.

    Args:
        annotated_df: Output of :func:`build_annotated_parquet`.
        level: One of ECOD_LEVELS (e.g. "h_group").

    Returns:
        Tuple of (same_mask, diff_mask, n_annotated) where masks are boolean numpy
        arrays of length len(annotated_df), and n_annotated is the number of pairs
        where both proteins have a valid classification at this level.
    """
    same_col = annotated_df[f"ecod_{level}_same"]
    same_mask = same_col.fill_null(False).to_numpy()
    diff_mask = (~same_col).fill_null(False).to_numpy()
    return same_mask, diff_mask, int(same_mask.sum() + diff_mask.sum())


# --------------------------------------------------------------------------- #
#                        DENSITY PLOTTING
# --------------------------------------------------------------------------- #


def _apply_plot_style() -> None:
    """Set consistent matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_density_comparison(
    distances: np.ndarray,
    same_mask: np.ndarray,
    diff_mask: np.ndarray,
    level: str,
    dist_col: str,
    output_path: Path,
    n_points: int = 500,
) -> None:
    """
    Plot KDE density distributions of embedding distances for same-group vs
    different-group pairs at one ECOD hierarchy level.

    Args:
        distances: Array of embedding distances for all pairs.
        same_mask: Boolean mask for same-group pairs.
        diff_mask: Boolean mask for different-group pairs.
        level: ECOD hierarchy level name (e.g. "h_group").
        dist_col: Name of the distance column (for axis label / title).
        output_path: Where to save the figure.
        n_points: Number of points for KDE evaluation grid.
    """
    same_dists = distances[same_mask]
    diff_dists = distances[diff_mask]

    # Filter out NaN values
    same_dists = same_dists[~np.isnan(same_dists)]
    diff_dists = diff_dists[~np.isnan(diff_dists)]

    if len(same_dists) < 10 or len(diff_dists) < 10:
        logger.warning(
            f"Too few valid pairs for {dist_col} / {level} "
            f"(same={len(same_dists)}, diff={len(diff_dists)}). Skipping plot."
        )
        return

    # Compute KDE on a shared x-axis range
    all_dists = np.concatenate([same_dists, diff_dists])
    x_min = np.percentile(all_dists, 0.5)
    x_max = np.percentile(all_dists, 99.5)
    x_grid = np.linspace(x_min, x_max, n_points)

    # Evaluate each KDE once and reuse the curve for both the fill and the line —
    # gaussian_kde is O(n_samples x n_grid), which at 10^6+ pairs is the dominant
    # cost of this figure.
    y_same = gaussian_kde(same_dists, bw_method="scott")(x_grid)
    y_diff = gaussian_kde(diff_dists, bw_method="scott")(x_grid)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(x_grid, y_same, alpha=0.3, color=SAME_GROUP_COLOR)
    ax.plot(
        x_grid, y_same, color=SAME_GROUP_COLOR, linewidth=1.5,
        label=f"Same {ECOD_LEVEL_LABELS[level]} (n={len(same_dists):,})",
    )

    ax.fill_between(x_grid, y_diff, alpha=0.3, color=DIFF_GROUP_COLOR)
    ax.plot(
        x_grid, y_diff, color=DIFF_GROUP_COLOR, linewidth=1.5,
        label=f"Different {ECOD_LEVEL_LABELS[level]} (n={len(diff_dists):,})",
    )

    # Prettify
    plm_name = dist_col.replace("dist_", "").replace("_", " ").title()
    ax.set_title(f"{plm_name} — {ECOD_LEVEL_LABELS[level]}")
    ax.set_xlabel("Embedding distance (Euclidean)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.8")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


def plot_level_overlay(
    distances: np.ndarray,
    masks_by_level: Dict[str, Tuple[np.ndarray, np.ndarray]],
    dist_col: str,
    output_path: Path,
    n_points: int = 500,
) -> None:
    """
    Plot overlaid same-group KDE curves for all ECOD levels on one figure.

    This is the "money plot" — it shows how structural similarity at different
    ECOD levels maps onto the pLM distance space. Finer levels (T/F-group)
    should show tighter, more left-shifted distributions.

    Args:
        distances: Array of embedding distances for all pairs.
        masks_by_level: Dict mapping level name -> (same_mask, diff_mask).
        dist_col: Name of the distance column.
        output_path: Where to save the figure.
        n_points: Number of points for KDE evaluation grid.
    """
    # Iterate in hierarchy order, but only over the levels the caller actually
    # asked for: `--levels h_group` builds a one-entry masks_by_level, and
    # indexing it by the full ECOD_LEVELS list raised KeyError('arch') before a
    # single curve was drawn.
    levels = [level for level in ECOD_LEVELS if level in masks_by_level]

    # Determine shared x range across all levels
    all_same_dists = []
    for level in levels:
        same_mask, _ = masks_by_level[level]
        d = distances[same_mask]
        d = d[~np.isnan(d)]
        if len(d) >= 10:
            all_same_dists.append(d)

    if not all_same_dists:
        logger.warning(f"No valid same-group distances for {dist_col}. Skipping overlay.")
        return

    combined = np.concatenate(all_same_dists)
    x_min = np.percentile(combined, 0.5)
    x_max = np.percentile(combined, 99.5)
    x_grid = np.linspace(x_min, x_max, n_points)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for level in levels:
        same_mask, _ = masks_by_level[level]
        d = distances[same_mask]
        d = d[~np.isnan(d)]

        if len(d) < 10:
            logger.info(f"  Skipping {level} overlay (n={len(d)} < 10)")
            continue

        y = gaussian_kde(d, bw_method="scott")(x_grid)
        color = LEVEL_COLORS[level]

        ax.plot(
            x_grid, y, color=color, linewidth=2.0,
            label=f"{ECOD_LEVEL_LABELS[level]} (n={len(d):,})",
        )
        ax.fill_between(x_grid, y, alpha=0.15, color=color)

    # Also plot the "different" distribution from the coarsest level as baseline
    _, diff_mask_arch = masks_by_level.get("arch", (np.array([]), np.array([])))
    if diff_mask_arch is not None and diff_mask_arch.sum() >= 10:
        diff_d = distances[diff_mask_arch]
        diff_d = diff_d[~np.isnan(diff_d)]
        if len(diff_d) >= 10:
            kde_diff = gaussian_kde(diff_d, bw_method="scott")
            ax.plot(
                x_grid, kde_diff(x_grid), color="0.5", linewidth=1.5,
                linestyle="--", label=f"Different Architecture (n={len(diff_d):,})",
            )

    plm_name = dist_col.replace("dist_", "").replace("_", " ").title()
    ax.set_title(f"{plm_name} — ECOD Level Overlay (same-group distances)")
    ax.set_xlabel("Embedding distance (Euclidean)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.8")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"  Saved overlay: {output_path}")


# --------------------------------------------------------------------------- #
#                        SUMMARY STATISTICS
# --------------------------------------------------------------------------- #


def compute_separation_stats(
    distances: np.ndarray,
    same_mask: np.ndarray,
    diff_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute summary statistics for the separation between same-group and
    different-group embedding distance distributions.

    Returns dict with:
        - mean_same, mean_diff: mean distances
        - median_same, median_diff: median distances
        - cohens_d: effect size (Cohen's d)
        - overlap_coefficient: approximate overlap from KDE (0 = no overlap, 1 = identical)
    """
    same_d = distances[same_mask]
    diff_d = distances[diff_mask]

    same_d = same_d[~np.isnan(same_d)]
    diff_d = diff_d[~np.isnan(diff_d)]

    if len(same_d) < 2 or len(diff_d) < 2:
        return {
            "mean_same": np.nan, "mean_diff": np.nan,
            "median_same": np.nan, "median_diff": np.nan,
            "cohens_d": np.nan, "overlap_coefficient": np.nan,
            "n_same": len(same_d), "n_diff": len(diff_d),
        }

    mean_same = float(np.mean(same_d))
    mean_diff = float(np.mean(diff_d))
    n_s, n_d = len(same_d), len(diff_d)

    # Positive d = different-group pairs sit at larger distances, i.e. good separation.
    separation_d = cohens_d(diff_d, same_d)

    # Overlap coefficient: integral of min(kde_same, kde_diff)
    all_vals = np.concatenate([same_d, diff_d])
    x_grid = np.linspace(np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5), 500)
    try:
        kde_s = gaussian_kde(same_d, bw_method="scott")
        kde_d = gaussian_kde(diff_d, bw_method="scott")
        overlap = float(np.trapz(np.minimum(kde_s(x_grid), kde_d(x_grid)), x_grid))
    except Exception:
        overlap = np.nan

    return {
        "mean_same": mean_same,
        "mean_diff": mean_diff,
        "median_same": float(np.median(same_d)),
        "median_diff": float(np.median(diff_d)),
        "cohens_d": separation_d,
        "overlap_coefficient": overlap,
        "n_same": n_s,
        "n_diff": n_d,
    }


# --------------------------------------------------------------------------- #
#                        PARQUET OUTPUT
# --------------------------------------------------------------------------- #


def build_annotated_parquet(
    pairs_df: pl.DataFrame,
    ecod_map: Dict[str, Dict[str, str]],
    output_path: Path,
) -> pl.DataFrame:
    """
    Add ECOD group label columns to the pairs dataframe and write to parquet.

    For each ECOD level, adds two columns:
        - ecod_{level}_query: group label for the query protein
        - ecod_{level}_target: group label for the target protein

    Also adds a boolean column ecod_{level}_same (True if same group).

    Args:
        pairs_df: Input pairs DataFrame.
        ecod_map: UniProt -> ECOD classification dict.
        output_path: Where to write the annotated parquet.

    Returns:
        The annotated DataFrame.
    """
    # Clean the labels once per *protein* (thousands) rather than once per pair row
    # (up to 10^8): everything below is then a hash join in polars instead of a
    # Python loop over the pair table.
    label_maps = {
        level: {
            acc: domain[level]
            for acc, domain in ecod_map.items()
            if _is_classified(domain[level])
        }
        for level in ECOD_LEVELS
    }

    result_df = pairs_df
    for level in ECOD_LEVELS:
        q_col, t_col = f"ecod_{level}_query", f"ecod_{level}_target"
        level_map = label_maps[level]

        if level_map:
            lookup = [
                pl.col(side)
                .replace_strict(level_map, default=None, return_dtype=pl.Utf8)
                .alias(name)
                for side, name in (("query", q_col), ("target", t_col))
            ]
        else:  # nothing classified at this level — every label is null
            lookup = [
                pl.lit(None, dtype=pl.Utf8).alias(q_col),
                pl.lit(None, dtype=pl.Utf8).alias(t_col),
            ]

        # `==` propagates null in polars, so ecod_{level}_same is null exactly when
        # either side is unclassified — the intended "excluded from both masks".
        result_df = result_df.with_columns(lookup).with_columns(
            (pl.col(q_col) == pl.col(t_col)).alias(f"ecod_{level}_same")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_path)
    logger.info(f"Wrote annotated parquet: {output_path} ({len(result_df)} rows)")

    return result_df


# --------------------------------------------------------------------------- #
#                                MAIN
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter protein pairs to ECOD homology groups and produce density "
            "distributions of embedding distances (same-group vs different-group)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs_parquet",
        type=Path,
        required=True,
        help="Parquet file with protein pairs and distance columns (query, target, dist_*)",
    )
    parser.add_argument(
        "--ecod_domains",
        type=Path,
        required=True,
        help="ECOD domain definitions file (TSV from prodata.swmed.edu/ecod/)",
    )
    parser.add_argument(
        "--distance_columns",
        nargs="+",
        required=True,
        help="Names of distance columns to analyze (e.g. dist_prott5 dist_esm2_650m)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for output plots and annotated parquet",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit number of pairs to process (for testing)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=ECOD_LEVELS,
        default=ECOD_LEVELS,
        help="ECOD hierarchy levels to analyze",
    )

    args = parser.parse_args()

    # --- Validate inputs ---
    if not args.pairs_parquet.exists():
        logger.error(f"Pairs parquet not found: {args.pairs_parquet}")
        sys.exit(1)
    if not args.ecod_domains.exists():
        logger.error(f"ECOD domain file not found: {args.ecod_domains}")
        sys.exit(1)

    _apply_plot_style()

    # --- Load data ---
    logger.info("=" * 60)
    logger.info("ECOD HOMOLOGY PAIR DENSITY ANALYSIS")
    logger.info("=" * 60)

    pairs_df = pl.read_parquet(args.pairs_parquet)
    logger.info(f"Loaded {len(pairs_df)} protein pairs from {args.pairs_parquet}")
    logger.info(f"Available columns: {pairs_df.columns}")

    if args.sample_size:
        pairs_df = pairs_df.head(args.sample_size)
        logger.info(f"Sampling {args.sample_size} pairs for testing")

    # Validate distance columns exist
    missing_cols = [c for c in args.distance_columns if c not in pairs_df.columns]
    if missing_cols:
        logger.error(
            f"Distance columns not found in parquet: {missing_cols}. "
            f"Available: {[c for c in pairs_df.columns if c.startswith('dist_')]}"
        )
        sys.exit(1)

    # --- Parse ECOD ---
    ecod_map = parse_ecod_domains(args.ecod_domains)

    # Check coverage against our protein pairs
    all_proteins = set(pairs_df["query"].unique().to_list()) | set(
        pairs_df["target"].unique().to_list()
    )
    mapped_proteins = all_proteins & set(ecod_map.keys())
    logger.info(
        f"ECOD coverage in pair dataset: {len(mapped_proteins)}/{len(all_proteins)} proteins "
        f"({len(mapped_proteins) / max(len(all_proteins), 1) * 100:.1f}%)"
    )

    # --- Write annotated parquet ---
    annotated_path = args.output_dir / "pairs_with_ecod.parquet"
    annotated_df = build_annotated_parquet(pairs_df, ecod_map, annotated_path)

    # --- Compute masks for all levels ---
    # The annotated frame already carries ecod_{level}_same for every level, so the
    # masks are a read of that column rather than a second pass over the pair table.
    masks_by_level: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for level in args.levels:
        same_mask, diff_mask, n_annotated = masks_from_annotation(annotated_df, level)
        masks_by_level[level] = (same_mask, diff_mask)
        logger.info(
            f"  {ECOD_LEVEL_LABELS[level]}: {n_annotated} annotated pairs "
            f"(same={same_mask.sum():,}, diff={diff_mask.sum():,})"
        )

    # --- Generate plots and stats for each distance column ---
    all_stats: List[Dict] = []

    for dist_col in args.distance_columns:
        logger.info(f"\nProcessing distance column: {dist_col}")
        distances = pairs_df[dist_col].to_numpy()

        # Per-level same vs different density plots
        for level in args.levels:
            same_mask, diff_mask = masks_by_level[level]

            # Plot
            plot_path = args.output_dir / dist_col / f"{level}_density.png"
            plot_density_comparison(
                distances, same_mask, diff_mask, level, dist_col, plot_path
            )

            # Stats
            stats = compute_separation_stats(distances, same_mask, diff_mask)
            stats["distance_column"] = dist_col
            stats["ecod_level"] = level
            all_stats.append(stats)

            if not np.isnan(stats["cohens_d"]):
                logger.info(
                    f"    {ECOD_LEVEL_LABELS[level]}: Cohen's d = {stats['cohens_d']:.3f}, "
                    f"overlap = {stats['overlap_coefficient']:.3f}, "
                    f"mean_same = {stats['mean_same']:.2f}, "
                    f"mean_diff = {stats['mean_diff']:.2f}"
                )

        # Overlay plot: all levels on one figure
        overlay_path = args.output_dir / dist_col / "ecod_level_overlay.png"
        plot_level_overlay(distances, masks_by_level, dist_col, overlay_path)

    # --- Write summary statistics ---
    stats_df = pl.DataFrame(all_stats) if all_stats else None
    if stats_df is not None:
        stats_path = args.output_dir / "ecod_separation_stats.csv"
        stats_df.write_csv(stats_path)
        logger.info(f"\nSaved separation statistics: {stats_path}")

    # --- Final summary ---
    logger.info("\n" + "=" * 60)
    logger.info("ECOD ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Annotated parquet: {annotated_path}")
    logger.info(f"Distance columns analyzed: {args.distance_columns}")
    logger.info(f"ECOD levels analyzed: {args.levels}")

    if stats_df is not None:
        # Pivot: show Cohen's d for each (dist_col, level) combination
        logger.info("\nCohen's d summary (higher = better separation):")
        for dist_col in args.distance_columns:
            subset = stats_df.filter(pl.col("distance_column") == dist_col)
            row_strs = []
            for row in subset.iter_rows(named=True):
                d = row["cohens_d"]
                d_str = f"{d:.2f}" if not np.isnan(d) else "N/A"
                row_strs.append(f"{ECOD_LEVEL_LABELS[row['ecod_level']]}={d_str}")
            logger.info(f"  {dist_col}: {', '.join(row_strs)}")


if __name__ == "__main__":
    main()
