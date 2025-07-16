#!/usr/bin/env python3
"""
Example script for generating pairwise embedding comparison visualizations.

This script provides a convenient way to run the comprehensive embedding comparison
visualizations on your data. It serves as a wrapper around the main visualization
class and demonstrates typical usage patterns.

Usage:
    # Generate all visualizations for a dataset
    uv run python scripts/create_pairwise_embedding_visualizations.py \
        --data_path data/processed/sprot_train/train.csv \
        --output_dir out/embedding_comparison \
        --sample_size 50000

    # Generate only specific visualizations
    uv run python scripts/create_pairwise_embedding_visualizations.py \
        --data_path data/processed/sprot_train/train.csv \
        --output_dir out/embedding_comparison \
        --visualizations hexagonal correlation distribution \
        --font_scale 1.2
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unknown_unknowns.visualization.pairwise_embedding_comparison import (
    EmbeddingComparisonVisualizer,
    logger,
)


def main():
    """Main function to run the embedding comparison visualizations."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive pairwise embedding comparison visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to CSV file containing the embedding distance data (e.g., train.csv, test.csv).",
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
        help="Scaling factor for all font sizes in visualizations (1.0 = default, 1.5 = 50% larger).",
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
        help="Specific visualizations to generate. Use 'all' for complete analysis.",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.data_path.exists():
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    logger.info(f"Starting pairwise embedding comparison visualization")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Sample size: {args.sample_size or 'All data'}")
    logger.info(f"Font scale: {args.font_scale}")
    logger.info(f"Visualizations: {args.visualizations}")

    try:
        # Create visualizer
        visualizer = EmbeddingComparisonVisualizer(
            data_path=args.data_path,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            font_scale=args.font_scale,
        )

        # Generate requested visualizations
        if "all" in args.visualizations:
            output_paths = visualizer.generate_all_visualizations(
                force_recompute=args.force_recompute
            )
        else:
            # Generate individual visualizations
            output_paths = {}
            cache_dir = args.output_dir / "cache"
            cache_dir.mkdir(exist_ok=True)

            if "hexagonal" in args.visualizations:
                logger.info("Generating hexagonal distance comparison...")
                hex_output = args.output_dir / "hexagonal_distance_comparison.png"
                visualizer.plot_hexagonal_distance_comparison(save_path=hex_output)
                output_paths["hexagonal"] = hex_output

            if "correlation" in args.visualizations:
                logger.info("Generating correlation heatmap...")
                corr_output = args.output_dir / "correlation_heatmap.png"
                visualizer.plot_correlation_heatmap(save_path=corr_output)
                output_paths["correlation"] = corr_output

            if "wasserstein" in args.visualizations:
                logger.info("Generating Wasserstein heatmap...")
                wass_output = args.output_dir / "wasserstein_heatmap.png"
                visualizer.plot_wasserstein_heatmap(save_path=wass_output)
                output_paths["wasserstein"] = wass_output

            if "distribution" in args.visualizations:
                logger.info("Generating distribution comparison...")
                dist_output = args.output_dir / "distribution_comparison.png"
                visualizer.plot_distributions(normalize=False, save_path=dist_output)
                output_paths["distribution"] = dist_output

            if "distribution_normalized" in args.visualizations:
                logger.info("Generating normalized distribution comparison...")
                norm_dist_output = (
                    args.output_dir / "distribution_comparison_normalized.png"
                )
                visualizer.plot_distributions(
                    normalize=True, save_path=norm_dist_output
                )
                output_paths["distribution_normalized"] = norm_dist_output

            if "violin" in args.visualizations:
                logger.info("Generating violin plot comparison...")
                violin_output = args.output_dir / "violin_plot_comparison.png"
                visualizer.create_violin_plot_comparison(save_path=violin_output)
                output_paths["violin"] = violin_output

        # Summary
        logger.info("=" * 60)
        logger.info("VISUALIZATION GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Generated {len(output_paths)} visualization(s):")

        for viz_type, path in output_paths.items():
            logger.info(f"  • {viz_type}: {path.name}")

        # Provide helpful next steps
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  • View visualizations in: {args.output_dir}")
        logger.info(f"  • Cached data available in: {args.output_dir / 'cache'}")
        logger.info("  • Use --force_recompute to regenerate cached data")
        logger.info("  • Adjust --font_scale for better readability")

    except Exception as e:
        logger.error(f"Error during visualization generation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
