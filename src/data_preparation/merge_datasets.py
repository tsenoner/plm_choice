#!/usr/bin/env python3
"""
Comprehensive pipeline to merge, analyse, and plot protein similarities.

This script processes MMSeqs2, Foldcomp, and FoldSeek data to analyze protein similarities.
It applies various filtering thresholds, merges datasets for comprehensive analysis, and
creates distribution plots to visualize data quality metrics.

Key features:
- Processes sequence similarity (MMSeqs2), structural confidence (FoldComp), and
  structural similarity (FoldSeek) data
- Computes HFSP scores
- Applies quality thresholds (coverage ≥0.8, PIDE ≥0.3, HFSP ≥0.0, TM-score ≥0.4)
- Removes self-matches and low-confidence structures
- Creates individual and combined violin plots for data distribution analysis
- Performs full outer joins to preserve all protein pairs from both datasets
- Supports test mode for pipeline validation with smaller datasets

Author: Tobias Senoner
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Configure Polars for better performance
pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(20)


class ProteinAnalysisPipeline:
    """Main pipeline class for protein similarity analysis."""

    def __init__(
        self, base_data_dir: str | Path = "data", output_dir: str | Path = "out"
    ):
        """Initialize pipeline with base directories."""
        self.base_data_dir = Path(base_data_dir)
        self.output_dir = Path(output_dir)
        self._setup_paths()

    def _setup_paths(self) -> None:
        """Setup all file paths used in the pipeline."""
        # Base directories
        self.interm_dir = self.base_data_dir / "interm" / "sprot_pre2024"

        # Input file paths
        self.mmseqs_tsv = self.interm_dir / "mmseqs" / "sprot_all_vs_all.tsv"
        self.foldcomp_plddt_tsv = self.interm_dir / "foldcomp" / "plddt.tsv"
        self.foldseek_tsv = (
            self.interm_dir / "foldseek" / "afdb_swissprot_v4_all_vs_all.tsv"
        )

        # Intermediate files
        self.foldcomp_low_plddt_ids = self.interm_dir / "foldcomp" / "ids_below_70.txt"

        # Output directories
        self.plots_dir = self.output_dir / "data_analysis"

    def get_file_paths(self, test_mode: bool = False) -> dict[str, Path]:
        """Get file paths, with test mode suffixes if needed."""
        if test_mode:
            # For test mode, modify the stem (filename without extension)
            mmseqs_parquet = self.mmseqs_tsv.with_stem(
                f"{self.mmseqs_tsv.stem}_test"
            ).with_suffix(".parquet")
            foldseek_parquet = self.foldseek_tsv.with_stem(
                f"{self.foldseek_tsv.stem}_test"
            ).with_suffix(".parquet")
            final_merged = self.interm_dir / "merged_protein_similarity_test.parquet"
            plots_dir = self.plots_dir.with_name(f"{self.plots_dir.name}_test")
        else:
            mmseqs_parquet = self.mmseqs_tsv.with_suffix(".parquet")
            foldseek_parquet = self.foldseek_tsv.with_suffix(".parquet")
            final_merged = self.interm_dir / "merged_protein_similarity.parquet"
            plots_dir = self.plots_dir

        return {
            "mmseqs_parquet": mmseqs_parquet,
            "foldseek_parquet": foldseek_parquet,
            "final_merged": final_merged,
            "plots_dir": plots_dir,
        }

    def run(self, test_mode: bool = False, test_size: int = 100_000) -> pl.DataFrame:
        """Run the complete analysis pipeline."""
        print("🧬 PROTEIN SIMILARITY ANALYSIS PIPELINE")
        print("=" * 70)

        if test_mode:
            print(f"🧪 TEST MODE: Processing {test_size:,} rows per dataset")
        else:
            print("🚀 FULL MODE: Processing complete datasets")

        file_paths = self.get_file_paths(test_mode)
        self._print_configuration(file_paths)

        # Execute pipeline steps
        mmseqs_df = self._process_mmseqs_data(
            file_paths["mmseqs_parquet"], test_mode, test_size
        )
        foldcomp_df = self._process_foldcomp_data(test_mode, test_size)
        self._save_low_plddt_ids(foldcomp_df)
        foldseek_df = self._process_foldseek_data(
            file_paths["foldseek_parquet"], test_mode, test_size
        )

        self._create_distribution_plots(
            mmseqs_df, foldcomp_df, foldseek_df, file_paths["plots_dir"]
        )
        final_df = self._merge_datasets(
            mmseqs_df, foldseek_df, file_paths["final_merged"]
        )

        self._print_completion_summary(final_df, file_paths)
        return final_df

    def _print_configuration(self, file_paths: dict[str, Path]) -> None:
        """Print pipeline configuration."""
        print("\n📁 Configuration:")
        print(f"   MMSeqs2 TSV:     {self.mmseqs_tsv}")
        print(f"   FoldComp TSV:    {self.foldcomp_plddt_tsv}")
        print(f"   FoldSeek TSV:    {self.foldseek_tsv}")
        print(f"   Plots output:    {file_paths['plots_dir']}")
        print(f"   Final merged:    {file_paths['final_merged']}")

    def _print_completion_summary(
        self, final_df: pl.DataFrame, file_paths: dict[str, Path]
    ) -> None:
        """Print completion summary."""
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"📊 Final dataset: {final_df.height:,} rows × {final_df.width} columns")
        print(f"📈 Plots saved to: {file_paths['plots_dir']}/")
        print(f"💾 Dataset saved to: {file_paths['final_merged']}")

        # Show final column summary
        print(f"🔍 Final columns: {list(final_df.columns)}")

        # Print rows where both HFSP and alnTMscore are present (not null)
        overlap_df = final_df.filter(
            (final_df["hfsp"].is_not_null()) & (final_df["alntmscore"].is_not_null())
        )
        print("Rows with both HFSP and alnTMscore present:")
        print(overlap_df.head())

        print(final_df.head())

    def _process_mmseqs_data(
        self, parquet_file: Path, test_mode: bool, test_size: int
    ) -> pl.DataFrame:
        """Process MMSeqs2 sequence similarity data."""
        print("\n🔄 Processing MMSeqs2 data...")

        df = self._load_or_convert_data(
            self.mmseqs_tsv, parquet_file, "MMSeqs2", test_mode, test_size
        )

        # Remove self-matches
        df = self._remove_self_matches(df, "MMSeqs2")

        # Compute HFSP scores efficiently
        df = self._compute_hfsp_scores(df)

        print(f"📊 MMSeqs2 final shape: {df.shape}")
        return df

    def _process_foldcomp_data(self, test_mode: bool, test_size: int) -> pl.DataFrame:
        """Process FoldComp pLDDT confidence scores."""
        print("\n🔄 Processing FoldComp data...")

        # Read with Polars for consistency
        df = pl.read_csv(
            self.foldcomp_plddt_tsv,
            separator="\t",
            has_header=False,
            new_columns=["id", "length", "plddt"],
        )

        if test_mode:
            df = df.sample(n=min(test_size, df.height), seed=42)
            print(f"🧪 Test mode: Using {df.height:,} rows")

        # Calculate average pLDDT using Polars
        df = df.with_columns(
            [
                pl.col("plddt")
                .str.split(",")
                .map_elements(
                    lambda x: sum(int(val) for val in x) // len(x),
                    return_dtype=pl.Int32,
                )
                .alias("avg_plddt")
            ]
        )

        print(f"📊 FoldComp final shape: {df.shape}")
        return df

    def _process_foldseek_data(
        self, parquet_file: Path, test_mode: bool, test_size: int
    ) -> pl.DataFrame:
        """Process FoldSeek structural similarity data."""
        print("\n🔄 Processing FoldSeek data...")

        if not parquet_file.exists():
            df = pl.read_csv(
                self.foldseek_tsv,
                separator="\t",
                columns=["query", "target", "qcov", "tcov", "alntmscore"],
            )

            if test_mode:
                df = df.sample(n=min(test_size, df.height), seed=42)
                print(f"🧪 Test mode: Using {df.height:,} rows")

            # Extract protein IDs and compute minimum coverage
            df = (
                self._extract_protein_ids(df)
                .with_columns([pl.min_horizontal(["qcov", "tcov"]).alias("min_cov")])
                .select(["query", "target", "min_cov", "alntmscore"])
            )

            parquet_file.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(parquet_file)
            print(f"💾 Saved parquet: {parquet_file}")
        else:
            df = pl.read_parquet(parquet_file)
            print(f"📖 Loaded parquet: {parquet_file}")

        # Remove self-matches and low confidence structures
        df = self._remove_self_matches(df, "FoldSeek")
        df = self._filter_low_confidence_structures(df)

        print(f"📊 FoldSeek final shape: {df.shape}")
        return df

    def _load_or_convert_data(
        self,
        tsv_file: Path,
        parquet_file: Path,
        dataset_name: str,
        test_mode: bool,
        test_size: int,
    ) -> pl.DataFrame:
        """Load data from parquet or convert from TSV."""
        if not parquet_file.exists():
            print(f"🔄 Converting {dataset_name} TSV → Parquet")
            df = pl.read_csv(tsv_file, separator="\t")

            if test_mode:
                df = df.sample(n=min(test_size, df.height), seed=42)
                print(f"🧪 Test mode: Using {df.height:,} rows")

            parquet_file.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(parquet_file)
            print(f"💾 Saved parquet: {parquet_file}")
        else:
            df = pl.read_parquet(parquet_file)
            print(f"📖 Loaded parquet: {parquet_file}")

        return df

    def _remove_self_matches(self, df: pl.DataFrame, dataset_name: str) -> pl.DataFrame:
        """Remove rows where query equals target."""
        before = df.height
        df = df.filter(pl.col("query") != pl.col("target"))
        removed = before - df.height
        print(f"🗑️  {dataset_name}: Removed {removed:,} self-matches")
        return df

    def _compute_hfsp_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute HFSP scores using vectorized Polars operations."""
        print("🧮 Computing HFSP scores...")

        return df.with_columns(
            [
                # Calculate ungapped alignment length
                (pl.col("nident") + pl.col("mismatch")).alias("ungapped_len")
            ]
        ).with_columns(
            [
                # Compute HFSP score with conditional logic
                pl.when(pl.col("ungapped_len") <= 11)
                .then(pl.col("fident") * 100 - 100)
                .when(pl.col("ungapped_len") <= 450)
                .then(
                    pl.col("fident") * 100
                    - 770
                    * pl.col("ungapped_len").pow(
                        -0.33 * (1 + pl.col("ungapped_len") / 1000).exp()
                    )
                )
                .otherwise(pl.col("fident") * 100 - 28.4)
                .alias("hfsp")
            ]
        )

    def _extract_protein_ids(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract protein IDs from AlphaFold format strings."""
        return df.with_columns(
            [
                pl.col(col).str.extract(r"AF-(.*?)-F1-model_v4", 1)
                for col in ["query", "target"]
            ]
        )

    def _save_low_plddt_ids(self, foldcomp_df: pl.DataFrame) -> None:
        """Save protein IDs with low confidence scores."""
        print("\n💾 Saving low confidence protein IDs...")

        # Filter and extract IDs with pLDDT < 70
        low_confidence_ids = (
            foldcomp_df.filter(pl.col("avg_plddt") < 70)
            .select("id")
            .with_columns(
                [
                    pl.col("id")
                    .str.extract(r"AF-(.*?)-F1-model_v4", 1)
                    .alias("parsed_id")
                ]
            )
            .select("parsed_id")
            .drop_nulls()
        )

        # Save to file
        self.foldcomp_low_plddt_ids.parent.mkdir(parents=True, exist_ok=True)
        with open(self.foldcomp_low_plddt_ids, "w") as f:
            for id_val in low_confidence_ids.get_column("parsed_id"):
                f.write(f"{id_val}\n")

        print(f"💾 Saved {low_confidence_ids.height:,} low confidence IDs")

    def _filter_low_confidence_structures(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove proteins with low structural confidence."""
        if not self.foldcomp_low_plddt_ids.exists():
            print("⚠️  Low confidence ID file not found, skipping filter")
            return df

        # Read low confidence IDs
        low_confidence_ids = set(
            self.foldcomp_low_plddt_ids.read_text().strip().split("\n")
        )

        before = df.height
        df = df.filter(
            ~pl.col("query").is_in(low_confidence_ids)
            & ~pl.col("target").is_in(low_confidence_ids)
        )
        removed = before - df.height

        print(
            f"🗑️  Removed {removed:,} rows with low confidence structures ({removed / before * 100:.1f}%)"
        )
        return df

    def _create_distribution_plots(
        self,
        mmseqs_df: pl.DataFrame,
        foldcomp_df: pl.DataFrame,
        foldseek_df: pl.DataFrame,
        plots_dir: Path,
    ) -> None:
        """Create all distribution visualization plots."""
        print("\n📊 Creating distribution plots...")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Create individual plots
        plot_configs = [
            # MMSeqs2 plots
            (
                mmseqs_df.select(pl.min_horizontal("qcov", "tcov"))
                .to_numpy()
                .flatten(),
                0.8,
                "MMSeqs2 Coverage",
                (0, 1),
                "mmseqs_coverage.png",
            ),
            (
                mmseqs_df.get_column("fident").to_numpy(),
                0.3,
                "MMSeqs2 PIDE",
                (0, 1),
                "mmseqs_pide.png",
            ),
            (
                mmseqs_df.get_column("hfsp").to_numpy(),
                0,
                "MMSeqs2 HFSP",
                (-60, 100),
                "mmseqs_hfsp.png",
            ),
            # FoldComp plot
            (
                foldcomp_df.get_column("avg_plddt").to_numpy(),
                70,
                "FoldComp Average pLDDT",
                (0, 100),
                "foldcomp_plddt.png",
            ),
            # FoldSeek plots
            (
                foldseek_df.get_column("min_cov").to_numpy(),
                0.8,
                "FoldSeek Coverage",
                (0, 1),
                "foldseek_coverage.png",
            ),
            (
                foldseek_df.get_column("alntmscore").to_numpy(),
                0.4,
                "FoldSeek TM-Score",
                (0, 1),
                "foldseek_tmscore.png",
            ),
        ]

        created_plots = 0
        for data, threshold, title, ylim, filename in plot_configs:
            plot_path = plots_dir / filename
            if not plot_path.exists():
                self._create_violin_plot(data, threshold, title, ylim, plot_path)
                created_plots += 1
            else:
                print(f"📊 Skipping existing plot: {filename}")

        # Create combined subplot figure
        combined_plot_path = plots_dir / "combined_distributions.png"
        if not combined_plot_path.exists():
            # Extract plot paths from plot_configs
            plot_paths = [plots_dir / filename for _, _, _, _, filename in plot_configs]
            self._create_combined_plot(plot_paths, combined_plot_path)
            created_plots += 1
        else:
            print(f"📊 Skipping existing combined plot: {combined_plot_path.name}")

        print(
            f"📊 Created {created_plots} new plots, {len(plot_configs) + 1 - created_plots} already existed in {plots_dir}"
        )

    def _create_combined_plot(
        self,
        plot_paths: list[Path],
        combined_plot_path: Path,
    ) -> None:
        """Create a combined 2x3 subplot figure by loading existing PNG files."""

        # Panel labels: only show A, B, C once per group
        panel_labels = ["A", "", "", "B", "C", ""]

        # Set font scaling for labels
        scale = 1.5
        fontsize = 12 * scale
        plt.rcParams.update(
            {
                "font.size": fontsize,
                "axes.titlesize": fontsize * 1.2,
                "axes.labelsize": fontsize,
                "xtick.labelsize": fontsize,
                "ytick.labelsize": fontsize,
            }
        )

        # Create the combined figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for idx, (filename, panel) in enumerate(zip(plot_paths, panel_labels)):
            row, col = divmod(idx, 3)
            ax = axes[row, col]

            # Load and display the PNG image
            img = mpimg.imread(filename)
            ax.imshow(img)
            ax.axis("off")  # Remove axes for clean appearance

            # Add panel label outside the plot area (publication style)
            if panel:
                # Position label outside the top-left corner of the plot
                ax.text(
                    -0.05,
                    0.9,
                    panel,
                    transform=ax.transAxes,
                    fontsize=32,
                    fontweight="bold",
                    va="bottom",
                    ha="right",
                )

        plt.tight_layout()
        plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _merge_datasets(
        self, mmseqs_df: pl.DataFrame, foldseek_df: pl.DataFrame, output_file: Path
    ) -> pl.DataFrame:
        """Apply quality thresholds and merge datasets."""
        print("\n🔗 Applying thresholds and merging datasets...")

        # Apply quality thresholds
        mmseqs_filtered = self._apply_mmseqs_thresholds(mmseqs_df)
        foldseek_filtered = self._apply_foldseek_thresholds(foldseek_df)

        # Prepare datasets for merging - select only required columns
        mmseqs_merge = mmseqs_filtered.select(["query", "target", "fident", "hfsp"])

        foldseek_merge = foldseek_filtered.select(["query", "target", "alntmscore"])

        # Print statistics before merging
        print("📊 Pre-merge statistics:")
        print(f"   MMSeqs2 filtered: {mmseqs_merge.height:,} rows")
        print(f"   FoldSeek filtered: {foldseek_merge.height:,} rows")

        # Debug: Check column names before join
        print(f"🔍 MMSeqs2 columns: {mmseqs_merge.columns}")
        print(f"🔍 FoldSeek columns: {foldseek_merge.columns}")

        # Perform full outer join to keep all pairs from both datasets
        merged_df = mmseqs_merge.join(
            foldseek_merge,
            on=["query", "target"],
            how="full",
            coalesce=True,
        )

        # Calculate matching statistics
        total_rows = merged_df.height
        both_data = merged_df.filter(
            pl.col("fident").is_not_null() & pl.col("alntmscore").is_not_null()
        ).height
        mmseqs_only = merged_df.filter(
            pl.col("fident").is_not_null() & pl.col("alntmscore").is_null()
        ).height
        foldseek_only = merged_df.filter(
            pl.col("fident").is_null() & pl.col("alntmscore").is_not_null()
        ).height

        # Save merged dataset
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_df.write_parquet(output_file)

        # Print detailed merge statistics
        print("\n📈 Merge results:")
        print(f"   Total unique pairs: {total_rows:,}")
        print(
            f"   🎯 Matches found in both datasets: {both_data:,} ({both_data / total_rows * 100:.1f}%)"
        )
        print(
            f"   📊 MMSeqs2 only: {mmseqs_only:,} ({mmseqs_only / total_rows * 100:.1f}%)"
        )
        print(
            f"   🔍 FoldSeek only: {foldseek_only:,} ({foldseek_only / total_rows * 100:.1f}%)"
        )
        print(
            f"   🔗 Overlap rate: {both_data / (mmseqs_merge.height + foldseek_merge.height - both_data) * 100:.1f}%"
        )

        # Verify final columns
        print(f"\n🔍 Final dataset columns: {list(merged_df.columns)}")

        return merged_df

    def _apply_mmseqs_thresholds(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply quality thresholds to MMSeqs2 data."""
        before = df.height

        filtered_df = df.with_columns(
            [pl.min_horizontal("qcov", "tcov").alias("min_cov")]
        ).filter(
            (pl.col("min_cov") >= 0.8)
            & (pl.col("fident") >= 0.3)
            & (pl.col("hfsp") >= 0.0)
        )

        removed = before - filtered_df.height
        print(
            f"🎯 MMSeqs2 thresholds: {removed:,} removed ({removed / before * 100:.1f}%), "
            f"{filtered_df.height:,} retained"
        )

        return filtered_df

    def _apply_foldseek_thresholds(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply quality thresholds to FoldSeek data."""
        before = df.height

        filtered_df = df.filter(
            (pl.col("min_cov") >= 0.8) & (pl.col("alntmscore") >= 0.4)
        )

        removed = before - filtered_df.height
        print(
            f"🎯 FoldSeek thresholds: {removed:,} removed ({removed / before * 100:.1f}%), "
            f"{filtered_df.height:,} retained"
        )

        return filtered_df

    @staticmethod
    def _create_violin_plot(
        data: np.ndarray,
        threshold: float,
        title: str,
        ylim: Tuple[float, float],
        output_path: Path,
        scale: float = 1.5,
    ) -> None:
        """Create a violin plot with threshold visualization."""
        # Scale all font sizes
        fontsize = 12 * scale
        plt.rcParams.update(
            {
                "font.size": fontsize,
                "axes.titlesize": fontsize * 1.2,
                "axes.labelsize": fontsize,
                "xtick.labelsize": fontsize,
                "ytick.labelsize": fontsize,
            }
        )

        # Calculate statistics
        count_below = (data < threshold).sum()
        percentage_below = (count_below / len(data)) * 100
        count_str = ProteinAnalysisPipeline._human_format(count_below)

        # Create plot
        plt.figure(figsize=(6, 6))

        # Create violin plot with enhanced styling
        sns.violinplot(
            y=data,
            color="grey",
            alpha=0.3,
            inner="quart",
            inner_kws=dict(linewidth=2, color=".2"),
            linecolor="none",
        )

        # Customize plot
        plt.xlim(-0.5, 0.5)
        plt.ylim(ylim)
        plt.gca().set_xticks([])

        # Add threshold line and annotation
        plt.axhline(threshold, linestyle="--", color="red", alpha=0.7)
        plt.fill_between(
            [-0.5, 0.5], threshold, ylim[0], color="red", ec="none", alpha=0.15
        )
        plt.text(
            -0.48,
            threshold,
            f"{count_str} ({percentage_below:.1f}%) < {threshold}",
            color="red",
            va="bottom",
        )

        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close to save memory

    @staticmethod
    def _human_format(num: int, precision: int = 1) -> str:
        """Format large numbers in human-readable format."""
        units = ["", "K", "M", "B", "T", "P", "E", "Z", "Y"]
        num_abs = abs(num)
        if num_abs < 1000:
            return str(num)

        magnitude = 0
        while num_abs >= 1000 and magnitude < len(units) - 1:
            num_abs /= 1000.0
            magnitude += 1

        if num_abs < 10:
            formatted = f"{num_abs:.{precision}f}{units[magnitude]}"
        else:
            formatted = f"{int(num_abs)}{units[magnitude]}"

        return f"-{formatted}" if num < 0 else formatted


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Comprehensive protein similarity analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_datasets.py              # Run full analysis
  python merge_datasets.py --test       # Run test mode (100K rows per dataset)
        """,
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with smaller datasets (100K rows per dataset)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data",
        help="Base data directory (default: data)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="out",
        help="Output directory for plots (default: out)",
    )

    args = parser.parse_args()

    print("🧬 PROTEIN SIMILARITY ANALYSIS PIPELINE")
    print("=" * 50)

    if args.test:
        print("🧪 TEST MODE: Processing 100K rows per dataset")
        pipeline = ProteinAnalysisPipeline(args.data_dir, args.output_dir)
        result_df = pipeline.run(test_mode=True, test_size=100_000)
    else:
        print("🚀 FULL MODE: Processing complete datasets")
        pipeline = ProteinAnalysisPipeline(args.data_dir, args.output_dir)
        result_df = pipeline.run(test_mode=False)

    print("✅ Pipeline completed successfully!")
    print(f"📊 Final dataset shape: {result_df.shape}")


if __name__ == "__main__":
    main()
