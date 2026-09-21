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

    #: Class-level default so the tests' ``object.__new__`` bypass of ``__init__``
    #: (which needs the on-disk data-directory layout) still sees the real default.
    #: One declaration here beats a defensive ``getattr`` at each read site.
    _dedupe: bool = True

    def __init__(
        self,
        base_data_dir: str | Path = "data",
        output_dir: str | Path = "out",
        dataset: str = "2024_new",
        dedupe: bool = True,
    ):
        """Initialize pipeline with base directories and dataset type.

        Args:
            base_data_dir: Base data directory path
            output_dir: Output directory path
            dataset: Dataset type - either "sprot_pre2024" or "2024_new"
            dedupe: Collapse both orientations of each pair into one canonical
                unordered pair. On by default. Turning it off reproduces the
                pre-2026-08 directional table (written to a distinct filename)
                so the effect of deduplication stays separately attributable.
        """
        self.base_data_dir = Path(base_data_dir)
        self.output_dir = Path(output_dir)
        self.dataset = dataset
        self._dedupe = dedupe

        # Validate dataset parameter
        if dataset not in ["sprot_pre2024", "2024_new"]:
            raise ValueError(
                f"Dataset must be 'sprot_pre2024' or '2024_new', got '{dataset}'"
            )

        self._setup_paths()

    def _setup_paths(self) -> None:
        """Setup all file paths used in the pipeline."""
        # Base directories
        self.interm_dir = self.base_data_dir / "interm" / self.dataset

        # Dataset-specific file configurations
        if self.dataset == "2024_new":
            mmseqs_filename = "2024_novelSeqs_all_vs_all.tsv"
            foldseek_filename = "pdb_all_vs_all.tsv"
        elif self.dataset == "sprot_pre2024":
            mmseqs_filename = "sprot_all_vs_all.tsv"
            foldseek_filename = "afdb_swissprot_v4_all_vs_all.tsv"

        # Input file paths
        self.mmseqs_tsv = self.interm_dir / "mmseqs" / mmseqs_filename

        # pLDDT file location depends on dataset type
        if self.dataset == "2024_new":
            self.foldcomp_plddt_tsv = self.interm_dir / "colabfold" / "plddt.tsv"
            self.foldcomp_low_plddt_ids = (
                self.interm_dir / "colabfold" / "ids_below_70.txt"
            )
        else:  # sprot_pre2024
            self.foldcomp_plddt_tsv = self.interm_dir / "foldcomp" / "plddt.tsv"
            self.foldcomp_low_plddt_ids = (
                self.interm_dir / "foldcomp" / "ids_below_70.txt"
            )

        self._test_mode = False
        self.foldseek_tsv = self.interm_dir / "foldseek" / foldseek_filename

        # Output directories - dataset-specific
        self.plots_dir = self.output_dir / f"data_analysis_{self.dataset}"

    def get_file_paths(self, test_mode: bool = False) -> dict[str, Path]:
        """Get file paths, with test mode suffixes if needed."""
        # A non-deduplicated run writes to its own filename so it can never silently
        # overwrite the canonical (deduplicated) pair table -- in test mode too, where
        # the two would otherwise collide on merged_protein_similarity_test.parquet.
        dedupe_suffix = "" if self._dedupe else "_nodedup"
        if test_mode:
            # For test mode, modify the stem (filename without extension)
            mmseqs_parquet = self.mmseqs_tsv.with_stem(
                f"{self.mmseqs_tsv.stem}_test"
            ).with_suffix(".parquet")
            foldseek_parquet = self.foldseek_tsv.with_stem(
                f"{self.foldseek_tsv.stem}_test"
            ).with_suffix(".parquet")
            final_merged = (
                self.interm_dir / f"merged_protein_similarity{dedupe_suffix}_test.parquet"
            )
            plots_dir = self.plots_dir.with_name(f"{self.plots_dir.name}{dedupe_suffix}_test")
            low_plddt_ids = self.foldcomp_low_plddt_ids.with_stem(
                f"{self.foldcomp_low_plddt_ids.stem}_test"
            )
        else:
            mmseqs_parquet = self.mmseqs_tsv.with_suffix(".parquet")
            foldseek_parquet = self.foldseek_tsv.with_suffix(".parquet")
            final_merged = (
                self.interm_dir / f"merged_protein_similarity{dedupe_suffix}.parquet"
            )
            plots_dir = self.plots_dir.with_name(f"{self.plots_dir.name}{dedupe_suffix}")
            low_plddt_ids = self.foldcomp_low_plddt_ids

        return {
            "mmseqs_parquet": mmseqs_parquet,
            "foldseek_parquet": foldseek_parquet,
            "final_merged": final_merged,
            "plots_dir": plots_dir,
            "low_plddt_ids": low_plddt_ids,
        }

    def run(
        self,
        test_mode: bool = False,
        test_size: int = 100_000,
        plots_only: bool = False,
        reuse_plots: bool = False,
    ) -> pl.DataFrame | None:
        """Run the complete analysis pipeline.

        ``plots_only`` stops after the distribution figure, which is what a figure
        rebuild needs: the merged table is unchanged by redrawing it, and rewriting a
        1.5 GB parquet to get a PNG is both slow and a chance to corrupt the canonical
        table. It returns ``None`` in that mode, because no merged frame was built.
        """
        self._test_mode = test_mode
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
            mmseqs_df,
            foldcomp_df,
            foldseek_df,
            file_paths["plots_dir"],
            reuse=reuse_plots,
        )
        if plots_only:
            print("\n✅ Plots only: stopping before the merge step.")
            return None
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

        # Collapse both orientations of each pair BEFORE scoring, so HFSP is
        # computed once per unordered pair from the surviving alignment.
        if self._dedupe:
            df = self._dedupe_mmseqs_pairs(df)

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

        # Collapse both orientations BEFORE thresholding, so the quality filter
        # is applied once to the canonical pair value rather than to whichever
        # orientation happened to be reported.
        if self._dedupe:
            df = self._dedupe_foldseek_pairs(df)

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

    @staticmethod
    def _canonicalise_pairs(df: pl.DataFrame) -> pl.DataFrame:
        """Rewrite (query, target) to the lexicographically ordered orientation.

        Every expression is evaluated against the *input* frame, so the assignments
        happen simultaneously -- a safe swap rather than a two-step clobber. Pinned by
        test_canonicalises_pair_orientation.

        Columns whose meaning is tied to which protein was the query have to travel
        with the swap. ``qcov`` is the QUERY's coverage; leaving it in place on a
        flipped row makes it the target's, silently, for roughly half the table. That
        is invisible today because every reader goes through
        ``min_horizontal("qcov", "tcov")``, which is swap-invariant -- but it is a trap
        armed for the first direction-sensitive filter or probe target anyone adds.
        """
        flip = pl.col("query") > pl.col("target")
        exprs = [
            pl.min_horizontal("query", "target").alias("query"),
            pl.max_horizontal("query", "target").alias("target"),
        ]
        for q_col, t_col in (("qcov", "tcov"), ("qlen", "tlen")):
            if q_col in df.columns and t_col in df.columns:
                exprs += [
                    pl.when(flip).then(pl.col(t_col)).otherwise(pl.col(q_col)).alias(q_col),
                    pl.when(flip).then(pl.col(q_col)).otherwise(pl.col(t_col)).alias(t_col),
                ]
        return df.with_columns(exprs)

    def _dedupe_mmseqs_pairs(self, df: pl.DataFrame) -> pl.DataFrame:
        """Collapse both orientations of a pair, keeping the lowest-E-value hit.

        The profile search (``--num-iterations 3``) is directional, so a pair may
        be reported once or twice, with different alignments. Keeping one whole
        real alignment -- rather than averaging -- matters here because HFSP is a
        non-linear function of (PIDE, L): averaging would fabricate an alignment
        that never existed and leave fident/nident/mismatch mutually inconsistent.
        """
        deduped = (
            self._canonicalise_pairs(df)
            .sort("evalue")
            .unique(subset=["query", "target"], keep="first", maintain_order=True)
        )
        return self._report_dedupe("MMSeqs2", df.height, deduped)

    def _dedupe_foldseek_pairs(self, df: pl.DataFrame) -> pl.DataFrame:
        """Collapse both orientations of a pair, averaging the structural scores.

        ``alntmscore`` is normalised over the alignment (not by query or target
        length), so the two reports are two estimates of one symmetric quantity
        and the mean is the unbiased combination. Taking the max instead would
        re-introduce an upward bias on precisely the duplicated subset that this
        deduplication exists to stop over-weighting.
        """
        deduped = (
            self._canonicalise_pairs(df)
            # maintain_order=True: polars' group_by is multithreaded and its default
            # emits groups in a nondeterministic order, so two identical runs wrote
            # byte-different parquets. That matters beyond tidiness --
            # create_subset_datasets.py draws the 10% training subset with a seeded but
            # POSITIONAL df.sample(), so a reshuffled table is a different training set.
            .group_by(["query", "target"], maintain_order=True)
            .agg([pl.col("min_cov").mean(), pl.col("alntmscore").mean()])
        )
        return self._report_dedupe("FoldSeek", df.height, deduped)

    @staticmethod
    def _report_dedupe(name: str, before: int, deduped: pl.DataFrame) -> pl.DataFrame:
        """Report one arm's collapse. Shared so the two arms' numbers stay comparable."""
        removed = before - deduped.height
        share = f"{removed / before * 100:.1f}%" if before else "n/a"
        print(
            f"🔁 {name}: collapsed {removed:,} duplicate orientations "
            f"({share}), {deduped.height:,} unordered pairs"
        )
        return deduped

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
                # Compute HFSP score with conditional logic.
                # Mahlich et al. 2018 (Bioinformatics 34:i304-i312) Eq. 4, for 11 < L <= 450:
                #   HFSP = PIDE*100 - 770 * L^(-0.33 * (1 + exp(-L/1000)))
                # The exponent is -0.33 * (1 + exp(-L/1000)) — the "1 +" is OUTSIDE exp(),
                # and the argument of exp() is -L/1000 (not 1 + L/1000). Getting this wrong
                # makes the L>450 branch (which subtracts 28.4, = the correct length term at
                # L=450) discontinuous with the main branch. See tests/test_merge_datasets_hfsp.py.
                pl.when(pl.col("ungapped_len") <= 11)
                .then(pl.col("fident") * 100 - 100)
                .when(pl.col("ungapped_len") <= 450)
                .then(
                    pl.col("fident") * 100
                    - 770
                    * pl.col("ungapped_len").pow(
                        -0.33 * (1 + (-pl.col("ungapped_len") / 1000).exp())
                    )
                )
                .otherwise(pl.col("fident") * 100 - 28.4)
                .alias("hfsp")
            ]
        )

    def _extract_protein_ids(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract protein IDs from format strings."""
        if self.dataset == "2024_new":
            # For 2024_new dataset with ColabFold data, IDs are already in simple format
            return df
        else:
            # For sprot_pre2024, extract from AlphaFold format (AF-{ID}-F1-model_v4)
            return df.with_columns(
                [
                    pl.col(col).str.extract(r"AF-(.*?)-F1-model_v4", 1)
                    for col in ["query", "target"]
                ]
            )

    def _save_low_plddt_ids(self, foldcomp_df: pl.DataFrame) -> None:
        """Save protein IDs with low confidence scores."""
        print("\n💾 Saving low confidence protein IDs...")

        # Filter IDs with pLDDT < 70
        low_confidence_df = foldcomp_df.filter(pl.col("avg_plddt") < 70).select("id")

        # Extract protein IDs based on dataset type
        if self.dataset == "2024_new":
            # For ColabFold data, IDs are already in simple format
            low_confidence_ids = low_confidence_df.select(
                pl.col("id").alias("parsed_id")
            )
        else:
            # For AlphaFold data, extract from AF-{ID}-F1-model_v4 format
            low_confidence_ids = (
                low_confidence_df.with_columns(
                    [
                        pl.col("id")
                        .str.extract(r"AF-(.*?)-F1-model_v4", 1)
                        .alias("parsed_id")
                    ]
                )
                .select("parsed_id")
                .drop_nulls()
            )

        # Save to file.
        #
        # This path is suffixed by get_file_paths() like every other artifact. It
        # previously was not, so `--test` overwrote the PRODUCTION low-pLDDT
        # exclusion list with one derived from a 100k random sample. A later run
        # entering at the foldseek stage would then apply a ~99%-incomplete
        # exclusion list and silently keep low-confidence structures.
        target = self.get_file_paths(self._test_mode)["low_plddt_ids"]

        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            for id_val in low_confidence_ids.get_column("parsed_id"):
                f.write(f"{id_val}\n")

        print(f"💾 Saved {low_confidence_ids.height:,} low confidence IDs -> {target}")

    def _filter_low_confidence_structures(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove proteins with low structural confidence."""
        source = self.get_file_paths(self._test_mode)["low_plddt_ids"]
        if self._test_mode and not source.exists():
            # No test-mode list yet — fall back to the production exclusion list.
            source = self.foldcomp_low_plddt_ids
        if not source.exists():
            print("⚠️  Low confidence ID file not found, skipping filter")
            return df

        # Read low confidence IDs
        low_confidence_ids = set(
            source.read_text().strip().split("\n")
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
        reuse: bool = False,
    ) -> None:
        """Create all distribution visualization plots.

        ``reuse`` keeps panels that already exist on disk. The default is to redraw:
        a stale PNG left in place while the pipeline reports success is how a
        July-2025 filtering figure survived both the 2026 HFSP correction and the
        deduplication, and was still in the manuscript a year later. Skipping work
        is opt-in; correctness is not.
        """
        print("\n📊 Creating distribution plots...")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Create individual plots. The panel letter is the one used in the
        # supplementary caption: A = sequence metrics, B = structure confidence,
        # C = structure similarity.
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
                "A",
                "min(qcov, tcov)",
                "pairs",
            ),
            (
                mmseqs_df.get_column("fident").to_numpy(),
                0.3,
                "MMSeqs2 PIDE",
                (0, 1),
                "mmseqs_pide.png",
                "A",
                "fident",
                "pairs",
            ),
            (
                mmseqs_df.get_column("hfsp").to_numpy(),
                0,
                "MMSeqs2 HFSP",
                (-60, 100),
                "mmseqs_hfsp.png",
                "A",
                "hfsp",
                "pairs",
            ),
            # FoldComp plot
            (
                foldcomp_df.get_column("avg_plddt").to_numpy(),
                70,
                "FoldComp Average pLDDT",
                (0, 100),
                "foldcomp_plddt.png",
                "B",
                "avg_plddt",
                "structures",
            ),
            # FoldSeek plots
            (
                foldseek_df.get_column("min_cov").to_numpy(),
                0.8,
                "FoldSeek Coverage",
                (0, 1),
                "foldseek_coverage.png",
                "C",
                "min(qcov, tcov)",
                "pairs",
            ),
            (
                foldseek_df.get_column("alntmscore").to_numpy(),
                0.4,
                "FoldSeek TM-Score",
                (0, 1),
                "foldseek_tmscore.png",
                "C",
                "alntmscore",
                "pairs",
            ),
        ]

        created_plots = 0
        stats_rows: list[dict[str, object]] = []
        plot_paths: list[Path] = []
        reused: list[str] = []
        for data, threshold, title, ylim, filename, panel, column, unit in plot_configs:
            plot_path = plots_dir / filename
            plot_paths.append(plot_path)
            if not (reuse and plot_path.exists()):
                self._create_violin_plot(data, threshold, title, ylim, plot_path)
                created_plots += 1
            else:
                print(f"⚠️  REUSING existing plot, NOT regenerated: {filename}")
                reused.append(filename)
            stats_rows.append(
                self._threshold_stats(data, threshold, title, panel, column, unit)
            )

        # The numbers the figure annotates, written out so the caption can be checked
        # against the run that produced the PNG rather than against memory. That only
        # holds if the PNG came from this run: under --reuse-plots these stats describe
        # today's data while the panel beside them was drawn from older data, so
        # writing the CSV anyway would have it certify exactly the stale figure the
        # redraw-by-default exists to kill.
        stats_path = plots_dir / "filtering_thresholds.csv"
        if reused:
            print(
                f"⚠️  NOT writing {stats_path.name}: {len(reused)} reused panel(s) "
                f"({', '.join(reused)}) were not drawn from this data."
            )
        else:
            pl.DataFrame(stats_rows).write_csv(stats_path)
            print(f"💾 Threshold statistics → {stats_path}")

        # Create combined subplot figure
        combined_plot_path = plots_dir / "combined_distributions.png"
        if not (reuse and combined_plot_path.exists()):
            self._create_combined_plot(plot_paths, combined_plot_path)
            created_plots += 1
        else:
            print(f"📊 Skipping existing combined plot: {combined_plot_path.name}")

        print(
            f"📊 Created {created_plots} new plots, {len(plot_configs) + 1 - created_plots} already existed in {plots_dir}"
        )

    @staticmethod
    def _threshold_stats(
        data: np.ndarray,
        threshold: float,
        title: str,
        panel: str,
        column: str,
        unit: str,
    ) -> dict[str, object]:
        """The annotation of one violin panel, as numbers.

        ``count_below`` is computed with the same strict ``<`` the panel annotates, so
        the CSV and the red text on the PNG can never drift apart.
        """
        finite = data[np.isfinite(data)]
        n_below = int((finite < threshold).sum())
        q = np.percentile(finite, [1, 25, 50, 75, 99])
        return {
            "panel": panel,
            "metric": title,
            "source_column": column,
            "unit": unit,
            "threshold": float(threshold),
            "n": int(finite.size),
            "n_non_finite": int(data.size - finite.size),
            "n_below_threshold": n_below,
            "pct_below_threshold": 100.0 * n_below / finite.size,
            "mean": float(finite.mean()),
            "min": float(finite.min()),
            "p1": float(q[0]),
            "q1": float(q[1]),
            "median": float(q[2]),
            "q3": float(q[3]),
            "p99": float(q[4]),
            "max": float(finite.max()),
        }

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

        for idx, (filename, panel) in enumerate(zip(plot_paths, panel_labels, strict=True)):
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
        ylim: tuple[float, float],
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

        # Calculate statistics over the finite values only -- the same denominator
        # _threshold_stats writes to filtering_thresholds.csv, and the same population
        # seaborn actually draws, since violinplot discards non-finite values. Counting
        # nulls in the denominator here (but not there) would make the red annotation
        # and the CSV disagree for any metric column carrying them.
        finite = data[np.isfinite(data)]
        count_below = (finite < threshold).sum()
        percentage_below = (count_below / finite.size) * 100
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
  python merge_datasets.py                                    # Run full analysis (2024_new dataset)
  python merge_datasets.py --test                             # Run test mode (2024_new dataset, 100K rows)
  python merge_datasets.py --dataset sprot_pre2024           # Run full analysis (sprot_pre2024 dataset)
  python merge_datasets.py --dataset sprot_pre2024 --test    # Run test mode (sprot_pre2024 dataset, 100K rows)
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

    parser.add_argument(
        "--dataset",
        type=str,
        default="2024_new",
        choices=["sprot_pre2024", "2024_new"],
        help="Dataset to process: 'sprot_pre2024' or '2024_new' (default: 2024_new)",
    )

    parser.add_argument(
        "--no-dedupe",
        dest="dedupe",
        action="store_false",
        help=(
            "Keep both orientations of every pair (the pre-2026-08 behaviour). "
            "The profile search is directional, so this double-weights the pairs "
            "reported both ways -- which are measurably the more similar ones. "
            "Writes to merged_protein_similarity_nodedup.parquet so it cannot "
            "overwrite the canonical table. Use only to reproduce the old numbers."
        ),
    )
    parser.set_defaults(dedupe=True)

    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "Stop after the distribution figure; do not rebuild or overwrite the "
            "merged pair table. Use this to redraw the supplementary filtering "
            "figure from an already-corrected pipeline."
        ),
    )

    parser.add_argument(
        "--reuse-plots",
        action="store_true",
        help=(
            "Keep panels whose PNG already exists instead of redrawing them. Off by "
            "default: a stale figure kept while the rerun reports success is how the "
            "published filtering funnel outlived two corrections to its own data."
        ),
    )

    args = parser.parse_args()

    # run() prints its own mode banner and defaults test_size; only the dedupe state
    # is decided out here, because it is what picks the output filename.
    print(f"📦 Dataset: {args.dataset}")
    print(
        f"🔁 Pair deduplication: {'ON (canonical unordered pairs)' if args.dedupe else 'OFF (directional, legacy)'}"
    )

    pipeline = ProteinAnalysisPipeline(
        args.data_dir, args.output_dir, args.dataset, dedupe=args.dedupe
    )
    result_df = pipeline.run(
        test_mode=args.test,
        plots_only=args.plots_only,
        reuse_plots=args.reuse_plots,
    )

    print("✅ Pipeline completed successfully!")
    if result_df is not None:
        print(f"📊 Final dataset shape: {result_df.shape}")


if __name__ == "__main__":
    main()
