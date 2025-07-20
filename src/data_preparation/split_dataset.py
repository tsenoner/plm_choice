#!/usr/bin/env python3
"""
Script to cluster sequences/structures by similarity and create train/val/test splits.
Ensures no similar sequences/structures appear in different splits to prevent data leakage.
Uses MMseqs2 for sequence clustering or FoldSeek for structure clustering with hardcoded parameters.
"""

import os
import sys
import subprocess
import argparse
from collections import defaultdict
from pathlib import Path
import polars as pl


class DatasetSplitter:
    """Class to handle clustering and splitting of datasets."""

    def __init__(self, database_path, output_dir, tool_type="mmseqs", threads=8):
        self.database_path = database_path
        self.output_dir = Path(output_dir)
        self.tool_type = tool_type
        self.threads = threads

        # Fixed parameters for reproducibility
        self.cluster_threshold = 0.3
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # Derived attributes
        self.item_type = "sequences" if tool_type == "mmseqs" else "structures"
        self.tool_name = "MMseqs2" if tool_type == "mmseqs" else "FoldSeek"

        # Initialize paths
        self.base_name = os.path.basename(database_path)
        self.tmp_dir = self.output_dir / "tmp_clustering"
        self.cluster_dir = self.output_dir / "clusters"
        self.split_dir = self.output_dir / "splits"

        # Protein pair splitting paths
        self.merged_protein_file = Path(
            "data/interm/sprot_pre2024/merged_protein_similarity.parquet"
        )
        self.processed_dir = Path("data/processed/sprot_pre2024/sets")

        # Results storage
        self.cluster_info = {}
        self.splits = {}

    def run_command(self, cmd, check=True, capture_output=False):
        """Run a shell command and handle errors."""
        if isinstance(cmd, str):
            cmd = cmd.split()

        try:
            if capture_output:
                result = subprocess.run(
                    cmd, check=check, capture_output=True, text=True
                )
                return result.stdout, result.stderr
            else:
                result = subprocess.run(cmd, check=check)
                return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            if capture_output:
                print(f"Stderr: {e.stderr}")
            raise

    def check_dependencies(self):
        """Check if required tools are available."""
        if self.tool_type == "mmseqs":
            required_tools = ["mmseqs"]
            install_instructions = (
                "Please install mmseqs2: https://github.com/soedinglab/MMseqs2"
            )
        elif self.tool_type == "foldseek":
            required_tools = ["foldseek"]
            install_instructions = (
                "Please install FoldSeek: https://github.com/steineggerlab/foldseek"
            )
        else:
            raise ValueError(f"Unknown tool type: {self.tool_type}")

        missing = []
        for tool in required_tools:
            try:
                subprocess.run([tool, "--help"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing.append(tool)

        if missing:
            print(f"Error: Missing required tools: {missing}")
            print(install_instructions)
            sys.exit(1)

    def validate_database(self):
        """Validate that the database exists."""
        if not os.path.exists(f"{self.database_path}.dbtype"):
            print(f"Error: {self.tool_name} database '{self.database_path}' not found.")
            print(f"Expected file: {self.database_path}.dbtype")
            sys.exit(1)

    def cluster_items(self):
        """Cluster sequences/structures using mmseqs2 or foldseek."""
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.cluster_dir.mkdir(parents=True, exist_ok=True)

        cluster_db = self.tmp_dir / f"{self.base_name}_cluster_db"
        cluster_tsv = self.cluster_dir / f"{self.base_name}_clusters.tsv"

        # Run clustering if not already done
        if not os.path.exists(f"{cluster_db}.dbtype"):
            cmd = [
                self.tool_type,
                "cluster",
                self.database_path,
                str(cluster_db),
                str(self.tmp_dir),
                "-s",
                "7.5",
                "-e",
                "0.001",
                "-a",
                "--min-seq-id",
                str(self.cluster_threshold),
                "--cov-mode",
                "1",
                "-c",
                "0.8",
                "--threads",
                str(self.threads),
            ]
            self.run_command(cmd)

        # Convert to TSV if not already done
        if not cluster_tsv.exists():
            cmd = [
                self.tool_type,
                "createtsv",
                self.database_path,
                self.database_path,
                str(cluster_db),
                str(cluster_tsv),
            ]
            self.run_command(cmd)

        return cluster_tsv

    def parse_clusters(self, cluster_tsv_path):
        """Parse cluster TSV file and return cluster information."""
        clusters = defaultdict(list)

        with open(cluster_tsv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                representative = parts[0]
                member = parts[1]
                clusters[representative].append(member)

        # Create cluster membership information
        for cluster_id, (rep, members) in enumerate(clusters.items()):
            cluster_size = len(members)
            self.cluster_info[cluster_id] = {
                "representative": rep,
                "size": cluster_size,
                "members": members,
            }

    def create_splits(self):
        """Create train/val/test splits from clusters."""
        # Validate ratios
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {self.train_ratio + self.val_ratio + self.test_ratio}"
            )

        # Get cluster IDs and sort by size (largest first)
        cluster_ids = list(self.cluster_info.keys())
        cluster_ids.sort(key=lambda x: self.cluster_info[x]["size"], reverse=True)

        total_items = sum(self.cluster_info[cid]["size"] for cid in cluster_ids)
        target_train = int(total_items * self.train_ratio)
        target_val = int(total_items * self.val_ratio)
        target_test = total_items - target_train - target_val

        # Assign clusters to splits using greedy approach
        train_clusters = []
        val_clusters = []
        test_clusters = []

        train_count = 0
        val_count = 0
        test_count = 0

        for cluster_id in cluster_ids:
            cluster_size = self.cluster_info[cluster_id]["size"]

            # Calculate relative deficits
            train_deficit = (
                (target_train - train_count) / target_train if target_train > 0 else 0
            )
            val_deficit = (target_val - val_count) / target_val if target_val > 0 else 0
            test_deficit = (
                (target_test - test_count) / target_test if target_test > 0 else 0
            )

            # Assign to split with highest relative deficit
            if (
                train_deficit >= val_deficit
                and train_deficit >= test_deficit
                and (target_train - train_count) > 0
            ):
                train_clusters.append(cluster_id)
                train_count += cluster_size
            elif val_deficit >= test_deficit and (target_val - val_count) > 0:
                val_clusters.append(cluster_id)
                val_count += cluster_size
            else:
                test_clusters.append(cluster_id)
                test_count += cluster_size

        self.splits = {
            "train": {"clusters": train_clusters, "count": train_count},
            "val": {"clusters": val_clusters, "count": val_count},
            "test": {"clusters": test_clusters, "count": test_count},
        }

    def save_splits(self):
        """Save split information to files."""
        self.split_dir.mkdir(parents=True, exist_ok=True)

        total_items = sum(self.splits[split]["count"] for split in self.splits)

        for split_name, split_data in self.splits.items():
            cluster_list = split_data["clusters"]

            # Write item IDs
            items_file = self.split_dir / f"{split_name}_{self.item_type}.txt"
            with open(items_file, "w") as f:
                for cluster_id in cluster_list:
                    for member in self.cluster_info[cluster_id]["members"]:
                        f.write(f"{member}\n")

            # Write cluster information
            clusters_file = self.split_dir / f"{split_name}_clusters.txt"
            with open(clusters_file, "w") as f:
                f.write("cluster_id\trepresentative\tsize\n")
                for cluster_id in cluster_list:
                    rep = self.cluster_info[cluster_id]["representative"]
                    size = self.cluster_info[cluster_id]["size"]
                    f.write(f"{cluster_id}\t{rep}\t{size}\n")

        # Save cluster membership details
        cluster_members_file = self.cluster_dir / "cluster_members.tsv"
        with open(cluster_members_file, "w") as f:
            f.write("cluster_id\trepresentative\tmember\tcluster_size\n")
            for cluster_id, info in self.cluster_info.items():
                rep = info["representative"]
                size = info["size"]
                for member in info["members"]:
                    f.write(f"{cluster_id}\t{rep}\t{member}\t{size}\n")

        # Write summary
        summary_file = self.split_dir / "split_summary.txt"
        with open(summary_file, "w") as f:
            f.write("Dataset Split Summary\n")
            f.write("===================\n\n")
            f.write(f"Total {self.item_type}: {total_items}\n")
            f.write(f"Total clusters: {len(self.cluster_info)}\n\n")
            f.write("Split Distribution:\n")
            for split_name, split_data in self.splits.items():
                count = split_data["count"]
                clusters = len(split_data["clusters"])
                ratio = count / total_items
                f.write(
                    f"  {split_name.capitalize()}: {count} {self.item_type} ({ratio:.3f}) in {clusters} clusters\n"
                )

            f.write("\nFiles created:\n")
            for split_name in self.splits.keys():
                f.write(f"  {self.split_dir}/{split_name}_{self.item_type}.txt\n")
                f.write(f"  {self.split_dir}/{split_name}_clusters.txt\n")
                f.write(f"  {self.split_dir}/{split_name}_pairs.parquet\n")
            f.write(f"  {self.split_dir}/inter_split_pairs.parquet\n")

    def split_protein_pairs(self):
        """Split protein pairs from merged dataset based on cluster assignments."""
        if not self.merged_protein_file.exists():
            print(
                f"Warning: Merged protein similarity file not found at {self.merged_protein_file}"
            )
            return

        print(f"Splitting protein pairs from {self.merged_protein_file}...")

        # Read the merged dataset
        df = pl.read_parquet(self.merged_protein_file)

        # Read item IDs for each split
        splits = {}
        for split_name in ["train", "val", "test"]:
            items_file = self.split_dir / f"{split_name}_{self.item_type}.txt"
            with open(items_file, "r") as f:
                splits[split_name] = set(line.strip() for line in f)

        # Split the dataset
        for split_name, item_ids in splits.items():
            if not item_ids:
                continue

            # Filter pairs where both query and target are in this split
            split_df = df.filter(
                (pl.col("query").is_in(item_ids)) & (pl.col("target").is_in(item_ids))
            )

            # Save split-specific pairs
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            final_file = self.processed_dir / f"{split_name}.parquet"
            split_df.write_parquet(final_file)
            print(f"  {split_name}: {final_file} ({split_df.height} pairs)")

        # Identify inter-split pairs (pairs between different splits)
        all_train = splits["train"]
        all_val = splits["val"]
        all_test = splits["test"]

        # Find pairs between different splits
        inter_pairs = df.filter(
            (
                # Train-Val pairs
                ((pl.col("query").is_in(all_train)) & (pl.col("target").is_in(all_val)))
                | (
                    (pl.col("query").is_in(all_val))
                    & (pl.col("target").is_in(all_train))
                )
                |
                # Train-Test pairs
                (
                    (pl.col("query").is_in(all_train))
                    & (pl.col("target").is_in(all_test))
                )
                | (
                    (pl.col("query").is_in(all_test))
                    & (pl.col("target").is_in(all_train))
                )
                |
                # Val-Test pairs
                ((pl.col("query").is_in(all_val)) & (pl.col("target").is_in(all_test)))
                | (
                    (pl.col("query").is_in(all_test))
                    & (pl.col("target").is_in(all_val))
                )
            )
        )

        if inter_pairs.height > 0:
            inter_file = self.split_dir / "inter_split_pairs.parquet"
            inter_pairs.write_parquet(inter_file)
            print(f"  inter-split: {inter_file} ({inter_pairs.height} pairs)")
        else:
            print("  No inter-split pairs found")

    def run(self):
        """Execute the complete clustering and splitting pipeline."""
        print(
            f"Clustering {self.item_type} with {self.tool_name} "
            f"(threshold: {self.cluster_threshold:.1%}, "
            f"train/val/test: {self.train_ratio:.0%}/{self.val_ratio:.0%}/{self.test_ratio:.0%})"
        )

        try:
            # Validate inputs
            self.check_dependencies()
            self.validate_database()

            # Step 1: Cluster items
            cluster_tsv = self.cluster_items()

            # Step 2: Parse clusters
            self.parse_clusters(cluster_tsv)

            # Step 3: Create splits
            self.create_splits()

            # Step 4: Save splits
            self.save_splits()

            # Step 5: Split protein pairs
            self.split_protein_pairs()

            print(f"Done. Results saved to {self.output_dir}")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Cluster sequences/structures and create train/val/test splits using MMseqs2 or FoldSeek",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "database",
        help="Path to MMseqs2 or FoldSeek database (without .dbtype extension)",
    )
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument(
        "--tool",
        choices=["mmseqs", "foldseek"],
        default="mmseqs",
        help="Tool to use for clustering (default: mmseqs)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=8, help="Number of threads"
    )

    args = parser.parse_args()

    # Create and run the splitter
    splitter = DatasetSplitter(
        database_path=args.database,
        output_dir=args.output_dir,
        tool_type=args.tool,
        threads=args.threads,
    )
    splitter.run()


if __name__ == "__main__":
    main()
