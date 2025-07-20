# Data Preparation Module

Comprehensive protein similarity analysis and dataset splitting pipeline that processes MMSeqs2, FoldComp, and FoldSeek data.

## Overview

This module provides two main pipelines:

1. **Protein Similarity Analysis** (`merge_datasets.py`) - Processes and merges similarity data
2. **Dataset Splitting** (`split_dataset.py`) - Creates train/val/test splits with clustering

## Configuration

### Directory Structure

```
data/
└── interm/
    └── sprot_pre2024/
        ├── mmseqs/
        │   └── sprot_all_vs_all.tsv
        ├── foldcomp/
        │   └── plddt.tsv
        ├── foldseek/
        │   └── afdb_swissprot_v4_all_vs_all.tsv
        └── merged_protein_similarity.parquet
```

### Data Preparation

Before running the pipeline, extract pLDDT scores from FoldComp data:

```bash
# Extract pLDDT scores from FoldComp database
./bin/foldcomp extract --plddt -p 2 data/interm/sprot_pre2024/foldcomp/afdb_swissprot_v4 data/interm/sprot_pre2024/foldcomp/plddt.tsv
```

## Scripts

### 1. merge_datasets.py

**Purpose**: Process and merge protein similarity data from multiple sources

**Quick Start**:

```bash
# Run complete pipeline
python merge_datasets.py

# Test mode (recommended first)
python merge_datasets.py --test
```

**Output**: `interm/merged_protein_similarity.parquet`

### 2. split_dataset.py

**Purpose**: Create train/val/test splits using clustering to prevent data leakage

**Quick Start**:

```bash
# Sequence-based clustering (MMseqs2)
python split_dataset.py database output_dir --tool mmseqs

# Structure-based clustering (FoldSeek)
python split_dataset.py database output_dir --tool foldseek

# Custom thread count
python split_dataset.py database output_dir --tool foldseek -t 16
```

**Output**: Split files in `output_dir/splits/` and cluster information in `output_dir/clusters/`

## Data Processing Details

### merge_datasets.py

**Features**:

- Computes HFSP (functional similarity score)
- Processes MMSeqs2 (sequence similarity), Foldcomp (compressed PDB structures), and Foldseek (structural similarity)
- Applies quality thresholds and filtering
- Creates distribution plots for data quality analysis
- Merges datasets with full outer joins

**Processing Components**:

- **MMSeqs2**: Sequence similarity analysis with coverage ≥ 0.8, PIDE ≥ 0.3, HFSP ≥ 0.0
- **FoldComp**: Structural confidence assessment with pLDDT ≥ 70
- **FoldSeek**: Structural similarity analysis with coverage ≥ 0.8, TM-Score ≥ 0.4

**HFSP Formula**:
The HFSP (functional similarity score) is computed as:

$$
\begin{align*}
\text{HFSP} &= \text{PIDE} - \begin{cases}
100 & \text{if } L \leq 11 \\
770 \cdot L^{-0.33(1 + e^{L/1000})} & \text{if } 11 < L \leq 450 \\
28.4 & \text{if } L > 450
\end{cases} \\[1em]
\text{PIDE} &= \text{percentage sequence identity of the alignment} \\
L &= \text{ungapped alignment length}
\end{align*}
$$

### split_dataset.py

**Features**:

- Supports both MMseqs2 (sequences) and FoldSeek (structures)
- Clusters similar items to prevent data leakage
- Creates balanced train/val/test splits (70%/15%/15%)
- Generates comprehensive split summaries
- Splits protein pairs from merged similarity dataset

**Processing Components**:

- **Clustering**: Groups similar sequences/structures with 30% similarity threshold
- **Splitting**: Greedy assignment to maintain split ratios while preventing data leakage
- **Pair Splitting**: Filters protein pairs from merged dataset based on cluster assignments
- **Output**: Cluster-based train/val/test splits with detailed statistics and pair files

## Output Files

### merge_datasets.py

| File                                | Description                           |
| ----------------------------------- | ------------------------------------- |
| `merged_protein_similarity.parquet` | Final merged dataset                  |
| `plots/`                            | Distribution plots and visualizations |

**Dataset Columns**:

- `query`, `target`: Protein identifiers
- `fident`: Sequence identity (≥ 0.3)
- `hfsp`: Functional similarity score (≥ 0.0)
- `alntmscore`: Structural TM-Score (≥ 0.4)

### split_dataset.py

| File                               | Description                    |
| ---------------------------------- | ------------------------------ |
| `splits/train_sequences.txt`       | Training set items             |
| `splits/val_sequences.txt`         | Validation set items           |
| `splits/test_sequences.txt`        | Test set items                 |
| `splits/split_summary.txt`         | Comprehensive statistics       |
| `clusters/cluster_members.tsv`     | Detailed cluster information   |
| `splits/inter_split_pairs.parquet` | Pairs between different splits |
| `sets/train.parquet`               | Training set protein pairs     |
| `sets/val.parquet`                 | Validation set protein pairs   |
| `sets/test.parquet`                | Test set protein pairs         |
