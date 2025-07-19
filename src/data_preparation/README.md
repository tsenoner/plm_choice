# Data Preparation Module

Comprehensive protein similarity analysis pipeline that processes MMSeqs2, FoldComp, and FoldSeek data to analyze protein similarities and structural qualities.

## Overview

This module provides a unified pipeline that:

- Processes sequence similarity (MMSeqs2), structural confidence (FoldComp), and structural similarity (FoldSeek) data
- Applies quality thresholds and filtering
- Creates distribution plots for data quality analysis
- Merges datasets with full outer joins to preserve all protein pairs

## Quick Start

```bash
# Run the complete pipeline (full datasets)
python merge_datasets.py

# Run in test mode (recommended first)
python merge_datasets.py --test
```

## Pipeline Components

### 1. MMSeqs2 Processing

- **Purpose**: Sequence similarity analysis
- **Filters**:
  - Coverage ≥ 0.8 (minimum of query/target coverage)
  - PIDE ≥ 0.3 (percentage identity)
  - HFSP ≥ 0.0 (functional similarity score)
- **Output**: Sequence similarity scores with computed HFSP values

### 2. FoldComp Processing

- **Purpose**: Structural confidence assessment
- **Filters**: pLDDT ≥ 70 (structural confidence threshold)
- **Output**: Average pLDDT scores for structural quality assessment

### 3. FoldSeek Processing

- **Purpose**: Structural similarity analysis
- **Filters**:
  - Coverage ≥ 0.8 (minimum of query/target coverage)
  - TM-Score ≥ 0.4 (structural alignment quality)
- **Output**: Structural similarity scores

### 4. Data Merging

- **Method**: Full outer join to preserve all protein pairs from both datasets
- **Output**: Comprehensive dataset with sequence and structural similarity metrics

## Configuration

The pipeline automatically configures paths based on the expected directory structure:

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

Before running the pipeline, you need to extract pLDDT scores from FoldComp data:

```bash
# Extract pLDDT scores from FoldComp database
./bin/foldcomp extract --plddt -p 2 data/interm/sprot_pre2024/foldcomp/afdb_swissprot_v4 data/interm/sprot_pre2024/foldcomp/plddt.tsv
```

## Output

### Dataset

The final merged dataset contains:

| Column       | Source   | Description                   | Threshold |
| ------------ | -------- | ----------------------------- | --------- |
| `query`      | Both     | Query protein identifier      | -         |
| `target`     | Both     | Target protein identifier     | -         |
| `fident`     | MMSeqs2  | Sequence identity             | ≥ 0.3     |
| `hfsp`       | Computed | Functional similarity score   | ≥ 0.0     |
| `alntmscore` | FoldSeek | Structural alignment TM-Score | ≥ 0.4     |

### Visualizations

The pipeline generates:

- Individual violin plots for each metric distribution
- Combined 2×3 subplot figure with all distributions
- Threshold annotations showing data quality statistics

## HFSP Formula

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

## Features

- **Test Mode**: Process smaller datasets for pipeline validation
- **Caching**: Automatic parquet conversion for improved performance
- **Quality Control**: Comprehensive filtering and thresholding
- **Visualization**: Distribution plots with threshold annotations
- **Statistics**: Detailed merge statistics and overlap analysis
- **Memory Efficient**: Uses Polars for high-performance data processing
