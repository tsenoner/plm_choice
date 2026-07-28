# All-vs-All Embedding Distance Analysis

This document describes the all-vs-all embedding distance computation script that provides comprehensive analysis of protein language model (PLM) embeddings.

## Overview

The `all_vs_all_distance_computation.py` script provides a comprehensive, standalone solution that:

1. **Discovers all proteins** from embedding files automatically
2. **Computes all-vs-all distances** between every protein pair
3. **Generates visualization cache files** automatically for downstream analysis
4. **Handles memory efficiently** with chunked processing
5. **Supports resuming** interrupted computations

## Key Features

### Comprehensive Coverage

- Analyzes ALL proteins present in embedding files
- Computes distances for every possible protein pair (N×N matrix)
- Ensures complete coverage across all embedding types

### Visualization Ready

- Generates all cache files needed for `pairwise_embedding_comparison.py`
- Includes hexbin, correlation, Wasserstein, and distribution data
- No additional preprocessing needed for visualization

### Memory Efficient

- Processes proteins in configurable chunks
- Supports sampling for testing/memory constraints
- Incremental saving with resume capability

### Flexible Input

- Works with any directory of H5 embedding files
- Configurable precision and batch sizes
- Simple interface with no dependencies on external data files

## Usage

### Basic Usage

```bash
# Analyze all proteins with default settings
uv run python src/data_preparation/all_vs_all_distance_computation.py \
    --embeddings_dir data/processed/sprot_embs \
    --output_dir out/all_vs_all_analysis
```

### With Protein Limit (Recommended for Testing)

```bash
# Limit analysis to first 1000 proteins
uv run python src/data_preparation/all_vs_all_distance_computation.py \
    --embeddings_dir data/processed/sprot_embs \
    --output_dir out/all_vs_all_analysis \
    --max_proteins 1000
```

### With Custom Settings

```bash
# Use custom chunk size and precision settings
uv run python src/data_preparation/all_vs_all_distance_computation.py \
    --embeddings_dir data/processed/sprot_embs \
    --output_dir out/all_vs_all_analysis \
    --max_proteins 2000 \
    --chunk_size 100 \
    --precision 3
```

### Resume Interrupted Computation

```bash
# Resume from existing partial results
uv run python src/data_preparation/all_vs_all_distance_computation.py \
    --embeddings_dir data/processed/sprot_embs \
    --output_dir out/all_vs_all_analysis \
    --max_proteins 1000 \
    --resume
```

### Skip Cache Generation (Distance Only)

```bash
# Only compute distances, skip visualization cache
uv run python src/data_preparation/all_vs_all_distance_computation.py \
    --embeddings_dir data/processed/sprot_embs \
    --output_dir out/all_vs_all_analysis \
    --max_proteins 1000 \
    --skip_cache
```

## Arguments

| Argument           | Type | Default  | Description                              |
| ------------------ | ---- | -------- | ---------------------------------------- |
| `--embeddings_dir` | Path | Required | Directory containing H5 embedding files  |
| `--output_dir`     | Path | Required | Directory to save outputs and cache      |
| `--max_proteins`   | int  | None     | Maximum proteins to analyze (None = all) |
| `--chunk_size`     | int  | 100      | Proteins per processing chunk            |
| `--precision`      | int  | 4        | Decimal places for distances             |
| `--resume`         | flag | False    | Resume from existing results             |
| `--skip_cache`     | flag | False    | Skip visualization cache generation      |

## Output Structure

```
output_dir/
├── all_vs_all_distances.parquet          # Main distance data
└── cache/                                 # Visualization cache files
    ├── hexbin_data.json
    ├── correlation_data.json
    ├── wasserstein_data.json
    ├── distribution_data.json
    └── distribution_normalized_data.json
```

### Main Output File

The `all_vs_all_distances.parquet` file contains:

- `query`: Source protein ID
- `target`: Target protein ID
- `dist_<embedding_name>`: Distance columns for each embedding type

### Cache Files

Generated cache files are compatible with `pairwise_embedding_comparison.py`:

- **hexbin_data.json**: Pre-computed hexagonal binning data
- **correlation_data.json**: Spearman correlation matrices with confidence intervals
- **wasserstein_data.json**: Wasserstein distance matrices between distributions
- **distribution_data.json**: Raw distribution data for plotting
- **distribution_normalized_data.json**: Normalized distribution data

## Memory Considerations

### Scaling

For N proteins and E embeddings, the script computes:

- **Distance computations**: N² × E
- **Memory usage**: Proportional to N × E (chunk-based processing)
- **Output size**: N² rows in final parquet file

### Recommendations

| Proteins | Memory    | Computation Time | Recommendation    |
| -------- | --------- | ---------------- | ----------------- |
| 100      | Low       | Minutes          | Good for testing  |
| 1,000    | Moderate  | Hours            | Typical analysis  |
| 10,000   | High      | Days             | Large-scale study |
| 100,000+ | Very High | Weeks            | Consider HPC      |

### Memory Management

```bash
# For large datasets, use smaller chunks
--chunk_size 50

# Monitor memory usage and adjust max_proteins as needed
--max_proteins 5000
```

## Integration with Visualization

Once the analysis completes, use the generated files directly with the visualization script:

```bash
# Generate all visualizations using cache files
uv run python src/visualization/pairwise_embedding_comparison.py \
    --data_path out/all_vs_all_analysis/all_vs_all_distances.parquet \
    --output_dir out/all_vs_all_analysis/visualizations \
    --visualizations all
```

The visualization script will automatically detect and use the cache files, significantly speeding up plot generation.

## Comparison with Standard Distance Computation

| Feature      | `distance_computation.py`  | `all_vs_all_distance_computation.py` |
| ------------ | -------------------------- | ------------------------------------ |
| **Input**    | Protein pairs from parquet | Pure embedding files only            |
| **Coverage** | Specific pairs only        | Complete all-vs-all matrix           |
| **Output**   | Distance columns only      | Distances + visualization cache      |
| **Memory**   | Batch-based                | Chunk-based with resume              |
| **Use Case** | Targeted analysis          | Comprehensive exploration            |

## Examples

### Quick Test Run

```bash
# Test with minimal dataset
uv run python src/data_preparation/all_vs_all_distance_computation.py \
    --embeddings_dir data/processed/sprot_embs \
    --output_dir out/test_analysis \
    --max_proteins 100 \
    --chunk_size 20
```

### Production Analysis

```bash
# Full analysis for publication
uv run python src/data_preparation/all_vs_all_distance_computation.py \
    --embeddings_dir data/processed/sprot_embs \
    --output_dir out/production_analysis \
    --max_proteins 10000 \
    --chunk_size 100 \
    --precision 4
```

### Resume Long-Running Job

```bash
# Resume after interruption
uv run python src/data_preparation/all_vs_all_distance_computation.py \
    --embeddings_dir data/processed/sprot_embs \
    --output_dir out/production_analysis \
    --max_proteins 10000 \
    --resume
```

## Performance Tips

1. **Start small**: Test with `--max_proteins 100` first
2. **Monitor memory**: Adjust `--chunk_size` based on available RAM
3. **Use resume**: Long jobs can be interrupted and resumed safely
4. **SSD storage**: Use fast storage for output directory
5. **Parallel I/O**: Ensure H5 files are on fast storage

## Troubleshooting

### Common Issues

**Memory Error**

```bash
# Reduce chunk size and/or max proteins
--chunk_size 50 --max_proteins 1000
```

**Slow Performance**

```bash
# Check storage speed and reduce I/O
# Use local SSD for embeddings_dir and output_dir
```

**Resume Not Working**

```bash
# Check that output_dir contains all_vs_all_distances.parquet
# Delete partial file if corrupted and restart
```

### Validation

After completion, verify the results:

```python
import polars as pl

# Load and inspect results
df = pl.read_parquet("out/all_vs_all_analysis/all_vs_all_distances.parquet")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")

# Check distance statistics
dist_cols = [c for c in df.columns if c.startswith("dist_")]
for col in dist_cols:
    valid = df[col].drop_nulls()
    print(f"{col}: {len(valid)} valid, mean={valid.mean():.3f}")
```

This comprehensive analysis provides the foundation for deep exploration of protein embedding spaces and their relationships.
