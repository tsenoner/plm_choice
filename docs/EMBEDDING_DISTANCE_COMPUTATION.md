# Embedding Distance Computation

This document describes the embedding distance computation script that creates the necessary CSV files for pairwise embedding comparison visualizations.

## Overview

The `compute_embedding_distances.py` script computes euclidean distances between protein pairs for all available protein language model (PLM) embeddings. It takes:

- **Input**: CSV file with protein pairs + directory of H5 embedding files
- **Output**: CSV file with added distance columns for visualization analysis

## 🚀 Quick Start

### Basic Usage

```bash
# Compute distances for all embeddings
uv run python scripts/compute_embedding_distances.py \
    --input_csv data/processed/sprot_train/test.csv \
    --embeddings_dir data/processed/sprot_embs \
    --output_csv data/processed/sprot_train/test_with_distances.csv
```

### With Options

```bash
# Limit to 10,000 pairs and use smaller batches
uv run python scripts/compute_embedding_distances.py \
    --input_csv data/processed/sprot_train/test.csv \
    --embeddings_dir data/processed/sprot_embs \
    --output_csv data/processed/sprot_train/test_with_distances.csv \
    --sample_size 10000 \
    --batch_size 500 \
    --overwrite
```

## 📊 How It Works

### 1. **Input Validation**

```python
# Required CSV columns
required_columns = ['query', 'target']

# Example input CSV:
#   query,target,fident,alntmscore,hfsp
#   protein_001,protein_002,0.85,0.72,45.3
#   protein_003,protein_004,0.42,0.61,-12.7
```

### 2. **Embedding Discovery**

- Automatically finds all `.h5` files in the embeddings directory
- Extracts embedding metadata (dimensions, protein count, shape)
- Handles both protein-level and sequence-level embeddings

### 3. **Distance Computation Process**

#### For Each Embedding Type:

1. **Load Required Proteins**: Only loads embeddings for proteins present in the CSV
2. **Batch Processing**: Processes protein pairs in configurable batches (default: 1000)
3. **Distance Calculation**: Computes euclidean distance: `||embedding_query - embedding_target||`
4. **Handle Missing Data**: Assigns `NaN` for proteins not found in embeddings

#### Memory Management:

- Loads only required protein embeddings (not entire H5 files)
- Processes in batches to control memory usage
- Explicit garbage collection between embeddings
- Handles both protein-level and sequence-level embeddings

### 4. **Output Generation**

Creates CSV with original columns plus distance columns:

```csv
query,target,fident,alntmscore,hfsp,dist_prott5,dist_esm2_8m,dist_ankh_base,...
protein_001,protein_002,0.85,0.72,45.3,12.45,8.92,15.33,...
protein_003,protein_004,0.42,0.61,-12.7,18.76,11.24,22.41,...
```

## 🔧 **Key Features**

### **Smart Embedding Handling**

- **Protein-level embeddings**: Used directly (e.g., single vector per protein)
- **Sequence-level embeddings**: Automatically averaged across sequence length
- **Mixed dimensions**: Handles different embedding sizes (320D, 768D, 1024D, etc.)

### **Robust Data Processing**

- **Missing proteins**: Gracefully handles proteins not found in embeddings
- **Error handling**: Continues processing even if individual embeddings fail
- **Progress tracking**: Shows detailed progress for each embedding
- **Statistics logging**: Reports coverage, mean, and std for each embedding

### **Performance Optimization**

- **Incremental saving**: Results saved after each embedding, preventing memory accumulation
- **Resumable computation**: Automatically resumes from partially completed files
- **Batch processing**: Configurable batch sizes for memory control
- **Selective loading**: Only loads required protein embeddings
- **Memory cleanup**: Explicit garbage collection between embeddings
- **Progress bars**: Real-time progress tracking with `tqdm`

## 📋 **Command Line Options**

| Option             | Required | Default | Description                                                      |
| ------------------ | -------- | ------- | ---------------------------------------------------------------- |
| `--input_csv`      | ✅       | -       | CSV file with protein pairs ('query', 'target' columns required) |
| `--embeddings_dir` | ✅       | -       | Directory containing H5 embedding files                          |
| `--output_csv`     | ✅       | -       | Output CSV file path                                             |
| `--sample_size`    | ❌       | None    | Limit number of rows (for testing/memory)                        |
| `--batch_size`     | ❌       | 1000    | Protein pairs per batch                                          |
| `--overwrite`      | ❌       | False   | Overwrite output file if exists                                  |

## 🧪 **Testing**

### Run Tests

```bash
# Test with synthetic data
uv run python scripts/test_embedding_distance_computation.py
```

**Test Features:**

- Creates synthetic H5 embedding files with different dimensions
- Generates synthetic protein pair data
- Validates distance computation logic
- Checks for real data availability
- Provides next-step instructions

### Expected Test Output

```
✅ All tests passed!
Test Results:
  Input pairs: 100
  Distance columns: 3
  dist_test_ankh: 100.0% coverage, mean=3.803, std=0.525
  dist_test_esm2_8m: 100.0% coverage, mean=25.340, std=0.960
  dist_test_prott5: 100.0% coverage, mean=45.158, std=1.039
```

## 💡 **Advanced Usage**

### **Large Dataset Processing**

```bash
# For large datasets, use smaller batches and limit size
uv run python scripts/compute_embedding_distances.py \
    --input_csv large_dataset.csv \
    --embeddings_dir data/processed/sprot_embs \
    --output_csv large_dataset_distances.csv \
    --sample_size 50000 \
    --batch_size 500
```

### **Memory-Constrained Environments**

```bash
# Reduce batch size for low-memory systems
uv run python scripts/compute_embedding_distances.py \
    --input_csv data.csv \
    --embeddings_dir embeddings/ \
    --output_csv output.csv \
    --batch_size 100
```

### **Development/Testing**

```bash
# Quick test with small sample
uv run python scripts/compute_embedding_distances.py \
    --input_csv data.csv \
    --embeddings_dir embeddings/ \
    --output_csv test_output.csv \
    --sample_size 1000 \
    --overwrite
```

### **Resuming Interrupted Computations**

The script automatically resumes from where it left off:

```bash
# First run (gets interrupted after 5 embeddings)
uv run python scripts/compute_embedding_distances.py \
    --input_csv large_dataset.csv \
    --embeddings_dir embeddings/ \
    --output_csv results.csv

# Second run (automatically resumes from embedding #6)
uv run python scripts/compute_embedding_distances.py \
    --input_csv large_dataset.csv \
    --embeddings_dir embeddings/ \
    --output_csv results.csv
    # Will skip already computed embeddings!

# Force restart from beginning
uv run python scripts/compute_embedding_distances.py \
    --input_csv large_dataset.csv \
    --embeddings_dir embeddings/ \
    --output_csv results.csv \
    --overwrite
```

## 🔄 **Integration with Visualization Pipeline**

### Complete Workflow

```bash
# Step 1: Compute distances
uv run python scripts/compute_embedding_distances.py \
    --input_csv data/processed/sprot_train/test.csv \
    --embeddings_dir data/processed/sprot_embs \
    --output_csv data/processed/sprot_train/test_with_distances.csv

# Step 2: Generate visualizations
uv run python scripts/create_pairwise_embedding_visualizations.py \
    --data_path data/processed/sprot_train/test_with_distances.csv \
    --output_dir out/embedding_analysis

# Step 3: View results
open out/embedding_analysis/
```

## 📈 **Performance Characteristics**

### **Time Complexity**

- **Per embedding**: O(P × B) where P = unique proteins, B = batch size
- **Total**: O(E × P × B) where E = number of embeddings

### **Memory Usage**

- **Per embedding**: ~unique_proteins × embedding_dimension × 8 bytes
- **Example**: 1000 proteins × 1024D × 8 bytes ≈ 8MB per embedding
- **Total memory**: Only one embedding in memory at a time (not cumulative)
- **Optimization**: Incremental processing prevents memory accumulation

### **Typical Performance**

- **Small dataset** (1K pairs, 5 embeddings): ~30 seconds
- **Medium dataset** (10K pairs, 10 embeddings): ~5 minutes
- **Large dataset** (100K pairs, 15 embeddings): ~45 minutes

## ⚠️ **Common Issues & Solutions**

### **Issue: Missing Proteins**

```
Warning: Only found 850/1000 proteins in embedding file
```

**Solution**: This is normal - not all proteins exist in all embeddings. Check coverage statistics in output.

### **Issue: Memory Errors**

```
Error: Out of memory during distance computation
```

**Solution**: Reduce `--batch_size` (try 100-500) and/or use `--sample_size` to limit data.

### **Issue: Incorrect CSV Format**

```
Error: Missing required columns: ['query', 'target']
```

**Solution**: Ensure your CSV has 'query' and 'target' columns with protein IDs.

### **Issue: No H5 Files Found**

```
Error: No H5 files found in embeddings directory
```

**Solution**: Check path and ensure `.h5` files exist in the specified directory.

## 🔍 **Output Statistics Interpretation**

The script provides detailed statistics for each embedding:

```
dist_prott5: 95.2% coverage, mean=12.456, std=3.421
```

- **Coverage**: Percentage of protein pairs with valid distances
- **Mean**: Average euclidean distance (varies by embedding dimension)
- **Std**: Standard deviation of distances

### **Typical Distance Ranges**

- **Low-dimensional** (320D): ~8-15 average distance
- **Medium-dimensional** (768D): ~15-25 average distance
- **High-dimensional** (1024D+): ~25-50 average distance

### **Coverage Expectations**

- **95-100%**: Excellent coverage, comprehensive embedding set
- **80-95%**: Good coverage, some proteins missing
- **<80%**: Limited coverage, may need different embedding source

## 🎯 **Best Practices**

### **Data Preparation**

1. **Validate CSV format** before processing
2. **Check protein ID consistency** between CSV and embeddings
3. **Start with small samples** for testing

### **Performance Optimization**

1. **Use appropriate batch sizes** (1000 for most cases)
2. **Limit sample size** for initial exploration
3. **Monitor memory usage** during processing

### **Quality Control**

1. **Check coverage statistics** for each embedding
2. **Validate distance ranges** (should be non-negative)
3. **Compare results** across different embeddings

### **Workflow Integration**

1. **Use consistent file naming** (e.g., `*_with_distances.csv`)
2. **Document parameters used** for reproducibility
3. **Save intermediate results** before visualization

## 🔮 **Future Enhancements**

Potential improvements:

1. **Distance metrics**: Support for cosine similarity, Manhattan distance
2. **Parallel processing**: Multi-threading for faster computation
3. **Incremental processing**: Add distances for new embeddings to existing CSV
4. **Quality metrics**: Automatic validation of computed distances
5. **Caching**: Save computed distances for reuse
6. **GPU acceleration**: CUDA support for large-scale processing

## 📚 **Related Documentation**

- [Pairwise Embedding Visualization](./PAIRWISE_EMBEDDING_VISUALIZATION_SUMMARY.md)
- [Project Specification](./SPECIFICATION.md)
- [Visualization README](../src/unknown_unknowns/visualization/README_pairwise_comparison.md)
