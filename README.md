# Protein Language Model Evaluation Framework

A framework for training and evaluating machine learning models that predict protein properties based on embeddings from various Protein Language Models (PLMs).

## Overview

This project enables systematic comparison of different model architectures (feed-forward networks, linear regression, distance-based models) across multiple PLM embeddings and target protein properties. It automates the experimental workflow from training to evaluation and visualization.

**Key capabilities:**

- Automated training/evaluation across model types and embeddings
- Multiple baseline comparisons (Euclidean distance, random embeddings)
- Comprehensive performance metrics and visualizations
- Organized output structure for systematic analysis

## Quick Start

### Prerequisites

- Python >= 3.12
- `uv` package manager

### Installation

```bash
git clone <repository-url>
cd unknown_unknowns
uv sync
```

### Basic Usage

1. **Prepare your data structure:**

   ```
   data/processed/sprot_train/     # CSV files (train.csv, val.csv, test.csv)
   data/processed/sprot_embs/      # HDF5 embedding files
   ```

2. **Run experiments:**

   ```bash
   # Train and evaluate all model combinations
   uv run python src/training/run_experiments.py \
     --csv_dir data/processed/sprot_train \
     --evaluate_after_train \
     --model_types fnn linear linear_distance euclidean
   ```

3. **Generate summary plots:**
   ```bash
   # Performance summary across all runs
   uv run python src/visualization/create_performance_summary_plots.py \
     --results_dir models/sprot_train \
     --output out/plots
   ```

## Model Types

- **`fnn`:** Feed-forward neural network
- **`linear`:** Linear regression on concatenated embeddings
- **`linear_distance`:** Linear regression on embedding differences
- **`euclidean`:** Euclidean distance baseline (no training)

## Output

Results are organized as:

```
models/<dataset>/<model_type>/<parameter>/<embedding>/<timestamp>/
├── checkpoints/         # Model weights
├── tensorboard/         # Training logs
└── evaluation_results/  # Plots and metrics
```

For detailed documentation, see `docs/SPECIFICATION.md`.
