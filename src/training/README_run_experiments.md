# Training Experiment Runner

A comprehensive training pipeline that automatically runs protein similarity prediction experiments across multiple models, embeddings, and target parameters.

## What It Does

Trains and evaluates models that predict protein similarity metrics (`fident`, `alntmscore`, `hfsp`) from protein embeddings, comparing different embedding methods and model architectures.

## How It Works

### Core Workflow

1. **Discovery**: Finds all `.h5` embedding files in `<data_dir>/embeddings/`
2. **Combination**: Creates experiments for every combination of:
   - Model types: `fnn`, `linear`, `euclidean`, `linear_distance`
   - Target parameters: `fident`, `alntmscore`, `hfsp`
   - Embedding files: All `.h5` files found
3. **Training**: Calls `train.py` for each combination
4. **Evaluation**: Optionally runs `evaluate.py` after successful training
5. **Progress**: Shows real-time progress with tqdm, logs everything

### Model Types

- **`fnn`**: Feed-forward neural network with hidden layers
- **`linear`**: Simple linear regression on concatenated embeddings
- **`linear_distance`**: Linear regression on element-wise squared differences
- **`euclidean`**: Baseline using raw Euclidean distance (no training)

## Structure

### Data Organization

```
{data_dir}/
├── sets/                    # Dataset splits (train/val/test.parquet)
└── embeddings/             # Protein embeddings (*.h5 files)
```

### Output Organization

```
models/
└── {data_dir_name}/        # e.g., sprot_pre2024/
    └── {model_type}/       # fnn, linear, etc.
        └── {param_name}/   # fident, alntmscore, hfsp
            └── {embedding}/ # esm2_650m, prott5, etc.
                ├── {timestamp}/     # 20240720_173549/
                │   ├── checkpoints/ # Best model weights
                │   ├── tensorboard/ # Training logs & metrics
                │   └── evaluation_results/ # Plots & metrics
                └── run_logs/        # Execution logs
```

## Usage

```bash
# Run all combinations with evaluation
uv run python src/training/run_experiments.py \
    --data_dir data/processed/sprot_pre2024 \
    --evaluate_after_train

# Run specific subset
uv run python src/training/run_experiments.py \
    --model_types fnn linear \
    --target_params fident \
    --evaluate_after_train
```

## Key Features

- **Smart Skipping**: Avoids re-running completed experiments
- **Robust Logging**: All output captured in timestamped log files
- **Progress Tracking**: Real-time progress bar with error counts
- **Automatic Evaluation**: Optional post-training evaluation with metrics/plots
- **Error Handling**: Continues on failures, reports issues clearly

## Dependencies

- Uses `train.py` (PyTorch Lightning training)
- Uses `evaluate.py` (model evaluation & visualization)
- Requires parquet datasets (created by `split_dataset.py`)
- Requires protein embeddings in HDF5 format
