# Research Repository Structure

This repository has been reorganized to provide a clean, workflow-based structure optimized for manuscript reproduction and research workflows. All code is organized under `src/` by functional purpose.

## Directory Structure

```
src/
├── data_preparation/              # Complete data processing pipeline
│   ├── embeddings/               # Embedding generation and processing
│   │   ├── embedding_generation.py    # PLM embedding generation
│   │   ├── batch_embedding_generation.sh # Batch processing of embeddings
│   │   └── random_embeddings.py       # Random baseline generation
│   ├── 2024_new_proteins/        # Novel protein discovery data pipeline
│   │   ├── extract_uniref_to_sqlite.py # UniRef database extraction
│   │   ├── get_uniref50.sh            # Download UniRef50 data
│   │   └── identify_novel_dissimilar_proteins.py # Novel protein identification
│   ├── preprocessing.py           # Data cleaning and filtering
│   ├── distance_computation.py   # Compute embedding distances
│   ├── run_mmseqs_all_vs_all.sh  # Sequence similarity search
│   ├── run_foldseek_all_vs_all.sh # Structure similarity search
│   ├── merge_mmseqs_foldseek_datasets.py # Combine similarity data
│   ├── get_best_pdbs.py          # Extract best ColabFold predictions
│   └── README.md                 # Data pipeline documentation
│
├── training/                      # Model training and experiments
│   ├── run_experiments.py        # 🎯 MAIN ORCHESTRATOR SCRIPT
│   ├── train.py                  # Individual model training
│   ├── models.py                 # Predictor model definitions
│   └── predict.py                # Inference module (future)
│
├── evaluation/                    # Model evaluation and metrics
│   ├── evaluate.py               # Single model evaluation
│   ├── evaluate_multiple.py     # Batch evaluation of multiple runs
│   └── metrics.py               # Regression metrics calculation
│
├── visualization/                 # All plotting and analysis
│   ├── create_performance_summary_plots.py # Performance summary plots
│   ├── create_evaluation_grid_plots.py # Evaluation grid layouts
│   ├── create_embedding_comparison_plots.py # Embedding comparison wrapper
│   ├── pairwise_embedding_comparison.py # Core embedding analysis
│   ├── plot_utils.py            # Core plotting utilities
│   ├── legacy_*.py              # Deprecated visualization scripts
│   └── README_pairwise_comparison.md # Pairwise analysis documentation
│
└── shared/                       # Shared utilities and components
    ├── datasets.py              # Data loading utilities
    ├── helpers.py               # Common helper functions
    └── configs/                 # Configuration management
```

## Quick Start

### 1. Main Experimental Workflow

The primary entry point for reproducing results is the experiment orchestrator:

```bash
# Run all model types for all parameters
python src/training/run_experiments.py \
    --csv_dir data/processed/sprot_train \
    --evaluate_after_train \
    --model_types fnn linear linear_distance euclidean

# Run specific combinations
python src/training/run_experiments.py \
    --csv_dir data/processed/sprot_train \
    --model_types fnn linear \
    --target_params fident alntmscore
```

### 2. Data Preparation Pipeline

```bash
# Generate embeddings for all proteins
python src/data_preparation/embeddings/embedding_generation.py sequences.fasta prott5

# Generate random embeddings for baseline comparison
python src/data_preparation/embeddings/random_embeddings.py \
    --template_h5 data/processed/sprot_embs/prott5.h5 \
    --output_dir data/processed/sprot_embs

# Compute distances between protein pairs
python src/data_preparation/distance_computation.py \
    --input_csv data/processed/sprot_train/test.csv \
    --embeddings_dir data/processed/sprot_embs \
    --output_csv data/processed/sprot_train/test_with_distances.csv

# Process sequence/structure similarity
./src/data_preparation/run_mmseqs_all_vs_all.sh sequences.fasta output/mmseqs
./src/data_preparation/run_foldseek_all_vs_all.sh pdb_dir/ output/foldseek
python src/data_preparation/merge_mmseqs_foldseek_datasets.py \
    --mmseqs_file output/mmseqs/results.tsv \
    --foldseek_file output/foldseek/results.tsv \
    --output_file merged_dataset.tsv
```

### 3. Evaluation and Analysis

```bash
# Evaluate a specific trained model
python src/evaluation/evaluate.py --run_dir models/sprot_train/fnn/fident/prott5/20241201_120000

# Batch evaluate multiple runs
python src/evaluation/evaluate_multiple.py --input_path models/sprot_train

# Generate performance summary plots
python src/visualization/create_performance_summary_plots.py \
    --results_dir models/sprot_train \
    --output out/summary_plots

# Create comparison grids
python src/visualization/create_evaluation_grid_plots.py \
    --input_path models/sprot_train \
    --output_dir out/grid_plots

# Generate embedding comparison analysis
python src/visualization/create_embedding_comparison_plots.py \
    --data_path data/processed/sprot_train/test_with_distances.csv \
    --output_dir out/embedding_analysis
```

## Key Features

- **📁 Workflow-based organization**: Clear separation by research function
- **🚀 Single-command reproduction**: `run_experiments.py` orchestrates everything
- **🔄 Resumable workflows**: Automatic detection of existing results
- **📊 Comprehensive analysis**: Multiple visualization and evaluation tools
- **🧪 Research-focused**: Optimized for manuscript reproduction, not package distribution
- **📈 Scalable**: Easy to add new models, metrics, or analyses

## File Naming Conventions

- **Executable scripts**: Use descriptive verbs (e.g., `run_experiments.py`, `create_embedding_comparison_plots.py`)
- **Library modules**: Use nouns (e.g., `models.py`, `metrics.py`, `datasets.py`)
- **Legacy files**: Prefixed with `legacy_` to indicate deprecated functionality
- **Documentation**: `README.md` files in each major directory

## Migration Notes

This structure replaces the previous mixed `scripts/` and `src/unknown_unknowns/` organization with a cleaner, function-based layout. Modules are imported by their top-level package name (`evaluation.*`, `training.*`, `shared.*`, `visualization.*`, `data_preparation.*`), which is exactly how the installed wheel exposes them.

## Dependencies

All dependencies are managed through `uv` and defined in `pyproject.toml`. The repository is properly configured so that all scripts can be run from the root directory without any path manipulation.

### Running Scripts

**Option 1: Using uv (recommended)**

```bash
# All scripts should be run from the project root using uv
uv run python src/training/run_experiments.py --help
uv run python src/evaluation/evaluate.py --help
uv run python src/visualization/create_embedding_comparison_plots.py --help
uv run python src/visualization/create_performance_summary_plots.py --help
uv run python src/visualization/create_evaluation_grid_plots.py --help
uv run python src/data_preparation/embeddings/embedding_generation.py --help
uv run python src/data_preparation/embeddings/random_embeddings.py --help
```

**Option 2: Activate virtual environment**

```bash
# Activate the environment first, then run scripts
source .venv/bin/activate  # or: uv shell
python src/training/run_experiments.py --help
python src/evaluation/evaluate.py --help
```

### Project Configuration

The repository uses proper Python packaging configuration in `pyproject.toml`:

- Build system: `hatchling`
- Package source: `src/` directory
- All imports use the `src.*` namespace
- No manual path manipulation required

For more details on specific components, see the README files in each subdirectory.
