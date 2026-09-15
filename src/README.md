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
│   ├── novel_2024/              # Novel protein discovery data pipeline
│   │   ├── extract_uniref_to_sqlite.py # UniRef database extraction
│   │   ├── get_uniref50.sh            # Download UniRef50 data
│   │   └── identify_novel_dissimilar_proteins.py # Novel protein identification
│   ├── distance_computation.py   # Compute embedding distances
│   ├── all_vs_all_distance_computation.py # All-vs-all distances + viz caches
│   ├── merge_datasets.py         # Combine MMseqs/Foldseek similarity data
│   ├── merge_parquet_columns.py  # Merge new target columns into the splits
│   ├── go_semantic_similarity.py # GO Wang similarity (C1)
│   ├── brenda_hfsp_validation.py # HFSP vs curated enzyme classes (C1)
│   ├── pdb_tmscore.py            # SIFTS → RCSB → TMalign experimental TM-scores
│   ├── ecod_homology_pairs.py    # Per-ECOD-group distance densities (C2)
│   ├── organism_landscape.py     # Distance distributions by organism group
│   ├── run_mmseqs_all_vs_all.sh  # Sequence similarity search
│   └── run_foldseek_all_vs_all.sh # Structure similarity search
│
├── training/                      # Model training and experiments
│   ├── run_experiments.py        # 🎯 MAIN ORCHESTRATOR SCRIPT
│   ├── train.py                  # Individual model training
│   └── models.py                 # Predictor model definitions
│
├── evaluation/                    # Model evaluation and metrics
│   ├── evaluate.py               # Single model evaluation
│   ├── evaluate_multiple.py     # Batch evaluation of multiple runs
│   ├── metrics.py               # Regression metrics calculation
│   ├── stats.py                 # Shared statistics (effect sizes, CIs, tests)
│   ├── recall_fp.py             # ⭐ CANONICAL recall-at-first-FP (+ barrier spec)
│   ├── retrieval_metrics.py     # Flat-vector retrieval metrics — NOT canonical
│   ├── classification_eval.py   # AUROC + recall@1FP per hierarchy level
│   └── overtraining_analysis.py # Probe-capacity diagnostics
│
├── visualization/                 # All plotting and analysis
│   ├── plm_constants.py         # ⭐ Shared pLM sizes, families, colours, labels
│   ├── create_performance_summary_plots.py # Performance summary plots
│   ├── create_evaluation_grid_plots.py # Evaluation grid layouts
│   ├── create_embedding_comparison_plots.py # Embedding comparison wrapper
│   ├── create_retrieval_plots.py # Retrieval/classification panels
│   ├── pairwise_embedding_comparison.py # Core embedding analysis
│   ├── plot_ecdf.py             # ECDF panels
│   ├── plot_utils.py            # Core plotting utilities
│   └── README_pairwise_comparison.md # Pairwise analysis documentation
│
└── shared/                       # Shared utilities — the layer both
    │                             # data_preparation and visualization may import
    ├── datasets.py              # Data loading utilities
    ├── helpers.py               # Common helper functions
    ├── experiment_manager.py    # Run directory / experiment bookkeeping
    ├── atomic_io.py             # Atomic writes + completeness-guarded skip
    ├── hierarchies.py           # ECOD/SCOP level names shared by producer + figures
    └── embedding_names.py       # i.i.d. random floor vs untrained architecture
```

> **Layering note.** `shared/` exists so that a constant needed by both a producer and a
> figure has one home: `visualization/` is *not* importable from `data_preparation/`.
> When the same table or predicate is needed on both sides, it belongs in `shared/`.

## Quick Start

### 0. The `plm` CLI — the preferred entry point

`pyproject.toml` registers a console script, so every module below is reachable by name
rather than by file path. Path-addressed invocations rot when a module moves; the CLI is
covered by `tests/test_cli.py`, so a rename that breaks it fails the suite.

```bash
uv run plm --help                 # all command groups
uv run plm data --help            # data_preparation entry points
uv run plm evaluate --help        # evaluation entry points
uv run plm figures --help         # visualization entry points

uv run plm data go-similarity --help
uv run plm evaluate classification --help
uv run plm figures retrieval --help
```

The `uv run python src/...` forms below still work and are what
`scripts/run_ivan_pipeline.sh` currently uses.

### 1. Main Experimental Workflow

The primary entry point for reproducing results is the experiment orchestrator:

```bash
# Run all model types for all parameters
python src/training/run_experiments.py \
    --data_dir data/processed/sprot_pre2024_subset \
    --evaluate_after_train \
    --model_types fnn linear linear_distance euclidean

# Run specific combinations
python src/training/run_experiments.py \
    --data_dir data/processed/sprot_pre2024_subset \
    --model_types fnn linear \
    --target_params fident alntmscore
```

`--data_dir` is required and has no default — it selects the cohort, which *is* the probe
budget. `sprot_pre2024_subset` is what every published cell used.

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
- Modules are imported by their **top-level package name** — `evaluation.*`, `training.*`,
  `shared.*`, `visualization.*`, `data_preparation.*`. There is **no `src.*` namespace**:
  `pyproject.toml` maps each `src/<pkg>` into the wheel at top level, and `pythonpath`
  gives pytest the same view. `from src.evaluation...` will fail outside pytest.
- No manual path manipulation required

For more details on specific components, see the README files in each subdirectory.
