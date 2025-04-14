# Project Specification: Protein Property Prediction

## 1. Overview

This project aims to train and evaluate machine learning models to predict specific protein properties based on pre-computed embeddings from various Protein Language Models (PLMs). The primary goal is to compare the performance of different model architectures (currently Feed-Forward Neural Network and Linear Regression) across different PLM embeddings and target parameters. Baselines using Euclidean distance and models trained on randomly generated embeddings are also included for comparison.

## 2. Core Workflow & Automation

-   **Experiment Execution:** Experiments are managed and run via the `run_experiments.py` script located at the project root.
-   **Automation:** This script automates the process of training and optionally evaluating models for all combinations of specified:
    -   Model types (`--model_types`, e.g., `fnn`, `linear`, `euclidean`)
    -   Target parameters (`--target_params`, e.g., `fident`, `alntmscore`, `hfsp`)
    -   Embedding files (found automatically as `.h5` files in `data/processed/sprot_embs/` - this includes both PLM and generated random embeddings).
-   **Redundancy Check:** Before launching a run, `run_experiments.py` checks if a corresponding output directory (`models/<model_type>_runs/<param_name>/<embedding_name>/`) already contains timestamped results. If so, it skips the combination to avoid redundant computations.
-   **Optional Evaluation:** The `--evaluate_after_train` flag can be passed to `run_experiments.py` to automatically trigger the evaluation script (`evaluate.py`) for each successfully completed run (including the Euclidean baseline setup).
-   **Usage Example:**
    ```bash
    # Generate random embeddings (if not already present)
    uv run python scripts/generate_random_embeddings.py

    # Run FNN, Linear, & Euclidean models for all params/embeddings using 'processed/sprot_train' data, evaluate after
    # This will include runs using the 'random_*.h5' files as input embeddings for FNN/Linear
    uv run python run_experiments.py --csv_dir data/processed/sprot_train --evaluate_after_train

    # Run only FNN for fident using only the prott5 embedding
    # (Requires manually adjusting run_experiments.py or providing specific embedding file path if needed)
    # uv run python run_experiments.py --csv_dir data/processed/sprot_train --model_types fnn --target_params fident --embedding_files data/processed/sprot_embs/prott5.h5
    ```

## 3. Key Scripts & Components

-   **`run_experiments.py`:** (Project Root) Orchestrates the entire experimental workflow. Iterates through specified model types, target parameters, and all `.h5` embedding files found in the embedding directory. Performs redundancy checks, calls `train.py`, and optionally calls `evaluate.py`.
-   **`scripts/generate_random_embeddings.py`:** (Scripts Directory) Generates HDF5 files (`random_<dim>.h5`) containing random embeddings (standard normal distribution) for protein IDs found in a template HDF5 file (e.g., `prott5.h5`). These files are saved to the standard embedding directory (`data/processed/sprot_embs/`) and are intended to be used as input embeddings for the standard training workflow (`fnn`, `linear`) to establish a baseline.
-   **`src/unknown_unknowns/train.py`:** Handles the training of a single model instance (`fnn`, `linear`) *or* sets up the run directory for the non-trainable Euclidean baseline.
    -   Takes parameters via command-line arguments (model type, data paths, hyperparameters).
    -   **If model type is `fnn` or `linear`:** Selects the appropriate model, uses PyTorch Lightning for training using the specified input embedding file (which could be a PLM embedding or a generated random embedding), saves checkpoints and TensorBoard logs.
    -   **If model type is `euclidean`:** Creates the standard run directory structure and saves an `hparams.yaml` file marking the model type. No training occurs; evaluation is handled directly by `evaluate.py`.
    -   Hyperparameter defaults for trainable models are defined directly within its `argparse` setup.
-   **`src/unknown_unknowns/evaluate.py`:** Evaluates a trained model checkpoint (`fnn`, `linear`) or the Euclidean baseline from a specific run directory.
    -   Takes the run directory (`--run_dir`) as input.
    -   Loads necessary hyperparameters from `hparams.yaml`.
    -   **If model type is `fnn` or `linear`:** Loads the model checkpoint (trained on either PLM or random embeddings) and runs inference on the test set.
    -   **If model type is `euclidean`:** Skips model loading and directly calculates the Euclidean distance between the specified PLM embeddings for the test set pairs using the `run_euclidean_distance` function.
    -   Runs inference/calculation on the test set (`test.csv` from the training data directory specified in `hparams.yaml` by default, can be overridden with `--test_csv`).
    -   Calculates and saves metrics (including Spearman, Pearson, R^2) and plots to an `evaluation_results` subdirectory.
-   **`src/unknown_unknowns/models/predictor.py`:** Contains the PyTorch Lightning model definitions (`FNNPredictor`, `LinearRegressionPredictor`).
-   **`src/unknown_unknowns/data/preprocessing.py`:** Contains scriptable logic to preprocess raw CSV data (e.g., applying NaN thresholds). Run via `uv run python src/unknown_unknowns/data/preprocessing.py --input_dir ... --output_dir ...`.
-   **`src/unknown_unknowns/data/datasets.py`:** Contains data loading logic (`create_single_loader`, `H5PyDataset`) for handling HDF5 embedding files and CSV data files.
-   **`src/unknown_unknowns/evaluation/metrics.py`:** Calculates evaluation metrics (MSE, RMSE, MAE, R2, Pearson, Pearson_r2, Spearman). Assumes inputs are NaN-free.
-   **`src/unknown_unknowns/visualization/plot.py`:** Generates plots (e.g., true vs. predicted).
-   **`src/unknown_unknowns/utils/helpers.py`:** Utility functions.

## 4. Data

-   **Raw Data:** Original CSV files are expected in `data/raw/<subdir_name>/` (e.g., `data/raw/training/`).
-   **Processed Data:** Preprocessing scripts save results to `data/processed/<subdir_name>/` (e.g., `data/processed/sprot_train/`). `run_experiments.py` uses the directory specified by `--csv_dir`.
-   **Embeddings:** Stored as HDF5 files (`.h5`) in `data/processed/sprot_embs/`. This directory contains both PLM embeddings (e.g., `prott5.h5`) and generated random embeddings (e.g., `random_512.h5`). `run_experiments.py` automatically discovers and uses all `.h5` files found here.
-   **CSV Files:** The directory specified via `--csv_dir` must contain `train.csv`, `val.csv`, and `test.csv`.

## 5. Output Structure

-   Runs are saved under `models/`.
-   Structure: `models/<model_type>_runs/<param_name>/<embedding_name>/<timestamp>/`
    -   `<model_type>`: `fnn`, `linear`, or `euclidean`.
    -   `<embedding_name>`: The stem of the HDF5 file used (e.g., `prott5`, `ankh_base`, `random_512`, `random_1024`).
-   Inside the `<timestamp>` directory:
    -   `checkpoints/`: Contains the saved model checkpoint (`.ckpt`) for trainable models (`fnn`, `linear` - trained on either PLM or random embeddings). Empty for Euclidean.
    -   `tensorboard/`: Contains TensorBoard logs (if applicable) and the `hparams.yaml` file.
    -   `evaluation_results/`: Contains evaluation metrics and plots if evaluation is run.

## 6. Key Design Decisions & Rationale

-   **Automation:** Shifted from manual script execution per experiment to an automated `run_experiments.py` script to handle numerous combinations efficiently and reduce errors.
-   **Configuration Management:** Eliminated the central `config.py` file. Configuration is handled via CLI arguments and `hparams.yaml` saved in run directories.
-   **Parameterization & Flexibility:** `run_experiments.py` and `train.py` are parameterized via CLI arguments.
-   **Path Handling:** `train.py` uses absolute paths passed via arguments. `run_experiments.py` resolves paths before calling `train.py`.
-   **Modularity:** Separated concerns into distinct scripts and modules.
-   **Model Selection:** `train.py` handles different trainable model architectures (`fnn`, `linear`) and sets up the Euclidean baseline.
-   **Random Embedding Baseline:** Implemented by generating random embedding files (`scripts/generate_random_embeddings.py`) and using them as standard inputs to the existing `fnn` and `linear` training workflows. This allows direct comparison of model performance on meaningful vs. random inputs using the same architecture.
-   **Output Organization:** Nested output directory structure clearly separates results.
-   **Progress Visibility:** Training progress bars are streamed live.
-   **Euclidean Baseline Integration:** Treated as a distinct non-trainable `model_type`.

## 7. Environment

-   The project uses `uv` for environment and task management. Commands are typically run using `uv run python ...`.

## 8. Potential Future Work

-   Implement a more sophisticated configuration system (e.g., YAML) for experiment sets.
-   Add more model types.
-   Add more evaluation metrics or plots.
-   Implement hyperparameter sweeping.
-   Add functionality to `run_experiments.py` to select specific embedding files via CLI instead of using all found `.h5` files.