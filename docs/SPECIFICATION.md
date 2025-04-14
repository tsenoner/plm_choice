# Project Specification: Protein Property Prediction

## 1. Overview

This project aims to train and evaluate machine learning models to predict specific protein properties based on pre-computed embeddings from various Protein Language Models (PLMs). The primary goal is to compare the performance of different model architectures (currently Feed-Forward Neural Network and Linear Regression) across different PLM embeddings and target parameters.

## 2. Core Workflow & Automation

-   **Experiment Execution:** Experiments are managed and run via the `run_experiments.py` script located at the project root.
-   **Automation:** This script automates the process of training and optionally evaluating models for all combinations of specified:
    -   Model types (`--model_types`, e.g., `fnn`, `linear`, `euclidean`)
    -   Target parameters (`--target_params`, e.g., `fident`, `alntmscore`, `hfsp`)
    -   Embedding files (found automatically as `.h5` files in `data/swissprot/embeddings/`)
-   **Redundancy Check:** Before launching a run, `run_experiments.py` checks if a corresponding output directory (`models/<model_type>_runs/<param_name>/<embedding_name>/`) already contains timestamped results. If so, it skips the combination to avoid redundant computations.
-   **Optional Evaluation:** The `--evaluate_after_train` flag can be passed to `run_experiments.py` to automatically trigger the evaluation script (`evaluate.py`) for each successfully completed run (including the Euclidean baseline setup).
-   **Usage Example:**
    ```bash
    # Run FNN, Linear, & Euclidean models for all params/embeddings using 'processed/training' data, evaluate after
    uv run python run_experiments.py --csv_subdir training --evaluate_after_train

    # Run only FNN for fident using 'processed/training' data, no evaluation
    uv run python run_experiments.py --csv_subdir training --model_types fnn --target_params fident
    ```

## 3. Key Scripts & Components

-   **`run_experiments.py`:** (Project Root) Orchestrates the entire experimental workflow, iterating through combinations, performing redundancy checks, calling `train.py` (for training or baseline setup), and optionally calling `evaluate.py`.
-   **`src/unknown_unknowns/train.py`:** Handles the training of a single model instance *or* sets up the run directory for the Euclidean baseline.
    -   Takes parameters via command-line arguments (model type, data paths, hyperparameters).
    -   **If model type is `fnn` or `linear`:** Selects the appropriate model, uses PyTorch Lightning for training, saves checkpoints and TensorBoard logs.
    -   **If model type is `euclidean`:** Creates the standard run directory structure and saves an `hparams.yaml` file marking the model type. No data loading or calculation occurs in this script for the baseline.
    -   Hyperparameter defaults for trainable models are defined directly within its `argparse` setup.
-   **`src/unknown_unknowns/evaluate.py`:** Evaluates a trained model checkpoint or the Euclidean baseline from a specific run directory.
    -   Takes the run directory (`--run_dir`) as input.
    -   Loads necessary hyperparameters from `hparams.yaml`.
    -   **If model type is `fnn` or `linear`:** Loads the model checkpoint and runs inference on the test set.
    -   **If model type is `euclidean`:** Skips model loading and directly calculates the Euclidean distance between embeddings for the test set pairs using the `run_euclidean_distance` function.
    -   Runs inference on the test set (`test.csv` from the training data directory specified in `hparams.yaml` by default, can be overridden with `--test_csv`).
    -   Calculates and saves metrics (including Spearman, Pearson, R^2) and plots to an `evaluation_results` subdirectory.
-   **`src/unknown_unknowns/models/predictor.py`:** Contains the PyTorch Lightning model definitions (`FNNPredictor`, `LinearRegressionPredictor`).
-   **`src/unknown_unknowns/data/preprocessing.py`:** Contains scriptable logic to preprocess raw CSV data (e.g., applying NaN thresholds). Run via `uv run python src/unknown_unknowns/data/preprocessing.py --input_dir ... --output_dir ...`.
-   **`src/unknown_unknowns/data/datasets.py`:** Contains data loading logic (`create_single_loader`, `H5PyDataset`) for handling HDF5 embedding files and CSV data files.
-   **`src/unknown_unknowns/evaluation/metrics.py`:** Calculates evaluation metrics (MSE, RMSE, MAE, R2, Pearson, Pearson_r2, Spearman). Assumes inputs are NaN-free.
-   **`src/unknown_unknowns/visualization/plot.py`:** Generates plots (e.g., true vs. predicted).
-   **`src/unknown_unknowns/utils/helpers.py`:** Utility functions.

## 4. Data

-   **Raw Data:** Original CSV files are expected in `data/raw/<subdir_name>/` (e.g., `data/raw/training/`).
-   **Processed Data:** Preprocessing scripts save results to `data/processed/<subdir_name>/` (e.g., `data/processed/training/`). `run_experiments.py` is configured to use this directory by default.
-   **Embeddings:** Stored as HDF5 files (`.h5`) in `data/swissprot/embeddings/`.
-   **CSV Files:** Subdirectories selected via `--csv_subdir` must contain `train.csv`, `val.csv`, and `test.csv`.

## 5. Output Structure

-   Runs are saved under `models/`.
-   Structure: `models/<model_type>_runs/<param_name>/<embedding_name>/<timestamp>/`
    -   `<model_type>`: `fnn`, `linear`, or `euclidean`.
-   Inside the `<timestamp>` directory:
    -   `checkpoints/`: Contains the saved model checkpoint (`.ckpt`) for trainable models (empty for Euclidean).
    -   `tensorboard/`: Contains TensorBoard logs (if applicable) and the `hparams.yaml` file.
    -   `evaluation_results/`: Contains evaluation metrics and plots if evaluation is run.

## 6. Key Design Decisions & Rationale

-   **Automation:** Shifted from manual script execution per experiment to an automated `run_experiments.py` script to handle numerous combinations efficiently and reduce errors.
-   **Configuration Management:** Eliminated the central `config.py` file.
    -   `train.py` uses command-line arguments for all necessary parameters and hyperparameters, with defaults set directly in `argparse`. This makes it self-contained.
    -   `evaluate.py` relies on the `hparams.yaml` file saved during training within the run directory, ensuring evaluation uses consistent parameters corresponding to the specific training run.
-   **Parameterization & Flexibility:** `train.py` and `run_experiments.py` are parameterized via CLI arguments, allowing easy selection of models, data subsets, and hyperparameters.
-   **Path Handling:** `train.py` requires absolute paths for embedding and CSV data directories passed via arguments, making it less dependent on relative paths and script location. `run_experiments.py` resolves these absolute paths before calling `train.py`.
-   **Modularity:** Separated concerns into distinct scripts (run, train, evaluate) and modules (models, data, evaluation, visualization).
-   **Model Selection:** Generalized `train.py` to load and train different model architectures based on the `--model_type` argument.
-   **Output Organization:** Adopted a nested output directory structure (`models/<model_type>_runs/...`) to clearly separate results from different models, parameters, and embeddings.
-   **Progress Visibility:** Ensured training progress bars are streamed live by avoiding output capture in `run_experiments.py` during the training subprocess call.
-   **Euclidean Baseline Integration:** Treated as a distinct `model_type`. `train.py` handles setup (directory + hparams), while `evaluate.py` performs the actual distance calculation. Standard metrics are reported:
    -   Correlations (`Pearson`, `Spearman`) are expected to be negative, indicating smaller distance corresponds to higher similarity.
    -   `Pearson_r2` (Pearson correlation squared, 0 to 1) quantifies the strength of this linear association.
    -   `R2` (Coefficient of Determination) is expected to be negative, as raw distance is a poor predictor of the target value's magnitude compared to a simple mean baseline.

## 7. Environment

-   The project uses `uv` for environment and task management. Commands are typically run using `uv run python ...`.

## 8. Potential Future Work

-   Implement a more sophisticated configuration system using YAML files to define experiment sets, potentially replacing the CLI loops in `run_experiments.py`.
-   Add more model types.
-   Add more evaluation metrics or plots.
-   Implement hyperparameter sweeping within `run_experiments.py`.