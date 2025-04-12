# Project Specification: Protein Property Prediction

## 1. Overview

This project aims to train and evaluate machine learning models to predict specific protein properties based on pre-computed embeddings from various Protein Language Models (PLMs). The primary goal is to compare the performance of different model architectures (currently Feed-Forward Neural Network and Linear Regression) across different PLM embeddings and target parameters.

## 2. Core Workflow & Automation

-   **Experiment Execution:** Experiments are managed and run via the `run_experiments.py` script located at the project root.
-   **Automation:** This script automates the process of training and optionally evaluating models for all combinations of specified:
    -   Model types (`--model_types`, e.g., `fnn`, `linear`)
    -   Target parameters (`--target_params`, e.g., `fident`, `alntmscore`, `hfsp`)
    -   Embedding files (found automatically as `.h5` files in `data/swissprot/embeddings/`)
-   **Redundancy Check:** Before launching a training run, `run_experiments.py` checks if a corresponding output directory (`models/<model_type>_runs/<param_name>/<embedding_name>/`) already contains timestamped results. If so, it skips the combination to avoid redundant computations.
-   **Optional Evaluation:** The `--evaluate_after_train` flag can be passed to `run_experiments.py` to automatically trigger the evaluation script (`evaluate.py`) for each successfully trained model, using its generated run directory.
-   **Usage Example:**
    ```bash
    # Run FNN & Linear models for all params/embeddings using 'training' data, evaluate after
    uv run python run_experiments.py --csv_subdir training --evaluate_after_train

    # Run only FNN for fident using 'train_sub' data, no evaluation
    uv run python run_experiments.py --csv_subdir train_sub --model_types fnn --target_params fident
    ```

## 3. Key Scripts & Components

-   **`run_experiments.py`:** (Project Root) Orchestrates the entire experimental workflow, iterating through combinations, performing redundancy checks, calling training, and optionally calling evaluation.
-   **`src/unknown_unknowns/train.py`:** Handles the training of a single model instance.
    -   Takes parameters via command-line arguments (model type, data paths, hyperparameters).
    -   Selects the appropriate model architecture.
    -   Uses PyTorch Lightning for the training loop.
    -   Saves checkpoints and TensorBoard logs to a structured output directory.
    -   Hyperparameter defaults are defined directly within its `argparse` setup.
-   **`src/unknown_unknowns/evaluate.py`:** Evaluates a trained model checkpoint from a specific run directory.
    -   Takes the run directory (`--run_dir`) as input.
    *   Loads necessary hyperparameters (model type, data paths, batch size, etc.) from `hparams.yaml` located within the run directory's `tensorboard` subdirectory.
    -   Loads the appropriate model class based on the hyperparameters.
    -   Loads the best checkpoint.
    -   Runs inference on the test set (`test.csv` from the original training data directory by default, can be overridden with `--test_csv`).
    -   Calculates and saves metrics and plots to an `evaluation_results` subdirectory within the run directory.
-   **`src/unknown_unknowns/models/predictor.py`:** Contains the PyTorch Lightning model definitions:
    -   `FNNPredictor`: A Feed-Forward Neural Network model.
    -   `LinearRegressionPredictor`: A simple linear regression model.
-   **`src/unknown_unknowns/data/datasets.py`:** Contains data loading logic (`create_single_loader`, `get_embedding_size`) for handling the HDF5 embedding files and CSV data files.
-   **`src/unknown_unknowns/evaluation/metrics.py`:** Calculates evaluation metrics (e.g., regression metrics).
-   **`src/unknown_unknowns/visualization/plot.py`:** Generates plots (e.g., true vs. predicted).
-   **`src/unknown_unknowns/utils/helpers.py`:** Utility functions (e.g., `get_device`).

## 4. Data

-   **Embeddings:** Stored as HDF5 files (`.h5`) in `data/swissprot/embeddings/`. Each file represents embeddings from a different PLM. The keys within the HDF5 file are expected to correspond to identifiers used in the CSV files.
-   **Training/Validation/Test Data:** Located in subdirectories within `data/swissprot/` (e.g., `data/swissprot/training/`, `data/swissprot/train_sub/`). The specific subdirectory is selected using the `--csv_subdir` argument in `run_experiments.py`.
    -   Each subdirectory must contain `train.csv` and `val.csv`.
    -   `test.csv` is required for evaluation (used by default by `evaluate.py` unless overridden).
    -   CSV files contain columns for query/target identifiers and the target parameters (e.g., `fident`, `alntmscore`, `hfsp`).

## 5. Output Structure

-   Training runs are saved under `models/`.
-   The structure is nested for clarity: `models/<model_type>_runs/<param_name>/<embedding_name>/<timestamp>/`
    -   `<model_type>`: `fnn` or `linear`.
    -   `<param_name>`: e.g., `fident`.
    -   `<embedding_name>`: Derived from the HDF5 filename, e.g., `prott5`.
    -   `<timestamp>`: `YYYYMMDD_HHMMSS` format.
-   Inside the `<timestamp>` directory:
    -   `checkpoints/`: Contains the saved model checkpoint (`.ckpt`).
    -   `tensorboard/`: Contains TensorBoard logs and the `hparams.yaml` file.
-   If evaluation is run, an `evaluation_results/` subdirectory is created within the `<timestamp>` directory containing metrics (`.txt`) and plots (`.png`).

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

## 7. Environment

-   The project uses `uv` for environment and task management. Commands are typically run using `uv run python ...`.

## 8. Potential Future Work

-   Implement a more sophisticated configuration system using YAML files to define experiment sets, potentially replacing the CLI loops in `run_experiments.py`.
-   Add more model types.
-   Add more evaluation metrics or plots.
-   Implement hyperparameter sweeping within `run_experiments.py`.