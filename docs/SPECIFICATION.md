# Project Specification: Protein Property Prediction

## 1. Overview

This project aims to train and evaluate machine learning models to predict specific protein properties based on pre-computed embeddings from various Protein Language Models (PLMs). The primary goal is to compare the performance of different model architectures (currently Feed-Forward Neural Network (`fnn`), Linear Regression (`linear`), and Linear Regression on embedding difference (`linear_distance`)) across different PLM embeddings and target parameters. Baselines using Euclidean distance (`euclidean`) and models trained on randomly generated embeddings are also included for comparison.

## 2. Core Workflow & Automation

-   **Experiment Execution:** Experiments are managed and run via the `run_experiments.py` script located at the project root.
-   **Automation:** This script automates the process of training and optionally evaluating models for all combinations of specified:
    -   Model types (`--model_types`, e.g., `fnn`, `linear`, `euclidean`, `linear_distance`)
    -   Target parameters (`--target_params`, e.g., `fident`, `alntmscore`, `hfsp`)
    -   Embedding files (found automatically as `.h5` files in `data/processed/sprot_embs/` - this includes both PLM and generated random embeddings).
-   **Redundancy Check:** Before launching a run, `run_experiments.py` checks if a corresponding output directory (`models/<train_data_sub_dir>/<model_type>/<param_name>/<embedding_name>/`) already contains timestamped results. If so, it skips the combination to avoid redundant computations. `<train_data_sub_dir>` is the base name of the directory specified by `--csv_dir`.
-   **Optional Evaluation:** The `--evaluate_after_train` flag can be passed to `run_experiments.py` to automatically trigger the evaluation script (`evaluate.py`) for each successfully completed run (including the Euclidean baseline setup).
-   **Usage Example:**
    ```bash
    # Generate random embeddings (if not already present)
    uv run python scripts/generate_random_embeddings.py

    # Run FNN, Linear, LinearDistance & Euclidean models for all params/embeddings using 'processed/sprot_train' data, evaluate after
    # This will include runs using the 'random_*.h5' files as input embeddings for FNN/Linear/LinearDistance
    uv run python run_experiments.py --csv_dir data/processed/sprot_train --evaluate_after_train --model_types fnn linear linear_distance euclidean

    # Run only LinearDistance for fident using only the prott5 embedding
    # (Requires manually adjusting run_experiments.py or providing specific embedding file path if needed)
    # uv run python run_experiments.py --csv_dir data/processed/sprot_train --model_types linear_distance --target_params fident --embedding_files data/processed/sprot_embs/prott5.h5
    ```

## 3. Key Scripts & Components

-   **`run_experiments.py`:** (Project Root) Orchestrates the entire experimental workflow. Iterates through specified model types (`fnn`, `linear`, `linear_distance`, `euclidean`), target parameters, and all `.h5` embedding files found in the embedding directory. Extracts the base name of the `--csv_dir` directory (`<train_data_sub_dir>`). Performs redundancy checks based on the expected output path (`models/<train_data_sub_dir>/<model_type>/<param_name>/<embedding_name>/`), constructs this base path, and calls `train.py`, passing the base path via the `--output_base_dir` argument. Optionally calls `evaluate.py`.
-   **`scripts/generate_random_embeddings.py`:** (Scripts Directory) Generates HDF5 files (`random_<dim>.h5`) containing random embeddings (standard normal distribution) for protein IDs found in a template HDF5 file (e.g., `prott5.h5`). These files are saved to the standard embedding directory (`data/processed/sprot_embs/`) and are intended to be used as input embeddings for the standard training workflow (`fnn`, `linear`, `linear_distance`) to establish a baseline.
-   **`scripts/evaluate_multiple_runs.py`:** (Scripts Directory) Provides a flexible way to re-evaluate multiple model runs. It takes an `--input_path` which can be a high-level directory (e.g., `models/sprot_train`), a mid-level directory (e.g., `models/sprot_train/fnn/fident`), or a specific timestamped run directory. It recursively searches for valid run directories (identified by the presence of `tensorboard/hparams.yaml`) and then calls `src/unknown_unknowns/evaluate.py --run_dir <path_to_run_dir>` for each. This is useful for updating plots or re-calculating metrics for multiple existing runs with new evaluation code or plotting styles. Includes a `--dry_run` option.
-   **`src/unknown_unknowns/train.py`:** Handles the training of a single model instance (`fnn`, `linear`, `linear_distance`) *or* sets up the run directory for the non-trainable Euclidean baseline within a specified output directory.
    -   Takes parameters via command-line arguments, including the base output directory (`--output_base_dir`) provided by `run_experiments.py`.
    -   Creates a timestamped subdirectory within the `--output_base_dir` for the specific run.
    -   **If model type is `fnn`, `linear`, or `linear_distance`:** Selects the appropriate model class (`FNNPredictor`, `LinearRegressionPredictor`, or `LinearDistancePredictor`), uses PyTorch Lightning for training using the specified input embedding file (which could be a PLM embedding or a generated random embedding), saves checkpoints and TensorBoard logs within the timestamped run directory.
    -   **If model type is `euclidean`:** Creates the necessary structure (`tensorboard/hparams.yaml`) within the timestamped run directory. No training occurs; evaluation is handled directly by `evaluate.py`.
    -   Hyperparameter defaults for trainable models are defined directly within its `argparse` setup.
-   **`src/unknown_unknowns/evaluate.py`:** Evaluates a trained model checkpoint (`fnn`, `linear`, `linear_distance`) or the Euclidean baseline from a specific run directory.
    -   Takes the run directory (`--run_dir`) and optionally the number of bootstrap samples (`--n_bootstrap`, default 1000) as input.
    -   Loads necessary hyperparameters from `hparams.yaml`.
    -   **If model type is `fnn`, `linear`, or `linear_distance`:** Loads the correct model checkpoint using the appropriate class and runs inference on the test set.
    -   **If model type is `euclidean`:** Skips model loading and directly calculates the Euclidean distance.
    -   Runs inference/calculation on the test set (`test.csv` from the training data directory specified in `hparams.yaml` by default, can be overridden with `--test_csv`).
    -   Calls `evaluation.metrics.calculate_regression_metrics` to compute standard regression metrics.
    -   If `n_bootstrap > 1`, metrics calculation includes bootstrapping to estimate Standard Error (SE) and Confidence Intervals (CI) for Pearson R^2 and Spearman Rho.
    -   Saves raw numerical metrics (including SE/CI if calculated) to a `.txt` file in an `evaluation_results` subdirectory.
    -   Generates and saves plots (e.g., true vs. predicted) to the same subdirectory. The true vs. predicted plot is now a hexagonal binning plot to show point density, includes a regression line with 95% CI, and features a customized grid. The ideal y=x line has been removed, and the plot aspect ratio is no longer forced to be equal.
-   **`src/unknown_unknowns/models/predictor.py`:** Contains the PyTorch Lightning model definitions. Includes a `BasePredictor` class with common logic (training/validation/prediction steps, optimizer configuration) and specific model classes inheriting from it: `FNNPredictor`, `LinearRegressionPredictor` (operates on concatenated embeddings), and `LinearDistancePredictor` (operates on squared embedding difference).
-   **`src/unknown_unknowns/data/preprocessing.py`:** Contains scriptable logic to preprocess raw CSV data (e.g., applying NaN thresholds). Run via `uv run python src/unknown_unknowns/data/preprocessing.py --input_dir ... --output_dir ...`.
-   **`src/unknown_unknowns/data/datasets.py`:** Contains data loading logic (`create_single_loader`, `H5PyDataset`) for handling HDF5 embedding files and CSV data files.
-   **`src/unknown_unknowns/evaluation/metrics.py`:** Calculates standard regression metrics (MSE, RMSE, MAE, R2, Pearson, Spearman, and associated p-values). Includes optional bootstrapping (`_bootstrap_stat` helper) for Pearson R^2 and Spearman correlation coefficients to estimate standard error and confidence intervals. Returns raw numerical results.
-   **`src/unknown_unknowns/visualization/plot.py`:** Generates plots (e.g., true vs. predicted).
-   **`src/unknown_unknowns/utils/helpers.py`:** Utility functions.

## 4. Data

-   **Raw Data:** Original CSV files are expected in `data/raw/<subdir_name>/` (e.g., `data/raw/training/`).
-   **Processed Data:** Preprocessing scripts save results to `data/processed/<subdir_name>/` (e.g., `data/processed/sprot_train/`). `run_experiments.py` uses the directory specified by `--csv_dir`.
-   **Embeddings:** Stored as HDF5 files (`.h5`) in `data/processed/sprot_embs/`. This directory contains both PLM embeddings (e.g., `prott5.h5`) and generated random embeddings (e.g., `random_512.h5`). `run_experiments.py` automatically discovers and uses all `.h5` files found here.
-   **CSV Files:** The directory specified via `--csv_dir` must contain `train.csv`, `val.csv`, and `test.csv`.

## 5. Output Structure

-   Runs are saved under `models/`.
-   Structure: `models/<train_data_sub_dir>/<model_type>/<param_name>/<embedding_name>/<timestamp>/`
    -   `<train_data_sub_dir>`: The base name of the directory provided via `--csv_dir` (e.g., `sprot_train`).
    -   `<model_type>`: `fnn`, `linear`, `euclidean`, or `linear_distance`.
    -   `<embedding_name>`: The stem of the HDF5 file used (e.g., `prott5`, `ankh_base`, `random_512`, `random_1024`).
-   Inside the `<timestamp>` directory:
    -   `checkpoints/`: Contains the saved model checkpoint (`.ckpt`) for trainable models. Empty for Euclidean.
    -   `tensorboard/`: Contains TensorBoard logs (if applicable) and the `hparams.yaml` file.
    -   `evaluation_results/`: Contains plots and the raw evaluation metrics (`_metrics.txt`), potentially including SE and CI values for correlation coefficients if bootstrapping was enabled during evaluation.

## 6. Key Design Decisions & Rationale

-   **Automation:** Shifted from manual script execution per experiment to an automated `run_experiments.py` script to handle numerous combinations efficiently and reduce errors.
-   **Configuration Management:** Eliminated the central `config.py` file. Configuration is handled via CLI arguments and `hparams.yaml` saved in run directories.
-   **Parameterization & Flexibility:** `run_experiments.py` and `train.py` are parameterized via CLI arguments.
-   **Path Handling:** `train.py` uses absolute paths passed via arguments. `run_experiments.py` resolves paths before calling `train.py`.
-   **Modularity:** Separated concerns into distinct scripts and modules.
-   **Model Selection:** `train.py` handles different trainable model architectures (`fnn`, `linear`, `linear_distance`) and sets up the Euclidean baseline.
-   **Predictor Refactoring:** Model classes in `predictor.py` inherit from a `BasePredictor` to reduce code duplication.
-   **Linear Distance Model:** Added `LinearDistancePredictor` to explore if predicting directly from the squared difference of embeddings is effective.
-   **Random Embedding Baseline:** Implemented by generating random embedding files (`scripts/generate_random_embeddings.py`) and using them as standard inputs to the existing trainable model workflows. This allows direct comparison of model performance on meaningful vs. random inputs using the same architecture.
-   **Output Organization:** Nested output directory structure clearly separates results.
-   **Progress Visibility:** Training progress bars are streamed live.
-   **Euclidean Baseline Integration:** Treated as a distinct non-trainable `model_type`.
-   **Metrics Calculation:** Separated metric calculation logic into `evaluation/metrics.py`. Includes robust calculation of standard regression metrics and optional bootstrapping via `_bootstrap_stat` helper to estimate standard errors and confidence intervals for Pearson R^2 and Spearman Rho, providing insight into the stability of these correlation measures.
-   **Output Format:** Evaluation results (metrics) are saved as raw numerical values in the `.txt` file for easier machine parsing and downstream analysis, while console output provides basic formatting.
-   **Plotting Enhancements:** The primary true vs. predicted plot has been updated to use hexagonal binning for better visualization of point density, especially with large datasets. It includes a clear regression line with its 95% confidence interval, and the grid is styled for better readability (transparent and behind plot elements). The y=x ideal line has been removed to reduce clutter, and the plot aspect ratio is no longer fixed, allowing axes to scale to their data.

## 7. Environment

-   The project uses `uv` for environment and task management. Commands are typically run using `uv run python ...`.

## 8. Potential Future Work

-   Implement a more sophisticated configuration system (e.g., YAML) for experiment sets.
-   Add more model types.
-   Add more evaluation metrics or plots.
-   Implement hyperparameter sweeping.
-   Add functionality to `run_experiments.py` to select specific embedding files via CLI instead of using all found `.h5` files.

## 9. Results Visualization

-   **Script:** `scripts/visualize_summary_results.py`
-   **Purpose:** Parses the evaluation metric files (`_metrics.txt`) generated by `evaluate.py` across different runs and creates summary plots visualizing performance.
-   **Inputs:**
    -   `--results_dir`: (Required) The base directory containing the experiment results (e.g., `models/train_sub`, `models/sprot_train`). It expects the standard output structure: `<results_dir>/<model_type>/<param_name>/<embedding_name>/<timestamp>/evaluation_results/*_metrics.txt`.
    -   `--output`: (Optional) The directory where the output plots and CSV file will be saved. Defaults to `./plots/`.
    -   `--ignore-random`: (Optional flag) If set, results associated with the 'Random' embedding family (specifically `random_1024`) will be excluded from the plots.
-   **Outputs:**
    -   **CSV File:** `parsed_metrics_all.csv` saved in the `--output` directory. This file contains the aggregated data parsed from all found metric files, including PLM size, embedding family, model type, parameter, and all extracted metrics (Pearson R2, MAE, Spearman, R2, and their SEs if applicable).
    -   **Plot Files (PNG):** Separate plots are generated for key metrics and saved in the `--output` directory. Filenames are based on the metric (e.g., `pearson_r2.png`, `spearman_rho.png`, `mae.png`, `r2.png`).
-   **Plot Characteristics:**
    -   **Type:** Faceted scatter plot (`seaborn.relplot`).
    -   **Facets:** Columns represent the different target parameters (e.g., `fident`, `alntmscore`, `hfsp`).
    -   **Y-axis:** The specific metric being plotted (e.g., Pearson R², MAE).
    -   **X-axis:** Categorical, representing the different PLM embeddings. The categories are ordered based on the PLM parameter size (from the `PLM_SIZES` dictionary in the script).
    -   **X-axis Labels:** Tick labels display the human-readable PLM parameter size (e.g., "8M", "1.5B") corresponding to each embedding category.
    -   **Color:** Points are colored based on the specific PLM embedding (`EMBEDDING_COLOR_MAP`).
    -   **Shape:** Point markers represent the model type (`MODEL_MARKER_MAP`, e.g., `fnn`, `linear`).
    -   **Error Bars:** Added for metrics with corresponding Standard Error columns parsed (Pearson R², Spearman Rho).
    -   **Connecting Lines:** Lines connect points within specified PLM families (`FAMILIES_TO_CONNECT`) for the same model type, following the PLM size order on the categorical axis.
    -   **Legend:** Two separate legends are placed outside the plot area: one for embedding colors (square markers) and one for model type shapes.
-   **Usage Example:**
    ```bash
    # Generate plots and CSV for results in models/train_sub, save to out/summary_plot/
    uv run python scripts/visualize_summary_results.py --results_dir models/sprot_train --output out/summary_plot

    # Generate plots ignoring the random baseline
    uv run python scripts/visualize_summary_results.py --results_dir models/sprot_train --output out/summary_plot_no_random --ignore-random
    ```
-   **Dependencies:** Requires `pandas`, `seaborn`, and `matplotlib` to be installed in the environment.