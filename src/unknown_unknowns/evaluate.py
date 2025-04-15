import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import pytorch_lightning as pl

# Project specific imports
from data.datasets import create_single_loader
from evaluation.metrics import calculate_regression_metrics
from models.predictor import (
    FNNPredictor,
    LinearRegressionPredictor,
    LinearDistancePredictor,
)
from visualization.plot import plot_true_vs_predicted
from utils.helpers import get_device


def load_model_from_checkpoint(
    checkpoint_path: Path, model_class: type
) -> pl.LightningModule:
    """Load the LightningModule model from a checkpoint file using the correct class."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(
        f"Loading model from checkpoint: {checkpoint_path} using {model_class.__name__}"
    )
    # Load using the specific class determined from hparams
    model = model_class.load_from_checkpoint(str(checkpoint_path))
    model.eval()  # Set to evaluation mode
    return model


def find_best_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the single checkpoint file saved by ModelCheckpoint(save_top_k=1)."""
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return None

    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))

    if not ckpt_files:
        print(f"Error: No checkpoint files (*.ckpt) found in {checkpoint_dir}")
        return None

    if len(ckpt_files) > 1:
        print(
            f"Warning: Found {len(ckpt_files)} checkpoint files in {checkpoint_dir}. "
            f"Expected only one due to save_top_k=1. Using the first one: {ckpt_files[0]}"
        )

    print(f"Found checkpoint file: {ckpt_files[0]}")
    return ckpt_files[0]


def load_hparams(run_dir: Path) -> Dict[str, Any]:
    """Load hyperparameters from hparams.yaml in the run directory's tensorboard folder."""
    hparams_file = run_dir / "tensorboard" / "hparams.yaml"

    if not hparams_file.is_file():
        raise FileNotFoundError(
            f"hparams.yaml not found at {hparams_file}. Cannot proceed."
        )

    print(f"Loading hyperparameters from: {hparams_file}")
    with open(hparams_file, "r") as f:
        hparams = yaml.safe_load(f)

    # Basic validation for required keys
    required_keys = [
        "model_type",
        "param_name",
        "embedding_file",
        "csv_dir",
        "batch_size",
    ]
    missing_keys = [key for key in required_keys if key not in hparams]
    if missing_keys:
        raise KeyError(f"Missing required keys in hparams.yaml: {missing_keys}")

    # Convert path strings to Path objects where necessary
    try:
        hparams["embedding_file"] = Path(hparams["embedding_file"])
        hparams["csv_dir"] = Path(hparams["csv_dir"])
        # Convert batch_size to int if it was saved as string somehow
        hparams["batch_size"] = int(hparams["batch_size"])
    except Exception as e:
        raise ValueError(f"Error converting hparams values: {e}") from e

    print("Hyperparameters loaded successfully.")
    return hparams


def prepare_evaluation_data(
    hparams: Dict[str, Any],  # Use loaded hparams
    test_csv_override: Optional[Path] = None,
) -> Tuple[DataLoader, Path]:
    """Prepare the DataLoader for the test set using parameters from hparams."""

    embeddings_file = hparams["embedding_file"]
    param_name = hparams["param_name"]
    batch_size = hparams["batch_size"]
    # The original CSV dir used for training
    original_csv_dir = hparams["csv_dir"]

    if test_csv_override:
        test_csv_path = test_csv_override.resolve()
        print(f"Using overridden test CSV: {test_csv_path}")
    else:
        # Default to using the test.csv from the original training csv_dir
        test_csv_path = (original_csv_dir / "test.csv").resolve()
        print(
            f"Using default test CSV from original training directory: {test_csv_path}"
        )

    if not embeddings_file.is_file():
        raise FileNotFoundError(
            f"Embeddings file not found (from hparams): {embeddings_file}"
        )
    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV file not found: {test_csv_path}")

    test_loader = create_single_loader(
        csv_file=str(test_csv_path),
        hdf_file=str(embeddings_file),
        param_name=param_name,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Prepared test data loader with {len(test_loader)} batches.")
    return test_loader, test_csv_path


def run_inference(
    model: pl.LightningModule, test_loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on the test set."""
    print("Running inference on test data...")
    device = get_device()
    model.to(device)

    predictions_list = []
    targets_list = []
    with torch.no_grad():
        for batch in test_loader:
            # Assuming batch structure: (query_emb, target_emb, target_val)
            # Adjust if your dataset yields different batch structures
            if len(batch) != 3:
                raise ValueError(
                    f"Unexpected batch structure: expected 3 elements, got {len(batch)}"
                )
            query_emb, target_emb, target_val = batch

            query_emb = query_emb.to(device)
            target_emb = target_emb.to(device)

            pred = model(query_emb, target_emb)
            predictions_list.append(pred.cpu())
            targets_list.append(target_val.cpu())

    predictions = torch.cat(predictions_list).numpy().flatten()
    targets = torch.cat(targets_list).numpy().flatten()
    print("Inference complete.")
    return predictions, targets


# --- NEW FUNCTION for Euclidean Baseline ---
def run_euclidean_distance(test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Euclidean distance for batches in the test loader."""
    print("Calculating Euclidean distances for baseline...")
    all_distances = []
    all_targets = []
    with torch.no_grad():  # Still good practice even if not using model
        for batch in test_loader:
            # Assuming batch structure: (query_emb, target_emb, target_val)
            # DataLoader currently returns numpy arrays
            if len(batch) != 3:
                raise ValueError(
                    f"Unexpected batch structure: expected 3 elements, got {len(batch)}"
                )
            query_emb_np, target_emb_np, target_val_np = batch

            # Calculate Euclidean distance using numpy
            distances = np.linalg.norm(query_emb_np - target_emb_np, axis=1)

            all_distances.append(distances)
            all_targets.append(target_val_np)

    predictions = np.concatenate(all_distances)
    targets = np.concatenate(all_targets)
    print("Euclidean distance calculation complete.")
    return predictions, targets


# -----------------------------------------


def generate_evaluation_results(
    predictions: np.ndarray,
    targets: np.ndarray,
    run_dir: Path,
    checkpoint_name: str,
    test_set_name: str,
    n_bootstrap: int = 1000,
):
    """Calculate metrics, save plot, and save raw metrics file."""
    print(f"Calculating metrics for test set '{test_set_name}'...")
    eval_dir = run_dir / "evaluation_results"
    eval_dir.mkdir(exist_ok=True)

    # Calculate raw metrics
    metrics = calculate_regression_metrics(
        targets, predictions, n_bootstrap=n_bootstrap
    )

    base_filename = f"test_{test_set_name}_{checkpoint_name}"
    results_png = eval_dir / f"{base_filename}_results.png"
    metrics_file = eval_dir / f"{base_filename}_metrics.txt"

    plot_title = f"Evaluation on '{test_set_name}' ({checkpoint_name})"
    plot_true_vs_predicted(
        targets, predictions, results_png, metrics=metrics, title=plot_title
    )
    print(f"Saved evaluation plot to: {results_png}")

    print(f"\nMetrics for Test Set '{test_set_name}':")
    for metric, value in metrics.items():
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            print(f"{metric}: NaN")
        elif value < 1e-4:
            print(f"{metric}: {value:.4e}")
        else:
            print(f"{metric}: {value:.4f}")

    with open(metrics_file, "w") as f:
        f.write(f"# Evaluation metrics for checkpoint: {checkpoint_name}\n")
        f.write(f"# Test Set: {test_set_name}\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    print(f"Saved metrics to: {metrics_file}")


def main(args):
    """Main workflow orchestrator for evaluation."""
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Specified run directory not found: {run_dir}")

    try:
        hparams = load_hparams(run_dir)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error loading hyperparameters: {e}")
        return

    model_type = hparams.get("model_type")
    predictions = None
    targets = None
    eval_identifier = ""

    try:
        # Load test data using hparams
        test_loader, test_csv_path = prepare_evaluation_data(hparams, args.test_csv)
        test_set_name = test_csv_path.stem
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error preparing evaluation data: {e}")
        return

    # --- Handle Euclidean Baseline ---
    if model_type == "euclidean":
        print("\nEvaluating Euclidean Distance Baseline...")
        try:
            predictions, targets = run_euclidean_distance(test_loader)
            eval_identifier = (
                "euclidean_baseline"  # Use as placeholder for checkpoint name
            )
        except Exception as e:
            print(f"Error during Euclidean distance calculation: {e}")
            return

    # --- Handle Trained Models (FNN, Linear, LinearDistance) ---
    else:
        print(f"\nEvaluating Trained Model (type: {model_type})...")
        best_checkpoint_path = find_best_checkpoint(run_dir)
        if not best_checkpoint_path:
            print("Evaluation aborted: Could not find a suitable checkpoint file.")
            return
        eval_identifier = best_checkpoint_path.stem  # Use checkpoint stem

        # Determine model class
        model_class = None  # Initialize
        if model_type == "fnn":
            model_class = FNNPredictor
        elif model_type == "linear":
            model_class = LinearRegressionPredictor
        elif model_type == "linear_distance":  # Add case for the new model
            model_class = LinearDistancePredictor
        # Handle potential old naming conventions if needed
        elif "FNNPredictor" in str(model_type):
            print(
                "Warning: Found old model type 'FNNPredictor' in hparams, using FNNPredictor class."
            )
            model_class = FNNPredictor
        elif "LinearRegressionPredictor" in str(model_type):
            print(
                "Warning: Found old model type 'LinearRegressionPredictor' in hparams, using LinearRegressionPredictor class."
            )
            model_class = LinearRegressionPredictor

        # Check if a model class was successfully determined
        if model_class is None:
            print(
                f"Error: Unsupported or unknown model_type '{model_type}' for loading checkpoint."
            )
            return

        # Load model and run inference
        try:
            model = load_model_from_checkpoint(best_checkpoint_path, model_class)
        except Exception as e:
            print(f"Error loading model from checkpoint {best_checkpoint_path}: {e}")
            return
        try:
            predictions, targets = run_inference(model, test_loader)
        except Exception as e:
            print(f"Error during model inference: {e}")
            return

    # --- Generate Results (Common for both paths if successful) ---
    if predictions is not None and targets is not None:
        try:
            generate_evaluation_results(
                predictions,
                targets,
                run_dir,
                eval_identifier,
                test_set_name,
                n_bootstrap=args.n_bootstrap,
            )
            print(
                f"\nEvaluation complete. Results saved in: {run_dir / 'evaluation_results'}"
            )
        except Exception as e:
            print(f"Error generating evaluation results: {e}")
    else:
        print("\nEvaluation could not be completed (predictions or targets missing).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model checkpoint or baseline from a run directory."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to the run directory containing checkpoints/hparams.yaml.",
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=None,
        help="Optional: Path to specific test CSV. Uses test.csv from hparams if not set.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for Pearson R^2 SE/CI. Set 0 to disable.",
    )

    args = parser.parse_args()
    main(args)
