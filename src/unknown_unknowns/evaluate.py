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
    metrics_path: Path,
    plot_path: Path,
    force_recompute: bool,
    n_bootstrap: int = 1000,
):
    """Calculate metrics, (re)generate plot, and save raw metrics file if it doesn't exist or recompute is forced for metrics."""

    # Plot is always regenerated if this function is called, unless metrics skip happens.
    # Metrics skip logic:
    if not force_recompute and metrics_path.is_file():
        print(
            f"Metrics file ({metrics_path}) found. Loading metrics and regenerating plot."
        )
        # Load existing metrics for plot title and console output
        metrics = {}
        try:
            with open(metrics_path, "r") as f:
                for line in f:
                    if line.startswith("#") or ":" not in line:
                        continue
                    key, value = line.strip().split(":", 1)
                    try:
                        metrics[key.strip()] = float(value.strip())
                    except ValueError:
                        metrics[key.strip()] = (
                            value.strip()
                        )  # Keep as string if not float (e.g. NaN)
            print(
                f"\nLoaded Metrics for Test Set '{test_set_name}' (from {metrics_path}):"
            )
            for (
                metric,
                value_p,
            ) in metrics.items():  # Use different var name to avoid conflict
                if isinstance(value_p, (float, np.floating)) and np.isnan(value_p):
                    print(f"{metric}: NaN")
                elif isinstance(value_p, float) and value_p < 1e-4:
                    print(f"{metric}: {value_p:.4e}")
                elif isinstance(value_p, float):
                    print(f"{metric}: {value_p:.4f}")
                else:
                    print(f"{metric}: {value_p}")  # Print string as is
        except Exception as e:
            print(
                f"Could not read existing metrics file ({metrics_path}): {e}. Will recompute metrics."
            )
            metrics = calculate_regression_metrics(
                targets, predictions, n_bootstrap=n_bootstrap
            )
            # Save recomputed metrics
            with open(metrics_path, "w") as f:
                f.write(f"# Evaluation metrics for checkpoint: {checkpoint_name}\n")
                f.write(f"# Test Set: {test_set_name}\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")
            print(f"Saved recomputed metrics to: {metrics_path}")

    else:  # Metrics need to be computed (or forced)
        print(f"Generating metrics and plot for test set '{test_set_name}'...")
        metrics = calculate_regression_metrics(
            targets, predictions, n_bootstrap=n_bootstrap
        )
        # Save metrics
        with open(metrics_path, "w") as f:
            f.write(f"# Evaluation metrics for checkpoint: {checkpoint_name}\n")
            f.write(f"# Test Set: {test_set_name}\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
        print(f"Saved metrics to: {metrics_path}")
        # Console print for newly computed metrics
        print(f"\nMetrics for Test Set '{test_set_name}':")
        for metric, value in metrics.items():
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                print(f"{metric}: NaN")
            elif value < 1e-4:
                print(f"{metric}: {value:.4e}")
            else:
                print(f"{metric}: {value:.4f}")

    # Plot is always generated here, using loaded or newly computed metrics
    plot_title = f"Evaluation on '{test_set_name}' ({checkpoint_name})"
    plot_true_vs_predicted(
        targets, predictions, plot_path, metrics=metrics, title=plot_title
    )
    print(f"Saved evaluation plot to: {plot_path}")


def main(args):
    """Main workflow orchestrator for evaluation."""
    try:
        hparams = load_hparams(args.run_dir)
    except Exception as e:
        print(f"Error loading hparams: {e}")
        return

    model_type = hparams["model_type"]
    test_loader, test_csv_path = prepare_evaluation_data(hparams, args.test_csv)
    test_set_name = test_csv_path.stem  # e.g., 'test' from 'test.csv'

    checkpoint_name: str
    if model_type in ["fnn", "linear", "linear_distance"]:
        best_checkpoint_path = find_best_checkpoint(args.run_dir)
        if not best_checkpoint_path:
            print("No checkpoint found, cannot evaluate model.")
            return
        checkpoint_name = best_checkpoint_path.stem
    elif model_type == "euclidean":
        checkpoint_name = "euclidean_baseline"
    else:
        print(f"Unknown model type '{model_type}' in hparams. Cannot evaluate.")
        return

    eval_dir = args.run_dir / "evaluation_results"
    eval_dir.mkdir(exist_ok=True)

    base_filename = f"test_{test_set_name}_{checkpoint_name}"
    preds_targets_path = eval_dir / f"{base_filename}_predictions_targets.npz"
    metrics_path = eval_dir / f"{base_filename}_metrics.txt"
    plot_path = eval_dir / f"{base_filename}_results.png"

    if (
        not args.force_recompute
        and preds_targets_path.is_file()
        and metrics_path.is_file()
        and plot_path.is_file()
    ):
        print(
            f"All evaluation artifacts (predictions, metrics, plot) found for {base_filename}. Skipping."
        )
        if metrics_path.is_file():
            print(f"\nExisting Metrics (from {metrics_path}):")
            try:
                with open(metrics_path, "r") as f:
                    lines = [
                        line.strip()
                        for line in f.readlines()
                        if not line.startswith("#")
                    ]
                    for line in lines:
                        print(line)
            except Exception as e:
                print(f"Could not read existing metrics file: {e}")
        return

    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    predictions_were_loaded = False

    if preds_targets_path.is_file() and not args.force_recompute:
        print(
            f"Loading pre-computed predictions and targets from: {preds_targets_path}"
        )
        try:
            data = np.load(preds_targets_path)
            predictions = data["predictions"]
            targets = data["targets"]
            if predictions is not None and targets is not None:
                print("Successfully loaded pre-computed predictions and targets.")
                predictions_were_loaded = True
            else:
                print(
                    "Warning: Loaded file is missing 'predictions' or 'targets' key. Recomputing."
                )
        except Exception as e:
            print(
                f"Warning: Could not load predictions/targets from {preds_targets_path}: {e}. Recomputing."
            )

    if not predictions_were_loaded:
        print("Predictions/targets not loaded or recompute forced. Computing now...")
        if model_type in ["fnn", "linear", "linear_distance"]:
            best_checkpoint_path_for_compute = find_best_checkpoint(args.run_dir)
            if not best_checkpoint_path_for_compute:
                print("Critical: No checkpoint found during recompute path.")
                return
            model_class_map = {
                "fnn": FNNPredictor,
                "linear": LinearRegressionPredictor,
                "linear_distance": LinearDistancePredictor,
            }
            ModelClass = model_class_map[model_type]
            model = load_model_from_checkpoint(
                best_checkpoint_path_for_compute, ModelClass
            )
            predictions, targets = run_inference(model, test_loader)
        elif model_type == "euclidean":
            predictions, targets = run_euclidean_distance(test_loader)

        if predictions is not None and targets is not None:
            try:
                np.savez_compressed(
                    preds_targets_path, predictions=predictions, targets=targets
                )
                print(
                    f"Saved freshly computed predictions and targets to: {preds_targets_path}"
                )
            except Exception as e:
                print(
                    f"Warning: Could not save freshly computed predictions/targets to {preds_targets_path}: {e}"
                )
        else:
            print(
                "Error: Failed to compute predictions/targets. Evaluation aborted before saving."
            )
            return

    if predictions is not None and targets is not None:
        generate_evaluation_results(
            predictions,
            targets,
            args.run_dir,
            checkpoint_name,
            test_set_name,
            metrics_path=metrics_path,
            plot_path=plot_path,
            force_recompute=args.force_recompute,
            n_bootstrap=args.n_bootstrap,
        )
    else:
        print(
            "Error: Predictions and targets could not be obtained. Evaluation aborted."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model or Euclidean baseline."
    )
    parser.add_argument(
        "--run_dir", type=Path, required=True, help="Path to the model run directory."
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=None,
        help="Optional path to a specific test CSV file to use for evaluation. "
        "If not provided, uses 'test.csv' from the original training data directory.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for calculating SE/CI for correlation metrics. "
        "Set to 1 or 0 to disable bootstrapping.",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force re-computation of predictions even if saved predictions/targets file exists.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
