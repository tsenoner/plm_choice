import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import pytorch_lightning as pl
from tqdm import tqdm

# Project specific imports
from src.shared.datasets import create_single_loader
from src.shared.experiment_manager import ExperimentManager
from src.evaluation.metrics import calculate_regression_metrics
from src.training.models import (
    FNNPredictor,
    LinearRegressionPredictor,
    LinearDistancePredictor,
)
from src.visualization.plot_utils import plot_true_vs_predicted
from src.shared.helpers import get_device


# --- Computation and Caching Helpers ---
def _compute_and_save_predictions_targets(
    model_type: str, experiment_dir: Path, test_loader: DataLoader, save_path: Path
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Computes predictions/targets via inference or baseline and saves them."""
    print("Computing predictions and targets...")
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None

    if model_type in ["fnn", "linear", "linear_distance"]:
        # Find best checkpoint directly
        checkpoints_dir = experiment_dir / "checkpoints"
        if not checkpoints_dir.exists():
            print("Critical: No checkpoints directory found.")
            return None

        # Look for best checkpoint (guaranteed by save_top_k=1)
        best_ckpt_files = list(checkpoints_dir.glob("best-*.ckpt"))
        if not best_ckpt_files:
            print("Critical: No best checkpoint found.")
            return None

        best_checkpoint_path = best_ckpt_files[0]
        model_class_map = {
            "fnn": FNNPredictor,
            "linear": LinearRegressionPredictor,
            "linear_distance": LinearDistancePredictor,
        }
        ModelClass = model_class_map.get(model_type)
        if not ModelClass:
            print(f"Error: Unknown model type '{model_type}' for loading.")
            return None
        try:
            model = load_model_from_checkpoint(best_checkpoint_path, ModelClass)
            predictions, targets = run_inference(model, test_loader)
        except Exception as e:
            print(f"Error during model loading or inference: {e}")
            return None
    elif model_type == "euclidean":
        try:
            predictions, targets = run_euclidean_distance(test_loader)
        except Exception as e:
            print(f"Error during Euclidean distance calculation: {e}")
            return None
    else:
        print(
            f"Error: Cannot compute predictions for unknown model type '{model_type}'."
        )
        return None

    if predictions is not None and targets is not None:
        try:
            # Ensure parent directory exists before saving
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_path, predictions=predictions, targets=targets)
            print(f"Saved freshly computed predictions and targets to: {save_path}")
            return predictions, targets
        except Exception as e:
            print(
                f"Warning: Could not save freshly computed predictions/targets to {save_path}: {e}"
            )
            return predictions, targets  # Still return computed values
    else:
        print("Error: Failed to compute predictions/targets.")
        return None


def _get_predictions_targets(
    preds_targets_path: Path,
    force_recompute: bool,
    model_type: str,
    experiment_dir: Path,
    hparams: Dict[str, Any],  # Needed for DataLoader if recomputing
    test_data_path: Path,  # Needed for DataLoader if recomputing
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Gets predictions and targets, using cache or computing/saving as needed."""
    if not force_recompute and preds_targets_path.is_file():
        print(f"Attempting to load predictions and targets from: {preds_targets_path}")
        try:
            data = np.load(preds_targets_path)
            predictions = data.get("predictions")
            targets = data.get("targets")
            if predictions is not None and targets is not None:
                print("Successfully loaded pre-computed predictions and targets.")
                return predictions, targets
            else:
                print(
                    "Warning: Cached file missing 'predictions' or 'targets' key. Recomputing."
                )
        except Exception as e:
            print(
                f"Warning: Could not load predictions/targets from {preds_targets_path}: {e}. Recomputing."
            )

    # If cache doesn't exist, is invalid, or force_recompute is True
    # Prepare DataLoader ONLY if computation is required
    try:
        test_loader = _prepare_dataloader(hparams, test_data_path)
    except Exception as e:
        print(f"Error preparing DataLoader for computation: {e}")
        return None

    return _compute_and_save_predictions_targets(
        model_type, experiment_dir, test_loader, preds_targets_path
    )


def _compute_and_save_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int,
    save_path: Path,
    checkpoint_name: str,
    test_set_name: str,
) -> Dict[str, Any]:
    """Computes metrics and saves them to a file."""
    print("Calculating metrics...")
    metrics = calculate_regression_metrics(
        targets, predictions, n_bootstrap=n_bootstrap
    )

    try:
        # Ensure parent directory exists before saving
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(f"# Evaluation metrics for checkpoint: {checkpoint_name}\n")
            f.write(f"# Test Set: {test_set_name}\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
        print(f"Saved metrics to: {save_path}")
    except Exception as e:
        print(f"Warning: Could not save metrics to {save_path}: {e}")

    # Console print for newly computed metrics
    print(f"\nMetrics for Test Set '{test_set_name}':")
    for metric, value in metrics.items():
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            print(f"{metric}: NaN")
        elif isinstance(value, float) and value < 1e-4:
            print(f"{metric}: {value:.4e}")
        elif isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return metrics


def _get_metrics(
    metrics_path: Path,
    force_recompute: bool,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int,
    checkpoint_name: str,
    test_set_name: str,
) -> Optional[Dict[str, Any]]:
    """Gets metrics, using cache or computing/saving as needed."""
    if not force_recompute and metrics_path.is_file():
        print(f"Attempting to load metrics from: {metrics_path}")
        metrics = {}
        try:
            with open(metrics_path, "r") as f:
                for line in f:
                    if line.startswith("#") or ":" not in line:
                        continue
                    key, value = line.strip().split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        metrics[key] = float("nan") if value.lower() == "nan" else value
            if metrics:  # Basic check if metrics were loaded
                print("Successfully loaded metrics from file.")
                # Print loaded metrics to console
                print(f"\nCached Metrics for Test Set '{test_set_name}':")
                for metric, val in metrics.items():
                    if isinstance(val, (float, np.floating)) and np.isnan(val):
                        print(f"{metric}: NaN")
                    elif isinstance(val, float) and val < 1e-4:
                        print(f"{metric}: {val:.4e}")
                    elif isinstance(val, float):
                        print(f"{metric}: {val:.4f}")
                    else:
                        print(f"{metric}: {val}")
                return metrics
            else:
                print("Warning: Metrics file seemed empty or invalid. Recomputing.")
        except Exception as e:
            print(
                f"Warning: Could not load metrics from {metrics_path}: {e}. Recomputing."
            )

    # If cache doesn't exist, is invalid, or force_recompute is True
    return _compute_and_save_metrics(
        predictions, targets, n_bootstrap, metrics_path, checkpoint_name, test_set_name
    )


# --- Data Loading and Model Helpers ---


def _prepare_dataloader(hparams: Dict[str, Any], test_data_path: Path) -> DataLoader:
    """Prepares the DataLoader using resolved paths and hparams."""
    # Simplified: assumes embeddings_file exists (or fails here)
    embeddings_file = hparams["embedding_file"]
    if not embeddings_file.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    test_loader = create_single_loader(
        parquet_file=str(test_data_path),
        hdf_file=str(embeddings_file),
        param_name=hparams["param_name"],
        batch_size=hparams["batch_size"],
        shuffle=False,
    )
    print(f"Prepared test data loader with {len(test_loader)} batches.")
    return test_loader


def load_model_from_checkpoint(
    checkpoint_path: Path, model_class: type
) -> pl.LightningModule:
    """Load the LightningModule model from a checkpoint file."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(
        f"Loading model from checkpoint: {checkpoint_path} using {model_class.__name__}"
    )
    model = model_class.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    return model


def load_hparams_from_wandb(experiment_dir: Path) -> Dict[str, Any]:
    """Load hyperparameters from wandb's config.yaml file."""
    wandb_dir = experiment_dir / "wandb"

    if not wandb_dir.exists():
        raise FileNotFoundError(f"Wandb directory not found: {wandb_dir}")

    # Look for config.yaml in wandb run directories
    config_files = []

    # Check latest-run first (if it exists)
    latest_run_dir = wandb_dir / "latest-run"
    if latest_run_dir.exists():
        config_file = latest_run_dir / "files" / "config.yaml"
        if config_file.exists():
            config_files.append(config_file)

    # Check all run directories
    for run_subdir in wandb_dir.iterdir():
        if run_subdir.is_dir() and run_subdir.name.startswith("run-"):
            config_file = run_subdir / "files" / "config.yaml"
            if config_file.exists():
                config_files.append(config_file)

    if not config_files:
        raise FileNotFoundError(f"No wandb config.yaml files found in {wandb_dir}")

    # Use the most recent config file
    config_file = max(config_files, key=lambda f: f.stat().st_mtime)

    print(f"Loading hyperparameters from wandb config: {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Convert wandb config format to expected format
    hparams = {}
    for key, value_dict in config.items():
        if isinstance(value_dict, dict) and "value" in value_dict:
            hparams[key] = value_dict["value"]
        else:
            hparams[key] = value_dict

    # Convert paths back to Path objects for compatibility
    if "embedding_file" in hparams:
        hparams["embedding_file"] = Path(hparams["embedding_file"])
    if "data_dir" in hparams:
        hparams["data_dir"] = Path(hparams["data_dir"])
    if "batch_size" in hparams:
        hparams["batch_size"] = int(hparams["batch_size"])

    return hparams


def load_hparams_from_local(experiment_dir: Path) -> Dict[str, Any]:
    """Fallback: Load hyperparameters from local hparams.yaml file."""
    hparams_file = experiment_dir / "hparams.yaml"
    if not hparams_file.is_file():
        raise FileNotFoundError(f"hparams.yaml not found at {hparams_file}")
    print(f"Loading hyperparameters from local file: {hparams_file}")
    with open(hparams_file, "r") as f:
        hparams = yaml.safe_load(f)

    required = ["model_type", "param_name", "embedding_file", "data_dir", "batch_size"]
    if missing := [k for k in required if k not in hparams]:
        raise KeyError(f"Missing keys in hparams.yaml: {missing}")

    try:
        hparams["embedding_file"] = Path(hparams["embedding_file"])
        hparams["data_dir"] = Path(hparams["data_dir"])
        hparams["batch_size"] = int(hparams["batch_size"])
    except Exception as e:
        raise ValueError(f"Error converting hparams values: {e}") from e
    return hparams


def load_hparams(experiment_dir: Path) -> Dict[str, Any]:
    """Load hyperparameters from wandb config files."""
    try:
        return load_hparams_from_wandb(experiment_dir)
    except FileNotFoundError as e:
        # For euclidean baseline or other cases where wandb config might not exist
        print(f"Could not load from wandb config: {e}")
        print("Trying local hparams.yaml as fallback...")
        return load_hparams_from_local(experiment_dir)


def run_inference(
    model: pl.LightningModule, test_loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on the test set with parallel processing."""
    print("Running inference...")
    device = get_device()
    model.to(device)
    model.eval()  # Ensure model is in eval mode

    preds, tgts = [], []

    # Use torch.no_grad() and optimized inference
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Model inference", unit="batch"):
            if len(batch) != 3:
                raise ValueError("Unexpected batch structure")
            q_emb, t_emb, val = batch

            # Move to device efficiently
            q_emb = q_emb.to(device, non_blocking=True)
            t_emb = t_emb.to(device, non_blocking=True)

            # Forward pass
            pred = model(q_emb, t_emb)

            # Move back to CPU and store
            preds.append(pred.cpu())
            tgts.append(val.cpu())

            # Clear cache periodically for large datasets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Inference complete.")
    return torch.cat(preds).numpy().flatten(), torch.cat(tgts).numpy().flatten()


def run_euclidean_distance(test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Euclidean distance baseline."""
    print("Calculating Euclidean distances...")
    dists, tgts = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) != 3:
                raise ValueError("Unexpected batch structure")
            q_emb, t_emb, val = batch
            d = np.linalg.norm(q_emb - t_emb, axis=1)
            dists.append(d)
            tgts.append(val)
    print("Euclidean distance calculation complete.")
    return np.concatenate(dists), np.concatenate(tgts)


# --- Main Orchestration ---


def main(args):
    """Main workflow orchestrator for evaluation."""
    try:
        # Compute project root for models_base_dir
        project_root = Path(__file__).parent.parent.parent
        models_base_dir = project_root / "models"

        hparams = load_hparams(args.experiment_dir)
        model_type = hparams["model_type"]

        # 1. Resolve Test Data Path (parquet only)
        original_data_dir = hparams["data_dir"]
        if args.test_file:
            test_data_path = args.test_file.resolve()
            print(f"Using overridden test file: {test_data_path}")
        else:
            # Look for parquet file only
            test_data_path = (original_data_dir / "test.parquet").resolve()

            if not test_data_path.is_file():
                raise FileNotFoundError(
                    f"Test parquet file not found: {test_data_path}"
                )

            print(f"Using default test file: {test_data_path}")

        if not test_data_path.is_file():
            raise FileNotFoundError(f"Test data file not found: {test_data_path}")
        test_set_name = test_data_path.stem

        # 2. Determine Checkpoint/Identifier using ExperimentManager
        checkpoint_name: str
        if model_type in ["fnn", "linear", "linear_distance"]:
            # Create a minimal ExperimentManager for checkpoint finding
            exp_manager = ExperimentManager.from_hparams(hparams, models_base_dir)
            ckpt_path = exp_manager.find_best_checkpoint(args.experiment_dir)
            if not ckpt_path:
                raise FileNotFoundError("No checkpoint found")
            checkpoint_name = ckpt_path.stem
        elif model_type == "euclidean":
            checkpoint_name = "euclidean_baseline"
        else:
            raise ValueError(f"Unknown model type '{model_type}' in hparams")

        # 3. Determine Artifact Paths
        eval_dir = args.experiment_dir / "evaluation_results"
        eval_dir.mkdir(exist_ok=True)
        base_filename = f"test_{test_set_name}_{checkpoint_name}"
        preds_targets_path = eval_dir / f"{base_filename}_predictions_targets.npz"
        metrics_path = eval_dir / f"{base_filename}_metrics.txt"
        plot_path = eval_dir / f"{base_filename}_results.png"

        # 4. Get Predictions & Targets (handles cache check or compute/save)
        preds_targets_tuple = _get_predictions_targets(
            preds_targets_path,
            args.force_recompute,
            model_type,
            args.experiment_dir,
            hparams,
            test_data_path,  # Pass needed info for potential recompute
        )
        if preds_targets_tuple is None:
            raise RuntimeError("Failed to obtain predictions/targets")
        predictions, targets = preds_targets_tuple

        # 5. Get Metrics (handles cache check or compute/save)
        metrics = _get_metrics(
            metrics_path,
            args.force_recompute,
            predictions,
            targets,
            args.n_bootstrap,
            checkpoint_name,
            test_set_name,
        )
        if metrics is None:
            raise RuntimeError("Failed to obtain metrics")

        # 6. Generate Plot (always)
        print("Generating evaluation plot...")
        plot_title = f"Evaluation on '{test_set_name}' ({checkpoint_name})"
        plot_true_vs_predicted(
            targets, predictions, plot_path, metrics=metrics, title=plot_title
        )
        print(f"Saved evaluation plot to: {plot_path}")

        print("\nEvaluation process complete.")

    except Exception as e:
        print("\n--- EVALUATION FAILED --- ")
        print(f"Error during evaluation for experiment {args.experiment_dir}: {e}")
        import traceback

        traceback.print_exc()
        print("-------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model or Euclidean baseline, using caching."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to the experiment directory.",
        dest="experiment_dir",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        default=None,
        help="Optional path to a specific test parquet file to use for evaluation. "
        "If not provided, uses 'test.parquet' from the original training data directory.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for calculating SE/CI for correlation metrics. "
        "Set to 1 or 0 to disable bootstrapping. (default: 1000)",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force re-computation of predictions and metrics, ignoring cache.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
