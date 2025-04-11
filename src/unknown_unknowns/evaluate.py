import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml  # To potentially load hparams.yaml

# Project specific imports
from configs.config import ProjectConfig, load_config
from data.datasets import create_single_loader
from evaluation.metrics import calculate_regression_metrics
from models.predictor import ModelPredictor
from visualization.plot import plot_true_vs_predicted
from utils.helpers import get_device  # If needed for device placement


def load_model_from_checkpoint(checkpoint_path: Path) -> ModelPredictor:
    """Load the ModelPredictor model from a checkpoint file."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"Loading model from checkpoint: {checkpoint_path}")
    # load_from_checkpoint handles moving the model to the correct device (usually CPU by default)
    model = ModelPredictor.load_from_checkpoint(str(checkpoint_path))
    model.eval()  # Set to evaluation mode
    return model


def find_best_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the single checkpoint file saved by ModelCheckpoint(save_top_k=1)."""
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return None

    # Look for any .ckpt file (expecting only one due to save_top_k=1)
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


def get_config_and_paths_from_run(
    run_dir: Path,
) -> Tuple[ProjectConfig, Path]:
    """Try to load config from hparams.yaml in the run directory's tensorboard folder."""
    # run_dir is now passed directly
    hparams_file = run_dir / "tensorboard" / "hparams.yaml"

    if not hparams_file.is_file():
        print(
            f"Warning: hparams.yaml not found at {hparams_file}. Using default config."
        )
        return load_config(), run_dir

    print(f"Loading hyperparameters from: {hparams_file}")
    with open(hparams_file, "r") as f:
        hparams = yaml.safe_load(f)

    # Create a default config and override with hparams
    # Note: This is basic, might need more robust handling for type conversions (Path)
    cfg = load_config()
    try:
        # Paths need careful handling - hparams saves them as strings
        cfg.paths.data_dir = Path(hparams.get("data_dir", cfg.paths.data_dir))
        cfg.paths.embeddings_file = Path(
            hparams.get("embeddings_file", cfg.paths.embeddings_file)
        )
        cfg.paths.csv_subdir = hparams.get("csv_subdir", cfg.paths.csv_subdir)
        # Output dir isn't strictly needed for eval config but good for consistency
        cfg.paths.output_dir = Path(hparams.get("output_dir", cfg.paths.output_dir))

        # Training params
        cfg.training.param_name = hparams.get("param_name", cfg.training.param_name)
        cfg.training.batch_size = int(
            hparams.get("batch_size", cfg.training.batch_size)
        )
        # Other training params usually not needed for eval, but load if desired
        cfg.training.hidden_size = int(
            hparams.get("hidden_size", cfg.training.hidden_size)
        )

    except Exception as e:
        print(
            f"Warning: Error parsing hparams.yaml ({e}). Using default config values."
        )
        cfg = load_config()  # Fallback to default

    # This assumes data_dir is relative to project root in the stored hparams
    # Attempt to reconstruct project_root based on where data_dir is assumed to be relative to.
    # This might be fragile if the hparam path format changes.
    try:
        project_root = (
            (run_dir.parent.parent.parent / hparams.get("data_dir", "."))
            .resolve()
            .parent.parent
        )
    except Exception:
        print(
            "Warning: Could not reliably determine project root from hparams. Using current directory."
        )
        project_root = Path.cwd()

    return cfg, project_root


def prepare_evaluation_data(
    cfg: ProjectConfig, project_root: Path, test_csv_override: Optional[Path] = None
) -> Tuple[DataLoader, Path]:
    """Prepare the DataLoader for the test set and return the path used."""
    data_dir = (project_root / cfg.paths.data_dir).resolve()
    embeddings_file = (data_dir / cfg.paths.embeddings_file).resolve()

    if test_csv_override:
        test_csv_path = test_csv_override.resolve()
        print(f"Using overridden test CSV: {test_csv_path}")
    else:
        csv_dir = (data_dir / cfg.paths.csv_subdir).resolve()
        test_csv_path = (csv_dir / "test.csv").resolve()
        print(f"Using test CSV from config: {test_csv_path}")

    if not embeddings_file.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV file not found: {test_csv_path}")

    # Use the new single loader function
    test_loader = create_single_loader(
        csv_file=str(test_csv_path),
        hdf_file=str(embeddings_file),
        param_name=cfg.training.param_name,
        batch_size=cfg.training.batch_size,
        shuffle=False,  # No need to shuffle test data
        # num_workers can be added here if desired, defaults to 4
    )

    print(f"Prepared test data loader with {len(test_loader)} batches.")
    return test_loader, test_csv_path


def run_inference(
    model: ModelPredictor, test_loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on the test set."""
    print("Running inference on test data...")
    device = get_device()  # Get the appropriate device
    model.to(device)

    predictions_list = []
    targets_list = []
    with torch.no_grad():
        for batch in test_loader:
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


def generate_evaluation_results(
    predictions: np.ndarray,
    targets: np.ndarray,
    run_dir: Path,
    checkpoint_name: str,
    test_set_name: str,
):
    """Calculate metrics, save plot, and save metrics file in the run directory."""
    print(f"Calculating metrics and saving results for test set '{test_set_name}'...")
    eval_dir = run_dir / "evaluation_results"
    eval_dir.mkdir(exist_ok=True)

    # Calculate metrics
    metrics = calculate_regression_metrics(targets, predictions)

    # Create filenames incorporating test set name
    base_filename = f"test_{test_set_name}_{checkpoint_name}"
    results_png = eval_dir / f"{base_filename}_results.png"
    metrics_file = eval_dir / f"{base_filename}_metrics.txt"

    # Save plot - Add test set name to title
    plot_title = f"Evaluation on '{test_set_name}' ({checkpoint_name})"
    plot_true_vs_predicted(
        targets, predictions, results_png, metrics=metrics, title=plot_title
    )
    print(f"Saved evaluation plot to: {results_png}")

    # Print metrics
    print(f"\nMetrics for Test Set '{test_set_name}':")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save metrics to file - Add test set name to header
    with open(metrics_file, "w") as f:
        f.write(f"# Evaluation metrics for checkpoint: {checkpoint_name}\n")
        f.write(f"# Test Set: {test_set_name}\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"Saved metrics to: {metrics_file}")


def main(args):
    """Main workflow orchestrator for evaluation."""
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Specified run directory not found: {run_dir}")

    # Find the best checkpoint within the run directory
    best_checkpoint_path = find_best_checkpoint(run_dir)
    if not best_checkpoint_path:
        print("Evaluation aborted: Could not find a suitable checkpoint file.")
        return

    model = load_model_from_checkpoint(best_checkpoint_path)
    cfg, project_root = get_config_and_paths_from_run(run_dir)

    # Use param_name directly from the loaded config
    if "param_name" not in cfg.training.__dict__:
        print(
            "Warning: 'param_name' not found in loaded hparams. Using default from config."
        )
        param_name = load_config().training.param_name  # Fallback to default
    else:
        param_name = cfg.training.param_name
        print(f"Using param_name from loaded config: {param_name}")
    # Update the config object used for data loading (in case it wasn't loaded from hparams)
    cfg.training.param_name = param_name

    # Prepare data and get the actual test_csv path used
    test_loader, test_csv_path = prepare_evaluation_data(
        cfg, project_root, args.test_csv
    )
    # Use the stem of the test csv file as the identifier
    test_set_name = test_csv_path.stem

    predictions, targets = run_inference(model, test_loader)
    generate_evaluation_results(
        predictions, targets, run_dir, best_checkpoint_path.stem, test_set_name
    )

    print(f"\nEvaluation complete. Results saved in: {run_dir / 'evaluation_results'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model checkpoint from a run directory."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to the training run directory (e.g., models/runs/TIMESTAMP).",
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=None,
        help="Optional: Path to a specific test CSV file to use, overriding the one from training config.",
    )
    # Add other arguments here if needed to override config (e.g., batch_size)

    args = parser.parse_args()
    main(args)
