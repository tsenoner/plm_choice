"""
This script trains a specified model or calculates the Euclidean baseline.

Usage:
python train.py --model_type fnn --embedding_file path/to/embeddings.h5 --data_dir path/to/data_dir --param_name param_name
"""

import argparse
from pathlib import Path
from typing import Tuple, Type
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb
import yaml

from src.shared.datasets import create_single_loader, get_embedding_size
from src.shared.experiment_manager import ExperimentManager, ExperimentPaths
from src.training.models import (
    FNNPredictor,
    LinearRegressionPredictor,
    LinearDistancePredictor,
)


def setup_environment(seed: int):
    """Set random seed and PyTorch settings."""
    pl.seed_everything(seed)
    if torch.cuda.is_available():
        print("CUDA available, setting matmul precision to medium.")
        torch.set_float32_matmul_precision("medium")
    else:
        print("CUDA not available.")


def prepare_data(
    param_name: str,
    batch_size: int,
    embeddings_file: Path,
    train_file: Path,
    val_file: Path,
    num_workers: int,
) -> Tuple[int, DataLoader, DataLoader]:
    """Load train/val datasets and return embedding size and dataloaders."""
    print("Preparing train and validation data loaders...")
    embedding_size = get_embedding_size(str(embeddings_file))
    print(f"Detected embedding size: {embedding_size}")

    loader_args = {
        "hdf_file": str(embeddings_file),
        "param_name": param_name,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    print(f"Using {num_workers} worker(s) for DataLoaders.")

    train_loader = create_single_loader(
        parquet_file=str(train_file), shuffle=True, **loader_args
    )
    val_loader = create_single_loader(
        parquet_file=str(val_file), shuffle=False, **loader_args
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return embedding_size, train_loader, val_loader


def train_model(
    model_class: Type[pl.LightningModule],
    model_kwargs: dict,
    trainer_kwargs: dict,
    paths: ExperimentPaths,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hparams: dict,
    wandb_project: str = "which-plm",
    wandb_entity: str = None,
    resume_from_checkpoint: bool = False,
) -> Tuple[str, WandbLogger, float]:
    """Configure and run the PyTorch Lightning training loop for a given model."""
    print(f"Configuring model ({model_class.__name__}) and trainer...")

    # Extract hyperparameters from hparams dict
    early_stopping_patience = hparams.get("early_stopping_patience", 10)
    val_check_interval = hparams.get("val_check_interval", 0.2)
    batch_size = hparams.get("batch_size", 1024)

    # Calculate actual patience based on val_check_interval
    if isinstance(val_check_interval, float) and 0 < val_check_interval <= 1:
        actual_patience = max(1, int(early_stopping_patience / val_check_interval))
    else:
        actual_patience = early_stopping_patience

    print(
        f"Early stopping patience: {early_stopping_patience} epochs = {actual_patience} validation checks"
    )

    # Instantiate the selected model
    model = model_class(**model_kwargs)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=paths.checkpoints_dir,
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        filename="best-{epoch:02d}-{step}-{val_loss:.3f}",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=actual_patience,
        mode="min",
        verbose=True,
    )
    callbacks = [early_stopping_callback, checkpoint_callback]

    # Weights & Biases logger configuration
    embedding_name = Path(hparams["embedding_file"]).stem
    run_name = f"{hparams['model_type']}-{hparams['param_name']}-{embedding_name}"

    # Handle wandb run resuming
    wandb_run_id = None
    wandb_id_file = paths.experiment_dir / "wandb_run_id.txt"

    if resume_from_checkpoint and wandb_id_file.exists():
        try:
            with open(wandb_id_file, "r") as f:
                wandb_run_id = f.read().strip()
            print(f"Resuming wandb run with ID: {wandb_run_id}")
        except Exception as e:
            print(f"Could not load wandb run ID: {e}. Starting new run.")
            wandb_run_id = None

    logger = WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        save_dir=str(paths.experiment_dir),
        log_model=True,
        id=wandb_run_id,
        resume="must" if wandb_run_id else None,
    )

    # Save wandb run ID for future resuming (only for new runs)
    if not wandb_run_id:
        try:
            # Access experiment to ensure run is created
            _ = logger.experiment
            new_run_id = logger.experiment.id
            with open(wandb_id_file, "w") as f:
                f.write(new_run_id)
            print(f"Saved wandb run ID: {new_run_id}")
        except Exception as e:
            print(f"Warning: Could not save wandb run ID: {e}")

    print(f"Using Weights & Biases logging - Project: {wandb_project}, Run: {run_name}")

    # Log hyperparameters to wandb
    logger.log_hyperparams(hparams)

    # --- Trainer Setup ---
    # Calculate logging frequency to be consistent with validation
    # If val_check_interval is a fraction, convert to steps for consistent logging
    if isinstance(val_check_interval, float) and 0 < val_check_interval <= 1:
        # Validation happens every val_check_interval * epoch_steps
        val_steps = max(1, int(len(train_loader) * val_check_interval))
        # Log at least as frequently as validation, but not more than every 10 steps
        logging_steps = max(10, min(val_steps // 2, len(train_loader) // 4))
    else:
        # val_check_interval is integer steps
        logging_steps = max(1, min(val_check_interval // 2, len(train_loader) // 4))

    print(
        f"Validation every {val_check_interval} ({'fraction of epoch' if isinstance(val_check_interval, float) else 'steps'})"
    )
    print(f"Logging every {logging_steps} steps")
    print(f"Batch size: {batch_size}")

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=logging_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        val_check_interval=val_check_interval,
        **trainer_kwargs,
    )

    # --- Training ---
    print("Starting training...")

    # Check for checkpoint to resume from if specified
    ckpt_path = None
    if resume_from_checkpoint and paths.last_checkpoint.exists():
        ckpt_path = paths.last_checkpoint
        print(f"Resuming training from: {ckpt_path}")
    elif resume_from_checkpoint:
        print("No last.ckpt found - starting fresh training")

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    best_model_path = checkpoint_callback.best_model_path
    best_model_score = (
        float(checkpoint_callback.best_model_score)
        if checkpoint_callback.best_model_score is not None
        else 0.0
    )

    print(f"\nTraining finished. Best model saved at: {best_model_path}")
    print(f"Best validation loss: {best_model_score:.6f}")

    return best_model_path, logger, best_model_score


def main(args):
    """Main workflow orchestrator for training or Euclidean baseline setup."""
    setup_environment(args.seed)

    # Initialize wandb for all model types except euclidean baseline
    if args.model_type != "euclidean":
        print("Initializing Weights & Biases...")
        try:
            wandb.login()
            print("Successfully logged into Weights & Biases")
        except Exception as e:
            print(f"Warning: Could not log into wandb automatically: {e}")
            print(
                "Please run 'wandb login' manually or set WANDB_API_KEY environment variable"
            )

    # Create experiment manager
    # Determine dataset directory - if data_dir is 'sets', get parent; otherwise use data_dir
    if args.data_dir.name == "sets":
        dataset_dir = args.data_dir.parent
    else:
        dataset_dir = args.data_dir

    exp_manager = ExperimentManager(
        dataset_dir=dataset_dir,
        embedding_name=args.embedding_file.stem,
        model_type=args.model_type,
        param_name=args.param_name,
        models_base_dir=args.output_base_dir.parents[
            3
        ],  # Go up 4 levels: embedding/param/model/dataset -> models
    )

    # Create experiment paths
    paths = exp_manager.create_experiment_paths(
        project_root=Path(__file__).parent.parent.parent
    )

    # --- Euclidean Baseline Setup Only ---
    if args.model_type == "euclidean":
        print("Setting up directory for Euclidean Distance Baseline...")

        # Euclidean baseline doesn't use wandb, so save hparams locally
        try:
            embedding_size = get_embedding_size(str(paths.embedding_file))
        except Exception as e:
            print(f"Warning: Could not get embedding size: {e}")
            embedding_size = -1

        hparams_to_log = {
            "model_type": args.model_type,
            "param_name": args.param_name,
            "embedding_file": str(paths.embedding_file),
            "data_dir": str(paths.data_dir),
            "embedding_size": embedding_size,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }

        hparams_path = paths.experiment_dir / "hparams.yaml"
        try:
            with open(hparams_path, "w") as f:
                yaml.dump(hparams_to_log, f)
            print(f"Saved hyperparameters to {hparams_path} (euclidean baseline)")
        except Exception as e:
            print(f"Warning: Could not save hyperparameters: {e}")

        print("\nEuclidean baseline setup complete.")
        print(f"Experiment directory: {paths.experiment_dir}")

        # Create completion marker for euclidean baseline
        exp_manager.create_completion_marker(
            paths.experiment_dir, "euclidean_baseline", 0.0
        )

        # Print the experiment dir path for the runner script
        print(str(paths.experiment_dir.resolve()))

    # --- Standard Model Training ---
    else:
        embedding_size, train_loader, val_loader = prepare_data(
            param_name=args.param_name,
            batch_size=args.batch_size,
            embeddings_file=paths.embedding_file,
            train_file=paths.train_file,
            val_file=paths.val_file,
            num_workers=args.num_workers,
        )

        # Prepare model arguments
        model_kwargs = {
            "embedding_size": embedding_size,
            "learning_rate": args.learning_rate,
        }
        if args.model_type == "fnn":
            model_class = FNNPredictor
            model_kwargs["hidden_size"] = args.hidden_size
        elif args.model_type == "linear":
            model_class = LinearRegressionPredictor
        elif args.model_type == "linear_distance":
            model_class = LinearDistancePredictor
        else:
            raise ValueError(f"Unknown trainable model_type: {args.model_type}")

        # Prepare trainer arguments
        trainer_kwargs = {
            "max_epochs": args.max_epochs,
            "accelerator": "auto",
            "devices": "auto",
        }

        # Prepare hyperparameters to log (subset of args + derived)
        hparams_to_log = {
            "model_type": args.model_type,
            "param_name": args.param_name,
            "embedding_file": str(paths.embedding_file),
            "data_dir": str(paths.data_dir),
            "embedding_size": embedding_size,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "val_check_interval": args.val_check_interval,
            "num_workers": args.num_workers,
            "seed": args.seed,
        }
        if args.model_type == "fnn":
            hparams_to_log["hidden_size"] = args.hidden_size

        # Train model
        best_model_path, _, best_model_score = train_model(
            model_class=model_class,
            model_kwargs=model_kwargs,
            trainer_kwargs=trainer_kwargs,
            paths=paths,
            train_loader=train_loader,
            val_loader=val_loader,
            hparams=hparams_to_log,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

        # Create completion marker using the experiment manager
        exp_manager.create_completion_marker(
            paths.experiment_dir,
            best_model_path,
            best_model_score,
        )

        # --- Final Output for Training ---
        print(
            f"\nRun completed successfully. Best checkpoint saved to: {best_model_path}"
        )
        print("View logs and metrics at: https://wandb.ai")
        print(
            "\nTo evaluate the best model, run evaluate.py using the experiment directory:"
        )
        print(
            f'uv run python src/evaluation/evaluate.py --run_dir "{paths.experiment_dir}"'
        )
        # Print the experiment dir path for the runner script
        print(str(paths.experiment_dir.resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a specified model or calculate Euclidean baseline."
    )

    # --- Model Selection ---
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["fnn", "linear", "euclidean", "linear_distance"],
        help="Type of model to train or baseline to set up.",
    )

    # --- Input Data Paths ---
    parser.add_argument(
        "--embedding_file",
        type=Path,
        required=True,
        help="Absolute path to the embedding HDF5 file.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Absolute path to the directory containing train/val/test files (CSV or parquet format).",
    )
    parser.add_argument(
        "--param_name",
        type=str,
        required=True,
        choices=["fident", "alntmscore", "hfsp"],
        help="Target parameter name to train the model for.",
    )

    # --- Output Location ---
    parser.add_argument(
        "--output_base_dir",
        type=Path,
        required=True,
        help="Absolute path to the experiment directory.",
    )

    # --- Core Training Hyperparameters (with defaults set here) ---
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size (default: 1024)"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs (default: 100)",
    )
    parser.add_argument(
        "-esp",
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Patience for early stopping (default: 5)",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.2,
        help="How often to run validation during training. Float = fraction of epoch (0.2 = 5 times per epoch), Int = every N steps (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)",
    )

    # --- Model Specific Hyperparameters ---
    # Only relevant for FNN
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden layer size for FNN model (default: 64) - Ignored for linear model.",
    )

    # --- Weights & Biases Configuration ---
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="unknown-unknowns",
        help="Weights & Biases project name (default: unknown-unknowns).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity (username/team). If not specified, uses your default entity.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume training from the latest checkpoint in the experiment directory.",
    )

    args = parser.parse_args()

    main(args)
