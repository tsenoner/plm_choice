import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from configs.config import ProjectConfig, load_config
from data.datasets import create_single_loader, get_embedding_size
from models.predictor import ModelPredictor


# Define a simple structure to hold paths
@dataclass
class ResolvedPaths:
    project_root: Path
    data_dir: Path
    embeddings_file: Path
    csv_dir: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    output_dir: Path
    run_dir: Path


def setup_environment(seed: int):
    """Set random seed and PyTorch settings."""
    pl.seed_everything(seed)
    if torch.cuda.is_available():
        print("CUDA available, setting matmul precision to medium.")
        torch.set_float32_matmul_precision("medium")
    else:
        print("CUDA not available.")


def create_run_directory(base_output_dir: Path) -> Path:
    """Create a unique timestamped run directory, cleaning up if it exists."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / timestamp
    print(f"Creating run directory: {run_dir}")

    # Use with caution: removes existing directory!
    if run_dir.exists():
        print(f"Warning: Run directory {run_dir} already exists. Removing.")
        shutil.rmtree(run_dir)

    # Create run structure
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

    return run_dir


def prepare_paths(cfg: ProjectConfig) -> ResolvedPaths:
    """Resolve all necessary absolute paths based on the configuration."""
    # Assumes the script is run from the project root or that paths in config are relative to root
    project_root = Path(__file__).parent.parent.parent
    data_dir = (project_root / cfg.paths.data_dir).resolve()
    embeddings_file = (data_dir / cfg.paths.embeddings_file).resolve()
    csv_dir = (data_dir / cfg.paths.csv_subdir).resolve()
    train_csv = (csv_dir / "train.csv").resolve()
    val_csv = (csv_dir / "val.csv").resolve()
    test_csv = (csv_dir / "test.csv").resolve()
    output_dir = (project_root / cfg.paths.output_dir).resolve()

    # Check essential files exist (only train/val needed for training script)
    if not embeddings_file.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    if not train_csv.is_file():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not val_csv.is_file():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")
    # Don't strictly need to check test_csv here, but good practice
    if not test_csv.is_file():
        print(f"Warning: Test CSV not found: {test_csv}")

    # Create run directory (needs output_dir resolved first)
    run_dir = create_run_directory(output_dir)

    return ResolvedPaths(
        project_root=project_root,
        data_dir=data_dir,
        embeddings_file=embeddings_file,
        csv_dir=csv_dir,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        output_dir=output_dir,
        run_dir=run_dir,
    )


def prepare_data(
    cfg: ProjectConfig, paths: ResolvedPaths
) -> Tuple[int, DataLoader, DataLoader]:
    """Load train/val datasets and return embedding size and dataloaders."""
    print("Preparing train and validation data loaders...")
    embedding_size = get_embedding_size(str(paths.embeddings_file))
    print(f"Detected embedding size: {embedding_size}")

    # Shared arguments for loaders
    loader_args = {
        "hdf_file": str(paths.embeddings_file),
        "param_name": cfg.training.param_name,
        "batch_size": cfg.training.batch_size,
        # num_workers can be added from config if needed
    }

    # Create train and validation loaders individually
    train_loader = create_single_loader(
        csv_file=str(paths.train_csv), shuffle=True, **loader_args
    )
    val_loader = create_single_loader(
        csv_file=str(paths.val_csv), shuffle=False, **loader_args
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return embedding_size, train_loader, val_loader


def train_model(
    cfg: ProjectConfig,
    paths: ResolvedPaths,
    embedding_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Tuple[str, TensorBoardLogger]:  # Returns best_model_path and logger
    """Configure and run the PyTorch Lightning training loop."""
    print("Configuring model and trainer...")

    model = ModelPredictor(
        embedding_size=embedding_size,
        hidden_size=cfg.training.hidden_size,
        learning_rate=cfg.training.learning_rate,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=paths.run_dir / "checkpoints",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="model-{epoch:02d}-{val_loss:.3f}",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.early_stopping_patience,
        mode="min",
        verbose=True,
    )
    callbacks = [early_stopping_callback, checkpoint_callback]

    # Logger
    logger = TensorBoardLogger(
        save_dir=str(paths.run_dir / "tensorboard"),
        name=None,
        version="",
        default_hp_metric=False,
    )

    # Log hyperparameters manually
    hparams_train = cfg.training.__dict__.copy()
    hparams_train["embedding_size"] = embedding_size

    hparams_paths = {k: str(v) for k, v in cfg.paths.__dict__.items()}
    logger.log_hyperparams({**hparams_paths, **hparams_train})

    # Trainer
    logging_steps = max(1, len(train_loader) // 4)  # Log 4 times per epoch
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        logger=logger,
        log_every_n_steps=logging_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        # default_root_dir=str(paths.run_dir), # Not needed when logger is specified
    )

    # --- Training ---
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    print(f"\nTraining finished. Best model saved at: {best_model_path}")

    return best_model_path, logger


def main(cfg: ProjectConfig):
    """Main workflow orchestrator for training."""
    setup_environment(cfg.training.seed)
    paths = prepare_paths(cfg)
    embedding_size, train_loader, val_loader = prepare_data(cfg, paths)
    best_model_path, _ = train_model(
        cfg, paths, embedding_size, train_loader, val_loader
    )

    print(f"\nRun completed successfully. Best checkpoint saved to: {best_model_path}")
    print("To view TensorBoard logs for this run, use:")
    print(f"tensorboard --logdir {paths.run_dir / 'tensorboard'}")
    print("\nTo evaluate the best model, run evaluate.py using the run directory:")
    print(f'uv run python src/unknown_unknowns/evaluate.py --run_dir "{paths.run_dir}"')


if __name__ == "__main__":
    # --- Configuration Loading & Argument Parsing --- #
    config = load_config()  # Load default config

    parser = argparse.ArgumentParser(
        description="Train model, allowing selection of training data subdirectory."
    )
    parser.add_argument(
        "--csv_subdir",
        type=str,
        default=config.paths.csv_subdir,
        help=f"Subdirectory within {config.paths.data_dir} containing train/val/test CSVs (default: {config.paths.csv_subdir})",
        choices=["training", "train_sub", "train_sub1"],  # Restrict choices
    )
    # Example of overriding another config value:
    # parser.add_argument(
    #     "--batch_size", type=int, help="Override default batch size."
    # )

    args = parser.parse_args()

    # Update config with parsed arguments
    config.paths.csv_subdir = args.csv_subdir
    # if args.batch_size is not None:
    #    config.training.batch_size = args.batch_size
    # -------------------------------------------- #

    main(config)  # Run the main workflow
