import os
import argparse
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import shutil

from models.predictor import ModelPredictor
from models.utils import create_data_loaders, get_embedding_size, plot_scatter

def get_run_dir(base_dir: str) -> str:
    """Create a unique run directory based on timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)

    # If directory exists, remove it completely
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    # Create fresh directories
    os.makedirs(run_dir)
    os.makedirs(os.path.join(run_dir, 'checkpoints'))

    return run_dir

def main(args):
    # Setup
    pl.seed_everything(args.seed)

    # Create unique run directory
    run_dir = get_run_dir(args.output_dir)
    print(f"Output directory: {run_dir}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Prepare data
    embedding_size = get_embedding_size(args.hdf_file)
    train_loader, val_loader, test_loader = create_data_loaders(
        os.path.join(args.data_dir, "train.csv"),
        os.path.join(args.data_dir, "val.csv"),
        os.path.join(args.data_dir, "test.csv"),
        args.hdf_file,
        args.param_name,
        args.batch_size
    )

    # Adjust logging interval based on dataset size
    logging_steps = max(1, len(train_loader) // 4)  # Log 4 times per epoch

    # Create model
    model = ModelPredictor(
        embedding_size=embedding_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate
    )

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            mode="min"
        ),
        ModelCheckpoint(
            dirpath=os.path.join(run_dir, 'checkpoints'),
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            filename="model-{epoch:02d}-{val_loss:.3f}"
        )
    ]

    # Setup TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(run_dir, 'tensorboard'),
        name=None,  # Don't create additional subdirectory
        default_hp_metric=False
    )

    # Log hyperparameters
    hparams = vars(args)
    logger.log_hyperparams(hparams)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        default_root_dir=run_dir,
        logger=logger,
        log_every_n_steps=logging_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train and evaluate
    trainer.fit(model, train_loader, val_loader)

    # Save best model path
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"\nBest model saved at: {best_model_path}")

    predictions = trainer.predict(dataloaders=test_loader, ckpt_path="best")
    predictions = torch.cat(predictions).cpu().numpy()
    targets = torch.cat([batch[2] for batch in test_loader]).cpu().numpy()

    # Save results
    metrics = plot_scatter(predictions, targets, os.path.join(run_dir, "results.png"))

    # Print and log metrics
    print("\nFinal Test Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        logger.experiment.add_scalar(f"test/{metric}", value, 0)

    # Save metrics to file
    metrics_file = os.path.join(run_dir, "test_metrics.txt")
    with open(metrics_file, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"\nRun completed. Results saved in: {run_dir}")
    print("To view TensorBoard logs, run:")
    print(f"tensorboard --logdir {run_dir}/tensorboard")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--hdf_file", required=True, help="Path to HDF5 embeddings")
    parser.add_argument("--param_name", required=True, help="Target parameter name")
    parser.add_argument("--output_dir", required=True, help="Base output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=5)

    main(parser.parse_args())