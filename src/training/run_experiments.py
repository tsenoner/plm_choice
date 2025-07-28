#!/usr/bin/env python
"""
This script runs training experiments for different models, embeddings, and parameters.
All training runs use Weights & Biases for logging and visualization.
Wandb automatically captures all console output, metrics, and model artifacts.
It also optionally evaluates the results after each training run.

Usage:
python run_experiments.py --data_dir data/processed/sprot_pre2024 --evaluate_after_train --model_types fnn linear euclidean --target_params fident alntmscore hfsp --wandb_project my-project
"""

import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

from src.shared.experiment_manager import ExperimentManager


def main(args):
    # Since this script is in src/training/, go up two levels to get project root
    project_root = Path(__file__).parent.parent.parent.resolve()

    # Use the provided data directory path
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"Error: Provided data directory not found at {data_dir}")
        return

    # Look for sets and embeddings subdirectories
    sets_dir = data_dir / "sets"
    embeddings_dir = data_dir / "embeddings"

    if not sets_dir.is_dir():
        print(f"Error: Sets directory not found at {sets_dir}")
        return
    if not embeddings_dir.is_dir():
        print(f"Error: Embeddings directory not found at {embeddings_dir}")
        return

    models_base_dir = project_root / "models"

    # --- Experiment Definitions ---
    model_types = args.model_types
    target_params = args.target_params
    # ---------------------------

    # Find all .h5 embedding files in the specified directory
    embedding_files = list(embeddings_dir.glob("*.h5"))
    if not embedding_files:
        print(f"Error: No HDF5 (.h5) embedding files found in {embeddings_dir}")
        return

    print(f"Using data directory: {data_dir}")  # Log the used path
    print(f"Using embeddings directory: {embeddings_dir}")
    print(f"Using sets directory: {sets_dir}")
    print(f"Found {len(embedding_files)} embedding files:")
    for f in embedding_files:
        print(f" - {f.name}")
    print(f"Target model types: {model_types}")
    print(f"Target parameters: {target_params}")
    print(f"Wandb project: {args.wandb_project}")
    if args.wandb_entity:
        print(f"Wandb entity: {args.wandb_entity}")
    if args.evaluate_after_train:
        print("Evaluation after each training run: Enabled")
    print(
        "All training logs and metrics are captured automatically by Weights & Biases."
    )
    print("\n")

    total_combinations = len(model_types) * len(target_params) * len(embedding_files)
    run_count = 0
    skipped_count = 0
    eval_success_count = 0
    eval_fail_count = 0
    error_count = 0  # Track general errors

    # Wrap the iteration with tqdm for a progress bar
    pbar = tqdm(total=total_combinations, desc="Running Experiments", unit="run")

    # Iterate through all combinations
    for model_type in model_types:
        for param_name in target_params:
            for embedding_file_path in embedding_files:
                embedding_name = embedding_file_path.stem

                # Update progress bar description
                pbar.set_description(
                    f"Checking: {model_type}/{param_name}/{embedding_name}"
                )

                # Create experiment manager for this combination
                exp_manager = ExperimentManager(
                    dataset_dir=data_dir,  # data/processed/sprot_pre2024
                    embedding_name=embedding_name,  # e.g., 'esm1b'
                    model_type=model_type,
                    param_name=param_name,
                    models_base_dir=models_base_dir,
                )

                # Check experiment status using the centralized manager
                status, experiment_dir = exp_manager.check_experiment_status()

                if status == "completed":
                    # Training completed - skip this experiment
                    skipped_count += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        skipped=skipped_count, errors=error_count, refresh=True
                    )
                    continue
                elif status == "interrupted":
                    # Training interrupted but has checkpoint - resume
                    print(
                        f"Resuming interrupted training: {model_type}/{param_name}/{embedding_name}"
                    )
                    resume_training = True
                else:
                    # Fresh start (not_started or empty)
                    resume_training = False

                # Update description before training
                pbar.set_description(
                    f"Training: {model_type}/{param_name}/{embedding_name}"
                )

                run_count += 1

                # Base command
                train_command = [
                    "uv",
                    "run",
                    "python",
                    str(project_root / "src" / "training" / "train.py"),
                    "--model_type",
                    model_type,
                    "--param_name",
                    param_name,
                    "--embedding_file",
                    str(embedding_file_path.resolve()),
                    "--data_dir",
                    str(sets_dir),
                    "--output_base_dir",
                    str(exp_manager.experiment_dir),
                    "--num_workers",
                    "10",
                    "--batch_size",
                    "1024",
                    "--learning_rate",
                    "0.0001",
                    "--max_epochs",
                    "100",
                    "--early_stopping_patience",
                    "10",
                    "--val_check_interval",
                    str(args.val_check_interval),  # Use the provided val_check_interval
                ]

                # Add wandb configuration
                if args.wandb_project:
                    train_command.extend(["--wandb_project", args.wandb_project])
                if args.wandb_entity:
                    train_command.extend(["--wandb_entity", args.wandb_entity])

                # Add resume flag if resuming
                if resume_training:
                    train_command.append("--resume_from_checkpoint")

                training_successful = False
                try:
                    # Run training - wandb captures all output automatically
                    subprocess.run(
                        train_command,
                        check=True,  # Will raise CalledProcessError on failure
                        cwd=project_root,
                        text=True,
                    )
                    training_successful = True

                except subprocess.CalledProcessError as e:
                    error_count += 1
                    tqdm.write(
                        f"Error during training for {model_type}/{param_name}/{embedding_name}: {e}"
                    )
                except KeyboardInterrupt:
                    pbar.close()
                    print("\nExperiment run interrupted by user.")
                    return
                except Exception as e:
                    error_count += 1
                    tqdm.write(
                        f"Unexpected error during training execution for {model_type}/{param_name}/{embedding_name}: {e}"
                    )

                # --- Optional Evaluation Step ---
                if training_successful and args.evaluate_after_train:
                    # Use the experiment directory for evaluation
                    pbar.set_description(
                        f"Evaluating: {model_type}/{param_name}/{embedding_name}"
                    )

                    eval_command = [
                        "uv",
                        "run",
                        "python",
                        str(project_root / "src" / "evaluation" / "evaluate.py"),
                        "--run_dir",
                        str(experiment_dir),
                    ]

                    try:
                        # Run evaluation - output goes to console/wandb
                        subprocess.run(
                            eval_command,
                            check=True,
                            cwd=project_root,
                            text=True,
                        )
                        eval_success_count += 1
                    except subprocess.CalledProcessError as e:
                        eval_fail_count += 1
                        error_count += 1  # Count eval errors too
                        tqdm.write(f"Error during evaluation for {experiment_dir}: {e}")
                    except KeyboardInterrupt:
                        pbar.close()
                        print("\nEvaluation interrupted by user.")
                        return
                    except Exception as e:
                        eval_fail_count += 1
                        error_count += 1  # Count eval errors too
                        tqdm.write(
                            f"Unexpected error during evaluation for {experiment_dir}: {e}"
                        )

                # Update progress bar after processing each combination
                pbar.update(1)
                pbar.set_postfix(
                    skipped=skipped_count,
                    errors=error_count,
                    eval_ok=eval_success_count,
                    eval_fail=eval_fail_count,
                    refresh=True,
                )

    pbar.close()  # Close the progress bar cleanly

    print("\n==== Experiment Runner Finished ====")
    print(f"Total combinations processed: {total_combinations}")
    print(f"Launched training runs: {run_count}")
    print(f"Skipped existing runs: {skipped_count}")
    if args.evaluate_after_train:
        print(f"Successful evaluations: {eval_success_count}")
        print(f"Failed evaluations: {eval_fail_count}")
    print("==================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training experiments for different models, embeddings, and parameters, optionally evaluating after each run."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(
            "data/processed/sprot_pre2024"
        ),  # Set the default to the new structure
        help="Path to the directory containing 'sets' and 'embeddings' subdirectories (default: data/processed/sprot_pre2024)",
    )
    parser.add_argument(
        "--evaluate_after_train",
        action="store_true",
        help="Run evaluate.py automatically after each successful training run.",
    )
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["fnn", "linear", "euclidean", "linear_distance"],
        choices=["fnn", "linear", "euclidean", "linear_distance"],
        help="List of model types to run (e.g., fnn linear euclidean linear_distance).",
    )
    parser.add_argument(
        "--target_params",
        nargs="+",
        default=["fident", "alntmscore", "hfsp"],
        choices=["fident", "alntmscore", "hfsp"],
        help="List of target parameters to run.",
    )
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
        help="Weights & Biases entity (username/team).",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.2,
        help="How often to run validation during training. 0.2 = 5 times per epoch (default: 0.2)",
    )

    args = parser.parse_args()
    main(args)
