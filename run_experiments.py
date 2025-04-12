#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path
import os  # Import os for stat
import re  # Import re for regular expressions
from typing import Optional


def find_latest_run_dir(base_dir: Path) -> Optional[Path]:
    """Find the most recently created timestamped subdirectory in base_dir."""
    if not base_dir.is_dir():
        return None

    timestamp_pattern = re.compile(r"\d{8}_\d{6}")  # Compile pattern YYYYMMDD_HHMMSS

    subdirs = [
        d
        for d in base_dir.iterdir()
        # Check if it's a directory AND its name matches the pattern
        if d.is_dir() and timestamp_pattern.fullmatch(d.name)
    ]

    if not subdirs:
        return None

    # Find the one with the latest creation time (st_ctime)
    latest_dir = max(subdirs, key=lambda d: os.stat(d).st_ctime)
    return latest_dir


def main(args):
    project_root = Path(__file__).parent.resolve()
    embeddings_dir = project_root / "data" / "swissprot" / "embeddings"
    models_base_dir = project_root / "models"

    # Resolve the absolute path for the CSV directory based on the subdir argument
    csv_dir_abs = (project_root / "data" / "swissprot" / args.csv_subdir).resolve()
    if not csv_dir_abs.is_dir():
        print(f"Error: CSV directory not found at {csv_dir_abs}")
        return

    # --- Experiment Definitions ---
    model_types = args.model_types  # Use arg
    target_params = args.target_params  # Use arg
    # ---------------------------

    # Find all .h5 embedding files
    embedding_files = list(embeddings_dir.glob("*.h5"))
    if not embedding_files:
        print(f"Error: No HDF5 (.h5) embedding files found in {embeddings_dir}")
        return

    print(f"Found {len(embedding_files)} embedding files:")
    for f in embedding_files:
        print(f" - {f.name}")
    print(f"Target model types: {model_types}")
    print(f"Target parameters: {target_params}")
    print(f"Using CSV directory: {csv_dir_abs}")
    if args.evaluate_after_train:
        print("Evaluation after each training run: Enabled")
    print("\n")

    total_combinations = len(model_types) * len(target_params) * len(embedding_files)
    run_count = 0
    skipped_count = 0
    eval_success_count = 0
    eval_fail_count = 0

    # Iterate through all combinations
    for model_type in model_types:
        for param_name in target_params:
            for embedding_file_path in embedding_files:
                embedding_name = embedding_file_path.stem
                embedding_file_abs = str(embedding_file_path.resolve())

                experiment_output_dir = (
                    models_base_dir / f"{model_type}_runs" / param_name / embedding_name
                )

                print("---")
                print(
                    f"Checking: Model={model_type}, Param={param_name}, Embedding={embedding_name}"
                )

                # --- Redundancy Check ---
                if experiment_output_dir.exists():
                    if any(item.is_dir() for item in experiment_output_dir.iterdir()):
                        print(
                            f"Skipping: Output directory {experiment_output_dir} already exists with subdirectories."
                        )
                        skipped_count += 1
                        continue
                    else:
                        print(
                            f"Warning: Output directory {experiment_output_dir} exists but is empty. Proceeding to run."
                        )
                # ------------------------

                print(
                    f"Running training for Model={model_type}, Param={param_name}, Embedding={embedding_name}..."
                )
                run_count += 1

                # Base command
                train_command = [
                    "uv",
                    "run",
                    "python",
                    str(project_root / "src" / "unknown_unknowns" / "train.py"),
                    "--model_type",
                    model_type,
                    "--param_name",
                    param_name,
                    "--embedding_file",
                    embedding_file_abs,
                    "--csv_dir",
                    str(csv_dir_abs),
                    "--num_workers",
                    "14",
                    "--batch_size",
                    "2048",
                    "--learning_rate",
                    "0.0001",
                    "--max_epochs",
                    "100",
                    "--early_stopping_patience",
                    "5",
                ]

                # Add other hyperparameter overrides here if needed

                print(f"Executing: {' '.join(train_command)}")
                training_successful = False
                try:
                    # Run training, let output stream directly
                    subprocess.run(
                        train_command,
                        check=True,  # Will raise CalledProcessError on failure
                        cwd=project_root,
                        text=True,
                        # Removed capture_output=True
                    )
                    print(
                        f"Successfully finished training for Model={model_type}, Param={param_name}, Embedding={embedding_name}"
                    )
                    training_successful = True

                except subprocess.CalledProcessError as e:
                    print(
                        f"Error during training for Model={model_type}, Param={param_name}, Embedding={embedding_name}:"
                    )
                    # Output was streamed, so error messages should already be visible
                    print(f"Return code: {e.returncode}")
                    # Optionally stop here
                    # return
                except KeyboardInterrupt:
                    print("\nExperiment run interrupted by user.")
                    return
                except Exception as e:
                    print(f"Unexpected error during training execution: {e}")

                # --- Optional Evaluation Step ---
                run_dir_to_evaluate = None
                if training_successful and args.evaluate_after_train:
                    print(f"Finding run directory in {experiment_output_dir}...")
                    run_dir_to_evaluate = find_latest_run_dir(experiment_output_dir)
                    if run_dir_to_evaluate:
                        print(f"Found latest run directory: {run_dir_to_evaluate}")
                    else:
                        print(
                            f"Error: Could not find timestamped run directory in {experiment_output_dir} after successful training."
                        )

                if run_dir_to_evaluate:
                    print(f"Running evaluation for run: {run_dir_to_evaluate}")
                    eval_command = [
                        "uv",
                        "run",
                        "python",
                        str(project_root / "src" / "unknown_unknowns" / "evaluate.py"),
                        "--run_dir",
                        str(run_dir_to_evaluate),
                    ]

                    print(f"Executing: {' '.join(eval_command)}")
                    try:
                        # Let evaluation output stream as well
                        subprocess.run(
                            eval_command, check=True, cwd=project_root, text=True
                        )
                        print(
                            f"Successfully finished evaluation for {run_dir_to_evaluate}"
                        )
                        eval_success_count += 1
                    except subprocess.CalledProcessError as e:
                        print(f"Error during evaluation for {run_dir_to_evaluate}:")
                        print(f"Return code: {e.returncode}")
                        eval_fail_count += 1
                    except KeyboardInterrupt:
                        print("\nEvaluation interrupted by user.")
                        return
                    except Exception as e:
                        print(f"Unexpected error during evaluation: {e}")
                        eval_fail_count += 1
                elif args.evaluate_after_train and training_successful:
                    # Log failure if we expected to evaluate but couldn't find dir
                    eval_fail_count += 1
                    print(
                        f"Skipping evaluation: Could not determine run directory for {experiment_output_dir}"
                    )
                elif args.evaluate_after_train:
                    # Don't evaluate if training failed
                    print(
                        f"Skipping evaluation: Training failed for Model={model_type}, Param={param_name}, Embedding={embedding_name}"
                    )

                print("---")

    print("\n==== Experiment Runner Finished ====")
    print(f"Total combinations checked: {total_combinations}")
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
        "--csv_subdir",
        type=str,
        default="training",
        help="Subdirectory name within data/swissprot containing train/val/test CSVs (default: training)",
        choices=["training", "train_sub", "train_sub1"],
    )
    parser.add_argument(
        "--evaluate_after_train",
        action="store_true",  # Make it a flag
        help="Run evaluate.py automatically after each successful training run.",
    )
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["fnn", "linear"],
        choices=["fnn", "linear"],
        help="List of model types to run.",
    )
    parser.add_argument(
        "--target_params",
        nargs="+",
        default=["fident", "alntmscore", "hfsp"],
        choices=["fident", "alntmscore", "hfsp"],
        help="List of target parameters to run.",
    )

    args = parser.parse_args()
    main(args)
