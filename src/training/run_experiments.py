#!/usr/bin/env python
"""
This script runs training experiments for different models, embeddings, and parameters.
It also optionally evaluates the results after each training run.

Usage:
python run_experiments.py --data_dir data/processed/sprot_pre2024 --evaluate_after_train --model_types fnn linear euclidean --target_params fident alntmscore hfsp
"""

import argparse
import subprocess
from pathlib import Path
import os
import re
from typing import Optional
import datetime  # Import datetime for timestamp
import tqdm  # Import tqdm for progress bar


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
    # Extract the last part of the data_dir path for the model subdirectory
    train_data_sub_dir = data_dir.name

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
    if args.evaluate_after_train:
        print("Evaluation after each training run: Enabled")
    # Add log directory info
    print("Detailed logs will be stored in 'run_logs' subdirectories.")
    print("\n")

    total_combinations = len(model_types) * len(target_params) * len(embedding_files)
    run_count = 0
    skipped_count = 0
    eval_success_count = 0
    eval_fail_count = 0
    error_count = 0  # Track general errors

    # Wrap the iteration with tqdm for a progress bar
    pbar = tqdm.tqdm(total=total_combinations, desc="Running Experiments", unit="run")

    # Iterate through all combinations
    for model_type in model_types:
        for param_name in target_params:
            for embedding_file_path in embedding_files:
                embedding_name = embedding_file_path.stem
                embedding_file_abs = str(embedding_file_path.resolve())

                # Update progress bar description
                pbar.set_description(
                    f"Checking: {model_type}/{param_name}/{embedding_name}"
                )

                experiment_output_dir = (
                    models_base_dir
                    / train_data_sub_dir  # Add the training data subdir
                    / model_type  # Use model_type directly
                    / param_name
                    / embedding_name
                )

                # --- Redundancy Check ---
                if experiment_output_dir.exists():
                    if any(item.is_dir() for item in experiment_output_dir.iterdir()):
                        # Removed print statement for skipping
                        # print(
                        #     f"Skipping: Output directory {experiment_output_dir} already exists with subdirectories."
                        # )
                        skipped_count += 1
                        pbar.update(1)  # Update progress bar
                        pbar.set_postfix(
                            skipped=skipped_count, errors=error_count, refresh=True
                        )
                        continue
                # ------------------------

                # --- Create Log Directory and Define Log Paths (AFTER check) ---
                log_dir = experiment_output_dir / "run_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                train_log_path = log_dir / f"train_{timestamp_str}.log"
                eval_log_path = log_dir / f"eval_{timestamp_str}.log"
                # -------------------------------------------------------------

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
                    embedding_file_abs,
                    "--data_dir",
                    str(sets_dir),
                    "--output_base_dir",
                    str(experiment_output_dir),
                    "--num_workers",
                    "10",
                    "--batch_size",
                    "4096",
                    "--learning_rate",
                    "0.0001",
                    "--max_epochs",
                    "200",
                    "--early_stopping_patience",
                    "5",
                ]

                # print(f"Executing: {' '.join(train_command)}")
                # print(f"Log file: {train_log_path}")  # Indicate log file path
                training_successful = False
                try:
                    # Run training, redirecting output to log file
                    with open(train_log_path, "a", encoding="utf-8") as log_file:
                        subprocess.run(
                            train_command,
                            check=True,  # Will raise CalledProcessError on failure
                            cwd=project_root,
                            text=True,
                            stdout=log_file,  # Redirect stdout
                            stderr=log_file,  # Redirect stderr
                        )
                    # print(
                    #     f"Successfully finished training for Model={model_type}, Param={param_name}, Embedding={embedding_name}"
                    # )
                    training_successful = True

                except subprocess.CalledProcessError as e:
                    error_count += 1
                    # Keep error message
                    tqdm.tqdm.write(  # Use tqdm.write to avoid messing up the progress bar
                        f"Error during training for {model_type}/{param_name}/{embedding_name}: Check log {train_log_path}\nDetails: {e}"
                    )
                except KeyboardInterrupt:
                    pbar.close()
                    print("\nExperiment run interrupted by user.")
                    return
                except Exception as e:
                    error_count += 1
                    # Keep error message
                    tqdm.tqdm.write(
                        f"Unexpected error during training execution for {model_type}/{param_name}/{embedding_name}: Check log {train_log_path}\nDetails: {e}"
                    )

                # --- Optional Evaluation Step ---
                run_dir_to_evaluate = None
                if training_successful and args.evaluate_after_train:
                    # print(f"Finding run directory in {experiment_output_dir}...") # Keep error finding dir, comment out info prints
                    run_dir_to_evaluate = find_latest_run_dir(experiment_output_dir)
                    if run_dir_to_evaluate:
                        # print(f"Found latest run directory: {run_dir_to_evaluate}") # Keep error finding dir, comment out info prints
                        pass
                    else:
                        print(
                            f"Error: Could not find timestamped run directory in {experiment_output_dir} after successful training."
                        )

                if run_dir_to_evaluate:
                    # Update description before evaluation
                    pbar.set_description(
                        f"Evaluating: {model_type}/{param_name}/{embedding_name}"
                    )

                    # print(f"Running evaluation for run: {run_dir_to_evaluate}")
                    eval_command = [
                        "uv",
                        "run",
                        "python",
                        str(project_root / "src" / "evaluation" / "evaluate.py"),
                        "--run_dir",
                        str(run_dir_to_evaluate),
                    ]

                    # print(f"Executing: {' '.join(eval_command)}")
                    # print(f"Log file: {eval_log_path}")  # Indicate log file path
                    try:
                        # Let evaluation output stream as well, redirecting output
                        with open(eval_log_path, "a", encoding="utf-8") as log_file:
                            subprocess.run(
                                eval_command,
                                check=True,
                                cwd=project_root,
                                text=True,
                                stdout=log_file,  # Redirect stdout
                                stderr=log_file,  # Redirect stderr
                            )
                        # print(
                        #     f"Successfully finished evaluation for {run_dir_to_evaluate}"
                        # )
                        eval_success_count += 1
                    except subprocess.CalledProcessError as e:
                        eval_fail_count += 1
                        error_count += 1  # Count eval errors too
                        # Keep error message
                        tqdm.tqdm.write(
                            f"Error during evaluation for {run_dir_to_evaluate}: Check log {eval_log_path}\nDetails: {e}"
                        )
                    except KeyboardInterrupt:
                        pbar.close()
                        print("\nEvaluation interrupted by user.")
                        return
                    except Exception as e:
                        eval_fail_count += 1
                        error_count += 1  # Count eval errors too
                        # Keep error message
                        tqdm.tqdm.write(
                            f"Unexpected error during evaluation for {run_dir_to_evaluate}: Check log {eval_log_path}\nDetails: {e}"
                        )
                elif args.evaluate_after_train and training_successful:
                    # Log failure if we expected to evaluate but couldn't find dir
                    eval_fail_count += 1
                    error_count += 1
                    # Optional: log this specific failure case
                    tqdm.tqdm.write(
                        f"Skipping evaluation: Could not determine run directory for {experiment_output_dir}"
                    )
                # Removed redundant skip messages unless training failed
                # elif args.evaluate_after_train:
                #     print(
                #         f"Skipping evaluation: Training failed for Model={model_type}, Param={param_name}, Embedding={embedding_name}"
                #     )

                # Update progress bar after processing each combination
                pbar.update(1)
                pbar.set_postfix(
                    skipped=skipped_count,
                    errors=error_count,
                    eval_ok=eval_success_count,
                    eval_fail=eval_fail_count,
                    refresh=True,
                )
                # # Removed: print("---") # Ensure commented out

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

    args = parser.parse_args()
    main(args)
