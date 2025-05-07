import argparse
from pathlib import Path
import subprocess
import os

# Maximum depth to search for run directories from the input_path.
# Example: models/sprot_train (0) / fnn (1) / fident (2) / prott5 (3) / timestamp (4)
MAX_SEARCH_DEPTH_FROM_INPUT = (
    5  # A little extra buffer just in case of slight variations
)


def is_valid_run_dir(path: Path) -> bool:
    """Checks if a directory is a valid run directory (contains tensorboard/hparams.yaml)."""
    if not path.is_dir():
        return False
    hparams_file = path / "tensorboard" / "hparams.yaml"
    return hparams_file.is_file()


def collect_run_dirs_recursive(
    current_path: Path,
    collected_dirs: set[Path],
    max_depth: int,
    current_depth: int = 0,
):
    """
    Recursively searches for valid run directories.
    - current_path: The directory currently being searched.
    - collected_dirs: A set to store the absolute paths of found valid run directories.
    - max_depth: Maximum recursion depth from the initial call (not from input_path, but from current_path in recursion).
    - current_depth: Current depth of recursion.
    """
    if not current_path.is_dir() or current_depth > max_depth:
        return

    # If current_path itself is a valid run directory, add it and stop searching deeper from this path.
    if is_valid_run_dir(current_path):
        collected_dirs.add(current_path.resolve())
        return  # Found a run directory, no need to look inside it for more run directories

    # If not a run directory itself, and not too deep, recurse into subdirectories.
    # The check current_depth < max_depth ensures we don't go too many levels down *from the initial input_path*.
    # If input_path is already deep, current_depth will start at 0 relative to input_path.
    if current_depth < max_depth:  # This condition allows exploring children
        for sub_dir in current_path.iterdir():
            if sub_dir.is_dir():
                # Skip known non-run-material subdirectories or hidden files/dirs
                if sub_dir.name == "__pycache__" or sub_dir.name.startswith("."):
                    continue
                if sub_dir.name == "run_logs":
                    # print(f"Explicitly skipping known directory: {sub_dir}") # Optional: for verbosity
                    continue

                # Recurse, incrementing current_depth
                collect_run_dirs_recursive(
                    sub_dir, collected_dirs, max_depth, current_depth + 1
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Flexibly finds and re-evaluates model runs to update plots and metrics. "
            "Takes an input path and intelligently finds all individual run directories "
            "(containing tensorboard/hparams.yaml) beneath it, then calls "
            "src/unknown_unknowns/evaluate.py for each."
        )
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="The input path to search for model runs. Can be a top-level directory "
        "(e.g., models/sprot_train), a mid-level directory "
        "(e.g., models/sprot_train/fnn/fident), or a specific run directory.",
    )
    parser.add_argument(
        "--evaluate_script_path",
        type=Path,
        default=Path("src/unknown_unknowns/evaluate.py"),
        help="Path to the evaluate.py script (default: src/unknown_unknowns/evaluate.py).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands that would be executed without actually running them.",
    )

    args = parser.parse_args()

    if not args.input_path.exists():  # Check if path exists at all
        print(f"Error: Input path not found: {args.input_path}")
        return

    if not args.input_path.is_dir():  # If it exists but not a dir (unless it *is* a run dir itself, handled by collect)
        # If it's a file, it cannot be a run directory in the sense of containing sub-elements
        # The recursive function will handle if it's a valid run_dir path that is_file() (not possible with current is_valid_run_dir)
        # For now, we assume input_path should broadly be a directory to search within or be a run_dir itself.
        if not is_valid_run_dir(
            args.input_path
        ):  # Check if the file path itself could be a run dir (not possible by current logic)
            print(
                f"Error: Input path {args.input_path} is not a directory and not recognized as a direct run directory structure."
            )
            return

    if not args.evaluate_script_path.is_file():
        print(f"Error: Evaluation script not found at {args.evaluate_script_path}")
        return

    print(f"Searching for run directories under or at: {args.input_path}")

    collected_run_dirs_set = set()
    collect_run_dirs_recursive(
        args.input_path, collected_run_dirs_set, MAX_SEARCH_DEPTH_FROM_INPUT
    )

    actual_run_dirs = sorted(list(collected_run_dirs_set))

    if not actual_run_dirs:
        print(f"No valid run directories found under or at {args.input_path}.")
        return

    print(f"Found {len(actual_run_dirs)} run director(y/ies) to evaluate:")
    for rd in actual_run_dirs:
        print(f"  {rd}")

    for run_dir in actual_run_dirs:
        command = [
            "uv",
            "run",
            "python",
            str(args.evaluate_script_path.resolve()),
            "--run_dir",
            str(run_dir.resolve()),  # run_dir is already resolved from the collection
        ]

        print(f"\nExecuting: {' '.join(command)}")
        if args.dry_run:
            print("(Dry run - command not executed)")
        else:
            try:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=os.getcwd(),
                )
                print("STDOUT:")
                print(process.stdout)
                if process.stderr:
                    print("STDERR:")
                    print(process.stderr)
                if process.returncode != 0:
                    print(
                        f"Error running evaluation for {run_dir}. Return code: {process.returncode}"
                    )
                else:
                    print(f"Successfully evaluated {run_dir}")
            except Exception as e:
                print(f"An exception occurred while evaluating {run_dir}: {e}")


if __name__ == "__main__":
    main()
