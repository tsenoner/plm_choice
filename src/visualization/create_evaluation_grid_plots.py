import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import argparse
from pathlib import Path


def create_grid_plots(input_path_str: str, output_dir_str: str):
    """
    Generates grid plots from individual evaluation result images.
    Searches for plot files within the input_path structure:
    ... / <model_type> / <param_name> / <embedding_name> / <timestamp> / evaluation_results / *_results.png
    And groups them by (model_type, param_name) for grid generation.
    """
    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)

    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        return

    # Ensure output directory exists and is writable
    output_dir.mkdir(parents=True, exist_ok=True)
    # The following check is a bit more robust for some edge cases like permission issues after mkdir.
    can_write_to_output = False
    try:
        with open(output_dir / ".check_writable", "w") as f:
            f.write("test")
        (output_dir / ".check_writable").unlink()
        can_write_to_output = True
        if not any(output_dir.iterdir()):  # Check if it was newly created
            print(f"Created output directory: {output_dir}")
        else:
            print(f"Output directory already exists or was just created: {output_dir}")
    except OSError:
        print(
            f"Error: Output directory {output_dir} could not be created or is not writable."
        )
        return
    if (
        not can_write_to_output
    ):  # Should have been caught by OSError, but as a safeguard.
        print(
            f"Error: Failed to confirm writability for output directory {output_dir}."
        )
        return

    all_plot_infos = []

    print(f"Searching for plot files under: {input_path}")
    # Find all potential plot files: .../evaluation_results/*_results.png
    found_plot_files = []
    if input_path.is_file():
        if (
            input_path.name.endswith("_results.png")
            and input_path.parent.name == "evaluation_results"
        ):
            found_plot_files = [input_path]
    elif input_path.is_dir():
        found_plot_files = list(input_path.rglob("**/evaluation_results/*_results.png"))

    if not found_plot_files:
        print(
            f"No '*_results.png' files found within 'evaluation_results' subdirectories under {input_path}"
        )
        return

    print(f"Found {len(found_plot_files)} potential plot files. Parsing paths...")

    for plot_file in found_plot_files:
        try:
            # Expected structure: ... / <model_type> / <param_name> / <embedding_name> / <timestamp> / evaluation_results / plot.png
            eval_results_dir = plot_file.parent
            timestamp_dir = eval_results_dir.parent
            embedding_dir = timestamp_dir.parent
            param_dir = embedding_dir.parent
            model_dir = (
                param_dir.parent
            )  # This could be <train_data_sub_dir> if structure is deeper

            # Validate timestamp directory name format
            if not (len(timestamp_dir.name) == 15 and timestamp_dir.name[8] == "_"):
                # print(f"Skipping {plot_file}: Parent directory '{timestamp_dir.name}' doesn't look like a timestamp.")
                continue

            # Ensure component names are not empty
            if not all(
                [model_dir.name, param_dir.name, embedding_dir.name, timestamp_dir.name]
            ):
                # print(f"Skipping {plot_file}: One of the path components (model, param, embedding, timestamp) is empty.")
                continue

            all_plot_infos.append(
                {
                    "path": plot_file,
                    "model_type": model_dir.name,  # If input_path is high, this might be train_data_sub_dir
                    "param_name": param_dir.name,  # And this would be model_type, etc.
                    "embedding_name": embedding_dir.name,  # This should be robustly embedding name
                    "timestamp": timestamp_dir.name,
                    # For more accurate model/param if input_path is high (e.g. "models/")
                    # we might need to adjust which .name we pick based on depth from a known root
                    # or by looking for known model/param names if available.
                    # Current logic assumes a fixed depth from plot file to these components.
                    # Let's refine this: The actual model_type is 3 levels above timestamp, param is 2, embedding is 1.
                    "actual_model_type": param_dir.parent.name,  # model_dir in the old context
                    "actual_param_name": embedding_dir.parent.name,  # param_dir in the old context
                    "actual_embedding_name": timestamp_dir.parent.name,  # embedding_dir in the old context
                }
            )
        except (
            IndexError
        ):  # Path too short, e.g. plot_file.parent.parent is already root
            print(
                f"Warning: Could not parse expected path structure for {plot_file} (path too short). Skipping."
            )
            continue
        except Exception as e:
            print(
                f"Warning: Could not parse path structure for {plot_file}: {e}. Skipping."
            )
            continue

    if not all_plot_infos:
        print(f"No valid plot structures found after parsing paths from {input_path}")
        return

    # Filter for latest timestamp for each (actual_model_type, actual_param_name, actual_embedding_name)
    latest_plots = {}
    for info in all_plot_infos:
        key = (
            info["actual_model_type"],
            info["actual_param_name"],
            info["actual_embedding_name"],
        )
        if (
            key not in latest_plots
            or info["timestamp"] > latest_plots[key]["timestamp"]
        ):
            latest_plots[key] = {"path": info["path"], "timestamp": info["timestamp"]}

    # Group by (actual_model_type, actual_param_name)
    grouped_for_grid = {}
    for (model_t, param_n, embedding_n), plot_data in latest_plots.items():
        group_key = (model_t, param_n)
        if group_key not in grouped_for_grid:
            grouped_for_grid[group_key] = []
        # Storing the actual embedding name derived from path parsing
        grouped_for_grid[group_key].append(
            {"embedding": embedding_n, "path": plot_data["path"]}
        )

    if not grouped_for_grid:
        print("No plots to arrange into grids after filtering and grouping.")
        return

    print(f"Generating {len(grouped_for_grid)} grid plot(s)...")
    for (model_type_name, param_name_val), image_files_data in grouped_for_grid.items():
        image_files_data.sort(key=lambda x: x["embedding"])

        num_images = len(image_files_data)
        if num_images == 0:
            continue

        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)

        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 5 + 1, rows * 5 + 1), squeeze=False
        )
        axes = axs.flatten()

        for i, img_data in enumerate(image_files_data):
            try:
                img = mpimg.imread(img_data["path"])
                axes[i].imshow(img)
                axes[i].set_title(img_data["embedding"], fontsize=22, y=1.025)
                axes[i].axis("off")
            except FileNotFoundError:
                print(f"Error: Image file not found at {img_data['path']}")
                axes[i].set_title(
                    f"{img_data['embedding']}\n(File not found)", fontsize=10
                )
                axes[i].axis("off")
            except Exception as e:
                print(f"Error loading image {img_data['path']}: {e}")
                axes[i].set_title(
                    f"{img_data['embedding']}\n(Error loading)", fontsize=10
                )
                axes[i].axis("off")

        for j in range(num_images, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            f"Model: {model_type_name} | Parameter: {param_name_val}",
            fontsize=26,
        )
        # Use tight_layout for overall fit.
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])

        output_filename = output_dir / f"grid_{model_type_name}_{param_name_val}.png"
        try:
            plt.savefig(output_filename)
            print(f"Saved grid plot: {output_filename}")
        except Exception as e:
            print(f"Error saving plot {output_filename}: {e}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create grid plots from evaluation image results. "
            "Searches for '*_results.png' files within 'evaluation_results' subdirectories "
            "under the input_path. The script expects a structure like: "
            ".../<model_type>/<param_name>/<embedding_name>/<timestamp>/evaluation_results/."
        )
    )
    parser.add_argument(
        "--input_path",  # Renamed from base_results_dir
        type=str,
        default="models/sprot_train",
        help=(
            "Input path to search for model run outputs. Can be a high-level directory "
            "(e.g., models/sprot_train), a specific model/parameter directory, "
            "or even a direct path to an evaluation_results directory or plot file."
            "(default: models/sprot_train)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/summary_grids",  # User's preferred default
        help="Directory where the output grid plots will be saved (default: out/summary_grids).",
    )
    args = parser.parse_args()

    create_grid_plots(args.input_path, args.output_dir)
