from pathlib import Path
import numpy as np
import shutil  # Added for copying the file
from unknown_unknowns.visualization.plot import plot_true_vs_predicted


def test_plot_generation(tmp_path):
    # Generate some sample data
    np.random.seed(42)  # for reproducibility
    num_points = 50000
    targets = np.linspace(0, 1, num_points)
    predictions = targets + np.random.normal(
        0, 0.1, num_points
    )  # predictions with some noise

    # Define the output file path using pytest's tmp_path fixture
    output_directory = tmp_path / "plot_outputs"
    output_directory.mkdir()  # Create subdirectory within tmp_path
    output_file = output_directory / "test_plot_generation.png"

    # Call the function
    plot_true_vs_predicted(
        targets, predictions, output_file, title="Test: Plot Generation"
    )

    assert output_file.exists(), f"Plot file was not generated at {output_file}"

    # For inspection: Print path and copy to a persistent location
    print(f"Temporary plot saved to: {output_file.resolve()}")

    persistent_output_dir = Path("test_outputs")
    persistent_output_dir.mkdir(parents=True, exist_ok=True)
    inspect_file_path = persistent_output_dir / "inspect_latest_plot.png"
    shutil.copy(output_file, inspect_file_path)
    print(f"Plot copied for inspection to: {inspect_file_path.resolve()}")
