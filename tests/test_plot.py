import numpy as np

from visualization.plot_utils import plot_true_vs_predicted


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
    assert output_file.stat().st_size > 0, "Plot file is empty"


def test_plot_generation_with_metrics(tmp_path):
    """The metrics annotation is an optional code path — exercise it too.

    plot_true_vs_predicted only reads the ``Pearson_r2`` / ``Pearson_r2_SE``
    keys, so use those rather than arbitrary metric names.
    """
    rng = np.random.default_rng(0)
    targets = np.linspace(0, 1, 1000)
    predictions = targets + rng.normal(0, 0.05, 1000)
    output_file = tmp_path / "with_metrics.png"

    plot_true_vs_predicted(
        targets,
        predictions,
        output_file,
        metrics={"Pearson_r2": 0.97, "Pearson_r2_SE": 0.01},
        title="Test: Plot With Metrics",
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0
