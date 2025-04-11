from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_true_vs_predicted(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_file: Path,
    metrics: Optional[dict[str, float]] = None,
    title: str = "True vs Predicted",
):
    """Generates a scatter plot of true vs predicted values."""
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5)

    # Determine plot limits based on data range
    min_val = min(np.min(targets), np.min(predictions)) * 0.95
    max_val = max(np.max(targets), np.max(predictions)) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")

    plot_title = title
    if metrics:
        pearson = metrics.get("Pearson", float("nan"))
        r2 = metrics.get("R2", float("nan"))
        plot_title += f"\nPearson: {pearson:.4f}, R²: {r2:.4f}"

    plt.title(plot_title)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")  # Ensure square plot
    plt.legend()
    plt.savefig(output_file)
    plt.close()
