from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm


def plot_true_vs_predicted(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_file: Path,
    metrics: Optional[dict[str, float]] = None,
    title: str = "True vs Predicted",
):
    """Generates a scatter plot of true vs predicted values."""
    plt.figure(figsize=(8, 8))

    ax = plt.gca()
    ax.set_axisbelow(True)  # Ensure grid is drawn below other artists
    ax.grid(True, alpha=0.6)  # Draw grid behind other elements and set alpha

    hb = ax.hexbin(
        targets,
        predictions,
        gridsize=100,
        cmap="viridis",
        mincnt=1,
        alpha=0.6,
        zorder=2,
        norm=LogNorm(),
    )  # Hexagonal binning to show point density
    cb = plt.colorbar(hb, label="log(counts)")
    cb.set_label("log(counts)", fontsize=20)
    cb.ax.tick_params(labelsize=18)

    # Calculate and plot regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(targets, predictions)

    # Define x-range for the regression line and CI, extending slightly beyond data range for better visuals.
    line_plot_x_min, line_plot_x_max = np.min(targets), np.max(targets)
    if line_plot_x_min != line_plot_x_max:
        padding = (line_plot_x_max - line_plot_x_min) * 0.05
        line_plot_x_min -= padding
        line_plot_x_max += padding
    else:  # Handle case where all target values are the same, providing a small default range
        padding_val = 0.1 * abs(line_plot_x_min) if line_plot_x_min != 0 else 0.1
        line_plot_x_min -= padding_val
        line_plot_x_max += padding_val

    line_x = np.array([line_plot_x_min, line_plot_x_max])
    line_y = slope * line_x + intercept
    ax.plot(
        line_x, line_y, color="black", linewidth=2, label="Regression Line", zorder=4
    )  # Ensure regression line is plotted on top

    # Calculate and plot the 95% confidence interval for the regression line.
    # This involves fitting a linear model, then using its parameters and residuals
    # to compute the CI boundaries based on t-distribution.
    def linear_func(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(linear_func, targets, predictions)
    residuals = predictions - linear_func(targets, *popt)
    s_err = np.sqrt(np.sum(residuals**2) / (len(targets) - 2))
    t_val = stats.t.ppf(0.975, len(targets) - 2)
    ci_x = np.linspace(line_plot_x_min, line_plot_x_max, 100)
    ci_y = linear_func(ci_x, *popt)
    ci_delta = (
        t_val
        * s_err
        * np.sqrt(
            1 / len(targets)
            + (ci_x - np.mean(targets)) ** 2 / np.sum((targets - np.mean(targets)) ** 2)
        )
    )
    ax.fill_between(
        ci_x,
        ci_y - ci_delta,
        ci_y + ci_delta,
        color="gray",
        alpha=0.5,
        label="95% CI",
        zorder=3,
    )  # Ensure CI is between hexbins and regression line

    ax.set_xlabel("True Values", fontsize=20)
    ax.set_ylabel("Predicted Values", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=18)

    plot_title = title

    if metrics:
        pearson_r_squared_val = metrics.get("Pearson_r2", float("nan"))
        pearson_r_squared_se = metrics.get("Pearson_r2_SE", float("nan"))

        # If both Pearson R-squared value and its SE are found, add to title
        if not np.isnan(pearson_r_squared_val) and not np.isnan(pearson_r_squared_se):
            plot_title += f"\nPearson R²: {pearson_r_squared_val:.4f} ± {pearson_r_squared_se:.4f}"
        elif not np.isnan(pearson_r_squared_val):  # Fallback: if only value is found
            plot_title += f"\nPearson R²: {pearson_r_squared_val:.4f}"

    ax.set_title(plot_title, fontsize=24)
    ax.legend(fontsize=18)
    plt.tight_layout(pad=0.5)
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.05)
    plt.close()
