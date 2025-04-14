import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_regression_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    """Calculates standard regression metrics, including Pearson r-squared.
    Assumes inputs (targets, predictions) do not contain NaNs.
    """
    # Calculate correlations first, as r2 depends on r
    pearson_corr, _ = pearsonr(targets, predictions)
    spearman_corr, _ = spearmanr(targets, predictions)

    return {
        "MSE": mean_squared_error(targets, predictions),
        "RMSE": np.sqrt(mean_squared_error(targets, predictions)),
        "MAE": mean_absolute_error(targets, predictions),
        "R2": r2_score(targets, predictions),  # Coefficient of Determination
        "Pearson": pearson_corr,
        "Pearson_r2": pearson_corr**2,  # Pearson r-squared
        "Spearman": spearman_corr,
    }
