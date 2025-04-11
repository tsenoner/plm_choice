import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_regression_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    """Calculates standard regression metrics."""
    r2 = r2_score(targets, predictions)
    pearson_corr, _ = pearsonr(targets, predictions)  # Note: Order matters for pearsonr
    spearman_corr, _ = spearmanr(targets, predictions)

    return {
        "MSE": mean_squared_error(targets, predictions),
        "RMSE": np.sqrt(mean_squared_error(targets, predictions)),
        "MAE": mean_absolute_error(targets, predictions),
        "R2": r2,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
    }
