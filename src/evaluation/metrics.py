import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Optional, Callable, List, Any
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _r2_ci_from_r_bounds(r_lower: float, r_upper: float) -> tuple[float, float]:
    """Map a confidence interval on the signed correlation r to one on R².

    Squaring is monotone only on |r|, so the R² interval is derived
    carefully from the r interval ``[r_lower, r_upper]``:

      * upper = max(r_lower², r_upper²)
      * lower = 0 if the r interval brackets 0 (R² can be 0 inside it),
        else min(r_lower², r_upper²)

    This is the coherent alternative to squaring r *inside* each bootstrap
    resample, which makes the resampled R² pile against the 0 boundary when
    the true r is near zero and yields a positive-artifact lower bound.
    """
    r2_upper = max(r_lower * r_lower, r_upper * r_upper)
    if r_lower <= 0.0 <= r_upper:
        r2_lower = 0.0
    else:
        r2_lower = min(r_lower * r_lower, r_upper * r_upper)
    # Guard against floating-point spill outside [0, 1].
    r2_lower = max(0.0, min(1.0, r2_lower))
    r2_upper = max(0.0, min(1.0, r2_upper))
    return r2_lower, r2_upper


def _bootstrap_worker(args):
    """Worker function for parallel bootstrap sampling."""
    targets, predictions, stat_func, value_transform, seed = args
    np.random.seed(seed)
    n_samples = len(targets)

    # Generate bootstrap indices
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    targets_boot, predictions_boot = targets[indices], predictions[indices]

    try:
        if np.var(targets_boot) < 1e-9 or np.var(predictions_boot) < 1e-9:
            return np.nan
        else:
            stat_val, _ = stat_func(targets_boot, predictions_boot)
            return (
                value_transform(stat_val)
                if value_transform and not np.isnan(stat_val)
                else stat_val
            )
    except ValueError:
        return np.nan


def _bootstrap_stat(
    targets: np.ndarray,
    predictions: np.ndarray,
    n_bootstrap: int,
    confidence_level: float,
    stat_func: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
    stat_name: str,
    value_transform: Optional[Callable[[float], float]] = None,
    square_after_ci: bool = False,
    se_key_suffix: str = "_SE",
    ci_key_suffix_lower: str = "_CI_lower",
    ci_key_suffix_upper: str = "_CI_upper",
    use_parallel: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Helper: Performs bootstrapping for a given statistic with optional parallel processing.

    When ``square_after_ci`` is True the resampled statistic is the signed
    correlation r; the percentile CI is computed on r and then mapped to an
    R² CI via :func:`_r2_ci_from_r_bounds` (instead of squaring r inside each
    resample). ``seed`` makes both the parallel and sequential resampling
    reproducible.
    """
    n_samples = len(targets)
    rng = np.random.default_rng(seed)

    key_base = stat_name
    se_key = f"{key_base}{se_key_suffix}"
    ci_lower_key = f"{key_base}_{int(confidence_level * 100)}{ci_key_suffix_lower}"
    ci_upper_key = f"{key_base}_{int(confidence_level * 100)}{ci_key_suffix_upper}"
    results = {se_key: np.nan, ci_lower_key: np.nan, ci_upper_key: np.nan}

    # Use parallel processing for large bootstrap samples
    if use_parallel and n_bootstrap >= 100:
        try:
            n_processes = min(cpu_count(), 8)  # Limit to 8 processes max

            # Prepare arguments for parallel processing
            seeds = rng.integers(0, 2**31, n_bootstrap)
            worker_args = [
                (targets, predictions, stat_func, value_transform, seed)
                for seed in seeds
            ]

            with Pool(processes=n_processes) as pool:
                bootstrap_stat_values = list(
                    tqdm(
                        pool.imap(_bootstrap_worker, worker_args),
                        total=n_bootstrap,
                        desc=f"Bootstrap {stat_name} (parallel)",
                        unit="sample",
                    )
                )

            bootstrap_stat_values = np.array(bootstrap_stat_values)

        except Exception as e:
            print(f"Parallel bootstrap failed ({e}), falling back to sequential...")
            use_parallel = False

    # Sequential processing (fallback or for small n_bootstrap)
    if not use_parallel or n_bootstrap < 100:
        bootstrap_stat_values = np.empty(n_bootstrap)

        # Bootstrap sampling with progress bar
        for i in tqdm(range(n_bootstrap), desc=f"Bootstrap {stat_name}", unit="sample"):
            indices = rng.integers(0, n_samples, size=n_samples)
            targets_boot, predictions_boot = targets[indices], predictions[indices]
            try:
                if np.var(targets_boot) < 1e-9 or np.var(predictions_boot) < 1e-9:
                    stat_val = np.nan
                else:
                    stat_val, _ = stat_func(targets_boot, predictions_boot)
            except ValueError:
                stat_val = np.nan
            final_val = (
                value_transform(stat_val)
                if value_transform and not np.isnan(stat_val)
                else stat_val
            )
            bootstrap_stat_values[i] = final_val

    valid_bootstrap_stats = bootstrap_stat_values[~np.isnan(bootstrap_stat_values)]
    num_valid = len(valid_bootstrap_stats)

    if num_valid < n_bootstrap * 0.5:
        print(
            f"Warning: >50% bootstrap samples failed ({n_bootstrap - num_valid} failures) for {stat_name}. SE/CI unreliable."
        )

    if num_valid > 1:
        alpha = (1.0 - confidence_level) / 2.0
        if square_after_ci:
            # ``valid_bootstrap_stats`` are signed r values. Take the percentile
            # CI on r, then map it to an R² CI (zero-crossing aware). SE is the
            # spread of the squared samples, kept only as a descriptor.
            r_lower = np.percentile(valid_bootstrap_stats, alpha * 100)
            r_upper = np.percentile(valid_bootstrap_stats, (1 - alpha) * 100)
            ci_lower, ci_upper = _r2_ci_from_r_bounds(float(r_lower), float(r_upper))
            se = float(np.std(valid_bootstrap_stats**2, ddof=1))
        else:
            se = float(np.std(valid_bootstrap_stats, ddof=1))
            ci_lower = float(np.percentile(valid_bootstrap_stats, alpha * 100))
            ci_upper = float(np.percentile(valid_bootstrap_stats, (1 - alpha) * 100))
        results[se_key] = se
        results[ci_lower_key] = ci_lower
        results[ci_upper_key] = ci_upper
        print(f"{stat_name} bootstrapping complete.")
    elif num_valid <= 1:
        print(
            f"Warning: Not enough valid bootstrap samples ({num_valid}) for {stat_name} SE/CI."
        )

    return results


def calculate_regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    n_bootstrap: Optional[int] = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Calculates regression metrics, optionally bootstrapping correlation stats.

    Pass ``seed`` to make the bootstrap SE/CIs reproducible. The Pearson R² CI
    is obtained by bootstrapping the signed r and mapping the r-interval to R²
    (see :func:`_r2_ci_from_r_bounds`), not by squaring r inside each resample.
    """
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have the same length.")

    # --- Define all expected metric keys --- #
    standard_keys = [
        "MSE",
        "RMSE",
        "MAE",
        "R2",
        "Pearson",
        "Pearson_p_value",
        "Pearson_r2",
        "Spearman",
        "Spearman_p_value",
    ]
    bootstrap_configs: List[Dict[str, Any]] = [
        {
            "stat_func": pearsonr,
            "stat_name": "Pearson_r2",
            "square_after_ci": True,
        },
        {"stat_func": spearmanr, "stat_name": "Spearman", "value_transform": None},
    ]
    bootstrap_keys = []
    if n_bootstrap and n_bootstrap > 1:
        for config in bootstrap_configs:
            name = config["stat_name"]
            se_suffix = config.get("se_key_suffix", "_SE")
            ci_low_suffix = config.get("ci_key_suffix_lower", "_CI_lower")
            ci_high_suffix = config.get("ci_key_suffix_upper", "_CI_upper")
            bootstrap_keys.extend(
                [
                    f"{name}{se_suffix}",
                    f"{name}_{int(confidence_level * 100)}{ci_low_suffix}",
                    f"{name}_{int(confidence_level * 100)}{ci_high_suffix}",
                ]
            )

    # Initialize metrics dict with NaNs for all keys
    all_keys = standard_keys + bootstrap_keys
    metrics: Dict[str, float] = {k: np.nan for k in all_keys}

    # --- Handle Insufficient Data --- #
    if len(targets) < 2:
        print("Warning: Need at least two samples for metrics. Returning NaNs.")
        return metrics

    # --- Standard Metrics Calculation (overwrite NaNs) --- #
    try:  # Group basic sklearn metrics
        mse = mean_squared_error(targets, predictions)
        metrics["MSE"] = mse
        metrics["RMSE"] = np.sqrt(mse)
        metrics["MAE"] = mean_absolute_error(targets, predictions)
        metrics["R2"] = r2_score(targets, predictions)
    except ValueError:
        pass  # Keep related NaNs if basic calculation fails

    try:  # Pearson calculation
        pearson_corr, p_pearson = pearsonr(targets, predictions)
        metrics["Pearson"] = pearson_corr
        metrics["Pearson_p_value"] = p_pearson
        metrics["Pearson_r2"] = pearson_corr**2
    except ValueError:
        pass  # Keep related NaNs (e.g., constant input)

    try:  # Spearman calculation
        spearman_corr, p_spearman = spearmanr(targets, predictions)
        metrics["Spearman"] = spearman_corr
        metrics["Spearman_p_value"] = p_spearman
    except ValueError:
        pass  # Keep related NaNs (e.g., constant input)

    # --- Bootstrapping (overwrite NaNs) --- #
    if n_bootstrap and n_bootstrap > 1:
        for config in bootstrap_configs:
            bootstrap_results = _bootstrap_stat(
                targets=targets,
                predictions=predictions,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
                **config,
            )
            metrics.update(bootstrap_results)
    elif n_bootstrap == 1:
        print("Warning: n_bootstrap=1, cannot calculate SE/CI. Skipping bootstrap.")
        # Relevant keys remain NaN as initialized

    return metrics
