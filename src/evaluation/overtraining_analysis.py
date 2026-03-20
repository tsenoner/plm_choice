# --- Ivan infrastructure (2026-03-20) ---
"""Overtraining analysis for pLM Choice training runs.

Investigates whether some pLMs show signs of overtraining during the
downstream prediction task — manifested as train/val loss divergence or
unusually peaked distance distributions. This is distinct from pLM
pre-training overfitting (the "PP2 problem"); here we ask whether certain
embeddings make the *downstream* model more prone to overfitting.

Approach:
    1. Recursively discover completed training runs under a models directory,
       extracting embedding name, model type, and target parameter from the
       directory hierarchy and/or hparams.yaml.
    2. Parse TensorBoard event files (via tbparse) to extract per-step
       train_loss and val_loss trajectories.
    3. Compute overtraining indicators per run:
       - Final train/val loss gap (larger = more overfit)
       - Late-stage divergence: val_loss trending up while train_loss trends
         down in the final 20% of training
       - Early stopping ratio: stopped_epoch / max_epochs (low = early stop
         due to overfitting or fast convergence; must correlate with gap to
         distinguish)
    4. Aggregate across runs: for each embedding x model_type x param, report
       mean overtraining metrics.
    5. Optionally check distance distribution kurtosis per embedding — an
       overtrained pLM may produce overly peaked distance distributions.
    6. Output: summary parquet + CSV + diagnostic plots (train/val loss curves,
       overtraining heatmap by pLM x param).

For Tobi's review: the key output is the heatmap showing which pLM x param
combinations exhibit val/train divergence. If any pLM consistently overfits
across params, that's a red flag worth mentioning in the paper. The kurtosis
check is secondary — it tests a different hypothesis (pLM-level
memorization during pre-training, not downstream overfitting).

Usage:
    uv run python src/evaluation/overtraining_analysis.py \\
        --models_dir models/sprot_pre2024 \\
        --pairs_parquet data/processed/sprot_pre2024/sets/test_with_distances.parquet \\
        --output_dir out/overtraining

    # Quick test on a subset:
    uv run python src/evaluation/overtraining_analysis.py \\
        --models_dir models/sprot_pre2024 \\
        --output_dir out/overtraining \\
        --sample_size 10

Optional dependency:
    tbparse — required for parsing TensorBoard event files.  If not
    installed, the script falls back to hparams-only analysis (no
    train/val loss curves).  Install with: ``uv add --dev tbparse``
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#                          DATA STRUCTURES
# ---------------------------------------------------------------------------


@dataclass
class RunInfo:
    """Metadata for a single training run discovered on disk."""

    path: Path
    embedding: str
    model_type: str
    param: str
    hparams: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"RunInfo({self.model_type}/{self.param}/{self.embedding})"


@dataclass
class TrainingCurves:
    """Train and validation loss arrays extracted from a single run."""

    steps: np.ndarray
    train_loss: np.ndarray
    val_loss: np.ndarray
    val_steps: np.ndarray  # validation may be logged at different steps


@dataclass
class OvertrainingMetrics:
    """Computed overtraining indicators for a single run."""

    embedding: str
    model_type: str
    param: str
    run_path: str

    # Core metrics
    final_train_loss: float
    final_val_loss: float
    train_val_gap: float  # val - train; positive = overfit
    train_val_ratio: float  # val / train; >1 = overfit

    # Late-stage divergence (final 20% of training)
    late_train_slope: float  # negative = still decreasing
    late_val_slope: float  # positive = increasing = overfit
    diverging: bool  # True if val up AND train down

    # Early stopping
    stopped_epoch: int
    max_epochs: int
    early_stop_ratio: float  # stopped / max; 1.0 = ran to completion

    # Best vs final
    best_val_loss: float
    best_val_step: int
    total_steps: int


# ---------------------------------------------------------------------------
#                          RUN DISCOVERY
# ---------------------------------------------------------------------------


def _load_hparams(run_dir: Path) -> Dict[str, Any]:
    """Load hyperparameters from the best available source in a run directory.

    Checks (in order): tensorboard/hparams.yaml, hparams.yaml at root,
    then wandb config.yaml.
    """
    # 1. tensorboard/hparams.yaml (most common for PL runs)
    tb_hparams = run_dir / "tensorboard" / "hparams.yaml"
    if tb_hparams.is_file():
        with open(tb_hparams) as f:
            return yaml.safe_load(f) or {}

    # 2. root hparams.yaml (euclidean baseline)
    root_hparams = run_dir / "hparams.yaml"
    if root_hparams.is_file():
        with open(root_hparams) as f:
            return yaml.safe_load(f) or {}

    # 3. wandb config.yaml
    wandb_dir = run_dir / "wandb"
    if wandb_dir.is_dir():
        config_files = list(wandb_dir.glob("**/config.yaml"))
        if config_files:
            latest = max(config_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                raw = yaml.safe_load(f) or {}
            # wandb wraps values in {"value": ...} dicts
            return {
                k: v["value"] if isinstance(v, dict) and "value" in v else v
                for k, v in raw.items()
            }

    return {}


def _infer_run_metadata(run_dir: Path, hparams: Dict[str, Any]) -> Optional[RunInfo]:
    """Infer embedding, model_type, param from hparams or directory structure.

    Directory convention:
        models/{dataset}/{model_type}/{param}/{embedding}/{timestamp}/
    """
    model_type = hparams.get("model_type")
    param = hparams.get("param_name")
    embedding = hparams.get("embedding_name")

    # Try to extract embedding from embedding_file path
    if not embedding and "embedding_file" in hparams:
        embedding = Path(str(hparams["embedding_file"])).stem

    # Fall back to directory structure
    parts = run_dir.parts
    if len(parts) >= 4:
        # Walk backwards: current = timestamp, -1 = embedding, -2 = param,
        # -3 = model_type, -4 = dataset
        if not embedding:
            embedding = parts[-1] if not model_type else parts[-1]
        # If run_dir is a timestamp subdir, shift by one
        if not model_type:
            # Heuristic: if the directory name looks like a timestamp or has
            # tensorboard/ inside, the parent is the embedding dir
            if (run_dir / "tensorboard").is_dir() or (run_dir / "checkpoints").is_dir():
                # This IS the run dir; parent structure = embedding/param/model/dataset
                if not embedding:
                    embedding = run_dir.parent.name
                if not param:
                    param = run_dir.parent.parent.name
                if not model_type:
                    model_type = run_dir.parent.parent.parent.name

    if not all([embedding, model_type, param]):
        logger.debug(
            f"Could not infer full metadata for {run_dir}: "
            f"embedding={embedding}, model_type={model_type}, param={param}"
        )
        return None

    return RunInfo(
        path=run_dir,
        embedding=embedding,
        model_type=model_type,
        param=param,
        hparams=hparams,
    )


def _is_run_dir(path: Path) -> bool:
    """Check if a directory looks like a training run directory."""
    if not path.is_dir():
        return False
    # Must have tensorboard events OR wandb logs OR checkpoints
    has_tb = (path / "tensorboard").is_dir()
    has_wandb = (path / "wandb").is_dir()
    has_ckpt = (path / "checkpoints").is_dir()
    has_hparams = (path / "hparams.yaml").is_file()
    has_tb_hparams = (path / "tensorboard" / "hparams.yaml").is_file()
    return has_tb_hparams or has_hparams or (has_tb and has_ckpt) or (has_wandb and has_ckpt)


def discover_runs(
    models_dir: Path, max_depth: int = 6
) -> List[RunInfo]:
    """Recursively discover training runs under a models directory.

    Walks the directory tree up to max_depth levels, identifies run
    directories by the presence of tensorboard/hparams.yaml or
    checkpoints/, loads metadata, and returns a list of RunInfo objects.

    Parameters
    ----------
    models_dir : Path
        Root directory to search (e.g., models/sprot_pre2024).
    max_depth : int
        Maximum directory depth to search from models_dir.

    Returns
    -------
    List[RunInfo]
        Discovered runs with metadata, sorted by (embedding, model_type, param).
    """
    if not models_dir.is_dir():
        logger.error(f"Models directory not found: {models_dir}")
        return []

    runs: List[RunInfo] = []
    visited: Set[Path] = set()

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth or not current.is_dir():
            return

        resolved = current.resolve()
        if resolved in visited:
            return
        visited.add(resolved)

        if _is_run_dir(current):
            hparams = _load_hparams(current)
            info = _infer_run_metadata(current, hparams)
            if info is not None:
                runs.append(info)
            return  # Don't recurse into run directories

        # Skip hidden dirs and __pycache__
        for child in sorted(current.iterdir()):
            if child.name.startswith(".") or child.name == "__pycache__":
                continue
            if child.is_dir():
                _walk(child, depth + 1)

    _walk(models_dir, 0)

    # Sort for deterministic ordering
    runs.sort(key=lambda r: (r.embedding, r.model_type, r.param))
    logger.info(f"Discovered {len(runs)} training runs under {models_dir}")
    return runs


# ---------------------------------------------------------------------------
#                     TRAINING CURVE PARSING
# ---------------------------------------------------------------------------


def _find_tfevents(run_dir: Path) -> List[Path]:
    """Find all TensorBoard event files in a run directory."""
    candidates: List[Path] = []

    # Check tensorboard/ subdirectory (PL default)
    tb_dir = run_dir / "tensorboard"
    if tb_dir.is_dir():
        candidates.extend(tb_dir.rglob("events.out.tfevents.*"))

    # Also check lightning_logs/ (alternate PL structure)
    ll_dir = run_dir / "lightning_logs"
    if ll_dir.is_dir():
        candidates.extend(ll_dir.rglob("events.out.tfevents.*"))

    # Check wandb run dirs
    wandb_dir = run_dir / "wandb"
    if wandb_dir.is_dir():
        candidates.extend(wandb_dir.rglob("events.out.tfevents.*"))

    return sorted(candidates)


def parse_training_curves(run_dir: Path) -> Optional[TrainingCurves]:
    """Parse TensorBoard event files to extract train_loss and val_loss per step.

    Uses tbparse (SummaryReader) for efficient event parsing. Falls back to
    manual scalar extraction if tbparse is not available.

    Parameters
    ----------
    run_dir : Path
        Path to the training run directory.

    Returns
    -------
    Optional[TrainingCurves]
        Extracted loss curves, or None if parsing fails.
    """
    event_files = _find_tfevents(run_dir)
    if not event_files:
        logger.debug(f"No TensorBoard event files found in {run_dir}")
        return None

    # Use the directory containing events, not individual files
    # tbparse SummaryReader expects a log directory
    log_dirs = list({ef.parent for ef in event_files})

    try:
        from tbparse import SummaryReader

        # Concatenate scalars from all log directories
        all_frames = []
        for log_dir in log_dirs:
            try:
                reader = SummaryReader(str(log_dir), pivot=False)
                scalars = reader.scalars
                if scalars is not None and len(scalars) > 0:
                    all_frames.append(scalars)
            except Exception as e:
                logger.debug(f"tbparse could not read {log_dir}: {e}")
                continue

        if not all_frames:
            logger.debug(f"No scalar data found in {run_dir}")
            return None

        import pandas as pd

        df = pd.concat(all_frames, ignore_index=True)

        # Extract train and val loss
        # PL logs: "train_loss" (per-step) and "val_loss" (per-validation)
        train_df = df[df["tag"].str.contains("train_loss", case=False, na=False)]
        val_df = df[df["tag"].str.contains("val_loss", case=False, na=False)]

        if train_df.empty and val_df.empty:
            # Try alternate tag names: "train/loss", "validation/loss"
            train_df = df[df["tag"].str.contains("train", case=False, na=False) &
                         df["tag"].str.contains("loss", case=False, na=False)]
            val_df = df[df["tag"].str.contains("val", case=False, na=False) &
                       df["tag"].str.contains("loss", case=False, na=False)]

        if train_df.empty and val_df.empty:
            logger.debug(f"No train/val loss tags found in {run_dir}. Tags: {df['tag'].unique()[:10]}")
            return None

        # Sort by step and extract arrays
        train_df = train_df.sort_values("step")
        val_df = val_df.sort_values("step")

        train_steps = train_df["step"].to_numpy()
        train_loss = train_df["value"].to_numpy()
        val_steps = val_df["step"].to_numpy() if not val_df.empty else np.array([])
        val_loss = val_df["value"].to_numpy() if not val_df.empty else np.array([])

        # Require at least a few data points
        if len(train_loss) < 3 and len(val_loss) < 3:
            logger.debug(f"Too few data points in {run_dir}")
            return None

        return TrainingCurves(
            steps=train_steps,
            train_loss=train_loss,
            val_steps=val_steps,
            val_loss=val_loss,
        )

    except ImportError:
        logger.warning(
            "tbparse not installed. Install with: pip install tbparse. "
            "Falling back to hparams-only analysis."
        )
        return None
    except Exception as e:
        logger.warning(f"Failed to parse training curves from {run_dir}: {e}")
        return None


# ---------------------------------------------------------------------------
#                   OVERTRAINING METRIC COMPUTATION
# ---------------------------------------------------------------------------


def _linear_slope(values: np.ndarray) -> float:
    """Fit a simple linear regression and return the slope."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.RankWarning)
        coeffs = np.polyfit(x, values, 1)
    return float(coeffs[0])


def compute_overtraining_metrics(
    run: RunInfo,
    curves: Optional[TrainingCurves],
) -> OvertrainingMetrics:
    """Compute overtraining indicators from training curves and run metadata.

    If curves are unavailable (e.g., euclidean baseline, missing events),
    returns a metrics object with NaN values for curve-derived fields.

    Parameters
    ----------
    run : RunInfo
        Run metadata.
    curves : Optional[TrainingCurves]
        Parsed training curves, or None.

    Returns
    -------
    OvertrainingMetrics
        Computed indicators.
    """
    max_epochs = int(run.hparams.get("max_epochs", 100))

    # Defaults for missing curves
    nan = float("nan")
    defaults = dict(
        embedding=run.embedding,
        model_type=run.model_type,
        param=run.param,
        run_path=str(run.path),
        final_train_loss=nan,
        final_val_loss=nan,
        train_val_gap=nan,
        train_val_ratio=nan,
        late_train_slope=nan,
        late_val_slope=nan,
        diverging=False,
        stopped_epoch=0,
        max_epochs=max_epochs,
        early_stop_ratio=nan,
        best_val_loss=nan,
        best_val_step=0,
        total_steps=0,
    )

    if curves is None:
        return OvertrainingMetrics(**defaults)

    # --- Extract final values ---
    train_loss = curves.train_loss
    val_loss = curves.val_loss

    has_train = len(train_loss) > 0
    has_val = len(val_loss) > 0

    final_train = float(train_loss[-1]) if has_train else nan
    final_val = float(val_loss[-1]) if has_val else nan

    # Train/val gap
    if has_train and has_val:
        gap = final_val - final_train
        ratio = final_val / final_train if final_train != 0 else nan
    else:
        gap = nan
        ratio = nan

    # --- Late-stage divergence (final 20% of training steps) ---
    late_frac = 0.2

    if has_train and len(train_loss) >= 5:
        n_late = max(2, int(len(train_loss) * late_frac))
        late_train = train_loss[-n_late:]
        late_train_slope = _linear_slope(late_train)
    else:
        late_train_slope = nan

    if has_val and len(val_loss) >= 5:
        n_late_val = max(2, int(len(val_loss) * late_frac))
        late_val = val_loss[-n_late_val:]
        late_val_slope = _linear_slope(late_val)
    else:
        late_val_slope = nan

    # Divergence = val going up while train going down
    diverging = False
    if not (np.isnan(late_train_slope) or np.isnan(late_val_slope)):
        diverging = (late_val_slope > 0) and (late_train_slope < 0)

    # --- Early stopping ---
    total_steps = int(curves.steps[-1]) if has_train else 0

    # Estimate stopped epoch from hparams if available
    # PL early stopping sets trainer.current_epoch
    stopped_epoch = int(run.hparams.get("stopped_epoch", 0))
    if stopped_epoch == 0 and total_steps > 0:
        # Rough estimate: total_steps / steps_per_epoch
        batch_size = int(run.hparams.get("batch_size", 1024))
        # We don't know dataset size, so use the training step count heuristic
        # If val_check_interval is 0.2, there are ~5 val checks per epoch
        val_check_interval = float(run.hparams.get("val_check_interval", 0.2))
        if has_val and len(val_loss) > 1 and val_check_interval > 0:
            # Number of val checks ~ len(val_loss)
            # epochs ~ val_checks * val_check_interval
            stopped_epoch = max(1, int(len(val_loss) * val_check_interval))
        else:
            stopped_epoch = max_epochs  # assume ran to completion

    early_stop_ratio = stopped_epoch / max_epochs if max_epochs > 0 else nan

    # --- Best validation loss ---
    if has_val:
        best_val_idx = int(np.argmin(val_loss))
        best_val_loss = float(val_loss[best_val_idx])
        best_val_step = int(curves.val_steps[best_val_idx]) if len(curves.val_steps) > best_val_idx else 0
    else:
        best_val_loss = nan
        best_val_step = 0

    return OvertrainingMetrics(
        embedding=run.embedding,
        model_type=run.model_type,
        param=run.param,
        run_path=str(run.path),
        final_train_loss=final_train,
        final_val_loss=final_val,
        train_val_gap=gap,
        train_val_ratio=ratio,
        late_train_slope=late_train_slope,
        late_val_slope=late_val_slope,
        diverging=diverging,
        stopped_epoch=stopped_epoch,
        max_epochs=max_epochs,
        early_stop_ratio=early_stop_ratio,
        best_val_loss=best_val_loss,
        best_val_step=best_val_step,
        total_steps=total_steps,
    )


# ---------------------------------------------------------------------------
#                     DISTANCE DISTRIBUTION KURTOSIS
# ---------------------------------------------------------------------------


def compute_distance_kurtosis(
    pairs_df: pl.DataFrame,
    dist_columns: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute kurtosis and related stats for distance distributions per embedding.

    An overtrained pLM may produce overly peaked (high kurtosis) distance
    distributions because it maps many protein pairs to very similar
    representations.

    Parameters
    ----------
    pairs_df : pl.DataFrame
        DataFrame with distance columns (dist_*).
    dist_columns : Optional[List[str]]
        Specific distance columns to analyze. If None, auto-detects all
        columns matching the dist_* pattern.

    Returns
    -------
    Dict[str, Dict[str, float]]
        For each embedding name: kurtosis, skewness, mean, std, min, max.
    """
    from scipy import stats as sp_stats

    if dist_columns is None:
        dist_columns = [c for c in pairs_df.columns if c.startswith("dist_")]

    if not dist_columns:
        logger.warning("No distance columns found in pairs DataFrame")
        return {}

    results: Dict[str, Dict[str, float]] = {}

    for col in dist_columns:
        if col not in pairs_df.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue

        values = pairs_df[col].drop_nulls().to_numpy().astype(np.float64)
        if len(values) < 4:
            logger.warning(f"{col}: too few values ({len(values)}), skipping")
            continue

        embedding_name = col.replace("dist_", "")

        kurt = float(sp_stats.kurtosis(values, fisher=True))  # excess kurtosis
        skew = float(sp_stats.skew(values))

        results[embedding_name] = {
            "kurtosis": kurt,
            "skewness": skew,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n_pairs": len(values),
        }

    return results


# ---------------------------------------------------------------------------
#                          VISUALIZATION
# ---------------------------------------------------------------------------


def plot_loss_curves(
    runs: List[RunInfo],
    curves_map: Dict[str, TrainingCurves],
    output_dir: Path,
    max_per_page: int = 12,
) -> List[Path]:
    """Plot train/val loss curves for discovered runs.

    Creates grid figures with train and val loss overlaid per run.

    Parameters
    ----------
    runs : List[RunInfo]
        Discovered runs.
    curves_map : Dict[str, TrainingCurves]
        Mapping from run path (str) to parsed curves.
    output_dir : Path
        Where to save figures.
    max_per_page : int
        Maximum subplots per figure page.

    Returns
    -------
    List[Path]
        Paths to saved figure files.
    """
    # Filter to runs with curves
    valid_runs = [r for r in runs if str(r.path) in curves_map]
    if not valid_runs:
        logger.info("No runs with training curves to plot")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    n_pages = (len(valid_runs) + max_per_page - 1) // max_per_page

    for page_idx in range(n_pages):
        page_runs = valid_runs[page_idx * max_per_page : (page_idx + 1) * max_per_page]
        n = len(page_runs)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for idx, run in enumerate(page_runs):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            curves = curves_map[str(run.path)]

            if len(curves.train_loss) > 0:
                ax.plot(curves.steps, curves.train_loss, label="train", alpha=0.7, linewidth=0.8)
            if len(curves.val_loss) > 0:
                ax.plot(curves.val_steps, curves.val_loss, label="val", alpha=0.9, linewidth=1.2, color="tab:orange")

            ax.set_title(f"{run.embedding}\n{run.model_type}/{run.param}", fontsize=9)
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("loss", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(f"Training Curves (page {page_idx + 1}/{n_pages})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        page_path = output_dir / f"loss_curves_page{page_idx + 1:02d}.png"
        fig.savefig(page_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(page_path)
        logger.info(f"Saved loss curves: {page_path}")

    return saved_paths


def plot_overtraining_heatmap(
    results_df: pl.DataFrame,
    output_dir: Path,
    metric: str = "train_val_gap",
) -> Optional[Path]:
    """Plot a heatmap of overtraining metrics: embedding x param, faceted by model_type.

    Parameters
    ----------
    results_df : pl.DataFrame
        Aggregated overtraining results with columns: embedding, model_type,
        param, and the metric to plot.
    output_dir : Path
        Where to save the figure.
    metric : str
        Column name to use as the heatmap value.

    Returns
    -------
    Optional[Path]
        Path to saved figure, or None on failure.
    """
    if metric not in results_df.columns:
        logger.warning(f"Metric '{metric}' not in results DataFrame")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    model_types = sorted(results_df["model_type"].unique().to_list())
    n_models = len(model_types)

    if n_models == 0:
        logger.warning("No model types found for heatmap")
        return None

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)

    for idx, mt in enumerate(model_types):
        ax = axes[0][idx]
        subset = results_df.filter(pl.col("model_type") == mt)

        if subset.is_empty():
            ax.set_title(f"{mt} (no data)")
            continue

        # Pivot to embedding x param
        pivot = subset.pivot(
            on="param", index="embedding", values=metric
        ).sort("embedding")

        embeddings = pivot["embedding"].to_list()
        params = [c for c in pivot.columns if c != "embedding"]
        data = pivot.select(params).to_numpy()

        sns.heatmap(
            data,
            ax=ax,
            xticklabels=params,
            yticklabels=embeddings,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",  # red = high gap (bad), green = low (good)
            center=0,
            linewidths=0.5,
            cbar_kws={"label": metric},
        )
        ax.set_title(f"{mt}", fontsize=14)
        ax.set_xlabel("Target Parameter", fontsize=11)
        ax.set_ylabel("Embedding" if idx == 0 else "", fontsize=11)
        ax.tick_params(labelsize=9)

    metric_label = metric.replace("_", " ").title()
    fig.suptitle(f"Overtraining: {metric_label} by Embedding x Param", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = output_dir / f"overtraining_heatmap_{metric}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved overtraining heatmap: {out_path}")
    return out_path


def plot_kurtosis_bar(
    kurtosis_results: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> Optional[Path]:
    """Bar chart of distance distribution kurtosis per embedding.

    Parameters
    ----------
    kurtosis_results : Dict
        Output of compute_distance_kurtosis().
    output_dir : Path
        Where to save the figure.

    Returns
    -------
    Optional[Path]
        Path to saved figure.
    """
    if not kurtosis_results:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = sorted(kurtosis_results.keys())
    kurtosis_vals = [kurtosis_results[e]["kurtosis"] for e in embeddings]

    fig, ax = plt.subplots(figsize=(max(8, len(embeddings) * 0.6), 5))
    colors = ["#d62728" if k > 3 else "#2ca02c" if k < 0 else "#ff7f0e" for k in kurtosis_vals]

    bars = ax.bar(range(len(embeddings)), kurtosis_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(embeddings)))
    ax.set_xticklabels(embeddings, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Excess Kurtosis", fontsize=12)
    ax.set_title("Distance Distribution Kurtosis by Embedding", fontsize=14)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Annotate values
    for bar, val in zip(bars, kurtosis_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    out_path = output_dir / "kurtosis_by_embedding.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved kurtosis plot: {out_path}")
    return out_path


def plot_overtraining_summary(
    results_df: pl.DataFrame,
    kurtosis_results: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> List[Path]:
    """Generate all overtraining diagnostic plots.

    Parameters
    ----------
    results_df : pl.DataFrame
        Aggregated overtraining results.
    kurtosis_results : Dict
        Output of compute_distance_kurtosis().
    output_dir : Path
        Where to save figures.

    Returns
    -------
    List[Path]
        All saved figure paths.
    """
    saved: List[Path] = []

    # Heatmap of train/val gap
    p = plot_overtraining_heatmap(results_df, output_dir, metric="train_val_gap")
    if p:
        saved.append(p)

    # Heatmap of early stop ratio
    p = plot_overtraining_heatmap(results_df, output_dir, metric="early_stop_ratio")
    if p:
        saved.append(p)

    # Heatmap of late val slope (positive = increasing = overfit)
    p = plot_overtraining_heatmap(results_df, output_dir, metric="late_val_slope")
    if p:
        saved.append(p)

    # Kurtosis bar chart
    p = plot_kurtosis_bar(kurtosis_results, output_dir)
    if p:
        saved.append(p)

    return saved


# ---------------------------------------------------------------------------
#                           AGGREGATION
# ---------------------------------------------------------------------------


def aggregate_metrics(
    metrics_list: List[OvertrainingMetrics],
) -> pl.DataFrame:
    """Aggregate per-run overtraining metrics into a summary DataFrame.

    Groups by (embedding, model_type, param) and computes mean metrics.

    Parameters
    ----------
    metrics_list : List[OvertrainingMetrics]
        Individual run metrics.

    Returns
    -------
    pl.DataFrame
        Aggregated summary with one row per embedding x model_type x param.
    """
    if not metrics_list:
        return pl.DataFrame()

    # Convert to polars DataFrame
    records = []
    for m in metrics_list:
        records.append({
            "embedding": m.embedding,
            "model_type": m.model_type,
            "param": m.param,
            "run_path": m.run_path,
            "final_train_loss": m.final_train_loss,
            "final_val_loss": m.final_val_loss,
            "train_val_gap": m.train_val_gap,
            "train_val_ratio": m.train_val_ratio,
            "late_train_slope": m.late_train_slope,
            "late_val_slope": m.late_val_slope,
            "diverging": m.diverging,
            "stopped_epoch": m.stopped_epoch,
            "max_epochs": m.max_epochs,
            "early_stop_ratio": m.early_stop_ratio,
            "best_val_loss": m.best_val_loss,
            "best_val_step": m.best_val_step,
            "total_steps": m.total_steps,
        })

    raw_df = pl.DataFrame(records)

    # Aggregate by embedding x model_type x param
    agg_df = (
        raw_df
        .group_by(["embedding", "model_type", "param"])
        .agg([
            pl.col("final_train_loss").mean().alias("final_train_loss"),
            pl.col("final_val_loss").mean().alias("final_val_loss"),
            pl.col("train_val_gap").mean().alias("train_val_gap"),
            pl.col("train_val_ratio").mean().alias("train_val_ratio"),
            pl.col("late_train_slope").mean().alias("late_train_slope"),
            pl.col("late_val_slope").mean().alias("late_val_slope"),
            pl.col("diverging").sum().alias("n_diverging"),
            pl.col("diverging").len().alias("n_runs"),
            pl.col("early_stop_ratio").mean().alias("early_stop_ratio"),
            pl.col("best_val_loss").mean().alias("best_val_loss"),
            pl.col("total_steps").mean().alias("total_steps"),
        ])
        .sort(["embedding", "model_type", "param"])
    )

    return agg_df


# ---------------------------------------------------------------------------
#                              MAIN
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Main orchestration: discover, parse, compute, aggregate, visualize."""
    models_dir = Path(args.models_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Discover runs ----
    logger.info(f"Discovering training runs under {models_dir}")
    runs = discover_runs(models_dir)

    if not runs:
        logger.error("No training runs found. Check --models_dir path.")
        sys.exit(1)

    # Optional sampling for quick testing
    if args.sample_size and args.sample_size < len(runs):
        import random
        random.seed(42)
        runs = random.sample(runs, args.sample_size)
        logger.info(f"Sampled {args.sample_size} runs for testing")

    # Skip euclidean baseline runs (no training curves)
    trainable_runs = [r for r in runs if r.model_type != "euclidean"]
    euclidean_runs = [r for r in runs if r.model_type == "euclidean"]
    logger.info(
        f"Found {len(trainable_runs)} trainable runs + "
        f"{len(euclidean_runs)} euclidean baselines"
    )

    # ---- 2. Parse training curves ----
    logger.info("Parsing training curves from TensorBoard events...")
    curves_map: Dict[str, TrainingCurves] = {}
    parse_failures = 0

    for run in trainable_runs:
        curves = parse_training_curves(run.path)
        if curves is not None:
            curves_map[str(run.path)] = curves
        else:
            parse_failures += 1

    logger.info(
        f"Parsed curves for {len(curves_map)}/{len(trainable_runs)} runs "
        f"({parse_failures} failures)"
    )

    # ---- 3. Compute overtraining metrics ----
    logger.info("Computing overtraining metrics...")
    all_metrics: List[OvertrainingMetrics] = []

    for run in runs:
        curves = curves_map.get(str(run.path))
        metrics = compute_overtraining_metrics(run, curves)
        all_metrics.append(metrics)

    # ---- 4. Aggregate ----
    logger.info("Aggregating results...")
    agg_df = aggregate_metrics(all_metrics)

    if agg_df.is_empty():
        logger.error("No metrics computed. Check run directories for valid training data.")
        sys.exit(1)

    # Save raw per-run results
    raw_records = []
    for m in all_metrics:
        raw_records.append({
            "embedding": m.embedding,
            "model_type": m.model_type,
            "param": m.param,
            "run_path": m.run_path,
            "final_train_loss": m.final_train_loss,
            "final_val_loss": m.final_val_loss,
            "train_val_gap": m.train_val_gap,
            "train_val_ratio": m.train_val_ratio,
            "late_train_slope": m.late_train_slope,
            "late_val_slope": m.late_val_slope,
            "diverging": m.diverging,
            "stopped_epoch": m.stopped_epoch,
            "max_epochs": m.max_epochs,
            "early_stop_ratio": m.early_stop_ratio,
            "best_val_loss": m.best_val_loss,
            "best_val_step": m.best_val_step,
            "total_steps": m.total_steps,
        })
    raw_df = pl.DataFrame(raw_records)

    # ---- 5. Distance kurtosis (optional) ----
    kurtosis_results: Dict[str, Dict[str, float]] = {}
    if args.pairs_parquet:
        pairs_path = Path(args.pairs_parquet).resolve()
        if pairs_path.is_file():
            logger.info(f"Loading pairs from {pairs_path} for kurtosis analysis...")
            try:
                pairs_df = pl.read_parquet(pairs_path)
                kurtosis_results = compute_distance_kurtosis(pairs_df)
                logger.info(f"Computed kurtosis for {len(kurtosis_results)} embeddings")
            except Exception as e:
                logger.warning(f"Failed to load pairs parquet: {e}")
        else:
            logger.warning(f"Pairs parquet not found: {pairs_path}")

    # ---- 6. Save outputs ----
    # Aggregated summary
    agg_parquet = output_dir / "overtraining_summary.parquet"
    agg_csv = output_dir / "overtraining_summary.csv"
    agg_df.write_parquet(agg_parquet)
    agg_df.write_csv(agg_csv)
    logger.info(f"Saved aggregated summary: {agg_parquet}")
    logger.info(f"Saved aggregated CSV: {agg_csv}")

    # Raw per-run results
    raw_parquet = output_dir / "overtraining_per_run.parquet"
    raw_csv = output_dir / "overtraining_per_run.csv"
    raw_df.write_parquet(raw_parquet)
    raw_df.write_csv(raw_csv)
    logger.info(f"Saved per-run results: {raw_parquet}")

    # Kurtosis
    if kurtosis_results:
        kurt_records = [
            {"embedding": emb, **stats}
            for emb, stats in kurtosis_results.items()
        ]
        kurt_df = pl.DataFrame(kurt_records)
        kurt_parquet = output_dir / "distance_kurtosis.parquet"
        kurt_csv = output_dir / "distance_kurtosis.csv"
        kurt_df.write_parquet(kurt_parquet)
        kurt_df.write_csv(kurt_csv)
        logger.info(f"Saved kurtosis results: {kurt_parquet}")

    # ---- 7. Plots ----
    logger.info("Generating diagnostic plots...")
    saved_plots: List[Path] = []

    # Loss curves
    curve_plots = plot_loss_curves(runs, curves_map, output_dir / "curves")
    saved_plots.extend(curve_plots)

    # Summary heatmaps and kurtosis
    summary_plots = plot_overtraining_summary(agg_df, kurtosis_results, output_dir)
    saved_plots.extend(summary_plots)

    # ---- 8. Console summary ----
    print("\n" + "=" * 70)
    print("OVERTRAINING ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Runs discovered:        {len(runs)}")
    print(f"Curves parsed:          {len(curves_map)}")
    print(f"Parse failures:         {parse_failures}")
    print(f"Unique embeddings:      {agg_df['embedding'].n_unique()}")
    print(f"Unique model types:     {agg_df['model_type'].n_unique()}")
    print(f"Unique params:          {agg_df['param'].n_unique()}")

    # Flag concerning runs
    if "n_diverging" in agg_df.columns:
        diverging = agg_df.filter(pl.col("n_diverging") > 0)
        if not diverging.is_empty():
            print(f"\nWARNING: {len(diverging)} embedding x model x param combos show late-stage divergence:")
            for row in diverging.iter_rows(named=True):
                print(f"  {row['embedding']}/{row['model_type']}/{row['param']}: "
                      f"{row['n_diverging']}/{row['n_runs']} runs diverging, "
                      f"gap={row['train_val_gap']:.4f}")

    if kurtosis_results:
        print(f"\nDistance distribution kurtosis ({len(kurtosis_results)} embeddings):")
        for emb in sorted(kurtosis_results, key=lambda e: kurtosis_results[e]["kurtosis"], reverse=True):
            k = kurtosis_results[emb]["kurtosis"]
            flag = " <-- high" if k > 3 else ""
            print(f"  {emb:20s}: kurtosis={k:.3f}{flag}")

    print(f"\nOutputs saved to: {output_dir}")
    print(f"Plots generated:  {len(saved_plots)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Investigate overtraining in pLM downstream models. "
            "Parses TensorBoard logs from training runs to detect "
            "train/val loss divergence and peaked distance distributions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full analysis\n"
            "  uv run python src/evaluation/overtraining_analysis.py \\\n"
            "      --models_dir models/sprot_pre2024 \\\n"
            "      --pairs_parquet data/processed/sprot_pre2024/sets/test_with_distances.parquet \\\n"
            "      --output_dir out/overtraining\n"
            "\n"
            "  # Quick test\n"
            "  uv run python src/evaluation/overtraining_analysis.py \\\n"
            "      --models_dir models/sprot_pre2024 \\\n"
            "      --output_dir out/overtraining \\\n"
            "      --sample_size 10\n"
        ),
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        required=True,
        help="Root directory containing training runs "
        "(e.g., models/sprot_pre2024). Searched recursively.",
    )
    parser.add_argument(
        "--pairs_parquet",
        type=Path,
        default=None,
        help="Parquet file with distance columns (dist_*) for kurtosis analysis. "
        "Optional — if omitted, kurtosis analysis is skipped.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for output files (parquet, CSV, plots).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Randomly sample N runs for quick testing. "
        "If omitted, all discovered runs are analyzed.",
    )

    args = parser.parse_args()
    main(args)
