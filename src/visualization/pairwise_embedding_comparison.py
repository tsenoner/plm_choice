#!/usr/bin/env python3
"""
Pairwise Embedding Comparison Visualization Script

This script generates comprehensive visualizations comparing different protein language model (PLM)
embeddings through various analytical approaches:

1. Hexagonal heatmap for distance comparisons across embedding pairs
2. Correlation heatmap with confidence intervals
3. Wasserstein distance heatmap between normalized distributions
4. Distribution comparison plots (both raw and normalized)
5. Violin plots for PLM distance differences

The script follows the project structure and uses consistent color schemes and styling
conventions from the project's visualization framework.
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import seaborn as sns
from diptest import diptest
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress matplotlib DEBUG messages
logging.getLogger("matplotlib").setLevel(logging.INFO)

# Bump when the shape or semantics of a cached JSON changes (e.g. the density
# grid size), so old caches are recomputed instead of silently reused.
CACHE_SCHEMA_VERSION = 2

# --- Project Constants & Configuration ---
# Shared with create_performance_summary_plots.py — see visualization/plm_constants.py.
from shared.embedding_names import is_iid_random_baseline
from visualization.plm_constants import (
    EMBEDDING_COLOR_MAP,
    EMBEDDING_DISPLAY_NAMES,
    EMBEDDING_FAMILY_COLOR_MAP,
    EMBEDDING_FAMILY_MAP,
    PLM_SIZES,
)


def _nice_tick_step(span: float) -> float:
    """A round tick interval giving roughly 5-10 ticks across ``span``."""
    if span <= 0:
        return 1.0
    raw = span / 7.0
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= mult * magnitude:
            return mult * magnitude
    return 10.0 * magnitude


def distance_column_sort_key(col: str) -> tuple:
    """Row order for every pairwise figure: family, then size within family.

    Shared with ``create_performance_summary_plots.py`` so a model sits in the same
    place in every panel of the paper.  Note what this is *not*: it is not a
    performance ranking.  The published Figure 2 caption claimed the rows were
    "ordered by performance ranking" and they never were -- ``ranking_csv`` is the
    opt-in that does that, and it was not passed.
    """
    embedding_name = col.replace("dist_", "").lower()
    return (
        EMBEDDING_FAMILY_MAP.get(embedding_name, "Unknown"),
        PLM_SIZES.get(embedding_name, 0),
        embedding_name,
    )


#: Per-row rescalings the ridge figure can draw under.  Each is an increasing affine
#: map ``(x - offset) / divisor``; both numbers are carried through to the statistics
#: CSV so a caption can state exactly what the axis was divided by.
#:
#: Why this is a choice at all: raw medians span four orders of magnitude across arms
#: (ankh_large 0.071 ... esm3_open 1160.6), so an unscaled shared axis shows fourteen
#: spikes at zero and one curve.  Something has to rescale each row -- the only question
#: is what it anchors on.
#:
#: ``offset`` names the summary field subtracted before dividing, ``stat`` the one
#: divided by; ``"max"`` and ``"min"`` are top-level summary fields, anything else is a
#: key of ``summary["quantiles"]``.
RIDGE_NORMALISATIONS: Dict[str, Dict[str, object]] = {
    # The published choice: sklearn's MinMaxScaler, (x - min) / (max - min).  Anchors on
    # the single most distant pair out of ~76 million, so one outlier fixes the scale for
    # the whole row; an arm whose tail reaches further gets its entire bulk squeezed
    # toward zero, which is how the rebuilt medians fell to 0.06-0.31 on the alignable
    # pairs without any embedding changing.  On uniformly random pairs the squeeze is
    # far milder (medians 0.14-0.59), which is why it is usable as a main-figure axis.
    "minmax": {
        "offset": "min",
        "stat": "max",
        # What the two ends mean belongs on the axis, not in the title: the title has
        # to fit the figure's own width (20 in) and a longer one silently widens the
        # PNG, which then has to be scaled down to the column and takes the whole plot
        # with it.
        "label": "Min-max normalized distance "
        "(0 = that model's closest pair, 1 = its most distant pair)",
        "xlim": (0.0, 1.0),
    },
    # Anchors on the 99th percentile: 50,000 pairs of the 5M decide the scale instead of
    # one, and the bulk keeps its shape.  Values above 1 are the top percent and are drawn.
    "p99": {"stat": "p99", "label": "Distance / 99th percentile", "xlim": (0.0, 1.2)},
    # Anchors on the row's own median, so every row is centred on 1 and the picture is
    # purely about spread -- the scale-free view that matches the QCD numbers.  The
    # limit is 3.0 and not 2.2 because at 2.2 the 99th percentile of the two ESM-C arms
    # (2.58 and 2.55 medians out) falls off the axis along with 3% of their pairs: an
    # axis that cannot show its own marked percentile is not a candidate.
    "median": {
        "stat": "p50",
        "label": "Distance / median (1.0 = that model's typical pair)",
        "xlim": (0.0, 3.0),
    },
    # No per-row rescaling at all: the raw euclidean distance on a log axis.  Honest about
    # the four orders of magnitude, at the cost of a log axis in a main figure.
    "log10": {"stat": None, "label": "Euclidean distance (log scale)", "xlim": None},
}

#: Percentiles the ridge rows can mark, in the order they are drawn.  ``q25``/``median``
#: /``q75`` are rules that run from the baseline up to the curve; ``p1``/``p99`` are
#: baseline ticks, because at those x the density is ~0 and a baseline-to-curve rule
#: would be invisible -- which is exactly why the tails were unmarked before.
RIDGE_TAIL_PERCENTILES = ("p1", "p99")

#: How the three quartile rules are drawn.  Darker and fully opaque compared with the
#: published figure's "0.3" at alpha 0.7: drawn over a saturated fill and then partly
#: covered by the row in front, a 70%-opaque mid-grey dotted line is the first thing to
#: disappear, and these three lines are the figure's quantitative content.  Each also
#: gets a white halo (path_effects at the draw site).
#:
#: It lives here rather than inside ``plot_ridge_distributions`` because the key at the
#: bottom of the figure has to draw the same glyphs.  With the widths and the dash
#: pattern typed out a second time for the legend, a change to one of them produces a
#: key that lies about the lines.
_RIDGE_QUARTILE_STYLE = {"color": "0.12", "linestyle": (0, (1.6, 1.4)), "linewidth": 2.6}
RIDGE_PERCENTILE_STYLES: Dict[str, Dict] = {
    "q25": _RIDGE_QUARTILE_STYLE,
    "median": {"color": "black", "linestyle": "-", "linewidth": 3.4},
    "q75": _RIDGE_QUARTILE_STYLE,
}


#: Diverging map for the Spearman cells of the combined fingerprint: purple on the
#: negative arm, cream at zero, OrRd's own reds on the positive arm.
#:
#: Why this exists.  The cells used to be drawn ``cmap="OrRd", vmin=0, vmax=1``.  On the
#: aligner-found pair population every rho was positive so the clipped floor never
#: showed; on the uniformly random population 12 of the 91 off-diagonal cells are
#: negative, down to rho = -0.238, and every one of them involves CLEAN -- which is the
#: result the figure exists to report.  Under ``vmin=0`` all twelve rendered identically
#: to rho = 0, i.e. the figure silently erased its own finding.
#:
#: Why purple and not the usual RdBu_r.  The upper triangle of the same figure is
#: already ``Blues`` on a different quantity with its own colourbar; a blue negative arm
#: would make two unrelated scales share a colour.
CORR_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "corr_diverging",
    [
        "#2D004B", "#542788", "#8073AC", "#B2ABD2", "#D8DAEB",  # negative arm
        "#FFF7EC",                                              # zero
        "#FDD49E", "#FC8D59", "#EF6548", "#B30000", "#7F0000",  # positive arm
    ],
)


def symmetric_corr_limit(*matrices: np.ndarray) -> float:
    """Smallest 0.01 step covering max|rho| over the off-diagonal of every matrix.

    Passing more than one matrix is how the main-text figure and its supplementary
    counterpart end up on one scale: equal colour then means equal rho in both, and the
    two are comparable cell for cell.  Scaled to each matrix's own maximum they are not.
    """
    worst = 0.0
    for m in matrices:
        m = np.asarray(m, dtype=float)
        iu = np.triu_indices(m.shape[0], 1)
        worst = max(worst, float(np.nanmax(np.abs(m[iu]))))
    return math.ceil(worst * 100) / 100


def _readable_text_color(cmap, norm, value: float) -> str:
    """Black or white, whichever the cell's own colour can carry.

    A fixed ``value > 0.5`` threshold assumes a ramp that darkens monotonically -- true
    for OrRd from 0, false for any diverging map, where both ends are dark and the
    middle is pale.  Under the diverging scale that rule painted white text on the cream
    cells around rho = 0.
    """
    r, g, b = mcolors.to_rgb(cmap(norm(value)))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.55 else "black"


def _place_legend_below_axis_label(fig, legend, bottom_ax, pad_in: float = 0.22) -> None:
    """Drop the ridge key just under the x-axis label, measured rather than guessed.

    The bottom margin's usable height is not a constant: it depends on the tick font,
    the label font, ``labelpad`` and the number of rows (which sets the figure height
    that the margin is a *fraction* of).  A hardcoded ``bbox_to_anchor`` y therefore
    works for one geometry and collides for the next.  Measuring the label after a draw
    and anchoring ``pad_in`` inches below it works for all of them.  ``bbox_inches=
    "tight"`` then grows the saved canvas if the key ends up below the figure edge,
    so a negative anchor is not a problem.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label = bottom_ax.xaxis.get_label()
    label_bottom_px = label.get_window_extent(renderer).y0
    y = (label_bottom_px - pad_in * fig.dpi) / fig.get_window_extent().height
    legend.set_bbox_to_anchor((0.5, y), transform=fig.transFigure)


def _assert_legend_clear(fig, legend, axes, dpi: int) -> None:
    """Fail if the ridge key overlaps any row's axes or the x-axis label.

    Measured in rendered pixels at the *save* dpi, because that is the only geometry
    that ships: a legend that clears a row on a 100-dpi screen preview can sit on its
    baseline in the 300-dpi PNG, which is how the published key came to overlap the
    ProtT5 row.  Raising rather than warning is deliberate -- a warning in a log nobody
    reads is how the overlap survived a figure refresh.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = dpi / float(fig.dpi)
    lb = legend.get_window_extent(renderer)
    collisions = []
    for idx, ax in enumerate(axes):
        if not ax.get_visible():
            continue
        if lb.overlaps(ax.get_window_extent(renderer)):
            collisions.append(f"row {idx} axes")
    xlabel = axes[-1].xaxis.get_label()
    if xlabel.get_text() and lb.overlaps(xlabel.get_window_extent(renderer)):
        collisions.append("x-axis label")
    if collisions:
        raise RuntimeError(
            "ridge legend overlaps "
            + ", ".join(collisions)
            + f" (legend px at {dpi} dpi: "
            f"x {lb.x0 * scale:.0f}-{lb.x1 * scale:.0f}, "
            f"y {lb.y0 * scale:.0f}-{lb.y1 * scale:.0f})"
        )
    logger.info(
        "Legend clear of all %d rows and the axis label; its box at %d dpi is "
        "x %.0f-%.0f, y %.0f-%.0f px",
        len(axes),
        dpi,
        lb.x0 * scale,
        lb.x1 * scale,
        lb.y0 * scale,
        lb.y1 * scale,
    )


def _ridge_anchor(
    summary: Dict, field: Optional[str], default: Optional[float] = None
) -> float:
    """Resolve one ``RIDGE_NORMALISATIONS`` anchor against a per-arm summary.

    ``"min"``/``"max"`` are top-level summary fields, everything else is a key of
    ``summary["quantiles"]``.  Resolving by name rather than by a chain of ``if``s is
    what keeps the offset and the divisor spelled the same way, so a normalisation can
    be added by naming two fields instead of by editing the rescaling arithmetic.
    """
    if field is None:
        if default is None:
            raise ValueError("no anchor field and no default")
        return float(default)
    if field in ("min", "max"):
        return float(summary[field])
    return float(summary["quantiles"][field])

#: The lattice the stored distances sit on: ``ridge_pair_distances.py`` writes
#: ``np.round(d, decimals=4)``.  Needed when rebinning, see
#: ``compute_distribution_data_from_summaries``.
DISTANCE_QUANTUM = 1e-4

DEFAULT_STYLE = {
    "figure_size": (15, 12),
    "dpi": 300,
    "font_scale": 1.0,
    "title_size": 16,
    "label_size": 12,
    "tick_size": 10,
    "legend_size": 10,
}


class EmbeddingComparisonVisualizer:
    """
    A comprehensive visualizer for comparing protein language model embeddings.

    Provides methods for various types of embedding comparison visualizations,
    including distance heatmaps, correlation analyses, and distribution comparisons.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        output_dir: Union[str, Path],
        sample_size: Optional[int] = None,
        font_scale: float = 1.0,
    ):
        """
        Initialize the visualizer.

        Args:
            data_path: Path to CSV file or pandas DataFrame containing the data
            output_dir: Directory where output files will be saved
            sample_size: Optional limit on number of rows to process
            font_scale: Scaling factor for all font sizes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.font_scale = font_scale

        # Load and process data
        self.df = self._load_data(data_path)
        self.dist_cols = self._identify_distance_columns()

        # Set up matplotlib styling
        self._setup_plotting_style()

    # --- Alternative construction: reduced summaries instead of a pair table ---

    @classmethod
    def from_distribution_summaries(
        cls,
        summary_dir: Union[str, Path],
        output_dir: Union[str, Path],
        arms: Optional[List[str]] = None,
        font_scale: float = 1.0,
    ) -> "EmbeddingComparisonVisualizer":
        """Build a visualizer from ``scripts/ridge_full_reduce.py`` output.

        The ordinary constructor reads a pair table into memory.  The full corrected
        cohort is 75,849,972 pairs over 15 arms -- 11.6 GB of parquet -- which is why
        the reduction runs on the cluster and only per-arm histograms and exact
        quantiles come home.  This constructor takes that directory instead, and the
        resulting object supports the ridge plot only: ``self.df`` is None, so any
        method that touches the pair table will fail loudly rather than quietly
        plotting something else.
        """
        summary_dir = Path(summary_dir)
        summaries: Dict[str, Dict] = {}
        for path in sorted(summary_dir.glob("*.json")):
            arm = path.stem
            if arms is not None and arm not in arms:
                continue
            payload = json.loads(path.read_text())
            npz_path = path.with_suffix(".npz")
            if not npz_path.exists():
                raise FileNotFoundError(f"{npz_path} missing next to {path}")
            payload["_npz"] = npz_path
            summaries[arm] = payload
        if not summaries:
            raise ValueError(f"no per-arm summaries found in {summary_dir}")

        self = cls.__new__(cls)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = None
        self.font_scale = font_scale
        self.df = None
        self.summaries = summaries
        # Same exclusions and the same row order as every other pairwise figure.
        cols = [
            f"dist_{arm}"
            for arm in summaries
            if not is_iid_random_baseline(arm) and arm.lower() != "prostt5"
        ]
        self.dist_cols = sorted(cols, key=distance_column_sort_key)
        dropped = sorted(set(summaries) - {c.replace("dist_", "") for c in self.dist_cols})
        if dropped:
            logger.warning(
                "EXCLUDED %d arm(s) from the ridge figure: %s (i.i.d. random baselines "
                "are excluded by design; prostt5 is a TEMPORARY exclusion)",
                len(dropped),
                ", ".join(dropped),
            )
        logger.info(
            "Loaded %d arm summaries, drawing %d rows in family-then-size order",
            len(summaries),
            len(self.dist_cols),
        )
        self._setup_plotting_style()
        return self

    @classmethod
    def from_matrices(
        cls,
        columns: List[str],
        output_dir: Union[str, Path],
        font_scale: float = 1.0,
    ) -> "EmbeddingComparisonVisualizer":
        """Build a visualizer that can draw the fingerprint and nothing else.

        The combined Wasserstein/correlation figure needs two 14x14 matrices and the
        row order; it never touches the pair table.  That table is 11.6 GB of parquet
        for the alignable cohort and 600 MB for the random-pair sample, so the
        reduction runs where the data is and only the matrices come home -- this
        constructor is what lets the drawing happen without them.

        ``self.df`` stays None so anything that does need the pairs fails loudly.
        """
        self = cls.__new__(cls)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = None
        self.font_scale = font_scale
        self.df = None
        self.dist_cols = sorted(
            (c if c.startswith("dist_") else f"dist_{c}" for c in columns),
            key=distance_column_sort_key,
        )
        logger.info(
            "No per-pair data loaded; drawing %d precomputed arms: %s",
            len(self.dist_cols),
            ", ".join(c.replace("dist_", "") for c in self.dist_cols),
        )
        self._setup_plotting_style()
        return self

    def _load_data(self, data_path: Union[str, Path]) -> pl.DataFrame:
        """Load data from various sources, returning polars DataFrame."""
        data_path = Path(data_path)
        logger.info(f"Loading data from {data_path}")

        # Support both CSV and Parquet files
        if data_path.suffix.lower() == ".parquet":
            df = pl.read_parquet(data_path)
            logger.info(f"Loaded Parquet file with {len(df)} rows")
        else:
            df = pl.read_csv(data_path)
            logger.info(f"Loaded CSV file with {len(df)} rows")

        if self.sample_size:
            df = df.head(self.sample_size)
            logger.info(f"Limited dataset to {len(df)} rows")

        return df

    def _identify_distance_columns(self) -> List[str]:
        """Identify and validate distance columns in the dataset, excluding random embeddings."""
        # Get all distance columns
        all_dist_cols = [
            col
            for col in self.df.columns
            if col.startswith("dist_") and not col.startswith("pca_")
        ]

        # Filter out the i.i.d. random baselines and temporarily exclude prostt5.
        #
        # `random_init_*` must survive this filter. Those are the untrained
        # *architectures* of reviewer R1.9 — a different control from the
        # i.i.d. `random_1024` noise floor, which is exactly why
        # plm_constants.py gives them their own "Untrained" family and colour.
        # A bare `startswith("random")` swallowed them, so the arm this branch
        # exists to add would have been absent from every pairwise figure while
        # the log claimed it was "excluded by design". The predicate is shared with
        # the all-vs-all cache builder, which had the identical trap.
        dist_cols = [
            col
            for col in all_dist_cols
            if not is_iid_random_baseline(col.replace("dist_", ""))
            and not col.replace("dist_", "").lower()
            == "prostt5"  # TEMPORARY: Remove this line to re-include prostt5
        ]

        dropped = sorted(set(all_dist_cols) - set(dist_cols))
        if dropped:
            logger.warning(
                "EXCLUDED %d embedding column(s) from every pairwise figure: %s "
                "(i.i.d. random baselines are excluded by design; prostt5 is a "
                "TEMPORARY exclusion — see the Data availability note in the "
                "README). random_init_* untrained architectures are NOT excluded.",
                len(dropped),
                ", ".join(c.replace("dist_", "") for c in dropped),
            )

        dist_cols = sorted(dist_cols, key=distance_column_sort_key)

        if not dist_cols:
            raise ValueError(
                "No valid distance columns found. Columns should start with 'dist_' and not be random embeddings"
            )

        filtered_count = len(all_dist_cols) - len(dist_cols)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} random embedding columns")

        logger.info(
            f"Found {len(dist_cols)} distance columns (sorted by PLM family, then size): {dist_cols}"
        )
        return dist_cols

    def _setup_plotting_style(self):
        """Configure matplotlib plotting parameters."""
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.size": DEFAULT_STYLE["font_scale"] * self.font_scale,
                "axes.titlesize": DEFAULT_STYLE["title_size"] * self.font_scale,
                "axes.labelsize": DEFAULT_STYLE["label_size"] * self.font_scale,
                "xtick.labelsize": DEFAULT_STYLE["tick_size"] * self.font_scale,
                "ytick.labelsize": DEFAULT_STYLE["tick_size"] * self.font_scale,
                "legend.fontsize": DEFAULT_STYLE["legend_size"] * self.font_scale,
                "figure.dpi": DEFAULT_STYLE["dpi"],
            }
        )

    def _get_embedding_color(self, dist_col: str) -> str:
        """Get color for a distance column based on embedding name."""
        embedding_name = dist_col.replace("dist_", "").lower()
        return EMBEDDING_COLOR_MAP.get(embedding_name, "#808080")

    def _get_embedding_family(self, dist_col: str) -> str:
        """Get family name for a distance column based on embedding name."""
        embedding_name = dist_col.replace("dist_", "").lower()
        return EMBEDDING_FAMILY_MAP.get(embedding_name, "Unknown")

    def _get_family_boundaries(self, dist_cols: List[str]) -> List[int]:
        """Get indices where family changes occur."""
        boundaries = [0]
        prev_family = None
        for i, col in enumerate(dist_cols):
            family = self._get_embedding_family(col)
            if prev_family is not None and family != prev_family:
                boundaries.append(i)
            prev_family = family
        boundaries.append(len(dist_cols))
        return boundaries

    def _cache_fingerprint(self) -> Dict:
        """Identify the inputs a cache entry was computed from.

        A cache keyed on filename alone cannot tell a full run from a
        ``--sample_size 5000`` debug run, or a cache written by an older version
        of this code from a current one. That is not hypothetical: the caches
        that produced the published ridge figure carry a 200-point density grid
        that no current code path emits, and nothing detected it.
        """
        return {
            "n_rows": int(len(self.df)),
            "dist_cols": list(self.dist_cols),
            "sample_size": self.sample_size,
            "schema_version": CACHE_SCHEMA_VERSION,
        }

    def _save_json_data(self, data: Dict, save_path: Path, description: str):
        """Helper method to save JSON data with consistent logging."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(data)
        meta = dict(data.get("metadata") or {})
        meta["_fingerprint"] = self._cache_fingerprint()
        data["metadata"] = meta
        with open(save_path, "w") as f:
            json.dump(data, f)
        logger.info(f"{description} saved to {save_path}")

    def _load_cached_data(
        self, cache_path: Path, force_recompute: bool
    ) -> Optional[Dict]:
        """Load a cache entry only if it was computed from the same inputs.

        Returns None (i.e. recompute) when the cache is stale or unfingerprinted,
        rather than silently reusing it.
        """
        if force_recompute or not cache_path.exists():
            return None

        with open(cache_path, "r") as f:
            data = json.load(f)

        stored = (data.get("metadata") or {}).get("_fingerprint")
        if stored is None:
            logger.warning(
                "IGNORING un-fingerprinted cache %s — it predates cache validation "
                "and there is no way to tell what data or code version wrote it. "
                "Recomputing.",
                cache_path.name,
            )
            return None

        current = self._cache_fingerprint()
        if stored != current:
            differing = [
                k for k in current if stored.get(k) != current.get(k)
            ]
            logger.warning(
                "IGNORING stale cache %s — %s differ(s) (cached=%s, current=%s). "
                "Recomputing.",
                cache_path.name,
                ", ".join(differing),
                {k: stored.get(k) for k in differing},
                {k: current.get(k) for k in differing},
            )
            return None

        return data

    # --- Hexagonal Distance Comparison ---

    def compute_hexbin_data(
        self, gridsize: int = 50, save_path: Optional[Path] = None
    ) -> Dict:
        """Pre-compute hexbin data for distance comparisons."""
        logger.info("Computing hexbin data for distance comparisons...")

        hexbin_data = {
            "metadata": {
                "dist_cols": self.dist_cols,
                "gridsize": gridsize,
                "max_count": 0,
            }
        }

        n = len(self.dist_cols)
        total_pairs = n * (n - 1)

        with tqdm(total=total_pairs, desc="Computing hexbin data") as pbar:
            for i, col1 in enumerate(self.dist_cols):
                for j, col2 in enumerate(self.dist_cols):
                    if i == j:
                        continue

                    mask = ~(self.df[col1].is_nan() | self.df[col2].is_nan())
                    if mask.sum() < 10:
                        pbar.update(1)
                        continue

                    filtered_df = self.df.filter(mask)
                    x_data = filtered_df[col1].to_numpy()
                    y_data = filtered_df[col2].to_numpy()

                    counts, xedges, yedges = np.histogram2d(
                        x_data, y_data, bins=gridsize
                    )

                    max_count = counts.max()
                    if max_count > hexbin_data["metadata"]["max_count"]:
                        hexbin_data["metadata"]["max_count"] = max_count

                    hexbin_data[f"{col1}_vs_{col2}"] = {
                        "counts": counts.tolist(),
                        "xedges": xedges.tolist(),
                        "yedges": yedges.tolist(),
                    }
                    pbar.update(1)

        if save_path:
            self._save_json_data(hexbin_data, save_path, "Hexbin data")

        return hexbin_data

    def plot_hexagonal_distance_comparison(
        self,
        hexbin_data: Optional[Dict] = None,
        gridsize: int = 50,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Create a grid of hexbin plots showing distance comparisons between embeddings."""
        if hexbin_data is None:
            hexbin_data = self.compute_hexbin_data(gridsize)

        logger.info("Creating hexagonal distance comparison plot...")

        dist_cols = hexbin_data["metadata"]["dist_cols"]
        n = len(dist_cols)
        vmax = hexbin_data["metadata"]["max_count"]

        fig, axes = plt.subplots(
            n, n, figsize=(20 * self.font_scale, 18 * self.font_scale)
        )
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        with tqdm(total=n * n, desc="Creating hexagonal plots") as pbar:
            for i, col1 in enumerate(dist_cols):
                for j, col2 in enumerate(dist_cols):
                    ax = axes[i, j]

                    if i == j:
                        # Diagonal: show embedding name with proper display name
                        embedding_key = col1.replace("dist_", "")
                        embedding_name = EMBEDDING_DISPLAY_NAMES.get(
                            embedding_key, embedding_key
                        )
                        ax.text(
                            0.5,
                            0.5,
                            embedding_name,
                            ha="center",
                            va="center",
                            # rotation=45,
                            fontsize=16 * self.font_scale,
                            weight="bold",
                            transform=ax.transAxes,
                        )
                        ax.axis("off")
                    elif i < j:
                        # Upper triangle: turn off (show only lower triangle)
                        ax.axis("off")
                    else:
                        # Lower triangle: show hexbin plots
                        key = f"{col1}_vs_{col2}"
                        if key in hexbin_data:
                            self._plot_hexbin_pair(ax, hexbin_data[key], vmax)
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                "No Data",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.axis("off")

                    # Add labels for edge plots with proper display names
                    if i == n - 1:
                        embedding_key = col2.replace("dist_", "")
                        label = EMBEDDING_DISPLAY_NAMES.get(
                            embedding_key, embedding_key
                        )
                        ax.set_xlabel(
                            label,
                            rotation=0,
                            ha="center",
                            fontsize=16 * self.font_scale,
                        )
                    if j == 0:
                        embedding_key = col1.replace("dist_", "")
                        label = EMBEDDING_DISPLAY_NAMES.get(
                            embedding_key, embedding_key
                        )
                        ax.set_ylabel(
                            label,
                            rotation=0,
                            ha="right",
                            fontsize=16 * self.font_scale,
                        )

                    pbar.update(1)

        # Add title and colorbar
        fig.suptitle(
            "PairwiseDistance Comparison",
            x=0.5,
            y=0.91,
            fontsize=22 * self.font_scale,
            weight="bold",
        )

        # Create a dummy mappable for colorbar
        im = plt.cm.ScalarMappable(cmap="pink", norm=plt.Normalize(vmin=1, vmax=vmax))
        cbar = fig.colorbar(
            im, ax=axes.ravel().tolist(), label="Count", shrink=1.0, aspect=30, pad=0.02
        )
        cbar.ax.tick_params(labelsize=14 * self.font_scale)
        cbar.set_label("Count", fontsize=16 * self.font_scale)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Hexagonal comparison plot saved to {save_path}")

        return fig, axes

    def _plot_hexbin_pair(self, ax: plt.Axes, data: Dict, vmax: float):
        """Plot a single hexbin pair on the given axes."""
        counts = np.array(data["counts"])
        xedges = np.array(data["xedges"])
        yedges = np.array(data["yedges"])

        X, Y = np.meshgrid(
            xedges[:-1] + np.diff(xedges) / 2, yedges[:-1] + np.diff(yedges) / 2
        )

        masked_counts = np.ma.masked_where(counts == 0, counts)
        ax.pcolormesh(X, Y, masked_counts.T, cmap="pink", vmin=1, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Correlation Analysis ---

    def compute_correlation_data(self, save_path: Optional[Path] = None) -> Dict:
        """Pre-compute Spearman correlations and confidence intervals for distance columns."""
        logger.info("Computing correlation data...")

        n = len(self.dist_cols)
        correlations = np.full((n, n), np.nan)
        ci_lower = np.full((n, n), np.nan)
        ci_upper = np.full((n, n), np.nan)

        with tqdm(total=(n * (n + 1)) // 2, desc="Calculating correlations") as pbar:
            for i in range(n):
                for j in range(i, n):
                    mask = ~(
                        self.df[self.dist_cols[i]].is_nan()
                        | self.df[self.dist_cols[j]].is_nan()
                    )
                    if mask.sum() > 3:
                        filtered_df = self.df.filter(mask)
                        correlation, _ = stats.spearmanr(
                            filtered_df[self.dist_cols[i]],
                            filtered_df[self.dist_cols[j]],
                        )
                        correlations[i, j] = correlations[j, i] = correlation

                        # Compute confidence intervals
                        if abs(correlation) > 0.9999:
                            ci_lower[i, j] = ci_lower[j, i] = correlation
                            ci_upper[i, j] = ci_upper[j, i] = correlation
                        else:
                            z = np.arctanh(correlation)
                            sigma = 1.0 / np.sqrt(mask.sum() - 3)
                            z_ci = stats.norm.interval(0.95, loc=z, scale=sigma)
                            ci = np.tanh(z_ci)
                            ci_lower[i, j] = ci_lower[j, i] = ci[0]
                            ci_upper[i, j] = ci_upper[j, i] = ci[1]

                    pbar.update(1)

        data = {
            "correlations": correlations.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
            "columns": [col.replace("dist_", "") for col in self.dist_cols],
        }

        if save_path:
            self._save_json_data(data, save_path, "Correlation data")

        return data

    def plot_correlation_heatmap(
        self,
        correlation_data: Optional[Dict] = None,
        show_ci: bool = False,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a correlation heatmap from pre-computed correlation data."""
        if correlation_data is None:
            correlation_data = self.compute_correlation_data()

        logger.info("Creating correlation heatmap...")

        correlations = np.array(correlation_data["correlations"])
        columns = correlation_data["columns"]
        n = len(columns)

        fig, ax = plt.subplots(figsize=(15 * self.font_scale, 12 * self.font_scale))

        masked_correlations = np.ma.array(
            correlations, mask=np.isnan(correlations) | np.eye(n, dtype=bool)
        )
        im = ax.imshow(masked_correlations, cmap="OrRd", vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, pad=0.02, aspect=50)
        cbar.ax.tick_params(labelsize=12 * self.font_scale)
        cbar.set_label("Spearman Correlation", fontsize=14 * self.font_scale)

        # Add annotations
        self._add_correlation_annotations(ax, correlation_data, show_ci, n)

        # Customize ticks and labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(
            columns, rotation=45, ha="right", fontsize=14 * self.font_scale
        )
        ax.set_yticklabels(columns, fontsize=14 * self.font_scale)

        title = "Spearman Correlations"
        if show_ci:
            title += " with 95% CI"
        plt.title(title, pad=20, fontsize=16 * self.font_scale, weight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Correlation heatmap saved to {save_path}")

        return fig, ax

    def _add_correlation_annotations(
        self, ax: plt.Axes, data: Dict, show_ci: bool, n: int
    ):
        """Add correlation annotations to the heatmap."""
        correlations = np.array(data["correlations"])
        columns = data["columns"]

        if show_ci:
            ci_lower = np.array(data["ci_lower"])
            ci_upper = np.array(data["ci_upper"])

        with tqdm(total=n * n, desc="Adding correlation annotations") as pbar:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        # Diagonal: show embedding name
                        embedding_name = columns[i].replace("_", "\n")
                        if embedding_name == "prottucker":
                            embedding_name = "prot-\ntucker"
                        elif embedding_name == "prostt5":
                            embedding_name = "prost-\nt5"
                        elif embedding_name == "prott5":
                            embedding_name = "prot-\nt5"
                        ax.text(
                            j,
                            i,
                            embedding_name,
                            ha="center",
                            va="center",
                            fontsize=14 * self.font_scale,
                            weight="bold",
                            color=self._get_embedding_color(columns[i]),
                        )
                    elif not np.isnan(correlations[i, j]):
                        if show_ci:
                            text = f"{correlations[i, j]:.2f}\n[{ci_lower[i, j]:.2f}, {ci_upper[i, j]:.2f}]"
                        else:
                            text = f"{correlations[i, j]:.2f}"

                        color = "white" if abs(correlations[i, j]) > 0.5 else "black"
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=16 * self.font_scale,
                        )
                    else:
                        ax.text(
                            j,
                            i,
                            "NA",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=16 * self.font_scale,
                        )
                    pbar.update(1)

    # --- Wasserstein Distance Analysis ---

    @staticmethod
    def normalize_distribution(x: pl.Series) -> np.ndarray:
        """Normalize a distribution to [0,1] range using MinMax scaling."""
        # Drop nulls and convert to numpy for compatibility with existing code
        x_clean = x.drop_nulls().to_numpy()
        if len(x_clean) == 0:
            return x_clean

        # Additional check for infinite values
        x_clean = x_clean[np.isfinite(x_clean)]
        if len(x_clean) == 0:
            return x_clean

        # Check if all values are the same (would cause division by zero)
        if np.all(x_clean == x_clean[0]):
            # Return array of 0.5s (middle of [0,1] range)
            return np.full_like(x_clean, 0.5)

        scaler = MinMaxScaler()
        return scaler.fit_transform(x_clean.reshape(-1, 1)).ravel()

    def _compute_wasserstein_pair(self, col1: str, col2: str) -> Tuple[float, int]:
        """Compute Wasserstein distance between two columns."""
        # Create a mask for rows where both columns have valid values
        mask = ~(self.df[col1].is_nan() | self.df[col2].is_nan())
        valid_df = self.df.filter(mask)

        if len(valid_df) < 10:
            return np.nan, len(valid_df)

        # Normalize the distributions using only the valid data
        dist1_normalized = self.normalize_distribution(valid_df[col1])
        dist2_normalized = self.normalize_distribution(valid_df[col2])

        if len(dist1_normalized) == 0 or len(dist2_normalized) == 0:
            return np.nan, len(valid_df)

        try:
            dist = wasserstein_distance(dist1_normalized, dist2_normalized)
            logger.debug(
                f"{col1} vs {col2}: {len(valid_df)} samples, distance = {dist:.4f}"
            )
            return dist, len(valid_df)
        except Exception as e:
            logger.warning(
                f"Error computing Wasserstein distance for {col1} vs {col2}: {e}"
            )
            return np.nan, len(valid_df)

    def compute_wasserstein_data(self, save_path: Optional[Path] = None) -> Dict:
        """Compute Wasserstein distances between all pairs of normalized distance distributions."""
        logger.info("Computing Wasserstein distances...")

        n = len(self.dist_cols)
        distances = np.zeros((n, n))

        # Compute pairwise distances
        with tqdm(total=(n * (n + 1)) // 2, desc="Computing distances") as pbar:
            for i in range(n):
                for j in range(i, n):
                    col1 = self.dist_cols[i]
                    col2 = self.dist_cols[j]

                    dist, sample_count = self._compute_wasserstein_pair(col1, col2)
                    distances[i, j] = distances[j, i] = dist

                    if np.isnan(dist) and sample_count < 10:
                        logger.warning(
                            f"Insufficient valid samples for {col1} vs {col2}: {sample_count} samples"
                        )

                    pbar.update(1)

        data = {
            "distances": distances.tolist(),
            "columns": [col.replace("dist_", "") for col in self.dist_cols],
        }

        if save_path:
            self._save_json_data(data, save_path, "Wasserstein data")

        return data

    def plot_wasserstein_heatmap(
        self,
        wasserstein_data: Optional[Dict] = None,
        show_values: bool = True,
        cmap: str = "Blues",
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a heatmap from pre-computed Wasserstein distance data."""
        if wasserstein_data is None:
            wasserstein_data = self.compute_wasserstein_data()

        logger.info("Creating Wasserstein distance heatmap...")

        distances = np.array(wasserstein_data["distances"])
        columns = wasserstein_data["columns"]
        n = len(columns)

        fig, ax = plt.subplots(figsize=(15 * self.font_scale, 12 * self.font_scale))

        masked_distances = np.ma.array(distances, mask=np.isnan(distances))
        im = ax.imshow(masked_distances, cmap=cmap)

        # Add colorbar
        cbar = plt.colorbar(im, pad=0.02, aspect=50)
        cbar.ax.tick_params(labelsize=10 * self.font_scale)
        cbar.set_label("Normalized Wasserstein Distance", fontsize=12 * self.font_scale)

        # Add annotations
        if show_values:
            self._add_wasserstein_annotations(ax, distances, columns, cmap, n)

        # Customize ticks and labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(
            columns, rotation=45, ha="right", fontsize=10 * self.font_scale
        )
        ax.set_yticklabels(columns, fontsize=10 * self.font_scale)

        plt.title(
            "Normalized Wasserstein Distances\nBetween Distance Embeddings",
            pad=20,
            fontsize=14 * self.font_scale,
            weight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Wasserstein heatmap saved to {save_path}")

        return fig, ax

    def _add_wasserstein_annotations(
        self, ax: plt.Axes, distances: np.ndarray, columns: List[str], cmap: str, n: int
    ):
        """Add annotations to Wasserstein heatmap with optimized text colors."""
        colormap = plt.colormaps[cmap]

        with tqdm(total=n * n, desc="Adding Wasserstein annotations") as pbar:
            for i in range(n):
                for j in range(n):
                    if not np.isnan(distances[i, j]):
                        if i == j:
                            text = columns[i]
                            weight = "bold"
                        else:
                            text = f"{distances[i, j]:.2f}"
                            weight = "normal"

                        # Optimize text color based on background
                        normalized_value = (distances[i, j] - np.nanmin(distances)) / (
                            np.nanmax(distances) - np.nanmin(distances)
                        )
                        rgba = colormap(normalized_value)
                        brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        color = "white" if brightness < 0.6 else "black"

                        rotation = 45 if i == j else 0
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            rotation=rotation,
                            color=color,
                            fontsize=8 * self.font_scale,
                            weight=weight,
                        )
                    else:
                        ax.text(
                            j,
                            i,
                            "NA",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=8 * self.font_scale,
                        )

    # --- Distribution Analysis ---

    def plot_ridge_distributions(
        self,
        distribution_data: Optional[Dict] = None,
        show_median: bool = True,
        alpha: float = 0.8,
        save_path: Optional[Path] = None,
        ranking_csv: Optional[Path] = None,
        overlap: float = 0.25,
        row_height: float = 1.0,
        xlim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        iqr_band: bool = False,
        quartile_labels: bool = False,
        quartile_legend: bool = False,
        tail_marks: bool = False,
        legend_loc: str = "row",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a ridge plot using pre-computed distribution data.

        The defaults reproduce the published Figure 2 exactly; every new argument is
        opt-in.  What each of them controls, and why the figure lives or dies on them:

        ``overlap``
            Rows are laid out with ``hspace=-overlap``, i.e. a *negative* gridspec gap,
            and the axes are transparent.  That negative gap is the entire joyplot
            effect -- each row's curve is allowed to rise into the row above it.  Set it
            to 0 and you get fifteen separate panels, which is what a plain
            ``plt.subplots`` grid gives you and why such a grid reads as a stack of
            strips rather than a landscape.
        ``row_height``
            Inches per row.  Each row is autoscaled to its own peak, so this is what
            decides how tall a distribution is drawn; at 0.6 in the curves flatten into
            smears.  The published figure uses 1.0.
        ``xlim``
            The published figure hardcodes (0, 1) because min-max scaling guarantees it.
            Any other normalisation needs its own limits, and hardcoding (0, 1) under a
            percentile scale silently clips the top percent of every row.
        ``iqr_band``
            Shades q25..q75 in a darker tint of the row colour.  The quartile *lines*
            alone are easy to lose against a saturated fill; the band makes the middle
            half of the distribution readable at a glance, which is the one thing the
            figure is supposed to show.
        ``tail_marks``
            Adds the 1st and 99th percentile to every row as short ticks standing on
            the baseline, completing the 1/25/50/75/99 set.  They are ticks and not
            full-height rules for a mechanical reason: at p1 and p99 the density is
            ~0, so a rule drawn from the baseline up to the curve -- how q25, the
            median and q75 are drawn -- would have no length at all.  A tick of fixed
            height is the only mark that is visible where the curve is not.  Left and
            right are unambiguous by construction (p1 < q25, p99 > q75), so both ticks
            share one glyph and one legend entry; the top row is additionally
            annotated "1st" and "99th" so no reader has to infer it.
        ``legend_loc``
            ``"row"`` keeps the key inside the bottom row's axes, where it sat before
            and where, at 300 dpi, it lands on the *second-to-last* row's baseline --
            the axes overlap, so "inside the last row" is not the same as "in empty
            space".  ``"figure"`` puts it in the figure's bottom margin under the axis
            label instead, which is outside every row's axes by construction; the
            placement is then asserted rather than eyeballed, see ``_assert_no_overlap``.
        """
        if legend_loc not in ("row", "figure"):
            raise ValueError(f"legend_loc must be 'row' or 'figure', got {legend_loc!r}")
        if distribution_data is None:
            distribution_data = self.compute_distribution_data(normalize=True)

        meta = distribution_data.get("metadata", {})
        if xlim is None:
            xlim = tuple(meta.get("xlim", (0.0, 1.0)))
        # The axis label belongs to the normalisation, so it travels with the data
        # rather than being passed alongside it.
        x_label = meta.get("label", "Min-Max Normalized Distances")
        if title is None:
            title = "Normalized Pairwise All-vs-All Distance Distributions"

        logger.info(
            "Creating optimized ridge plot using pre-computed distribution data..."
        )

        # Extract PLM names - default order is by family and parameter size
        # (from _identify_distance_columns sorting)
        plm_names = [col.replace("dist_", "") for col in self.dist_cols]

        # Optionally sort by performance ranking if provided
        if ranking_csv is not None and ranking_csv.exists():
            logger.info(f"Loading PLM ranking from {ranking_csv}")
            ranking_df = pl.read_csv(ranking_csv)
            # Create mapping from embedding name to rank
            rank_map = {
                row["Embedding"]: row["Final_Rank"]
                for row in ranking_df.iter_rows(named=True)
            }
            # Sort plm_names by rank (lower rank = better, so put at top)
            plm_names = sorted(
                plm_names,
                key=lambda x: rank_map.get(x, 999),  # Unranked goes to bottom
                reverse=False,  # Best (rank 1) is at top
            )
            logger.info("Sorted PLMs by performance ranking (best at top)")
        else:
            logger.info(
                "No ranking CSV provided, using default order (sorted by PLM family, then parameter size)"
            )

        # Set style for ridge plot
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Extract or estimate percentile values from pre-computed distribution data
        percentile_data = {}
        tail_data: Dict[str, Dict[str, float]] = {}
        if show_median:
            logger.info(
                "Extracting/estimating percentiles from precomputed distribution data..."
            )
            for col in self.dist_cols:
                plm_name = col.replace("dist_", "")
                if col in distribution_data["distributions"]:
                    dist_data = distribution_data["distributions"][col]

                    # Exact quantiles when the producer computed them on the values
                    # (the summary path does).  Estimating a quartile off the cumulative
                    # sum of a smoothed 500-point grid is accurate to about one cell,
                    # which is fine for drawing a line and not fine for a number quoted
                    # in a caption -- and these are the same numbers the caption quotes.
                    if all(k in dist_data for k in ("q25", "median", "q75")):
                        percentile_data[plm_name] = {
                            "q25": float(dist_data["q25"]),
                            "median": float(dist_data["median"]),
                            "q75": float(dist_data["q75"]),
                        }
                        # The tails come from the same exact-quantile source or not at
                        # all: estimating a 99th percentile off a 500-cell smoothed
                        # curve would put the tick wherever the bandwidth put it.
                        if all(k in dist_data for k in RIDGE_TAIL_PERCENTILES):
                            tail_data[plm_name] = {
                                k: float(dist_data[k]) for k in RIDGE_TAIL_PERCENTILES
                            }
                        continue

                    # First try to get precomputed median
                    median_val = dist_data.get("median")

                    # Estimate percentiles from KDE
                    x_range = np.array(dist_data["x_range"])
                    density = np.array(dist_data["density"])

                    if len(x_range) > 1 and len(density) > 1:
                        # Normalize density to integrate to 1
                        dx = x_range[1] - x_range[0]
                        density_normalized = density / (np.sum(density) * dx)

                        # Compute cumulative distribution
                        cumulative = np.cumsum(density_normalized) * dx

                        # Estimate percentiles
                        percentiles = {}
                        for p_name, p_val in [
                            ("q25", 0.25),
                            ("median", 0.5),
                            ("q75", 0.75),
                        ]:
                            # Use precomputed median if available
                            if (
                                p_name == "median"
                                and median_val is not None
                                and not np.isnan(median_val)
                            ):
                                percentiles[p_name] = median_val
                            else:
                                p_idx = np.searchsorted(cumulative, p_val)
                                if p_idx < len(x_range):
                                    percentiles[p_name] = float(x_range[p_idx])
                                else:
                                    percentiles[p_name] = None

                        if all(v is not None for v in percentiles.values()):
                            percentile_data[plm_name] = percentiles
                            logger.debug(
                                f"Estimated percentiles for {col}: "
                                f"Q25={percentiles['q25']:.4f}, "
                                f"Median={percentiles['median']:.4f}, "
                                f"Q75={percentiles['q75']:.4f}"
                            )
                        else:
                            logger.debug(
                                f"Could not estimate all percentiles for {col}"
                            )
                    else:
                        logger.warning(
                            f"Insufficient data to estimate percentiles for {col}"
                        )
                else:
                    logger.warning(f"No distribution data found for {col}")

        # Set up the figure - use a single large figure and manually position subplots
        fig = plt.figure(
            figsize=(20 * self.font_scale, len(plm_names) * row_height * self.font_scale)
        )

        # Negative hspace is the ridgeline: it lets each row's curve rise into the row
        # above it.  With the transparent axes facecolor set just above, that overlap is
        # what turns fifteen strips into one landscape.
        gs = fig.add_gridspec(len(plm_names), 1, hspace=-abs(overlap))

        axes = []
        tail_marks_to_draw: List[Tuple] = []
        for i, plm_name in enumerate(plm_names):
            # Create subplot
            ax = fig.add_subplot(gs[i])
            axes.append(ax)

            col = f"dist_{plm_name}"

            if col not in distribution_data["distributions"]:
                logger.warning(f"No distribution data found for {col}")
                ax.axis("off")
                continue

            dist_data = distribution_data["distributions"][col]
            x_range = np.array(dist_data["x_range"])
            density = np.array(dist_data["density"])
            color = self._get_embedding_color(col)

            # Plot the distribution - key insight: plot normally, let overlap create ridge effect
            ax.fill_between(x_range, density, alpha=alpha, color=color)

            # Darker tint over the middle half.  The quartile lines alone disappear into
            # a saturated fill; the band is what makes "where does the middle 50% sit"
            # legible without reading three thin lines off a coloured background.
            if iqr_band and show_median and plm_name in percentile_data:
                pc = percentile_data[plm_name]
                inside = (x_range >= pc["q25"]) & (x_range <= pc["q75"])
                if inside.any():
                    ax.fill_between(
                        x_range,
                        density,
                        where=inside,
                        color=mcolors.to_rgb(color),
                        alpha=min(1.0, alpha + 0.18),
                        linewidth=0,
                        zorder=1.2,
                    )
                    ax.fill_between(
                        x_range,
                        density,
                        where=inside,
                        color="black",
                        alpha=0.16,
                        linewidth=0,
                        zorder=1.3,
                    )

            ax.plot(x_range, density, color="white", linewidth=2, zorder=2)

            # Draw baseline at y=0
            ax.axhline(y=0, color="black", linewidth=2, clip_on=False)

            # Add percentile lines if requested
            if show_median and plm_name in percentile_data:
                percentiles = percentile_data[plm_name]

                for p_name, p_val in percentiles.items():
                    # The guard used to be a hardcoded ``0 <= p_val <= 1``, which is only
                    # correct because min-max scaling happens to produce that range.
                    # Under any other normalisation it silently drops the lines this
                    # figure exists to show.
                    if p_val is None or np.isnan(p_val):
                        continue
                    if not (xlim[0] <= p_val <= xlim[1]):
                        logger.warning(
                            "%s %s = %.4g falls outside the drawn range %s; not drawn",
                            plm_name,
                            p_name,
                            p_val,
                            xlim,
                        )
                        continue
                    # Interpolate to find the height at this percentile
                    p_y = np.interp(p_val, x_range, density)
                    # Only draw if interpolated y is valid
                    if not np.isnan(p_y) and p_y >= 0:
                        style = RIDGE_PERCENTILE_STYLES[p_name]
                        ax.plot(
                            [p_val, p_val],
                            [0, p_y],
                            **style,
                            alpha=1.0,
                            clip_on=False,
                            zorder=3,
                            solid_capstyle="butt",
                            path_effects=[
                                pe.withStroke(linewidth=style["linewidth"] + 2.2,
                                              foreground="white", alpha=0.85)
                            ],
                        )
                        if quartile_labels and p_name == "median":
                            ax.annotate(
                                f"{p_val:.2f}",
                                xy=(p_val, p_y),
                                xytext=(3, 2),
                                textcoords="offset points",
                                fontsize=int(13 * self.font_scale),
                                fontweight="bold",
                                color="black",
                                ha="left",
                                va="bottom",
                                zorder=4,
                                path_effects=[
                                    pe.withStroke(linewidth=3, foreground="white")
                                ],
                            )

            # --- 1st and 99th percentile: ticks standing on the baseline -----------
            # Height is a fraction of the ROW, not of the density, for the reason in
            # the docstring: the density at p1/p99 is ~0, so a mark scaled by it has no
            # length.  The row's peak density is what the row's axes autoscales to, so
            # a fixed fraction of it is a fixed fraction of the row's drawn height --
            # the ticks come out the same size on every row whatever the density units
            # of that row happen to be.
            if tail_marks and plm_name in tail_data:
                tick_h = 0.20 * float(np.nanmax(density))
                for p_name in RIDGE_TAIL_PERCENTILES:
                    p_val = tail_data[plm_name][p_name]
                    if p_val is None or np.isnan(p_val):
                        continue
                    if not (xlim[0] <= p_val <= xlim[1]):
                        logger.warning(
                            "%s %s = %.4g falls outside the drawn range %s; tick not "
                            "drawn",
                            plm_name,
                            p_name,
                            p_val,
                            xlim,
                        )
                        continue
                    # Queued, not drawn here.  A tick added to this row's axes is
                    # painted before the next row's axes and therefore *under* the
                    # next row's fill -- which in the ridgeline rises above this row's
                    # baseline and swallowed the tick whole wherever the two lined up
                    # (Ankh Base's 1st percentile disappeared under Ankh Large).  They
                    # are drawn at figure level after the loop instead, where nothing
                    # can be painted over them.
                    tail_marks_to_draw.append((ax, p_val, tick_h, p_name, i == 0))

            # Add PLM label on the left (remove \n for ridge plot)
            plm_display_name = EMBEDDING_DISPLAY_NAMES.get(plm_name, plm_name).replace(
                "\n", " "
            )
            ax.text(
                -0.02,
                0.1,
                plm_display_name,
                fontweight="bold",
                color=color,
                ha="right",
                va="center",
                transform=ax.transAxes,
                fontsize=int(26 * self.font_scale),
            )

            # Style the subplot
            ax.set_xlim(*xlim)
            ax.set_ylim(bottom=0)  # Start y-axis at 0

            # Remove y-axis elements
            ax.set_yticks([])
            ax.set_ylabel("")

            # Remove unnecessary spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # X-axis: only show on bottom subplot
            if i == len(plm_names) - 1:
                # Bottom subplot: full x-axis
                ax.spines["bottom"].set_visible(True)
                # Ticks follow the actual limits.  ``np.arange(0, 1.1, 0.2)`` is only
                # right for a [0, 1] axis; on a wider or a log axis it labels part of it
                # and leaves the rest bare.
                step = _nice_tick_step(xlim[1] - xlim[0])
                ax.set_xticks(
                    np.arange(
                        np.ceil(xlim[0] / step) * step,
                        xlim[1] + 0.5 * step,
                        step,
                    )
                )
                if meta.get("normalisation") == "log10":
                    ax.xaxis.set_major_formatter(
                        mticker.FuncFormatter(lambda v, _: f"$10^{{{v:g}}}$")
                    )
                ax.set_xlabel(
                    x_label,
                    fontsize=int(28 * self.font_scale),
                    labelpad=15,
                )
                ax.tick_params(
                    axis="x", which="major", labelsize=int(20 * self.font_scale)
                )
            else:
                # Other subplots: no x-axis
                ax.spines["bottom"].set_visible(False)
                ax.set_xticks([])
                ax.tick_params(axis="x", which="both", length=0, labelbottom=False)

        # A key for the three lines.  The published figure drew q25/median/q75 with no
        # legend at all, so a reader had no way to know what the dotted lines were --
        # they are the figure's only quantitative content and they were unlabelled.
        # --- tail ticks, drawn above every row ---------------------------------
        # Figure-level artists are painted after all axes, so a tick added here cannot
        # be covered by the row below rising into this row's band.  The transform stays
        # the owning row's data transform, so the tick still stands exactly on that
        # row's baseline at exactly that row's percentile.
        foot = 0.012 * (xlim[1] - xlim[0])
        for ax, p_val, tick_h, p_name, is_top_row in tail_marks_to_draw:
            # The transform is the owning row's, so the mark is placed in that row's
            # data coordinates even though the artist belongs to the figure.
            tick_style = dict(
                transform=ax.transData,
                color="0.12",
                linewidth=2.2,
                zorder=5,
                solid_capstyle="butt",
                path_effects=[
                    pe.withStroke(linewidth=4.4, foreground="white", alpha=0.9)
                ],
            )
            # A short piece of the row's own baseline under the tick.  Without it the
            # tick reads as floating: where the row below rises past this row's
            # baseline it hides it, so at exactly the x values where the tick most
            # needs an anchor there is no visible line for it to stand on (ESM 1b's
            # 99th percentile sits in the middle of ESM2 8M's body).
            fig.add_artist(
                plt.Line2D([p_val - foot, p_val + foot], [0, 0], **tick_style)
            )
            fig.add_artist(plt.Line2D([p_val, p_val], [0, tick_h], **tick_style))
            # Annotate the top row only.  Repeating "1st"/"99th" on fourteen rows would
            # be the clutter the ticks exist to avoid, and one labelled row is enough
            # to fix the reading of the other thirteen.  Set outward -- p1's label left
            # of its tick, p99's right -- so neither is written across the curve whose
            # tail the tick is marking.
            if is_top_row:
                outward = -1 if p_name == "p1" else 1
                # ...unless the tick already sits against the axis edge, where there is
                # no room outward: the label crosses into the row-label gutter and reads
                # as part of the model name ("Ankh Base 1st" on the aligner-found twin,
                # whose top-row p1 is 0.020 against 0.191 in Figure 2).  Flip it inward.
                lo, hi = ax.get_xlim()
                margin = 0.05 * (hi - lo)
                if (p_val - lo) < margin:
                    outward = 1
                elif (hi - p_val) < margin:
                    outward = -1
                fig.add_artist(
                    mtext.Annotation(
                        {"p1": "1st", "p99": "99th"}[p_name],
                        xy=(p_val, tick_h),
                        xycoords=ax.transData,
                        xytext=(7 * outward, -2),
                        textcoords="offset points",
                        # Same size as the key.  At 15 pt this label read on screen and
                        # died in print: the figure is 20 in wide and goes into a
                        # 6.5 in column, so every point size on it is multiplied by
                        # 0.325 and 15 pt lands at 4.9 pt on paper.
                        fontsize=int(20 * self.font_scale),
                        fontweight="bold",
                        color="0.12",
                        ha="right" if outward < 0 else "left",
                        va="top",
                        zorder=5,
                        annotation_clip=False,
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                    )
                )

        legend = None
        if quartile_legend and show_median and axes:
            # Same style objects the rules were drawn with, so the key cannot drift
            # away from what is on the rows.
            handles = [
                plt.Line2D([], [], label="median (50th)",
                           **RIDGE_PERCENTILE_STYLES["median"]),
                plt.Line2D([], [], label="25th / 75th percentile",
                           **RIDGE_PERCENTILE_STYLES["q25"]),
            ]
            if tail_marks and tail_data:
                # Drawn as a tick, not as a line segment: a short solid rule in the key
                # would read as a thinner median, which is the one confusion the key
                # exists to prevent.
                handles.append(
                    plt.Line2D([], [], color="0.12", linestyle="none", marker="|",
                               markersize=18, markeredgewidth=2.2,
                               label="1st / 99th percentile (baseline tick)")
                )
            if legend_loc == "figure":
                # The figure's own bottom margin, under the axis label.  This is the
                # only placement that is outside every row's axes *by construction*:
                # the rows are laid out with a negative hspace, so any anchor
                # expressed in one row's axes coordinates can still land on top of a
                # neighbouring row -- which is exactly what "inside the last row,
                # upper right" did to the ProtT5 baseline.  Horizontal (one column
                # per entry) keeps the margin's height free for the axis label.
                legend = fig.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.0),  # refined below, once measurable
                    ncol=len(handles),
                    frameon=False,
                    facecolor="none",
                    edgecolor="none",
                    framealpha=0.0,
                    fontsize=int(20 * self.font_scale),
                    handlelength=2.6,
                    columnspacing=3.0,
                    borderaxespad=0.0,
                )
            else:
                # The published placement, kept for reproducing the old figure: bottom
                # row, upper right, transparent so it does not paint over the baseline
                # of the row above it.  It still *sits* on that row; see legend_loc.
                legend = axes[-1].legend(
                    handles=handles,
                    loc="upper right",
                    bbox_to_anchor=(1.0, 0.92),
                    frameon=False,
                    facecolor="none",
                    edgecolor="none",
                    framealpha=0.0,
                    fontsize=int(20 * self.font_scale),
                    handlelength=2.6,
                    labelspacing=0.35,
                )

        # Add y-axis label on the left side (figure-level)
        fig.text(
            -0.075,  # x position - far left
            0.5,  # y position - middle
            "pLM Models",
            fontsize=int(28 * self.font_scale),
            va="center",
            ha="left",
            rotation=90,
        )

        # Add overall title
        fig.suptitle(
            title,
            fontsize=int(27 * self.font_scale),
            fontweight="bold",
            y=0.92,
        )

        # Check the key really is clear of the plotting area, at the dpi the file is
        # written at.  "It looked fine" is how a legend ends up on a baseline: the old
        # placement was chosen from a screen-resolution preview and overlapped ProtT5
        # in the 300-dpi PNG.
        if legend is not None and legend_loc == "figure":
            _place_legend_below_axis_label(fig, legend, axes[-1])
            _assert_legend_clear(fig, legend, axes, dpi=DEFAULT_STYLE["dpi"])

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Optimized ridge plot saved to {save_path}")

            # Save percentile statistics to CSV
            if percentile_data:
                stats_path = save_path.parent / f"{save_path.stem}_statistics.csv"

                # Convert nested dict to flat structure for CSV
                # Emit the machine key AND the label, the same convention the dip-test
                # output below uses. Writing only the label loses the join key, which
                # forced consumers to guess which spelling of the label a given CSV
                # carried.
                rows = []
                for plm_name, percentiles in percentile_data.items():
                    dist_data = distribution_data["distributions"].get(
                        f"dist_{plm_name}", {}
                    )
                    row = {
                        "plm_name": plm_name,
                        "plm_display_name": EMBEDDING_DISPLAY_NAMES.get(
                            plm_name, plm_name
                        ).replace("\n", " "),
                        "q25": percentiles["q25"],
                        "median": percentiles["median"],
                        "q75": percentiles["q75"],
                        "iqr": percentiles["q75"] - percentiles["q25"],
                        "normalisation": meta.get("normalisation", "minmax"),
                    }
                    # The scale-free columns, when the producer knows them.  QCD =
                    # (q75-q25)/(q75+q25) is invariant under any positive rescaling, so
                    # it is the only honest way to say one arm is "broader" than another
                    # -- the normalised IQR is not, and the published claim that
                    # task-specific training broadens the distribution came from reading
                    # the normalised axis.
                    for key in (
                        "n_used",
                        "n_zero_identical",
                        "p1",
                        "p99",
                        "offset",
                        "divisor",
                        "qcd",
                        "raw_min",
                        "raw_p1",
                        "raw_q25",
                        "raw_median",
                        "raw_q75",
                        "raw_p99",
                        "raw_max",
                        "frac_beyond_xlim",
                    ):
                        if key in dist_data:
                            row[key] = dist_data[key]
                    rows.append(row)

                # Create DataFrame and save to CSV
                stats_df = pl.DataFrame(rows)
                stats_df.write_csv(stats_path)
                logger.info(f"Ridge plot percentile statistics saved to {stats_path}")

        return fig, axes

    def compute_distribution_data_from_summaries(
        self,
        normalisation: str = "p99",
        grid: int = 500,
        xlim: Optional[Tuple[float, float]] = None,
        save_path: Optional[Path] = None,
    ) -> Dict:
        """Build ridge-plot densities from the cluster-reduced histograms.

        Why a histogram and not a KDE over the values.  ``compute_distribution_data``
        calls ``gaussian_kde`` on every distance in the column; at 75.8M points per arm
        that is neither affordable nor necessary, because a 50,000-bin histogram of 75.8M
        points already *is* the population density to far better precision than any
        bandwidth choice.  Re-binning that histogram onto the display grid and applying
        Scott's bandwidth as a Gaussian blur reproduces what the KDE would have drawn,
        from every pair rather than from a sample of them.

        Why the raw histogram can serve every normalisation.  min-max, ``/p99`` and
        ``/median`` are all positive affine maps, so rescaling the bin edges is exactly
        equivalent to rescaling the values.  ``log10`` uses the second, log-spaced
        histogram the reduction wrote for it.

        Quartiles are the exact ones from ``np.percentile`` over the full cohort, carried
        through the same rescaling -- not read off the smoothed curve.
        """
        # The test is whether the summaries are here, full stop.  Pairing it with
        # ``self.df is not None`` missed every other way of building this object that
        # leaves df None -- from_matrices() is one, and it got a bare AttributeError
        # from deep inside the loop instead of the sentence naming the constructor
        # it should have used.
        if not hasattr(self, "summaries"):
            raise RuntimeError(
                "compute_distribution_data_from_summaries needs the summary-backed "
                "constructor: EmbeddingComparisonVisualizer.from_distribution_summaries"
            )
        if normalisation not in RIDGE_NORMALISATIONS:
            raise ValueError(
                f"unknown normalisation {normalisation!r}; "
                f"choose from {sorted(RIDGE_NORMALISATIONS)}"
            )
        spec = RIDGE_NORMALISATIONS[normalisation]
        logger.info("Building ridge densities under normalisation=%s", normalisation)

        # A log axis has no per-row divisor, so its limits come from the data.
        if xlim is None:
            xlim = spec["xlim"]
        if xlim is None:
            # Bound by percentiles, not by min and max.  The minimum of a rounded
            # distance column can be one quantum (1e-4), which would open the axis by
            # four empty decades to accommodate a handful of pairs.
            drawn = [self.summaries[c.replace("dist_", "")] for c in self.dist_cols]
            lo = min(s["quantiles"]["p0.1"] for s in drawn)
            hi = max(s["quantiles"]["p99.9"] for s in drawn)
            xlim = (float(np.floor(np.log10(lo))), float(np.ceil(np.log10(hi))))

        x_edges = np.linspace(xlim[0], xlim[1], grid + 1)
        x_range = 0.5 * (x_edges[:-1] + x_edges[1:])
        dx = float(x_edges[1] - x_edges[0])

        distributions: Dict[str, Dict] = {}
        for col in self.dist_cols:
            arm = col.replace("dist_", "")
            summary = self.summaries[arm]
            with np.load(summary["_npz"]) as npz:
                if normalisation == "log10":
                    counts = npz["hist_log10"].astype(np.float64)
                    edges = npz["edges_log10"]
                    offset, divisor = 0.0, 1.0
                else:
                    counts = npz["hist"].astype(np.float64)
                    edges = npz["edges"]
                    # ``offset`` is what makes min-max actually min-max.  This path used
                    # to divide by ``max`` alone, which is a different map whenever the
                    # smallest distance is not 0 -- and it never is: the closest of 5M
                    # random pairs sits at min/max = 0.0014 (esm1b) to 0.0065 (CLEAN).
                    # The published figure used sklearn's MinMaxScaler, i.e.
                    # (x - min) / (max - min), so dividing by max was also a silent
                    # disagreement with the definition the caption claims.
                    offset = _ridge_anchor(summary, spec.get("offset"), default=0.0)
                    divisor = _ridge_anchor(summary, spec["stat"]) - offset
            centres = 0.5 * (edges[:-1] + edges[1:])

            # Undo the storage quantisation before re-binning.  The distances were
            # written with ``np.round(d, decimals=4)``, so every value sits on a 1e-4
            # lattice.  For an arm with a compressed range that lattice is coarse
            # relative to a display cell -- ankh_large spans 0.379, so a display cell
            # holds four or five lattice points depending on where its edges fall, and
            # the beat between the two draws a visible ripple along the curve.
            # Convolving the fine histogram with a boxcar one quantum wide spreads each
            # lattice point back over the interval it was rounded from, which is exactly
            # the information the rounding destroyed.  Where a raw bin is already wider
            # than the quantum (esm3_open spans 9958) the kernel is one bin and this is
            # a no-op.
            raw_bin_width = float(edges[1] - edges[0])
            if normalisation != "log10":
                quantum_bins = int(round(DISTANCE_QUANTUM / raw_bin_width))
                if quantum_bins > 1:
                    counts = uniform_filter1d(
                        counts, size=quantum_bins, mode="nearest"
                    )
                centres = (centres - offset) / divisor

            n_used = float(summary["n_used"])
            binned, _ = np.histogram(centres, bins=x_edges, weights=counts)
            density = binned / (n_used * dx)

            # Scott's bandwidth, the rule gaussian_kde uses by default, expressed in
            # display cells.  Moments are taken from the histogram so this works for the
            # log axis too, where the summary's mean/std are on the linear scale.
            total = counts.sum()
            mu = float((counts * centres).sum() / total)
            var = float((counts * (centres - mu) ** 2).sum() / total)
            sigma_cells = max((n_used ** (-1 / 5)) * np.sqrt(var) / dx, 0.8)
            density = gaussian_filter1d(density, sigma_cells, mode="nearest")

            # ``offset``/``divisor`` are bound as defaults rather than captured: the
            # closure is rebuilt per arm inside this loop, and a late-binding capture
            # would silently rescale every row by the last arm's divisor if this ever
            # stopped being called in the same iteration.
            def _scale(
                value: float, offset: float = offset, divisor: float = divisor
            ) -> float:
                if normalisation == "log10":
                    return float(np.log10(value)) if value > 0 else float("nan")
                return (float(value) - offset) / divisor

            q = summary["quantiles"]
            tail = float(counts[centres > xlim[1]].sum() / total)
            peak_idx = int(np.argmax(density))
            distributions[col] = {
                "x_range": x_range.tolist(),
                "density": density.tolist(),
                "peak_x": float(x_range[peak_idx]),
                "peak_y": float(density[peak_idx]),
                "min": _scale(summary["min"]),
                "max": _scale(summary["max"]),
                "median": _scale(summary["median"]),
                "q25": _scale(summary["q25"]),
                "q75": _scale(summary["q75"]),
                # The tails, on the drawn axis.  p99 is here because the axis this
                # figure used to be drawn on was built on it, so a reader has to be
                # able to see where it lands under any other axis; p1 is its mirror,
                # and the two together say how wide the drawn middle really is.
                "p1": _scale(q["p1"]),
                "p99": _scale(q["p99"]),
                "offset": offset,
                "divisor": divisor,
                "n_used": int(summary["n_used"]),
                "n_zero_identical": int(summary["n_zero_identical"]),
                "qcd": float(summary["qcd"]),
                "frac_beyond_xlim": tail,
                "raw_p1": float(q["p1"]),
                "raw_q25": float(summary["q25"]),
                "raw_median": float(summary["median"]),
                "raw_q75": float(summary["q75"]),
                "raw_p99": float(q["p99"]),
                "raw_min": float(summary["min"]),
                "raw_max": float(summary["max"]),
                "bandwidth_cells": float(sigma_cells),
            }
            if tail > 0.005:
                logger.info(
                    "%s: %.2f%% of pairs sit beyond the drawn x-limit %.2f",
                    arm,
                    100 * tail,
                    xlim[1],
                )

        data = {
            "metadata": {
                "normalized": normalisation != "log10",
                "normalisation": normalisation,
                "label": spec["label"],
                "xlim": [float(xlim[0]), float(xlim[1])],
                "grid": grid,
                "source": "ridge_full_reduce summaries",
                "identical_sequence_pairs_excluded": True,
            },
            "distributions": distributions,
        }
        if save_path:
            self._save_json_data(data, save_path, "Distribution data (from summaries)")
        return data

    def compute_distribution_data(
        self, normalize: bool = False, save_path: Optional[Path] = None
    ) -> Dict:
        """Pre-compute distribution data for plotting."""
        logger.info(f"Computing distribution data (normalize={normalize})...")

        distribution_data = {"metadata": {"normalized": normalize}, "distributions": {}}

        for col in tqdm(self.dist_cols, desc="Processing distributions"):
            # Get data using polars operations
            col_series = self.df.select(col).drop_nulls().get_column(col)
            data = col_series.to_numpy()

            if normalize:
                # Use normalized data for KDE
                data_normalized = self.normalize_distribution(col_series)
                data_clean = (
                    data_normalized[np.isfinite(data_normalized)]
                    if len(data_normalized) > 0
                    else np.array([])
                )
                x_range = np.linspace(0, 1, 500)
            else:
                # Use original data for KDE
                data_clean = data[np.isfinite(data)] if len(data) > 0 else np.array([])
                if len(data_clean) > 0:
                    x_range = np.linspace(data_clean.min(), data_clean.max(), 500)
                else:
                    x_range = np.linspace(0, 1, 500)

            if len(data_clean) > 1:  # Need at least 2 points for KDE
                kernel = stats.gaussian_kde(data_clean)
                density = kernel(x_range)
                peak_idx = np.argmax(density)
                peak_x = float(x_range[peak_idx])
                peak_y = float(density[peak_idx])
            else:
                density = np.zeros_like(x_range)
                peak_x = peak_y = 0.0

            # Compute median for normalized data
            if normalize and len(data_clean) > 0:
                median_val = float(np.median(data_clean))
            elif not normalize and len(data_clean) > 0:
                # For non-normalized, we still compute it but it won't be used in ridge plots
                median_val = float(np.median(data_clean))
            else:
                median_val = None

            distribution_data["distributions"][col] = {
                "x_range": x_range.tolist(),
                "density": density.tolist(),
                "peak_x": peak_x,
                "peak_y": peak_y,
                "min": float(data_clean.min()) if len(data_clean) > 0 else 0.0,
                "max": float(data_clean.max()) if len(data_clean) > 0 else 0.0,
                "median": median_val,
            }

        if save_path:
            self._save_json_data(distribution_data, save_path, "Distribution data")

        return distribution_data

    def compute_hartigan_dip_test(
        self, save_path: Optional[Path] = None
    ) -> pl.DataFrame:
        """
        Compute Hartigan's dip test on normalized distance distributions for each PLM.

        The dip test measures multimodality - a significant p-value indicates the distribution
        is multimodal (non-unimodal). Lower dip statistics indicate more unimodal distributions.

        Args:
            save_path: Optional path to save the results as CSV

        Returns:
            Polars DataFrame with columns: plm_name, dip_statistic, p_value, n_samples
        """

        logger.info(
            "Computing Hartigan's dip test for normalized distance distributions..."
        )

        results = []

        for col in tqdm(self.dist_cols, desc="Computing dip tests"):
            plm_name = col.replace("dist_", "")

            # Get normalized data
            col_series = self.df.select(col).drop_nulls().get_column(col)
            data_normalized = self.normalize_distribution(col_series)

            if len(data_normalized) < 10:
                logger.warning(
                    f"Insufficient data for {plm_name} ({len(data_normalized)} samples)"
                )
                results.append(
                    {
                        "plm_name": plm_name,
                        "plm_display_name": EMBEDDING_DISPLAY_NAMES.get(
                            plm_name, plm_name
                        ).replace("\n", " "),
                        "dip_statistic": np.nan,
                        "p_value": np.nan,
                        "n_samples": len(data_normalized),
                    }
                )
                continue

            # Compute dip test
            try:
                dip_stat, p_value = diptest(data_normalized)

                results.append(
                    {
                        "plm_name": plm_name,
                        "plm_display_name": EMBEDDING_DISPLAY_NAMES.get(
                            plm_name, plm_name
                        ).replace("\n", " "),
                        "dip_statistic": float(dip_stat),
                        "p_value": float(p_value),
                        "n_samples": len(data_normalized),
                    }
                )

                logger.info(
                    f"{plm_name}: dip={dip_stat:.6f}, p={p_value:.4f}, "
                    f"n={len(data_normalized)}, "
                    f"unimodal={'Yes' if p_value > 0.05 else 'No'}"
                )

            except Exception as e:
                logger.error(f"Error computing dip test for {plm_name}: {e}")
                results.append(
                    {
                        "plm_name": plm_name,
                        "plm_display_name": EMBEDDING_DISPLAY_NAMES.get(
                            plm_name, plm_name
                        ).replace("\n", " "),
                        "dip_statistic": np.nan,
                        "p_value": np.nan,
                        "n_samples": len(data_normalized),
                    }
                )

        # Create DataFrame with family information
        df_results = pl.DataFrame(results)

        # Add family and size columns for sorting
        df_results = df_results.with_columns(
            [
                pl.col("plm_name")
                .map_elements(
                    lambda x: EMBEDDING_FAMILY_MAP.get(x, "Unknown"),
                    return_dtype=pl.Utf8,
                )
                .alias("family"),
                pl.col("plm_name")
                .map_elements(lambda x: PLM_SIZES.get(x, 0), return_dtype=pl.Int64)
                .alias("size"),
            ]
        )

        # Sort by family, then by size within family (for manuscript consistency)
        df_results = df_results.sort(["family", "size"])

        # Analyze p-values
        logger.info("\n=== Hartigan's Dip Test Summary ===")
        logger.info(f"Total PLMs tested: {len(df_results)}")

        # Count p-values that are exactly 0
        p_zero = (df_results["p_value"] == 0.0).sum()
        logger.info(f"P-values = 0.0 (p < machine epsilon): {p_zero}")

        if p_zero == len(df_results):
            logger.info("All p-values are < machine epsilon (~2.2e-16)")
            logger.info("This indicates EXTREMELY strong evidence against unimodality")
            logger.info("All distributions are significantly multimodal")

        # Report dip statistic range
        dip_min = df_results["dip_statistic"].min()
        dip_max = df_results["dip_statistic"].max()
        dip_mean = df_results["dip_statistic"].mean()
        logger.info(f"\nDip statistics range: [{dip_min:.6f}, {dip_max:.6f}]")
        logger.info(f"Mean dip statistic: {dip_mean:.6f}")
        logger.info("\nNote: Dip statistic measures departure from unimodality.")
        logger.info("      Smaller values = more unimodal (closer to single peak)")
        logger.info("      Larger values = more multimodal (multiple peaks)")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Create manuscript-ready table
            df_manuscript = df_results.select(
                [
                    pl.col("plm_display_name").alias("Embedding"),
                    pl.col("dip_statistic").round(4).alias("Dip Statistic"),
                    pl.lit("< 2.2e-16").alias("p-value"),
                ]
            )

            # Save manuscript table
            df_manuscript.write_csv(save_path)
            logger.info(f"\nManuscript table saved to {save_path}")
            logger.info(f"Columns: Embedding, Dip Statistic (4 decimals), p-value")
            logger.info(f"Sorted by: PLM family, then parameter size within family")

        return df_results

    def plot_distributions(
        self,
        distribution_data: Optional[Dict] = None,
        normalize: bool = False,
        alpha: float = 0.05,
        linewidth: float = 2.5,
        show_peaks: bool = True,
        y_break: Optional[Tuple[float, float]] = None,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]]:
        """Create distribution plots from pre-computed data."""
        if distribution_data is None:
            distribution_data = self.compute_distribution_data(normalize)

        logger.info("Creating distribution comparison plot...")

        color_dict = {col: self._get_embedding_color(col) for col in self.dist_cols}

        # Font sizes
        font_sizes = {
            "title": int(24 * self.font_scale),
            "axis": int(22 * self.font_scale),
            "legend": int(16 * self.font_scale),
            "tick": int(14 * self.font_scale),
        }

        figsize = (12 * self.font_scale, 10 * self.font_scale)

        if y_break is None:
            fig, ax = self._create_single_distribution_plot(
                distribution_data,
                color_dict,
                alpha,
                linewidth,
                show_peaks,
                font_sizes,
                figsize,
            )
            axes = ax
        else:
            fig, (ax1, ax2) = self._create_broken_distribution_plot(
                distribution_data,
                color_dict,
                alpha,
                linewidth,
                show_peaks,
                y_break,
                font_sizes,
                figsize,
            )
            axes = (ax1, ax2)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Distribution plot saved to {save_path}")

        return fig, axes

    def _create_single_distribution_plot(
        self,
        distribution_data: Dict,
        color_dict: Dict,
        alpha: float,
        linewidth: float,
        show_peaks: bool,
        font_sizes: Dict,
        figsize: Tuple,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a single distribution plot."""
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Collect peak data for smart positioning
        peak_data = []

        for dist_name, color in tqdm(color_dict.items(), desc="Plotting distributions"):
            if dist_name not in distribution_data["distributions"]:
                continue

            dist_data = distribution_data["distributions"][dist_name]
            x_range = np.array(dist_data["x_range"])
            density = np.array(dist_data["density"])

            ax.fill_between(x_range, density, alpha=alpha, color=color)
            ax.plot(
                x_range,
                density,
                color=color,
                linewidth=linewidth,
                label=dist_name.replace("dist_", ""),
            )

            if show_peaks:
                peak_data.append(
                    (
                        dist_data["peak_x"],
                        dist_data["peak_y"],
                        dist_name.replace("dist_", ""),
                        color,
                    )
                )

        # Add peak labels with smart positioning
        if show_peaks:
            self._add_peak_labels_smart(ax, peak_data)

        self._customize_distribution_plot(ax, distribution_data, font_sizes)

        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fontsize=font_sizes["legend"],
            markerscale=2.0,  # Make legend markers 2x larger
        )
        fig.subplots_adjust(right=0.85)

        return fig, ax

    def _create_broken_distribution_plot(
        self,
        distribution_data: Dict,
        color_dict: Dict,
        alpha: float,
        linewidth: float,
        show_peaks: bool,
        y_break: Tuple[float, float],
        font_sizes: Dict,
        figsize: Tuple,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """Create a distribution plot with broken y-axis."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.08)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        fig.patch.set_facecolor("white")
        ax1.set_facecolor("white")
        ax2.set_facecolor("white")

        # Collect peak data for smart positioning
        peak_data_ax1 = []
        peak_data_ax2 = []

        # Plot on both axes
        for dist_name, color in tqdm(color_dict.items(), desc="Plotting distributions"):
            if dist_name not in distribution_data["distributions"]:
                continue

            dist_data = distribution_data["distributions"][dist_name]
            x_range = np.array(dist_data["x_range"])
            density = np.array(dist_data["density"])

            for ax in [ax1, ax2]:
                ax.fill_between(x_range, density, alpha=alpha, color=color)
                ax.plot(
                    x_range,
                    density,
                    color=color,
                    linewidth=linewidth,
                    label=dist_name.replace("dist_", ""),
                )

            if show_peaks:
                if dist_data["peak_y"] > y_break[1]:
                    peak_data_ax1.append(
                        (
                            dist_data["peak_x"],
                            dist_data["peak_y"],
                            dist_name.replace("dist_", ""),
                            color,
                        )
                    )
                elif dist_data["peak_y"] < y_break[0]:
                    peak_data_ax2.append(
                        (
                            dist_data["peak_x"],
                            dist_data["peak_y"],
                            dist_name.replace("dist_", ""),
                            color,
                        )
                    )

        # Add peak labels with smart positioning
        if show_peaks:
            self._add_peak_labels_smart(ax1, peak_data_ax1)
            self._add_peak_labels_smart(ax2, peak_data_ax2)

        # Set y-limits and add broken axis marks
        ax1.set_ylim(y_break[1], None)
        ax2.set_ylim(0, y_break[0])
        self._add_broken_axis_marks(ax1, ax2)

        # Customize plots
        ax1.set_xticklabels([])
        ax1.tick_params(axis="x", which="both", length=0)

        is_normalized = distribution_data["metadata"]["normalized"]
        xlabel = "Normalized Distance" if is_normalized else "Distance"
        title = (
            "Distribution Comparison of\n"
            + ("Normalized " if is_normalized else "")
            + "Distance Embeddings"
        )

        ax2.set_xlabel(xlabel, fontsize=font_sizes["axis"], labelpad=15)
        ax1.set_title(title, fontsize=font_sizes["title"], pad=20, weight="bold")
        ax2.set_ylabel("Density", fontsize=font_sizes["axis"], labelpad=15)

        for ax in [ax1, ax2]:
            ax.tick_params(axis="both", which="major", labelsize=font_sizes["tick"])
            ax.grid(True, alpha=0.2)

        ax1.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            fontsize=font_sizes["legend"],
            markerscale=2.0,  # Make legend markers 2x larger
        )
        fig.subplots_adjust(right=0.85)

        return fig, (ax1, ax2)

    def _add_peak_labels_smart(
        self, ax: plt.Axes, peak_data: List[Tuple[float, float, str, str]]
    ):
        """Add peak labels with smart positioning to avoid overlaps."""
        if not peak_data:
            return

        # Sort by x position
        peak_data = sorted(peak_data, key=lambda x: x[0])

        # Calculate positions with overlap avoidance
        positions = []
        for i, (x, y, label, color) in enumerate(peak_data):
            base_text_y = y + 0.05 * y

            # Check for overlaps with previous labels
            final_text_y = base_text_y
            for prev_x, prev_y in positions:
                # If x positions are close, adjust y position
                if abs(x - prev_x) < 0.15 * (ax.get_xlim()[1] - ax.get_xlim()[0]):
                    final_text_y = max(final_text_y, prev_y + 0.05 * y)

            positions.append((x, final_text_y))

            # Add the label
            ax.text(
                x,
                final_text_y,
                label,
                color=color,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8 * self.font_scale,  # Smaller font for better fit
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
                rotation=15,  # Slight rotation
            )
            ax.plot(
                [x, x],
                [y, final_text_y],
                color=color,
                linestyle=":",
                linewidth=1,
                alpha=0.5,
            )

    def _add_peak_label(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        label: str,
        color: str,
        offset: float = 0.05,
    ):
        """Add a label above the peak with a connecting line."""
        text_y = y + offset * y
        ax.text(
            x,
            text_y,
            label,
            color=color,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8 * self.font_scale,  # Smaller font to reduce overlap
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
            rotation=15,  # Reduced rotation for better readability
        )
        ax.plot([x, x], [y, text_y], color=color, linestyle=":", linewidth=1, alpha=0.5)

    def _customize_distribution_plot(
        self, ax: plt.Axes, distribution_data: Dict, font_sizes: Dict
    ):
        """Apply common customizations to distribution plots."""
        is_normalized = distribution_data["metadata"]["normalized"]
        xlabel = "Normalized Distance" if is_normalized else "Distance"
        title = (
            "Distribution Comparison of\n"
            + ("Normalized " if is_normalized else "")
            + "Distance Embeddings"
        )

        ax.set_xlabel(xlabel, fontsize=font_sizes["axis"], labelpad=15)
        ax.set_ylabel("Density", fontsize=font_sizes["axis"], labelpad=15)
        ax.set_title(title, fontsize=font_sizes["title"], pad=20, weight="bold")
        ax.tick_params(axis="both", which="major", labelsize=font_sizes["tick"])
        ax.grid(True, alpha=0.2)

    def _add_broken_axis_marks(self, ax1: plt.Axes, ax2: plt.Axes):
        """Add diagonal marks to indicate broken axis."""
        d = 0.015
        kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax1.spines["bottom"].set_color("#D3D3D3")
        ax2.spines["top"].set_color("#D3D3D3")

    # --- Violin Plot Analysis ---

    def create_violin_plot_comparison(
        self, sample_size: int = 10_000, save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Create violin plots comparing PLM distance differences.

        ``sample_size`` subsamples the pair table because the violins are drawn
        from every row. Pass ``None`` to use all pairs.

        Two things used to be wrong here and are worth keeping fixed: the
        subsample was taken with ``.head()``, which returns the FIRST n rows of
        a table that is not in random order (so the violins described whichever
        proteins sorted first, not the cohort), and nothing was logged, so a
        plot built from 10,000 of 113,186,256 pairs was indistinguishable from
        one built from all of them.
        """
        logger.info("Creating violin plot comparison...")

        if sample_size is not None and len(self.df) > sample_size:
            logger.warning(
                "violin plots use a random subsample of %s of %s pairs "
                "(seed=0). Pass sample_size=None to use every pair.",
                f"{sample_size:,}",
                f"{len(self.df):,}",
            )
            df_sample = self.df.sample(n=sample_size, seed=0)
        else:
            df_sample = self.df

        # Normalize data
        # Build normalized data using with_columns
        normalized_data = df_sample
        for col in tqdm(self.dist_cols, desc="Normalizing for violin plots"):
            if not df_sample[col].is_nan().all():
                normalized_col = (df_sample[col] - df_sample[col].min()) / (
                    df_sample[col].max() - df_sample[col].min()
                )
                normalized_data = normalized_data.with_columns(
                    normalized_col.alias(col)
                )

        n_models = len(self.dist_cols)
        fig, axes = plt.subplots(
            n_models, n_models, figsize=(20 * self.font_scale, 20 * self.font_scale)
        )
        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        # Compute differences and medians
        differences, row_ylims, min_median, max_median = self._compute_violin_data(
            normalized_data, n_models
        )

        # Create plots
        self._create_violin_plots(
            axes, differences, row_ylims, min_median, max_median, n_models
        )

        # Add title and colorbar
        self._add_violin_plot_decorations(fig, min_median, max_median)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Violin plot comparison saved to {save_path}")

        return fig, axes

    def _compute_violin_data(
        self, normalized_data: pl.DataFrame, n_models: int
    ) -> Tuple[Dict, Dict, float, float]:
        """Compute differences and statistics for violin plots."""
        all_medians = []
        differences = {}
        row_ylims = {}

        with tqdm(
            total=(n_models * (n_models - 1)) // 2, desc="Computing differences"
        ) as pbar:
            for i in range(n_models):
                row_diffs = []
                for j in range(n_models):
                    if i != j:
                        diff = abs(
                            normalized_data[self.dist_cols[i]]
                            - normalized_data[self.dist_cols[j]]
                        )
                        row_diffs.extend(diff.drop_nulls().to_numpy())
                        if i < j:
                            median = diff.median()
                            all_medians.append(median)
                            differences[(i, j)] = (diff, median)
                            pbar.update(1)

                if row_diffs:
                    row_ylims[i] = (0, max(row_diffs) * 1.1)

        min_median = min(all_medians) if all_medians else 0
        max_median = max(all_medians) if all_medians else 1

        return differences, row_ylims, min_median, max_median

    def _create_violin_plots(
        self,
        axes: np.ndarray,
        differences: Dict,
        row_ylims: Dict,
        min_median: float,
        max_median: float,
        n_models: int,
    ):
        """Create the violin plot grid."""
        total_plots = ((n_models * (n_models - 1)) // 2) + n_models

        with tqdm(total=total_plots, desc="Creating violin plots") as pbar:
            # Diagonal plots
            for i in range(n_models):
                ax = axes[i, i]
                ax.text(
                    0.5,
                    0.5,
                    self.dist_cols[i].replace("dist_", ""),
                    ha="center",
                    va="center",
                    rotation=45,
                    fontsize=12 * self.font_scale,
                    weight="bold",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                pbar.update(1)

            # Off-diagonal plots
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    diff, median = differences[(i, j)]
                    gray_val = 0.9 - 0.8 * (median - min_median) / (
                        max_median - min_median
                    )

                    self._create_single_violin_plot(
                        axes[i, j], diff, median, gray_val, row_ylims[i]
                    )
                    self._create_single_violin_plot(
                        axes[j, i], diff, median, gray_val, row_ylims[j]
                    )
                    pbar.update(1)

            # Format labels
            for i in range(n_models):
                for j in range(n_models):
                    if j == 0:
                        axes[i, j].set_ylabel(self.dist_cols[i].replace("dist_", ""))
                    else:
                        axes[i, j].set_ylabel("")
                        axes[i, j].set_yticklabels([])

                    if i == n_models - 1:
                        axes[i, j].set_xlabel(
                            self.dist_cols[j].replace("dist_", ""), rotation=45
                        )
                    else:
                        axes[i, j].set_xlabel("")

    def _create_single_violin_plot(
        self,
        ax: plt.Axes,
        diff: pl.Series,
        median: float,
        gray_val: float,
        ylim: Optional[Tuple[float, float]],
    ):
        """Create a single violin plot with consistent styling."""
        sns.violinplot(y=diff.drop_nulls(), ax=ax, inner="box", color=str(gray_val))
        if ylim:
            ax.set_ylim(ylim)
        ax.text(
            0,
            ax.get_ylim()[1],
            f"{median:.3f}",
            ha="center",
            va="bottom",
            fontsize=8 * self.font_scale,
        )
        ax.set_xticks([])
        ax.tick_params(axis="y", length=0)

    def _add_violin_plot_decorations(
        self, fig: plt.Figure, min_median: float, max_median: float
    ):
        """Add title and colorbar to violin plot."""
        plt.suptitle(
            "All-vs-All PLM Distance Differences\n(darker = larger median difference)",
            fontsize=16 * self.font_scale,
            y=0.925,
            weight="bold",
        )

        ax_legend = fig.add_axes([0.94, 0.1, 0.02, 0.8])
        gradient = np.linspace(0, 1, 256).reshape(256, 1)
        ax_legend.imshow(gradient, aspect="auto", cmap="gray")
        ax_legend.set_xticks([])

        n_ticks = 5
        tick_positions = np.linspace(0, 255, n_ticks)
        tick_values = np.linspace(max_median, min_median, n_ticks)
        ax_legend.set_yticks(tick_positions)
        ax_legend.set_yticklabels([f"{val:.3f}" for val in tick_values])
        ax_legend.set_title("Median\nDifference", fontsize=10 * self.font_scale)

    # --- Main Generation Method ---

    def generate_all_visualizations(
        self, force_recompute: bool = False
    ) -> Dict[str, Path]:
        """Generate all visualization types and save them to the output directory."""
        logger.info("Generating all embedding comparison visualizations...")

        output_paths = {}
        cache_dir = self.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        # Define visualization configurations
        viz_configs = [
            {
                "name": "hexagonal_comparison",
                "description": "hexagonal distance comparison",
                "cache_file": "hexbin_data.json",
                "output_file": "hexagonal_distance_comparison.png",
                "compute_func": self.compute_hexbin_data,
                "plot_func": lambda data: self.plot_hexagonal_distance_comparison(
                    data, save_path=output_paths.get("temp_path")
                ),
            },
            {
                "name": "correlation_heatmap",
                "description": "correlation heatmap",
                "cache_file": "correlation_data.json",
                "output_file": "correlation_heatmap.png",
                "compute_func": self.compute_correlation_data,
                "plot_func": lambda data: self.plot_correlation_heatmap(
                    data, show_ci=False, save_path=output_paths.get("temp_path")
                ),
            },
            {
                "name": "wasserstein_heatmap",
                "description": "Wasserstein distance heatmap",
                "cache_file": "wasserstein_data.json",
                "output_file": "wasserstein_heatmap.png",
                "compute_func": self.compute_wasserstein_data,
                "plot_func": lambda data: self.plot_wasserstein_heatmap(
                    data, cmap="Blues", save_path=output_paths.get("temp_path")
                ),
            },
            {
                "name": "distribution_comparison",
                "description": "raw distribution comparison",
                "cache_file": "distribution_data.json",
                "output_file": "distribution_comparison.png",
                "compute_func": lambda: self.compute_distribution_data(normalize=False),
                "plot_func": lambda data: self.plot_distributions(
                    data, save_path=output_paths.get("temp_path")
                ),
            },
            {
                "name": "distribution_comparison_normalized",
                "description": "normalized distribution comparison",
                "cache_file": "distribution_normalized_data.json",
                "output_file": "distribution_comparison_normalized.png",
                "compute_func": lambda: self.compute_distribution_data(normalize=True),
                "plot_func": lambda data: self.plot_distributions(
                    data, save_path=output_paths.get("temp_path")
                ),
            },
            {
                "name": "distribution_comparison_normalized_ridge",
                "description": "normalized distribution ridge plot",
                "cache_file": "distribution_normalized_data.json",
                "output_file": "distribution_comparison_normalized_ridge.png",
                "compute_func": lambda: self.compute_distribution_data(normalize=True),
                "plot_func": lambda data: self.plot_ridge_distributions(
                    data, alpha=0.8, save_path=output_paths.get("temp_path")
                ),
            },
        ]

        # Generate cacheable visualizations
        for config in viz_configs:
            logger.info(f"=== Generating {config['description']} ===")

            cache_path = cache_dir / config["cache_file"]
            output_path = self.output_dir / config["output_file"]
            output_paths["temp_path"] = output_path

            # Load or compute data
            data = self._load_cached_data(cache_path, force_recompute)
            if data is None:
                if "save_path" in config["compute_func"].__code__.co_varnames:
                    data = config["compute_func"](save_path=cache_path)
                else:
                    data = config["compute_func"]()
                    self._save_json_data(
                        data, cache_path, config["description"].title() + " data"
                    )

            # Generate plot
            config["plot_func"](data)
            output_paths[config["name"]] = output_path

        # Generate violin plot (no caching)
        logger.info("=== Generating violin plot comparison ===")
        violin_output = self.output_dir / "violin_plot_comparison.png"
        self.create_violin_plot_comparison(save_path=violin_output)
        output_paths["violin_plot_comparison"] = violin_output

        # Clean up temp key
        output_paths.pop("temp_path", None)

        logger.info("=== All visualizations complete ===")
        for viz_type, path in output_paths.items():
            logger.info(f"{viz_type}: {path}")

        return output_paths

    def plot_combined_wasserstein_correlation(
        self,
        wasserstein_data: Optional[Dict] = None,
        correlation_data: Optional[Dict] = None,
        gridsize: int = 50,
        save_path: Optional[Path] = None,
        corr_vlim: Optional[float] = None,
        wass_vmax: Optional[float] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Create a combined plot with Wasserstein distance (upper triangle),
        model names (diagonal), and correlation values (lower triangle).

        ``corr_vlim``
            Half-width of the *symmetric* correlation scale: the lower triangle and its
            colourbar run from ``-corr_vlim`` to ``+corr_vlim`` through
            ``CORR_DIVERGING_CMAP``, so equal colour distance means equal difference in
            rho on both sides of zero.  Default: ``symmetric_corr_limit`` of this
            matrix.  Pass an explicit value to put two figures on one scale.
        ``wass_vmax``
            Top of the Wasserstein scale.  Default: this matrix's own maximum, which is
            what the figure used to hardcode in three separate places (the cell vmax,
            the white/black text threshold and the colourbar).  Pass an explicit value
            to share the scale with another figure -- without it two figures scale their
            blues to their own maxima and identical shades mean different distances.
        """

        # Data should be provided by the caller (from viz_map)
        if wasserstein_data is None or correlation_data is None:
            raise ValueError(
                "Both wasserstein_data and correlation_data must be provided"
            )

        logger.info("Creating combined Wasserstein-correlation plot...")

        dist_cols = [f"dist_{col}" for col in wasserstein_data["columns"]]
        correlations = np.array(correlation_data["correlations"])
        wasserstein_distances = np.array(wasserstein_data["distances"])
        n = len(dist_cols)

        # What the two halves actually span.  Off the diagonal only: the diagonal
        # carries rho = 1 and W1 = 0 by construction and is drawn as model names.
        upper = np.triu_indices(n, 1)
        rho_lo = float(np.nanmin(correlations[upper]))
        rho_hi = float(np.nanmax(correlations[upper]))
        n_negative = int((correlations[upper] < 0).sum())
        wass_top = float(np.nanmax(wasserstein_distances[upper]))

        # One scale per quantity, resolved once.  Both used to be recomputed inline at
        # each use site, which is how the cells and their colourbar could disagree.
        # A scale derived here covers its data by construction; only an override can
        # be too small, so only an override is checked.
        if corr_vlim is None:
            corr_vlim = symmetric_corr_limit(correlations)
        else:
            # The scale is SYMMETRIC, so either end can run off it: the test is on
            # max|rho|, not on the minimum.  Testing only the minimum let a matrix
            # whose rho reaches +0.95 through under corr_vlim=0.10, and +0.95 and
            # +0.80 then rendered as the identical saturated red -- the same silent
            # saturation as the vmin=0 hardcode this scale was built to replace.  It
            # also called a positive number "the most negative correlation" whenever
            # every rho happened to be positive.
            worst_rho = rho_lo if abs(rho_lo) >= abs(rho_hi) else rho_hi
            if corr_vlim < abs(worst_rho):
                raise ValueError(
                    f"corr_vlim={corr_vlim} clips rho={worst_rho:+.3f}; the "
                    f"off-diagonal runs [{rho_lo:+.3f}, {rho_hi:+.3f}], so it needs "
                    f"at least {math.ceil(abs(worst_rho) * 100) / 100}"
                )
        if wass_vmax is None:
            wass_vmax = float(np.nanmax(wasserstein_distances))
        elif wass_vmax < wass_top:
            # The Wasserstein override had no check at all, so --wass-vmax 0.01
            # against W1 reaching 0.30 drew every upper-triangle cell in one blue and
            # said nothing.  One override policed and the other not is worse than
            # neither, because it reads as though both were.
            raise ValueError(
                f"wass_vmax={wass_vmax} clips the largest Wasserstein distance "
                f"{wass_top:.4f}"
            )
        corr_norm = plt.Normalize(vmin=-corr_vlim, vmax=corr_vlim)
        logger.info(
            "Correlation scale: symmetric +/-%.2f over rho in [%+.3f, %+.3f], "
            "%d of %d off-diagonal cells negative. Wasserstein top: %.4f",
            corr_vlim,
            rho_lo,
            rho_hi,
            n_negative,
            n * (n - 1) // 2,
            wass_vmax,
        )

        # Calculate figure size to ensure square cells
        # Base size per cell to ensure readability
        cell_size = 1.5 * self.font_scale
        grid_size = n * cell_size
        cbar_width = 0.8 * self.font_scale  # Reduced colorbar width
        spacing = 1.0 * self.font_scale  # Extra spacing between grid and colorbars

        # Total figure dimensions
        fig_width = grid_size + spacing + cbar_width
        fig_height = grid_size

        # Create figure
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Calculate margins to keep grid square
        left_margin = 0.08
        right_margin = 0.02
        top_margin = 0.08
        bottom_margin = 0.08

        # Available space for grid + spacing + colorbar
        available_width = 1 - left_margin - right_margin
        available_height = 1 - top_margin - bottom_margin

        # Fraction of figure width for grid, spacing, and colorbar
        grid_frac = grid_size / fig_width
        spacing_frac = spacing / fig_width
        cbar_frac = cbar_width / fig_width

        # Create grid with space for colorbars on the right
        # Add a spacing column between grid and colorbars
        gs = fig.add_gridspec(
            n,
            n + 2,  # n for grid, 1 for spacing, 1 for colorbar
            width_ratios=[1] * n
            + [spacing_frac / grid_frac * n, cbar_frac / grid_frac * n],
            wspace=0,
            hspace=0,
            left=left_margin,
            right=left_margin + grid_frac + spacing_frac + cbar_frac,
            top=1 - top_margin,
            bottom=bottom_margin,
        )

        # Create axes for the main plot
        axes = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                axes[i, j] = fig.add_subplot(gs[i, j])

        with tqdm(total=n * n, desc="Creating combined plot") as pbar:
            for i, col1 in enumerate(dist_cols):
                for j, col2 in enumerate(dist_cols):
                    ax = axes[i, j]

                    # Remove all ticks and spines for cleaner look
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    if i == j:
                        # Diagonal: show embedding name with proper display name
                        embedding_key = col1.replace("dist_", "")
                        embedding_name = EMBEDDING_DISPLAY_NAMES.get(
                            embedding_key, embedding_key
                        )
                        ax.text(
                            0.5,
                            0.5,
                            embedding_name,
                            ha="center",
                            va="center",
                            fontsize=28 * self.font_scale,
                            weight="bold",
                            color="black",  # All diagonal text in black
                            transform=ax.transAxes,
                        )
                    elif i < j:
                        # Upper triangle: Wasserstein distance
                        if not np.isnan(wasserstein_distances[i, j]):
                            wasserstein_val = wasserstein_distances[i, j]
                            # Create a heatmap cell with proper extent
                            im = ax.imshow(
                                [[wasserstein_val]],
                                cmap="Blues",
                                vmin=0,
                                vmax=wass_vmax,
                                aspect="auto",
                                extent=[0, 1, 0, 1],
                            )
                            # Determine text color based on background
                            normalized_val = wasserstein_val / wass_vmax
                            text_color = "white" if normalized_val > 0.5 else "black"
                            # Display value multiplied by 100 (no "0." prefix, no decimal)
                            ax.text(
                                0.5,
                                0.5,
                                f"{wasserstein_val * 100:.0f}",
                                ha="center",
                                va="center",
                                color=text_color,
                                fontsize=32 * self.font_scale,
                                weight="bold",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.set_facecolor("lightgray")
                            ax.text(
                                0.5,
                                0.5,
                                "NA",
                                ha="center",
                                va="center",
                                color="black",
                                fontsize=32 * self.font_scale,
                                transform=ax.transAxes,
                            )
                    else:
                        # Lower triangle: correlation values
                        if not np.isnan(correlations[i, j]):
                            correlation_val = correlations[i, j]
                            # Create a heatmap cell with proper extent
                            im = ax.imshow(
                                [[correlation_val]],
                                cmap=CORR_DIVERGING_CMAP,
                                vmin=-corr_vlim,
                                vmax=corr_vlim,
                                aspect="auto",
                                extent=[0, 1, 0, 1],
                            )
                            text_color = _readable_text_color(
                                CORR_DIVERGING_CMAP, corr_norm, correlation_val
                            )
                            # Display value multiplied by 100 (no "0." prefix, no decimal)
                            ax.text(
                                0.5,
                                0.5,
                                f"{correlation_val * 100:.0f}",
                                ha="center",
                                va="center",
                                color=text_color,
                                fontsize=32 * self.font_scale,
                                weight="bold",
                                transform=ax.transAxes,
                            )
                        else:
                            ax.set_facecolor("lightgray")
                            ax.text(
                                0.5,
                                0.5,
                                "NA",
                                ha="center",
                                va="center",
                                color="black",
                                fontsize=32 * self.font_scale,
                                transform=ax.transAxes,
                            )

                    pbar.update(1)

        # Add row and column labels outside the grid with proper display names
        for i, col in enumerate(dist_cols):
            embedding_key = col.replace("dist_", "")
            label = EMBEDDING_DISPLAY_NAMES.get(embedding_key, embedding_key)
            # Y-axis labels (left side)
            axes[i, 0].set_ylabel(
                label,
                rotation=0,
                ha="right",
                va="center",
                fontsize=30 * self.font_scale,
                labelpad=10,
            )
            # X-axis labels (bottom)
            axes[-1, i].set_xlabel(
                label,
                rotation=0,
                ha="center",
                va="top",
                fontsize=30 * self.font_scale,
                labelpad=5,
            )

        # Add family grouping
        self._add_family_grouping_combined(fig, axes, dist_cols)

        # Add title
        fig.suptitle(
            "Combined Pairwise Comparison (values ×100)\n(Upper: Wasserstein Distance, Diagonal: Models, Lower: Spearman Correlation)",
            fontsize=32 * self.font_scale,
            weight="bold",
            y=0.98,
        )

        # Create a sub-grid for the colorbar column (rightmost) to split it into two equal parts
        gs_right = gs[:, -1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.2)

        # Add colorbar for Wasserstein distances (top half)
        cbar_wass_ax = fig.add_subplot(gs_right[0])
        im_wass = plt.cm.ScalarMappable(
            cmap="Blues",
            norm=plt.Normalize(vmin=0, vmax=wass_vmax * 100),
        )
        cbar_wass = plt.colorbar(im_wass, cax=cbar_wass_ax, orientation="vertical")
        cbar_wass.ax.tick_params(labelsize=32 * self.font_scale)
        cbar_wass.set_label(
            "Wasserstein Distance (×100)", fontsize=36 * self.font_scale, labelpad=30
        )

        # Add colorbar for correlations (bottom half)
        cbar_corr_ax = fig.add_subplot(gs_right[1])
        im_corr = plt.cm.ScalarMappable(
            cmap=CORR_DIVERGING_CMAP,
            norm=plt.Normalize(vmin=-corr_vlim * 100, vmax=corr_vlim * 100),
        )
        cbar_corr = plt.colorbar(im_corr, cax=cbar_corr_ax, orientation="vertical")
        cbar_corr.ax.tick_params(labelsize=32 * self.font_scale)
        cbar_corr.set_label(
            "Spearman Correlation (×100)", fontsize=36 * self.font_scale, labelpad=30
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=DEFAULT_STYLE["dpi"])
            logger.info(f"Combined Wasserstein-correlation plot saved to {save_path}")

        return fig, axes

    def _add_family_grouping_combined(
        self, fig: plt.Figure, axes: np.ndarray, dist_cols: List[str]
    ):
        """Add thick black frames around PLM families (2+ members only) in combined plot."""
        boundaries = self._get_family_boundaries(dist_cols)
        n = len(dist_cols)

        # Configuration
        border_color = "black"
        border_width = 10

        # First, hide all internal borders
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                for spine in ax.spines.values():
                    spine.set_visible(False)

        # Draw thick frames around family blocks (only families with 2+ members)
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            family_size = end - start

            # Skip families with only 1 member
            if family_size < 2:
                continue

            # Get family color for subtle background tint
            family = self._get_embedding_family(dist_cols[start])
            color = EMBEDDING_FAMILY_COLOR_MAP.get(family, "#808080")
            rgb = mcolors.to_rgb(color)
            light_color = tuple(0.97 + 0.03 * c for c in rgb)  # Very subtle tint

            # Add subtle background color to family block
            for row in range(start, end):
                for col in range(start, end):
                    axes[row, col].set_facecolor(light_color)

            # Draw frame using Rectangle patch in figure coordinates
            # Get the bounding box of the family block
            # Use the first and last axes to determine the extent
            ax_first = axes[start, start]
            ax_last = axes[end - 1, end - 1]

            # Get positions in figure coordinates
            bbox_first = ax_first.get_position()
            bbox_last = ax_last.get_position()

            # Calculate rectangle position and size in figure coordinates
            # Left edge of leftmost axes
            rect_x = bbox_first.x0
            # Bottom edge of bottom axes
            rect_y = bbox_last.y0
            # Width spans from left of first to right of last
            rect_width = bbox_last.x1 - bbox_first.x0
            # Height spans from bottom of last to top of first
            rect_height = bbox_first.y1 - bbox_last.y0

            # Create rectangle patch (hollow, just the border)
            rect = mpatches.Rectangle(
                (rect_x, rect_y),
                rect_width,
                rect_height,
                linewidth=border_width,
                edgecolor=border_color,
                facecolor="none",
                transform=fig.transFigure,
                zorder=1000,  # Draw on top of everything
                clip_on=False,
            )

            # Add rectangle to figure
            fig.patches.append(rect)


def main():
    """Main function to parse arguments and run the visualization script."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive pairwise embedding comparison visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add arguments
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to CSV file containing the embedding distance data.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("out/embedding_comparison"),
        help="Directory to save output visualizations and cache files.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit the number of rows to process (for testing or memory constraints).",
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=1.0,
        help="Scaling factor for all font sizes in visualizations.",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of all cached intermediate data.",
    )
    parser.add_argument(
        "--visualizations",
        nargs="+",
        choices=[
            "hexagonal",
            "correlation",
            "wasserstein",
            "distribution",
            "distribution_normalized",
            "distribution_normalized_ridge",
            "violin",
            "combined",
            "dip_test",
            "all",
        ],
        default=["all"],
        help="Specific visualizations to generate.",
    )
    parser.add_argument(
        "--ranking_csv",
        type=Path,
        default=None,
        help="Path to PLM ranking CSV to sort ridge plot by performance (e.g., plm_ranking_by_spearman.csv).",
    )

    args = parser.parse_args()

    # Create visualizer and generate visualizations
    visualizer = EmbeddingComparisonVisualizer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        font_scale=args.font_scale,
    )

    if "all" in args.visualizations:
        output_paths = visualizer.generate_all_visualizations(
            force_recompute=args.force_recompute
        )
    else:
        # Generate individual visualizations
        output_paths = {}
        cache_dir = args.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        viz_map = {
            "hexagonal": (
                visualizer.compute_hexbin_data,
                visualizer.plot_hexagonal_distance_comparison,
                "hexbin_data.json",
                "hexagonal_distance_comparison.png",
            ),
            "correlation": (
                visualizer.compute_correlation_data,
                visualizer.plot_correlation_heatmap,
                "correlation_data.json",
                "correlation_heatmap.png",
            ),
            "wasserstein": (
                visualizer.compute_wasserstein_data,
                visualizer.plot_wasserstein_heatmap,
                "wasserstein_data.json",
                "wasserstein_heatmap.png",
            ),
            "distribution": (
                lambda: visualizer.compute_distribution_data(normalize=False),
                visualizer.plot_distributions,
                "distribution_data.json",
                "distribution_comparison.png",
            ),
            "distribution_normalized": (
                lambda: visualizer.compute_distribution_data(normalize=True),
                visualizer.plot_distributions,
                "distribution_normalized_data.json",
                "distribution_comparison_normalized.png",
            ),
            "distribution_normalized_ridge": (
                lambda: visualizer.compute_distribution_data(normalize=True),
                visualizer.plot_ridge_distributions,
                "distribution_normalized_data.json",
                "distribution_comparison_normalized_ridge.png",
            ),
            "violin": (
                None,
                visualizer.create_violin_plot_comparison,
                None,
                "violin_plot_comparison.png",
            ),
            "combined": (
                [
                    visualizer.compute_correlation_data,
                    visualizer.compute_wasserstein_data,
                ],
                visualizer.plot_combined_wasserstein_correlation,
                ["correlation_data.json", "wasserstein_data.json"],
                "combined_wasserstein_correlation.png",
            ),
            "dip_test": (
                None,
                visualizer.compute_hartigan_dip_test,
                None,
                "hartigan_dip_test_results.csv",
            ),
        }

        for viz_type in args.visualizations:
            if viz_type in viz_map:
                compute_func, plot_func, cache_file, output_file = viz_map[viz_type]
                output_path = args.output_dir / output_file

                logger.info(f"Generating {viz_type}...")

                if isinstance(compute_func, list):
                    # Handle combined plot with multiple compute functions
                    data_list = []
                    for i, (func, cache_file) in enumerate(
                        zip(compute_func, cache_file)
                    ):
                        cache_path = cache_dir / cache_file
                        data = visualizer._load_cached_data(
                            cache_path, args.force_recompute
                        )
                        if data is None:
                            data = func()
                            visualizer._save_json_data(
                                data, cache_path, f"{viz_type.title()} data {i + 1}"
                            )
                        data_list.append(data)

                    # Pass data to plot function
                    plot_func(
                        wasserstein_data=data_list[1],
                        correlation_data=data_list[0],
                        save_path=output_path,
                    )
                elif cache_file and compute_func:
                    cache_path = cache_dir / cache_file
                    data = visualizer._load_cached_data(
                        cache_path, args.force_recompute
                    )
                    if data is None:
                        data = compute_func()
                        visualizer._save_json_data(
                            data, cache_path, f"{viz_type.title()} data"
                        )
                    # Special handling for ridge plot to pass ranking_csv
                    if viz_type == "distribution_normalized_ridge":
                        plot_func(
                            data, save_path=output_path, ranking_csv=args.ranking_csv
                        )
                    else:
                        plot_func(data, save_path=output_path)
                else:
                    plot_func(save_path=output_path)

                output_paths[viz_type] = output_path

    logger.info("=== Visualization generation complete ===")
    for viz_type, path in output_paths.items():
        logger.info(f"{viz_type}: {path}")


if __name__ == "__main__":
    main()
