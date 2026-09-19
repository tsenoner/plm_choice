"""Each hexbin panel must be drawn the way its own axis labels say it is.

The supplementary fingerprint figure is a lower-triangular grid: the panel at
(row ``i``, column ``j``) shows the joint distance distribution of ``dist_cols[i]``
against ``dist_cols[j]``, and the grid writes exactly two labels per panel -- an
x label along the bottom row, taken from the *column* arm, and a y label down the
first column, taken from the *row* arm.

``_plot_hexbin_pair`` puts the arm named FIRST in the cache key on the horizontal
axis: it histograms ``(x=col1, y=col2)`` and then draws ``counts.T`` against
``meshgrid(xcentres, ycentres)``. The lookup used to be ``f"{col1}_vs_{col2}"``
with ``col1`` the row arm, so every panel came out transposed with respect to the
labels printed beside it. Nothing about the figure looked broken -- a transposed
joint density is still a plausible joint density -- which is why it survived.

The test makes the two arms distinguishable by giving them disjoint value ranges,
so the axis limits alone say which arm went where.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from visualization.pairwise_embedding_comparison import EmbeddingComparisonVisualizer

#: Value ranges chosen so that no bin edge of one arm can be confused for the other's.
X_RANGE = (0.0, 1.0)
Y_RANGE = (100.0, 200.0)


def _hexbin_cache(arms: list[str], gridsize: int = 5) -> dict:
    """A cache holding both orderings of every pair, as the cluster reducer emits."""
    ranges = {arms[0]: X_RANGE, arms[1]: Y_RANGE}
    cache: dict = {
        "metadata": {"dist_cols": arms, "gridsize": gridsize, "max_count": 10}
    }
    for a, b in ((arms[0], arms[1]), (arms[1], arms[0])):
        cache[f"{a}_vs_{b}"] = {
            "counts": np.ones((gridsize, gridsize)).tolist(),
            "xedges": np.linspace(*ranges[a], gridsize + 1).tolist(),
            "yedges": np.linspace(*ranges[b], gridsize + 1).tolist(),
        }
    return cache


def test_panel_axes_match_their_labels(tmp_path):
    """The bottom-row x axis carries the column arm, not the row arm."""
    arms = ["dist_ankh_base", "dist_prott5"]
    visualizer = EmbeddingComparisonVisualizer(data_path=None, output_dir=tmp_path)

    _, axes = visualizer.plot_hexagonal_distance_comparison(
        hexbin_data=_hexbin_cache(arms)
    )

    # The single lower-triangle panel: row 1 (dist_prott5), column 0 (dist_ankh_base).
    panel = axes[1, 0]
    assert panel.get_xlabel().replace("\n", " ") == "Ankh Base"
    assert panel.get_ylabel().replace("\n", " ") == "Prot T5"

    # dist_ankh_base was cached over X_RANGE and dist_prott5 over Y_RANGE, so the
    # axis limits say which arm actually reached which axis.
    assert panel.get_xlim() == pytest.approx(X_RANGE, abs=0.2)
    assert panel.get_ylim() == pytest.approx(Y_RANGE, abs=20.0)


def test_plot_only_visualizer_needs_no_source_frame(tmp_path):
    """``data_path=None`` is what lets the full-cohort cache be drawn here at all."""
    visualizer = EmbeddingComparisonVisualizer(data_path=None, output_dir=tmp_path)
    assert visualizer.df is None
    assert visualizer.dist_cols == []
