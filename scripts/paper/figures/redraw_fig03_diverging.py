#!/usr/bin/env python3
"""Redraw Figure 3 with a diverging correlation scale centred at zero.

Why. The shipped combined plot draws the lower triangle with
``imshow(..., cmap="OrRd", vmin=0, vmax=1)`` and its colourbar with
``Normalize(vmin=0, vmax=100)`` (pairwise_embedding_comparison.py:2158 and :2245).
On the alignable-pair population every rho was positive, so the clipped floor never
showed. On the random-pair population 12 of the 91 cells are negative, down to
rho = -0.238, and all 12 involve CLEAN -- the result the figure exists to show. Under
vmin=0 they render identical to rho = 0.

What changes, and only this: the correlation cells and their colourbar get a
diverging map centred at 0 with SYMMETRIC limits, so equal colour distance means
equal difference in rho on both sides of zero. Everything else -- geometry, family
boxes, Wasserstein triangle, printed values, fonts, dpi -- is the tracked code,
untouched. The tracked file is not edited: the two call signatures are intercepted
at the matplotlib level for the duration of the draw.

The limits are shared by BOTH populations (VLIM below), so the main-text figure and
its supplementary counterpart are directly comparable cell for cell. VLIM is the
smallest 0.01 step covering max|rho| over both matrices.

Colour choice: the negative arm is purple and the positive arm reproduces OrRd's
ramp. A blue negative arm (RdBu_r, the usual correlation map) would collide with the
upper triangle, which is already Blues on a different quantity and its own colourbar.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

import matplotlib


# These paths were absolute to one machine. They are environment variables now, so an
# unset one fails here by name rather than as a FileNotFoundError further down.
import os


def _need(var: str) -> str:
    """The value of `var`, or a message naming what to set."""
    try:
        return os.environ[var]
    except KeyError:
        raise SystemExit(f"set {var} before running this script") from None


ARTEFACTS = _need("PAPER_ARTEFACTS")
REPO = _need("REPO")

matplotlib.use("Agg")
import matplotlib.cm as mcm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import sys  # noqa: E402

W = Path(REPO)
sys.path.insert(0, str(W / "src"))
from visualization.pairwise_embedding_comparison import (  # noqa: E402
    EmbeddingComparisonVisualizer,
)

B = Path(ARTEFACTS)
R = B / "final_figures_2026-09-20"

RANDOM_JSON = R / "fp/out/fingerprint_full.json"
ALIGNABLE_JSON = B / "figures_2026-09-19/fig03/fingerprint_full.json"

#: Diverging map: purple (negative) -> cream (0) -> OrRd reds (positive).
CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr_diverging",
    [
        "#2D004B", "#542788", "#8073AC", "#B2ABD2", "#D8DAEB",  # negative arm
        "#FFF7EC",                                              # zero
        "#FDD49E", "#FC8D59", "#EF6548", "#B30000", "#7F0000",  # positive arm
    ],
)


def corr_limit(*mats: np.ndarray) -> float:
    """Smallest 0.01 step covering max|rho| over every matrix, off-diagonal only."""
    worst = 0.0
    for m in mats:
        n = m.shape[0]
        iu = np.triu_indices(n, 1)
        worst = max(worst, float(np.abs(m[iu]).max()))
    return math.ceil(worst * 100) / 100


@contextmanager
def shared_scales(vlim: float, wmax: float, wass: np.ndarray):
    """Intercept the correlation scale, and pin the Wasserstein scale to a shared top.

    Correlation: the cells are the only imshow in this figure with cmap="OrRd" and
    vmax=1 (the Wasserstein cells are Blues with a data-dependent vmax), and the
    correlation colourbar is the only ScalarMappable with cmap="OrRd" and vmax=100,
    so both matches are exact rather than positional.

    Wasserstein: the function calls ``np.nanmax(wasserstein_distances)`` in exactly
    three places -- the cell vmax, the white/black text threshold, and the colourbar
    -- all three meaning "top of the Wasserstein scale". Overriding that one value
    for that one matrix therefore moves all three together and cannot leave the text
    contrast disagreeing with the colours. Without it the two figures would scale
    their blues to their own maxima (44.2 vs 20.1) and identical shades would mean
    different distances across the pair.
    """
    real_imshow = Axes.imshow
    real_mappable = mcm.ScalarMappable
    real_nanmax = np.nanmax

    def imshow(self, X, *a, **kw):
        if kw.get("cmap") == "OrRd" and kw.get("vmax") == 1 and kw.get("vmin") == 0:
            kw = {**kw, "cmap": CORR_CMAP, "vmin": -vlim, "vmax": vlim}
        return real_imshow(self, X, *a, **kw)

    def mappable(*a, **kw):
        norm = kw.get("norm")
        if kw.get("cmap") == "OrRd" and norm is not None and norm.vmax == 100:
            kw = {**kw, "cmap": CORR_CMAP,
                  "norm": plt.Normalize(vmin=-vlim * 100, vmax=vlim * 100)}
        return real_mappable(*a, **kw)

    def nanmax(x, *a, **kw):
        arr = np.asarray(x)
        if arr.shape == wass.shape and np.array_equal(arr, wass):
            return wmax
        return real_nanmax(x, *a, **kw)

    Axes.imshow, mcm.ScalarMappable, np.nanmax = imshow, mappable, nanmax
    try:
        yield
    finally:
        Axes.imshow, mcm.ScalarMappable, np.nanmax = (
            real_imshow, real_mappable, real_nanmax
        )


def reorder(payload_path: Path, out_dir: Path):
    """Same load-and-reorder path as the --precomputed CLI branch."""
    payload = json.loads(payload_path.read_text())
    viz = EmbeddingComparisonVisualizer(output_dir=out_dir, columns=payload["columns"])
    order = [payload["columns"].index(c.replace("dist_", "")) for c in viz.dist_cols]
    cols = [payload["columns"][i] for i in order]
    corr = np.asarray(payload["correlations"], float)[np.ix_(order, order)]
    wass = np.asarray(payload["distances"], float)[np.ix_(order, order)]
    return viz, cols, corr, wass


def draw(payload_path: Path, out_png: Path, vlim: float, wmax: float) -> None:
    viz, cols, corr, wass = reorder(payload_path, out_png.parent)

    with shared_scales(vlim, wmax, wass):
        fig, _ = viz.plot_combined_wasserstein_correlation(
            wasserstein_data={"distances": wass.tolist(), "columns": cols},
            correlation_data={"correlations": corr.tolist(), "columns": cols},
            save_path=out_png,
        )
    plt.close(fig)

    iu = np.triu_indices(len(cols), 1)
    print(f"  {out_png.name}: {len(cols)} arms, rho in "
          f"[{corr[iu].min():+.3f}, {corr[iu].max():+.3f}], "
          f"{int((corr[iu] < 0).sum())} negative cells, "
          f"W1x100 in [{wass[iu].min() * 100:.1f}, {wass[iu].max() * 100:.1f}]")


def main() -> None:
    rnd = json.loads(RANDOM_JSON.read_text())
    aln = json.loads(ALIGNABLE_JSON.read_text())
    rc = np.asarray(rnd["correlations"], float)
    ac = np.asarray(aln["correlations"], float)
    vlim = corr_limit(rc, ac)

    # Shared Wasserstein top = the larger of the two populations' maxima, so the two
    # figures' blues mean the same distance. The random-pair figure sets it, and is
    # therefore drawn exactly as it would have been on its own scale.
    wmax = max(
        float(np.asarray(rnd["distances"], float)[np.triu_indices(14, 1)].max()),
        float(np.asarray(aln["distances"], float)[np.triu_indices(14, 1)].max()),
    )
    print(f"shared symmetric correlation limits: [{-vlim:+.2f}, {vlim:+.2f}] "
          f"(x100 on the colourbar: {-vlim * 100:.0f} to {vlim * 100:.0f})")
    print(f"shared Wasserstein top: {wmax:.4f} (x100: {wmax * 100:.1f})\n")

    draw(RANDOM_JSON, R / "fig03_randompairs_diverging.png", vlim, wmax)
    draw(ALIGNABLE_JSON, R / "fig03_alignable_diverging.png", vlim, wmax)
    print("\ndone")


if __name__ == "__main__":
    main()
