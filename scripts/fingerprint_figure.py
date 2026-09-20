#!/usr/bin/env python3
"""Draw the combined Wasserstein/correlation fingerprint (Figure 3 and its twin).

Two populations, one pair of scales.  ``--payload`` takes one or more
``{columns, correlations, distances}`` JSONs -- the output of the cluster
fingerprint reduction -- and draws each one.  When more than one is given, the
correlation half-width and the Wasserstein top are computed over *all* of them and
shared, so the main-text figure and the supplementary one are comparable cell for
cell.  With a single payload each figure would be scaled to its own maximum and equal
shades would mean different numbers in the two.

This replaces ``redraw_fig03_diverging.py``, a script that lived in the results
directory and monkey-patched ``Axes.imshow``, ``ScalarMappable`` and ``np.nanmax`` for
the duration of the draw because the tracked plotter hardcoded ``vmin=0`` on the
correlation cells.  The hardcoding is gone (see ``CORR_DIVERGING_CMAP`` and the
``corr_vlim``/``wass_vmax`` arguments), so the figure is reproducible from the repo.

    python scripts/fingerprint_figure.py \
        --payload random=<fingerprint_full.json> alignable=<fingerprint_full.json> \
        --out-dir <figures>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from visualization.pairwise_embedding_comparison import (  # noqa: E402
    EmbeddingComparisonVisualizer,
    symmetric_corr_limit,
)


def load(path: Path, out_dir: Path):
    """Load a payload and put both matrices into the figure's row order.

    Reordering rather than trusting the producer's order is not paranoia: a permuted
    or transposed matrix still draws a perfectly plausible-looking figure, and the row
    order here has to match every other panel of the paper (family, then size).
    """
    payload = json.loads(path.read_text())
    for key in ("columns", "correlations", "distances"):
        if key not in payload:
            raise SystemExit(f"{path} has no {key!r}")
    viz = EmbeddingComparisonVisualizer.from_matrices(
        columns=payload["columns"], output_dir=out_dir
    )
    order = [payload["columns"].index(c.replace("dist_", "")) for c in viz.dist_cols]
    cols = [payload["columns"][i] for i in order]
    corr = np.asarray(payload["correlations"], float)[np.ix_(order, order)]
    wass = np.asarray(payload["distances"], float)[np.ix_(order, order)]
    return viz, cols, corr, wass, payload.get("metadata", {})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--payload",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="One or more named payloads. NAME becomes the output file stem.",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--corr-vlim",
        type=float,
        default=None,
        help="Override the shared symmetric correlation half-width.",
    )
    ap.add_argument(
        "--wass-vmax",
        type=float,
        default=None,
        help="Override the shared Wasserstein top.",
    )
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for spec in args.payload:
        if "=" not in spec:
            raise SystemExit(f"--payload wants NAME=PATH, got {spec!r}")
        name, _, path = spec.partition("=")
        loaded[name] = load(Path(path), args.out_dir)

    corr_vlim = args.corr_vlim or symmetric_corr_limit(
        *(corr for _, _, corr, _, _ in loaded.values())
    )
    wass_vmax = args.wass_vmax or max(
        float(np.nanmax(wass[np.triu_indices(wass.shape[0], 1)]))
        for _, _, _, wass, _ in loaded.values()
    )
    print(
        f"shared correlation scale: [{-corr_vlim:+.2f}, {corr_vlim:+.2f}] "
        f"(x100 on the colourbar: {-corr_vlim * 100:.0f} to {corr_vlim * 100:.0f})"
    )
    print(f"shared Wasserstein top: {wass_vmax:.4f} (x100: {wass_vmax * 100:.1f})\n")

    for name, (viz, cols, corr, wass, meta) in loaded.items():
        out_png = args.out_dir / f"fingerprint_{name}.png"
        fig, _ = viz.plot_combined_wasserstein_correlation(
            wasserstein_data={"distances": wass.tolist(), "columns": cols},
            correlation_data={"correlations": corr.tolist(), "columns": cols},
            save_path=out_png,
            corr_vlim=corr_vlim,
            wass_vmax=wass_vmax,
        )
        plt.close(fig)
        iu = np.triu_indices(len(cols), 1)
        print(
            f"  {out_png.name}: {len(cols)} arms, "
            f"rho in [{corr[iu].min():+.3f}, {corr[iu].max():+.3f}], "
            f"{int((corr[iu] < 0).sum())} negative cells, "
            f"W1x100 in [{wass[iu].min() * 100:.1f}, {wass[iu].max() * 100:.1f}]"
            + (f", population: {meta['population']}" if "population" in meta else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
