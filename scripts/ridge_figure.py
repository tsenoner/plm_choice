#!/usr/bin/env python3
"""Draw the distance-distribution ridge (Figure 2) from the full-cohort summaries.

This is a thin driver: every plotting decision lives in
``src/visualization/pairwise_embedding_comparison.py``, which is the code that made the
published figure.  The point of going back through it rather than writing a standalone
plotter is that the published figure's readability is not an accident of taste -- it
comes from four specific things in ``plot_ridge_distributions`` (negative gridspec
hspace, transparent axes facecolors, one inch of height per row, and a white curve
outline over a 0.8-alpha fill), and a fresh ``plt.subplots`` grid has none of them.

    python scripts/ridge_figure.py \
        --summary-dir <ridge_full_summary> --out-dir <figures> \
        --norm p99 minmax median log10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from visualization.pairwise_embedding_comparison import (  # noqa: E402
    RIDGE_NORMALISATIONS,
    EmbeddingComparisonVisualizer,
)

#: Titles, per normalisation.  None of them says "all-vs-all": these are the pairs
#: MMseqs2 and Foldseek found over the corrected, deduplicated cohort, and a true
#: all-vs-all over 526,871 proteins would be 1.4e11 pairs, not 7.6e7.
TITLES = {
    "p99": "Pairwise embedding distances, scaled by each model's 99th percentile",
    "minmax": "Pairwise embedding distances, min-max scaled per model",
    "median": "Pairwise embedding distances, scaled by each model's median",
    "log10": "Pairwise embedding distances (raw euclidean, log axis)",
}


def _title(norm: str, suffix: str) -> str:
    base = TITLES.get(norm, "Pairwise embedding distance distributions")
    return f"{base}\n{suffix}" if suffix else base


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--norm", nargs="+", default=["p99", "minmax", "median", "log10"])
    ap.add_argument("--grid", type=int, default=500)
    ap.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help="Override the normalisation's default x-limits. Applies to every --norm "
        "given in the same call, so pass one --norm at a time when using it.",
    )
    ap.add_argument(
        "--overlap",
        type=float,
        default=0.45,
        help="Negative gridspec hspace. The published figure used 0.25; more overlap "
        "means taller, more interleaved curves, which is the joyplot look.",
    )
    ap.add_argument("--row-height", type=float, default=1.15)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--no-iqr-band", action="store_true")
    ap.add_argument("--quartile-labels", action="store_true")
    ap.add_argument(
        "--no-tail-marks",
        action="store_true",
        help="Drop the 1st/99th-percentile baseline ticks. They are on by default: the "
        "quartile rules alone say nothing about how far the tail reaches, which is the "
        "whole question under a min-max axis anchored on the single most distant pair.",
    )
    ap.add_argument(
        "--legend",
        choices=("figure", "row"),
        default="figure",
        help="Where the key goes. 'figure' is the bottom margin, outside every row; "
        "'row' is the published placement inside the bottom row, which overlaps the "
        "row above it at 300 dpi.",
    )
    ap.add_argument(
        "--published-look",
        action="store_true",
        help="Reproduce the published geometry exactly (overlap 0.25, 1.0 in rows, no "
        "IQR band) so the only difference from Figure 2 is the data.",
    )
    ap.add_argument("--tag", default="", help="Suffix for the output file names.")
    ap.add_argument(
        "--title-suffix",
        default="",
        help="Appended to the title on a second line. Use it to name the pair "
        "population -- the same axis over random pairs and over aligner-found pairs "
        "are two different figures and the title is the only place that says which.",
    )
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overlap = 0.25 if args.published_look else args.overlap
    row_height = 1.0 if args.published_look else args.row_height
    iqr_band = False if (args.published_look or args.no_iqr_band) else True

    viz = EmbeddingComparisonVisualizer.from_distribution_summaries(
        summary_dir=args.summary_dir, output_dir=args.out_dir
    )

    written: list[Path] = []
    for norm in args.norm:
        if norm not in RIDGE_NORMALISATIONS:
            raise SystemExit(f"unknown --norm {norm!r}")
        data = viz.compute_distribution_data_from_summaries(
            normalisation=norm,
            grid=args.grid,
            xlim=tuple(args.xlim) if args.xlim else None,
        )
        stem = f"ridge_{norm}{args.tag}"
        save_path = args.out_dir / f"{stem}.png"
        fig, _ = viz.plot_ridge_distributions(
            distribution_data=data,
            alpha=args.alpha,
            save_path=save_path,
            overlap=overlap,
            row_height=row_height,
            iqr_band=iqr_band,
            quartile_labels=args.quartile_labels,
            quartile_legend=not args.published_look,
            tail_marks=not (args.published_look or args.no_tail_marks),
            legend_loc="row" if args.published_look else args.legend,
            title=_title(norm, args.title_suffix),
        )
        plt.close(fig)
        (args.out_dir / f"{stem}_distribution_data.json").write_text(
            json.dumps(data["metadata"], indent=2)
        )
        written.append(save_path)
        print(f"wrote {save_path}")

    # One dip table across arms, straight from the summaries.
    rows = []
    for arm, s in sorted(viz.summaries.items()):
        dip = s["dip"]
        row = {
            "plm_name": arm,
            "n_pairs": s["n_used"],
            "dip_full_cohort": dip["full"]["dip"] if dip.get("full") else None,
            "p_value_full_cohort": dip["full"]["p_value"] if dip.get("full") else None,
        }
        for key, label in (("n1000000", "1M"), ("n100000", "100k")):
            if key in dip:
                row[f"dip_n{label}_mean"] = dip[key]["dip_mean"]
                row[f"dip_n{label}_sd"] = dip[key]["dip_sd"]
                row[f"p_n{label}_min"] = dip[key]["p_min"]
        if "affine_invariance_check" in dip:
            row["dip_affine_max_abs_diff"] = dip["affine_invariance_check"]["max_abs_diff"]
        rows.append(row)
    dip_path = args.out_dir / f"hartigan_dip_full_cohort{args.tag}.csv"
    pl.DataFrame(rows).write_csv(dip_path)
    print(f"wrote {dip_path}")

    print("\n".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
