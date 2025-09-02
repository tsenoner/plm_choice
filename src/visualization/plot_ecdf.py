#!/usr/bin/env python3
"""
plot_ecdf.py
============

Draw ECDF curves of |prediction - truth| for equally-sized quantile slices.

Single-file mode:
  INPUT  : .npz file written by infer_pairs.py
  OUTPUT : PNG (or any Matplotlib backend format)

Batch mode:
  INPUT  : Directory containing parameter subdirectories (e.g., fident, alntmscore, hfsp),
           each with multiple *.npz files (one per PLM).
  OUTPUT :
    - One ECDF PNG per NPZ (same stem, saved alongside unless --out-dir is set)
    - One grid PNG per parameter with all PLMs (rows x cols grid)

Examples
--------
python plot_ecdf.py results/inference/test_ankh_base_pred.npz --slices 3 --out-dir out/single_ecdf

python plot_ecdf.py --dir out/sprot_pre2024_subset/inference --slices 3 --rows 4 --cols 4 --out-dir out/sprot_pre2024_subset/ecdf
"""

import argparse
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ECDF of absolute prediction error.")
    # Single-file mode (positional is optional to allow --dir batch mode)
    p.add_argument("npz", nargs="?", type=Path, help=".npz from infer_pairs.py")
    # Batch mode
    p.add_argument(
        "--dir", type=Path, help="Root directory with per-parameter subdirs of NPZs"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        help="Base directory to write ECDFs and grids",
    )
    # Common options
    p.add_argument(
        "--slices",
        type=int,
        default=3,
        help="Number of equal-size quantiles (default: 3)",
    )
    p.add_argument("--rows", type=int, default=4, help="Grid rows for batch mode")
    p.add_argument("--cols", type=int, default=4, help="Grid cols for batch mode")
    p.add_argument("--output", type=Path, help="Output path (single-file mode)")
    return p.parse_args()


def _compute_ecdf_data(preds: np.ndarray, targets: np.ndarray, num_slices: int):
    if preds.size != targets.size:
        raise ValueError("predictions and targets differ in length")
    abs_err = np.abs(preds - targets)
    labels = [f"Q{i + 1}" for i in range(num_slices)]
    df = pd.DataFrame({"truth": targets, "abs_err": abs_err})
    df["slice"], bins = pd.qcut(df["truth"], q=num_slices, labels=labels, retbins=True)
    return df, labels, bins


def _plot_single_ecdf(
    ax: plt.Axes,
    df: pd.DataFrame,
    labels: Sequence[str],
    *,
    show_legend: bool,
    legend_title: str | None = None,
):
    for label in labels:
        e = df.loc[df["slice"] == label, "abs_err"].to_numpy()
        if e.size == 0:
            continue
        x = np.sort(e)
        y = np.arange(1, e.size + 1) / e.size * 100
        ax.step(x, y, where="post", label=label)
    ax.set_xlabel("|prediction − truth|")
    ax.set_ylabel("Cumulative % of pairs")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(True)
    if show_legend:
        ax.legend(title=legend_title)


def plot_single_file(npz_path: Path, slices: int, out_dir: Path | None) -> Path:
    data = np.load(npz_path)
    preds = data["predictions"].astype(float)
    targets = data["targets"].astype(float)
    df, labels, bins = _compute_ecdf_data(preds, targets, slices)
    fig, ax = plt.subplots()
    _plot_single_ecdf(ax, df, labels, show_legend=False)

    # Legend on top, horizontal, non-overlapping with grid
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        legend_labels = [
            f"{labels[i]} ({bins[i]:.2g}–{bins[i + 1]:.2g})"
            for i in range(len(bins) - 1)
        ]
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout()

    out = (
        (out_dir / (npz_path.stem + ".png"))
        if out_dir
        else npz_path.with_suffix(".png")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"✓ ECDF figure →  {out}")
    return out


def plot_grid_for_param(
    npz_files: Sequence[Path],
    slices: int,
    rows: int,
    cols: int,
    output_path: Path,
    panel_label: str | None = None,
) -> Path:
    npz_files = list(sorted(npz_files))
    if not npz_files:
        raise ValueError("No NPZ files provided for grid plot")

    # Prepare shared legend labels by using the first file
    first = np.load(npz_files[0])
    df_first, labels, bins = _compute_ecdf_data(
        first["predictions"].astype(float), first["targets"].astype(float), slices
    )
    legend_labels = [
        f"{labels[i]} ({bins[i]:.2g}-{bins[i + 1]:.2g})" for i in range(len(bins) - 1)
    ]

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    flat_axes = axes.flat

    for ax, npz in zip(flat_axes, npz_files):
        data = np.load(npz)
        preds = data["predictions"].astype(float)
        targets = data["targets"].astype(float)
        df, _, _ = _compute_ecdf_data(preds, targets, slices)
        _plot_single_ecdf(ax, df, labels, show_legend=False)
        # Title should be just the PLM name parsed from the file stem
        stem = npz.stem
        if stem.endswith("_pred"):
            stem = stem[: -len("_pred")]
        # Expected stem pattern: <pairs_stem>_<plm> (pairs_stem may contain underscores)
        plm = stem.split("_", 1)[1] if "_" in stem else stem
        ax.set_title(plm, fontsize=14, fontweight="bold")

    # Turn off unused subplots
    total = rows * cols
    for ax in list(flat_axes)[len(npz_files) : total]:
        ax.axis("off")

    # Add a single legend to the figure (top, horizontal)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, 0.99),
        )
    # Panel letter (e.g., A, B, C) in the upper-left corner
    if panel_label:
        fig.text(
            0.02, 0.98, panel_label, fontsize=32, fontweight="bold", va="top", ha="left"
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✓ ECDF grid →  {output_path}")
    return output_path


def _iter_param_subdirs(root: Path) -> Iterable[tuple[str, list[Path]]]:
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(sub.glob("*.npz"))
        if files:
            yield sub.name, files


def main() -> None:
    args = parse_cli()
    _apply_plot_style()

    # Single-file mode
    if args.npz and not args.dir:
        out_dir = args.out_dir if args.out_dir else None
        plot_single_file(args.npz, args.slices, out_dir)
        return

    # Batch mode
    if not args.dir:
        raise SystemExit("Provide either a single NPZ or --dir for batch mode")

    out_dir = args.out_dir if args.out_dir else args.dir
    grid_out_dir = out_dir

    for param_name, npz_list in _iter_param_subdirs(args.dir):
        # 1) per-file ECDFs
        for npz in npz_list:
            # save alongside unless --out-dir provided
            dest_dir = (out_dir / param_name) if out_dir else None
            plot_single_file(npz, args.slices, dest_dir)

        # 2) grid for the parameter
        grid_path = grid_out_dir / f"{param_name}_ecdf_grid.png"
        # Panel lettering mapping
        panel_map = {"fident": "A", "alntmscore": "B", "hfsp": "C"}
        panel_label = panel_map.get(param_name)
        # Grid size defaults can be overridden via --rows/--cols
        plot_grid_for_param(
            npz_list,
            args.slices,
            args.rows,
            args.cols,
            grid_path,
            panel_label=panel_label,
        )


if __name__ == "__main__":
    main()
