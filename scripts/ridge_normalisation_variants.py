#!/usr/bin/env python3
"""Draw the distance-distribution ridge under different normalisations.

Why this exists. The published figure min-max scales every arm, so each row's axis is
pinned to that arm's single most distant pair: one outlier sets the scale for six million
points. On the corrected pair set that squeezes every distribution into the left third of
the axis and drags the published medians from 0.20-0.54 down to 0.06-0.31 -- a change that
says nothing about the embeddings and everything about the tail. A percentile scale keeps
the same shape without letting one pair define it.

It also drops identical-sequence pairs. 23,369 pairs (0.39%) sit at exactly distance 0 in
twelve of the fifteen arms; spot-checking eight of them against sprot.fasta, all eight are
the same sequence deposited under two accessions. They are not a property of any embedding
space -- they are the same protein twice -- and they are what produces the spike at 0 and
pins the min-max minimum. random_1024 has none, which is the tell.

    python scripts/ridge_normalisation_variants.py \
        --pairs <merged distances>.parquet --out-dir <dir> [--norm p99 minmax]
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
import polars as pl  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from visualization.plm_constants import (  # noqa: E402
    EMBEDDING_COLOR_MAP,
    EMBEDDING_DISPLAY_NAMES,
)

GRID = 500
# Same exclusions as the shipped plotter: the i.i.d. floor and the excluded arm.
SKIP = {"random_1024", "prostt5"}


def normalise(x: np.ndarray, how: str) -> tuple[np.ndarray, float]:
    """Return the scaled values and the divisor, so the caption can state it."""
    if how == "minmax":
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo), float(hi)
    if how == "p99":
        hi = float(np.percentile(x, 99))
        return x / hi, hi
    if how == "p999":
        hi = float(np.percentile(x, 99.9))
        return x / hi, hi
    raise ValueError(f"unknown normalisation {how!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--norm", nargs="+", default=["p99", "minmax"])
    ap.add_argument(
        "--keep-identical",
        action="store_true",
        help="keep the exact-zero (identical-sequence) pairs instead of dropping them",
    )
    ap.add_argument("--xmax", type=float, default=1.6, help="x-axis limit for the plot")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_parquet(args.pairs)
    arms = [
        c[len("dist_") :]
        for c in lf.collect_schema().names()
        if c.startswith("dist_") and c[len("dist_") :] not in SKIP
    ]
    stats: dict[str, dict] = {}

    for how in args.norm:
        fig, axes = plt.subplots(
            len(arms), 1, figsize=(11, 0.62 * len(arms)), sharex=True
        )
        for ax, arm in zip(axes, arms, strict=True):
            col = f"dist_{arm}"
            x = lf.select(pl.col(col)).collect().to_series().drop_nulls().to_numpy()
            n_raw = x.size
            if not args.keep_identical:
                x = x[x > 0]
            scaled, divisor = normalise(x, how)
            # KDE on a subsample: at 6M points the density is the population density and
            # a full-sample gaussian_kde evaluation is pure cost.
            rng = np.random.default_rng(42)
            sample = scaled if scaled.size <= 200_000 else rng.choice(scaled, 200_000, replace=False)
            grid = np.linspace(0, args.xmax, GRID)
            dens = gaussian_kde(sample)(grid)
            q25, med, q75 = np.percentile(scaled, [25, 50, 75])
            stats.setdefault(how, {})[arm] = {
                "n_pairs": int(scaled.size),
                "n_identical_dropped": int(n_raw - scaled.size),
                "divisor": divisor,
                "q25": float(q25),
                "median": float(med),
                "q75": float(q75),
            }
            colour = EMBEDDING_COLOR_MAP.get(arm, "#808080")
            ax.fill_between(grid, dens, color=colour, alpha=0.85, linewidth=0)
            ax.plot(grid, dens, color="white", linewidth=0.8)
            for v, style in ((med, "--"), (q25, ":"), (q75, ":")):
                if v <= args.xmax:
                    ax.axvline(v, color="black" if style == "--" else "grey",
                               linestyle=style, linewidth=1.0)
            ax.set_yticks([])
            ax.set_ylabel(
                EMBEDDING_DISPLAY_NAMES.get(arm, arm).replace("\n", " "),
                rotation=0, ha="right", va="center", fontsize=9, color=colour,
                fontweight="bold",
            )
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
        label = {
            "minmax": "Min–max normalised distance",
            "p99": "Distance / 99th percentile",
            "p999": "Distance / 99.9th percentile",
        }[how]
        axes[-1].set_xlabel(label, fontsize=11)
        axes[0].set_title(
            f"Pairwise embedding distance distributions — {label}"
            + ("" if args.keep_identical else " (identical-sequence pairs removed)"),
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        out = args.out_dir / f"ridge_{how}{'_withidentical' if args.keep_identical else ''}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")

    (args.out_dir / "variant_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {args.out_dir / 'variant_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
