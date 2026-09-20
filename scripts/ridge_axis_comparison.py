#!/usr/bin/env python3
"""Put the candidate ridge axes side by side on the numbers, not on impressions.

The ridge figure has to rescale each row -- raw medians span four orders of magnitude
across arms -- and the choice of what to anchor on changes what the figure can be read
for.  Rendering three full ridgelines and eyeballing them compares the smoothing as
much as the axis.  This strips each arm to its five-number summary under each
candidate and draws them on one page, so the only thing that differs between panels is
the rescaling.

Per arm and per axis it draws p1 ---- q25 ==== median ==== q75 ---- p99, all exact
quantiles from the cluster reduction, plus two summary numbers per axis:

  * **median spread** -- how far apart the fourteen medians sit on the drawn axis, as a
    fraction of that axis.  This is the axis's power to separate models, and it has to
    be a fraction because the three candidates do not span the same range: 0.45 out of
    [0, 1] and 0.45 out of [0, 3] are not the same picture.  It is 0 by construction
    under /median, which centres every row on 1.0.
  * **axis used** -- the fraction of the drawn range between the smallest p1 and the
    largest p99.  What is left over is axis spent on nothing.

    python scripts/ridge_axis_comparison.py \
        --summary-dir <summaries> --out-dir <figures> --norm minmax p99 median
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from visualization.pairwise_embedding_comparison import (  # noqa: E402
    RIDGE_NORMALISATIONS,
    EmbeddingComparisonVisualizer,
)
from visualization.plm_constants import EMBEDDING_DISPLAY_NAMES  # noqa: E402

KEYS = ("p1", "q25", "median", "q75", "p99")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--norm", nargs="+", default=["minmax", "p99", "median"])
    ap.add_argument("--population", default="", help="Named in the figure title.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    viz = EmbeddingComparisonVisualizer.from_distribution_summaries(
        summary_dir=args.summary_dir, output_dir=args.out_dir
    )
    arms = [c.replace("dist_", "") for c in viz.dist_cols]

    per_norm = {}
    for norm in args.norm:
        if norm not in RIDGE_NORMALISATIONS:
            raise SystemExit(f"unknown --norm {norm!r}")
        data = viz.compute_distribution_data_from_summaries(normalisation=norm)
        per_norm[norm] = data

    rows = []
    fig, axs = plt.subplots(
        1, len(args.norm), figsize=(6.2 * len(args.norm), 7.2), sharey=True
    )
    axs = np.atleast_1d(axs)

    for ax, norm in zip(axs, args.norm, strict=True):
        data = per_norm[norm]
        spec = RIDGE_NORMALISATIONS[norm]
        xlim = tuple(data["metadata"]["xlim"])
        vals = {}
        for y, arm in enumerate(arms):
            d = data["distributions"][f"dist_{arm}"]
            v = {k: float(d[k]) for k in KEYS}
            vals[arm] = v
            color = viz._get_embedding_color(f"dist_{arm}")
            yy = len(arms) - 1 - y
            ax.plot([v["p1"], v["p99"]], [yy, yy], color=color, lw=1.6, zorder=2,
                    solid_capstyle="butt")
            for k in ("p1", "p99"):
                ax.plot([v[k], v[k]], [yy - 0.22, yy + 0.22], color=color, lw=1.6,
                        zorder=2)
            ax.plot([v["q25"], v["q75"]], [yy, yy], color=color, lw=7.5, zorder=3,
                    solid_capstyle="butt")
            ax.plot([v["median"], v["median"]], [yy - 0.3, yy + 0.3], color="black",
                    lw=2.2, zorder=4)
            rows.append({"normalisation": norm, "plm_name": arm,
                         "plm_display_name":
                             EMBEDDING_DISPLAY_NAMES.get(arm, arm).replace("\n", " "),
                         **v,
                         "divisor": float(d["divisor"]),
                         "offset": float(d.get("offset", 0.0))})

        medians = np.array([vals[a]["median"] for a in arms])
        spread = float(medians.max() - medians.min())
        used = (
            max(vals[a]["p99"] for a in arms) - min(vals[a]["p1"] for a in arms)
        ) / (xlim[1] - xlim[0])
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.8, len(arms) - 0.2)
        ax.set_xlabel(spec["label"], fontsize=11)
        ax.set_title(
            f"{norm}\nmedian spread {spread:.3f} = {spread / (xlim[1] - xlim[0]):.0%} "
            f"of the axis   |   axis used {used:.0%}",
            fontsize=12, fontweight="bold",
        )
        ax.grid(axis="x", color="0.9", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    axs[0].set_yticks(range(len(arms)))
    axs[0].set_yticklabels(
        [EMBEDDING_DISPLAY_NAMES.get(a, a).replace("\n", " ") for a in reversed(arms)],
        fontsize=11, fontweight="bold",
    )
    for tick, arm in zip(axs[0].get_yticklabels(), reversed(arms), strict=True):
        tick.set_color(viz._get_embedding_color(f"dist_{arm}"))

    handles = [
        plt.Line2D([], [], color="0.35", lw=1.6, label="1st - 99th percentile"),
        plt.Line2D([], [], color="0.35", lw=7.5, label="25th - 75th percentile"),
        plt.Line2D([], [], color="black", lw=2.2, label="median"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.005))
    title = "Candidate axes for Figure 2, same five exact quantiles under each"
    if args.population:
        title += f"\n{args.population}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))

    png = args.out_dir / f"axis_comparison{args.tag}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    csv = args.out_dir / f"axis_comparison{args.tag}.csv"
    pl.DataFrame(rows).write_csv(csv)
    print(f"wrote {png}\nwrote {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
