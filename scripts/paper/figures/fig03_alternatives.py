#!/usr/bin/env python3
"""Figure 3 alternatives for the Wasserstein decision (ledger M-42, D14 open decisions).

    PAPER_ARTEFACTS=<manuscript>/paper_inputs python scripts/paper/figures/fig03_alternatives.py \
        --out <manuscript>/paper_inputs/d14_open_2026-10-17/fig03_alternatives

Figure 3's upper triangle is the Wasserstein distance W between min-max scaled distance
distributions; its ordering changes with the scale (z-score/IQR vs min-max rho 0.34/0.31,
``wasserstein_scaling.py``). Drawn here, from the published matrices only (nothing is
recomputed):

* ``B_p99.png`` -- the published figure with W on the 99th-percentile scale (``distances_p99``,
  which the reducer already wrote beside the min-max matrix). Drawn by the tracked plotter
  with the published shared scales, so it differs from Figure 3 in the upper triangle only.
* ``C1_rho_random.png`` -- the rank-correlation triangle alone (the published lower triangle).
* ``C2_rho_two_populations.png`` -- rho on aligner-found pairs (upper) against rho on random
  pairs (lower), one diverging scale: both triangles scale-free, the family fingerprint and
  its collapse on unrelated proteins in one panel. The upper triangle is the lower triangle
  of the published Figure S(fingerprint_aligned).
* ``S_W_four_scales.png`` -- the Supplement panel option C would move W to: the rank of each
  of the 91 W values under min-max, p99, z-score and IQR scaling, family blocks outlined.

Option A is the published ``bib_2026/figures/fig03.png``, copied as ``A_minmax.png``.
``summary.json`` holds the numbers the recommendation quotes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from visualization.pairwise_embedding_comparison import (  # noqa: E402
    CORR_DIVERGING_CMAP,
    EMBEDDING_DISPLAY_NAMES,
    EmbeddingComparisonVisualizer,
    _readable_text_color,
    symmetric_corr_limit,
)

SCALES = ("minmax", "p99", "zscore", "iqr")
SCALE_LABEL = {
    "minmax": "min–max (published)",
    "p99": "99th-percentile divisor",
    "zscore": "z-score",
    "iqr": "IQR",
}


def load(path: Path, out_dir: Path):
    payload = json.loads(path.read_text())
    viz = EmbeddingComparisonVisualizer.from_matrices(columns=payload["columns"], output_dir=out_dir)
    order = [payload["columns"].index(c.replace("dist_", "")) for c in viz.dist_cols]
    cols = [payload["columns"][i] for i in order]
    pick = np.ix_(order, order)
    mats = {k: np.asarray(payload[k], float)[pick] for k in ("correlations", "distances", "distances_p99")}
    return viz, cols, mats, order


def family_blocks(viz, cols):
    b = viz._get_family_boundaries([f"dist_{c}" for c in cols])
    return [(b[k], b[k + 1]) for k in range(len(b) - 1) if b[k + 1] - b[k] >= 2]


def family_auc(mat: np.ndarray, viz, cols, higher_is_closer: bool) -> float:
    """P(a within-family cell is closer than a cross-family cell), over the 91 cells."""
    fam = [viz._get_embedding_family(f"dist_{c}") for c in cols]
    iu = np.triu_indices(len(cols), 1)
    v = mat[iu] if higher_is_closer else -mat[iu]
    within = np.array([fam[i] == fam[j] for i, j in zip(*iu, strict=True)])
    u = stats.mannwhitneyu(v[within], v[~within], alternative="two-sided").statistic
    return float(u / (within.sum() * (~within).sum()))


def draw_rho(cols, viz, lower, upper, vlim, title, cbar_label, path, upper_label=None, lower_label=None):
    """One n x n grid, diagonal = model names; both triangles on the diverging rho scale."""
    n = len(cols)
    norm = plt.Normalize(-vlim, vlim)
    fig, ax = plt.subplots(figsize=(15.5, 13))
    grid = np.full((n, n), np.nan)
    il = np.tril_indices(n, -1)
    grid[il] = lower[il]
    if upper is not None:
        iu = np.triu_indices(n, 1)
        grid[iu] = upper[iu]
    ax.imshow(np.ma.masked_invalid(grid), cmap=CORR_DIVERGING_CMAP, norm=norm, aspect="equal")
    for i in range(n):
        for j in range(n):
            if i == j:
                name = EMBEDDING_DISPLAY_NAMES.get(cols[i], cols[i])
                ax.text(j, i, name, ha="center", va="center", fontsize=10.5, weight="bold")
            elif not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j] * 100:.0f}", ha="center", va="center", fontsize=14,
                        weight="bold", color=_readable_text_color(CORR_DIVERGING_CMAP, norm, grid[i, j]))
    for s, e in family_blocks(viz, cols):
        ax.add_patch(Rectangle((s - 0.5, s - 0.5), e - s, e - s, fill=False, lw=4, ec="black"))
    labels = [EMBEDDING_DISPLAY_NAMES.get(c, c) for c in cols]
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(n), labels, fontsize=12)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if upper_label or lower_label:
        title += "\n" + "   ·   ".join(x for x in (upper_label, lower_label) if x)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vlim * 100, vlim * 100), cmap=CORR_DIVERGING_CMAP),
                      ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label, fontsize=14)
    ax.set_title(title, fontsize=15, weight="bold", pad=14)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_w_ranks(cols, viz, matrices, aucs, rhos, path):
    n = len(cols)
    iu = np.triu_indices(n, 1)
    fig, axes = plt.subplots(2, 2, figsize=(17, 16))
    cmap = plt.get_cmap("Blues_r")
    for ax, scale in zip(axes.flat, SCALES, strict=True):
        m = matrices[scale]
        ranks = np.full((n, n), np.nan)
        ranks[iu] = stats.rankdata(m[iu])
        ax.imshow(np.ma.masked_invalid(ranks), cmap=cmap, vmin=1, vmax=len(iu[0]), aspect="equal")
        for i, j in zip(*iu, strict=True):
            r = ranks[i, j]
            ax.text(j, i, f"{r:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if r < 40 else "black")
        for i in range(n):
            ax.text(i, i, EMBEDDING_DISPLAY_NAMES.get(cols[i], cols[i]),
                    ha="center", va="center", fontsize=6.5, weight="bold")
        for s, e in family_blocks(viz, cols):
            ax.add_patch(Rectangle((s - 0.5, s - 0.5), e - s, e - s, fill=False, lw=2.5, ec="black"))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        rho = "" if scale == "minmax" else f"; ρ with min–max {rhos[scale]:.2f}"
        ax.set_title(f"{SCALE_LABEL[scale]}\nfamily AUC {aucs[scale]:.2f}{rho}", fontsize=13)
    fig.suptitle("Rank of each Wasserstein distance among the 91 model pairs (1 = most alike),\n"
                 "5,000,000 random pairs; black frames = families", fontsize=15, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    art = Path(os.environ.get("PAPER_ARTEFACTS", REPO / "manuscript" / "paper_inputs"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", type=Path, default=art / "final_figures_2026-09-20/fp/out/fingerprint_full.json")
    ap.add_argument("--aligned", type=Path, default=art / "figures_2026-09-19/fig03/fingerprint_full.json")
    ap.add_argument("--scaling", type=Path,
                    default=art / "d14_2026-10-16/wasserstein_scaling/wasserstein_scaling.json")
    ap.add_argument("--published", type=Path, default=REPO / "manuscript/bib_2026/figures/fig03.png")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    viz, cols, rnd, order = load(args.random, args.out)
    _, cols_a, aln, _ = load(args.aligned, args.out)
    assert cols == cols_a
    vlim = symmetric_corr_limit(rnd["correlations"], aln["correlations"])

    # A: the published figure, as is.
    if args.published.is_file():
        shutil.copyfile(args.published, args.out / "A_minmax.png")

    # B: same plotter, same shared scales as the published pair, W on the p99 scale.
    wtop = max(float(np.nanmax(m["distances_p99"][np.triu_indices(14, 1)])) for m in (rnd, aln))
    fig, _ = viz.plot_combined_wasserstein_correlation(
        wasserstein_data={"distances": rnd["distances_p99"].tolist(), "columns": cols},
        correlation_data={"correlations": rnd["correlations"].tolist(), "columns": cols},
        corr_vlim=vlim, wass_vmax=wtop,
    )
    fig._suptitle.set_text("Combined Pairwise Comparison (values ×100)\n(Upper: Wasserstein Distance, each model "
                           "divided by its 99th percentile; Diagonal: Models; Lower: Spearman Correlation)")
    fig.savefig(args.out / "B_p99.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # C1 / C2: rho only.
    draw_rho(cols, viz, rnd["correlations"], None, vlim,
             "Spearman correlation between models' pairwise distances (×100)\n5,000,000 random pairs",
             "Spearman ρ (×100)", args.out / "C1_rho_random.png")
    draw_rho(cols, viz, rnd["correlations"], aln["correlations"], vlim,
             "Spearman correlation between models' pairwise distances (×100)",
             "Spearman ρ (×100)", args.out / "C2_rho_two_populations.png",
             upper_label="upper right: 74.6 M pairs MMseqs2/Foldseek could align",
             lower_label="lower left: 5,000,000 random pairs")

    # Supplement panel: W ranks on four scales (matrices from wasserstein_scaling.py, same order).
    sc = json.loads(args.scaling.read_text())
    assert sc["columns"] == json.loads(args.random.read_text())["columns"]
    wm = {s: np.asarray(sc["matrices"][s], float)[np.ix_(order, order)] for s in SCALES}
    assert np.allclose(wm["minmax"], rnd["distances"], atol=1e-6), "scaling JSON is not Figure 3's matrix"
    iu = np.triu_indices(14, 1)
    aucs = {s: family_auc(wm[s], viz, cols, higher_is_closer=False) for s in SCALES}
    rhos = {s: float(stats.spearmanr(wm[s][iu], wm["minmax"][iu]).statistic) for s in SCALES}
    draw_w_ranks(cols, viz, wm, aucs, rhos, args.out / "S_W_four_scales.png")

    rho_auc = {
        "random": family_auc(rnd["correlations"], viz, cols, higher_is_closer=True),
        "aligned": family_auc(aln["correlations"], viz, cols, higher_is_closer=True),
    }
    summary = {
        "corr_vlim": vlim,
        "w_family_auc": aucs,
        "w_rank_rho_vs_minmax": rhos,
        "rho_family_auc": rho_auc,
        "rho_random_vs_aligned_spearman": float(
            stats.spearmanr(rnd["correlations"][iu], aln["correlations"][iu]).statistic
        ),
        "mean_rho": {"random": float(rnd["correlations"][iu].mean()), "aligned": float(aln["correlations"][iu].mean())},
        "w_p99_range_x100": [float(rnd["distances_p99"][iu].min() * 100), float(rnd["distances_p99"][iu].max() * 100)],
        "w_minmax_range_x100": [float(rnd["distances"][iu].min() * 100), float(rnd["distances"][iu].max() * 100)],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
