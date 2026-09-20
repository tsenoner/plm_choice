#!/usr/bin/env python
"""Draw Supplementary Figure S2: the probe architecture, generated FROM the model.

Why this script exists
----------------------
The figure it replaces was a hand-drawn raster with no source. It could not be checked
against the code and it drifted from it: it drew a "+" between the two branches (the
code concatenates), it never labelled the 128-wide concatenation, and its input
dimensions ("128 / 1024 / 2560") were an arbitrary three of the eleven native widths
actually used. A schematic of a network should be derived from the network.

So every layer width and every parameter count printed here is read off a live
``FNNPredictor`` at draw time (see ``_facts``). If ``src/training/models.py`` changes,
this figure changes with it, or it fails loudly. The one number *not* derived that way
is the input range n = 128…2,560, which comes from the ``NATIVE_DIMS`` literal below:
the arms' embedding widths are a property of the arms, not of the probe, so the model
has nothing to say about them.

Tool choice: matplotlib. The repo's whole figure pipeline is matplotlib on the committed
``.venv``, so this regenerates with the same interpreter as every other figure and needs
no LaTeX or Inkscape on a co-author's machine; it emits a vector PDF and SVG (editable)
next to the 600-dpi PNG the pandoc->docx build embeds.

Regenerate
----------
From the repository root:

    PYTHONPATH=src ./.venv/bin/python scripts/make_architecture_figure.py

    # write straight into the manuscript (a separate, private repo):
    PYTHONPATH=src ./.venv/bin/python scripts/make_architecture_figure.py \
        --out-dir manuscript/bib_2026/figures --stem som_figS02

Outputs ``<stem>.png`` (600 dpi), ``<stem>.pdf`` and ``<stem>.svg``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from training.models import FNNPredictor

# --- the facts, read off the model -------------------------------------------------
# Native per-protein embedding widths of the fifteen arms (Table S1). The probe is also
# run on 128-dimensional PCA features, which is why 128 is both the minimum and the PCA
# setting.
#
# NOTE: unlike the layer widths and parameter counts, this tuple is a literal that
# models.py cannot contradict -- _facts()'s guards check the layer structure, not this.
# It is checked by hand against the Embedding-dim column of Table S1
# (90.supplementary.md), whose distinct values are exactly these eleven.
NATIVE_DIMS = (128, 320, 480, 640, 768, 960, 1024, 1152, 1280, 1536, 2560)

# train.py --hidden_size default; scripts/lrz/probe_grid.sbatch never overrides it.
HIDDEN_SIZE = 64

TARGETS = ("PIDE", "TM-score", "HFSP")


def _facts(hidden_size: int = HIDDEN_SIZE) -> dict:
    """Read the layer widths and parameter counts off a live model.

    Nothing in the drawing is allowed to be a literal that the code could contradict.
    """
    probe = FNNPredictor(embedding_size=NATIVE_DIMS[0], hidden_size=hidden_size)

    shared = [m for m in probe.individual_layers if hasattr(m, "in_features")]
    combined = [m for m in probe.combined_layers if hasattr(m, "in_features")]
    if len(shared) != 1 or len(combined) != 3:
        raise SystemExit(
            "models.FNNPredictor no longer has 1 shared + 3 combined linear layers; "
            "the figure must be redrawn, not relabelled."
        )
    if combined[0].in_features != 2 * shared[0].out_features:
        raise SystemExit(
            "the combined trunk no longer takes twice the shared width; the "
            "concatenation this figure draws is no longer what the model does."
        )

    def n_params(dim: int) -> int:
        m = FNNPredictor(embedding_size=dim, hidden_size=hidden_size)
        return sum(p.numel() for p in m.parameters())

    return {
        "proj_out": shared[0].out_features,
        "concat": combined[0].in_features,
        "widths": [ly.out_features for ly in combined],  # 64, 32, 1
        "per_dim": hidden_size,  # params grow as hidden_size * n
        "fixed": sum(p.numel() for p in probe.combined_layers.parameters())
        + hidden_size,  # trunk + the shared layer's bias
        "p_min": n_params(min(NATIVE_DIMS)),
        "p_max": n_params(max(NATIVE_DIMS)),
        "n_min": min(NATIVE_DIMS),
        "n_max": max(NATIVE_DIMS),
    }


# --- palette -----------------------------------------------------------------------
A_FILL, A_EDGE = "#aecde2", "#2c6d94"  # query lane
B_FILL, B_EDGE = "#f3cba6", "#b3652c"  # target lane
T_FILL, T_EDGE = "#c6d9c0", "#497b52"  # shared trunk, after concatenation
E_FILL, E_EDGE = "#fbf3e2", "#a2862f"  # training-free Euclidean read-out
FROZEN_BAND = "#f2f2f2"
INK = "#1a1a1a"
MUTED = "#555555"

# --- geometry ----------------------------------------------------------------------
# Data units are square (aspect equal), so a unit is FIG_W_IN / X_MAX inches in both axes.
FIG_W_IN = 6.5
X_MAX = 172.0
Y_MAX = 92.0
FIG_H_IN = FIG_W_IN * Y_MAX / X_MAX

LANE_A_Y = 72.0
LANE_B_Y = 28.0
TRUNK_Y = 50.0
EUCL_Y = 8.0

UNITS_PER_DIM = 18.0 / 128.0  # a 128-wide tensor is 18 units tall
BAR_W = 6.0
INPUT_BAR_H = 28.0  # n varies 20-fold, so this bar is drawn broken, not to scale

X_PROT, W_PROT = 2.0, 18.0
X_INPUT = 42.0
X_SHARED, W_SHARED = 60.0, 14.0
X_CONCAT, W_CONCAT = 88.0, 8.0
X_H1 = 112.0
X_H2 = 132.0
X_OUT = 152.0

FS_OP = 6.4
FS_NOTE = 6.1
FS_BOX = 6.8


def _bar(ax, x, ycenter, height, fill, edge, width=BAR_W, zorder=4):
    ax.add_patch(
        Rectangle(
            (x, ycenter - height / 2),
            width,
            height,
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.9,
            zorder=zorder,
        )
    )


def _arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.9, ls="-", zorder=3):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            linestyle=ls,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def _op(ax, x0, x1, y, lines, fs=FS_OP, gap=1.6, lh=3.2, backdrop=None):
    """An operation: a horizontal arrow with its label stacked above it.

    ``backdrop`` fills the label's box with a colour, so a line routed behind the label
    is interrupted by it rather than drawn through the glyphs.  Pass the colour of
    whatever the label sits on, not white, or the patch itself becomes visible.
    """
    _arrow(ax, x0, y, x1, y)
    bbox = None if backdrop is None else dict(
        facecolor=backdrop, edgecolor="none", pad=0.6
    )
    for i, line in enumerate(reversed(lines)):
        ax.text(
            (x0 + x1) / 2,
            y + gap + i * lh,
            line,
            ha="center",
            va="bottom",
            fontsize=fs,
            color=INK,
            bbox=bbox,
            zorder=6,
        )


def draw(out_dir: Path, stem: str, dpi: int) -> list[Path]:
    f = _facts()
    proj, concat = f["proj_out"], f["concat"]
    w1, w2, w3 = f["widths"]
    half = proj * UNITS_PER_DIM

    SANS = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.family"] = SANS
    # One family for prose AND math. The default "dejavusans" mathtext set renders the
    # math labels in DejaVu while the prose is Arial, and \mathbb has no DejaVu glyph so
    # it fell back again to STIXGeneral -- three typefaces, two of them 2 mm apart in the
    # same panel. "custom" points every math alphabet at the same family; \mathbb is not
    # used, because no sans family has blackboard-bold.
    plt.rcParams["mathtext.fontset"] = "custom"
    for _k in ("rm", "it", "bf", "sf", "tt", "cal"):
        plt.rcParams[f"mathtext.{_k}"] = SANS[0] + (":italic" if _k in ("it", "cal") else "")
    plt.rcParams["mathtext.bf"] = SANS[0] + ":bold"
    plt.rcParams["mathtext.default"] = "it"
    # Type 42 (TrueType), not matplotlib's default Type 3: Type 3 text cannot be
    # selected or re-set in Illustrator and several publishers reject it outright.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    # Keep SVG text as text rather than outlines, so the vector copy is editable.
    plt.rcParams["svg.fonttype"] = "none"
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- the two regions: what is frozen and what is trained --------------------
    ax.add_patch(
        Rectangle((0, 18), 54, 70, facecolor=FROZEN_BAND, edgecolor="none", zorder=0)
    )
    ax.text(
        22, 89.0, "frozen — no gradient reaches the pLM",
        ha="center", va="bottom", fontsize=FS_NOTE, color=MUTED, style="italic",
    )
    ax.add_patch(
        FancyBboxPatch(
            (57, 18), 109, 70,
            boxstyle="round,pad=0,rounding_size=2",
            facecolor="none", edgecolor="#888888",
            linestyle=(0, (4, 3)), linewidth=0.8, zorder=1,
        )
    )
    ax.text(
        111.5, 89.0,
        f"trained probe — {f['per_dim']}n + {f['fixed']:,} parameters "
        f"({f['p_min']:,} at n = {f['n_min']}, {f['p_max']:,} at n = {f['n_max']:,})",
        ha="center", va="bottom", fontsize=FS_NOTE, color=MUTED, style="italic",
    )

    # ---- inputs: one protein per lane -------------------------------------------
    for y, label, fill, edge in (
        (LANE_A_Y, "Protein A\n(query)", A_FILL, A_EDGE),
        (LANE_B_Y, "Protein B\n(target)", B_FILL, B_EDGE),
    ):
        ax.add_patch(
            FancyBboxPatch(
                (X_PROT, y - 6.0), W_PROT, 12.0,
                boxstyle="round,pad=0,rounding_size=2",
                facecolor="white", edgecolor=edge, linewidth=0.9, zorder=4,
            )
        )
        ax.text(X_PROT + W_PROT / 2, y, label, ha="center", va="center",
                fontsize=FS_BOX, color=INK, zorder=6)
        _op(ax, X_PROT + W_PROT, X_INPUT, y, ["frozen pLM", "mean-pooled"],
            backdrop=FROZEN_BAND)
        _bar(ax, X_INPUT, y, INPUT_BAR_H, fill, edge)
        # a break mark: the bar's height is not to scale, because n varies 20-fold
        for off in (-1.3, 1.3):
            ax.plot(
                [X_INPUT - 0.7, X_INPUT + BAR_W + 0.7], [y + off, y + off + 1.5],
                color="white", lw=1.5, solid_capstyle="butt", zorder=5,
            )

    ax.text(X_INPUT + BAR_W / 2, LANE_A_Y - INPUT_BAR_H / 2 - 1.8,
            "$x_A$", ha="center", va="top",
            fontsize=FS_BOX, color=INK, zorder=6)
    ax.text(X_INPUT + BAR_W / 2, LANE_B_Y + INPUT_BAR_H / 2 + 7.2,
            "$x_B$", ha="center", va="bottom",
            fontsize=FS_BOX, color=INK, zorder=6)
    ax.text(X_INPUT + BAR_W / 2, LANE_B_Y + INPUT_BAR_H / 2 + 1.0,
            f"n = {f['n_min']}…{f['n_max']:,}\n(Table S1)",
            ha="center", va="bottom", fontsize=FS_NOTE, color=MUTED, zorder=6)

    # ---- the shared projection, drawn as ONE layer spanning both lanes ----------
    top, bot = LANE_A_Y + 6.0, LANE_B_Y - 6.0
    ax.add_patch(
        FancyBboxPatch(
            (X_SHARED, bot), W_SHARED, top - bot,
            boxstyle="round,pad=0,rounding_size=2",
            facecolor=T_FILL, edgecolor=T_EDGE, linewidth=0.9, zorder=4,
        )
    )
    ax.text(X_SHARED + W_SHARED / 2, (top + bot) / 2,
            f"Linear(n → {proj}) + ReLU", rotation=90,
            ha="center", va="center", fontsize=FS_OP + 0.4, color=INK, zorder=6)
    ax.text(X_SHARED + W_SHARED / 2, top + 1.6,
            "shared: one layer,\napplied to both",
            ha="center", va="bottom", fontsize=FS_NOTE, color=MUTED,
            style="italic", zorder=6)
    for y in (LANE_A_Y, LANE_B_Y):
        _arrow(ax, X_INPUT + BAR_W, y, X_SHARED, y)

    # ---- concatenation: h_A stacked on h_B --------------------------------------
    _bar(ax, X_CONCAT, TRUNK_Y + half / 2, half, A_FILL, A_EDGE, width=W_CONCAT)
    _bar(ax, X_CONCAT, TRUNK_Y - half / 2, half, B_FILL, B_EDGE, width=W_CONCAT)
    _arrow(ax, X_SHARED + W_SHARED, LANE_A_Y, X_CONCAT, TRUNK_Y + half / 2)
    _arrow(ax, X_SHARED + W_SHARED, LANE_B_Y, X_CONCAT, TRUNK_Y - half / 2)
    ax.text(X_SHARED + W_SHARED + 1.5, LANE_A_Y + 1.2, f"$h_A$ ({proj})",
            ha="left", va="bottom", fontsize=FS_OP, color=INK, zorder=6)
    ax.text(X_SHARED + W_SHARED + 1.5, LANE_B_Y - 1.2, f"$h_B$ ({proj})",
            ha="left", va="top", fontsize=FS_OP, color=INK, zorder=6)
    ax.text(X_CONCAT + W_CONCAT / 2 + 8, TRUNK_Y + half + 2.0,
            f"concatenate → {concat}", ha="center", va="bottom",
            fontsize=FS_OP, color=INK, zorder=6)
    ax.text(X_CONCAT + W_CONCAT / 2 + 8, TRUNK_Y - half - 2.0,
            "$[\\,h_A\\,;\\,h_B\\,]$, ordered:\n$f(A,B)$ ≠ $f(B,A)$",
            ha="center", va="top", fontsize=FS_NOTE, color=MUTED,
            style="italic", zorder=6)

    # ---- the trunk ---------------------------------------------------------------
    _op(ax, X_CONCAT + W_CONCAT, X_H1, TRUNK_Y, [f"{concat} → {w1}", "ReLU"])
    _bar(ax, X_H1, TRUNK_Y, w1 * UNITS_PER_DIM, T_FILL, T_EDGE)
    _op(ax, X_H1 + BAR_W, X_H2, TRUNK_Y, [f"{w1} → {w2}", "ReLU"])
    _bar(ax, X_H2, TRUNK_Y, w2 * UNITS_PER_DIM, T_FILL, T_EDGE)
    _op(ax, X_H2 + BAR_W, X_OUT, TRUNK_Y, [f"{w2} → {w3}", "no activation"])
    _bar(ax, X_OUT, TRUNK_Y, 2.4, "white", "#333333", width=3.2)
    ax.text(X_OUT + 5.0, TRUNK_Y, "$\\hat{y}$", ha="left", va="center",
            fontsize=9, color=INK, zorder=6)
    ax.text(X_OUT - 2.0, TRUNK_Y - 5.0,
            "trained separately for\n" + ", ".join(TARGETS),
            ha="center", va="top", fontsize=FS_NOTE, color=MUTED, zorder=6)
    # Kept to two short lines: the one-line version ran off the right edge of the figure
    # and through the shared box.  It must not say "every arrow is a fully connected
    # layer" -- of the ten arrows drawn, seven only route vectors, and the fourth fully
    # connected layer is a box.
    ax.text(131, 20.0,
            "fully connected layers: the shared Linear(n → 64) box\n"
            "and the three trunk arrows labelled in → out",
            ha="center", va="bottom", fontsize=FS_NOTE, color=MUTED,
            style="italic", linespacing=1.5, zorder=6)

    # ---- the training-free Euclidean read-out, off the same frozen embeddings ----
    # 6.5 units left of the bars, not 5.0: at 5.0 the rail clipped the leading "n" of
    # the "n = 128…2,560" label, whose box starts at ~37.7.  The two lane labels it still
    # passes behind are given the band's own colour as a backdrop, below.
    rail = X_INPUT - 6.5
    for y in (LANE_A_Y, LANE_B_Y):
        ax.plot([X_INPUT, rail], [y, y], color=E_EDGE, lw=0.9,
                ls=(0, (3, 2)), zorder=2)
    ax.plot([rail, rail], [LANE_A_Y, EUCL_Y], color=E_EDGE, lw=0.9,
            ls=(0, (3, 2)), zorder=2)
    _arrow(ax, rail, EUCL_Y, X_INPUT + 6.0, EUCL_Y, color=E_EDGE, ls=(0, (3, 2)))
    ax.add_patch(
        FancyBboxPatch(
            (X_INPUT + 6.0, EUCL_Y - 6.5), 108, 13,
            boxstyle="round,pad=0,rounding_size=2",
            facecolor=E_FILL, edgecolor=E_EDGE, linewidth=0.9, zorder=4,
        )
    )
    ax.text(X_INPUT + 60, EUCL_Y + 2.0,
            "Euclidean read-out:   ‖$x_A - x_B$‖$_2$",
            ha="center", va="center", fontsize=FS_BOX + 0.4, color=INK, zorder=5)
    ax.text(X_INPUT + 60, EUCL_Y - 3.2,
            "the same frozen embeddings, no trained parameters, symmetric",
            ha="center", va="center", fontsize=FS_NOTE, color=MUTED,
            style="italic", zorder=5)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext, kw in (("png", {"dpi": dpi}), ("pdf", {}), ("svg", {})):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    p.add_argument("--out-dir", type=Path, default=repo / "out" / "figures")
    p.add_argument("--stem", default="som_figS02")
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    for path in draw(a.out_dir, a.stem, a.dpi):
        print(path)


if __name__ == "__main__":
    main()
