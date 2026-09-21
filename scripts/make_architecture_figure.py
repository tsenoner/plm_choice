#!/usr/bin/env python
"""Draw Supplementary Figure S2: the probe architecture, generated FROM the model.

Why this script exists
----------------------
The figure it replaces was a hand-drawn raster with no source. It could not be checked
against the code and it drifted from it: it drew a "+" between the two branches (the
code concatenates), it never labelled the 128-wide concatenation, and its input
dimensions ("128 / 1024 / 2560") were an arbitrary three of the eleven native widths
actually used. A schematic of a network should be derived from the network.

So every layer width printed here is read off a live ``FNNPredictor`` at draw time (see
``_facts``). If ``src/training/models.py`` changes, this figure changes with it, or it
fails loudly. The one number *not* derived that way is the input range n = 128…2,560,
which comes from the ``NATIVE_DIMS`` literal below: the arms' embedding widths are a
property of the arms, not of the probe, so the model has nothing to say about them.

What the drawing may say
------------------------
The caption in ``sections/90.supplementary.md`` already states the parameter counts, the
n range, the ordered-concatenation asymmetry, the three targets and the fact that the
Euclidean read-out is symmetric and trains nothing. A figure that repeats its own caption
is read twice and understood once, so this one carries the *shape* and nothing else:
region, layer, tensor. Two rules make that possible without a legend --

  * a box is a layer, an arrow is a tensor. Nothing is a layer because a label sits on
    the arrow next to it. (The version before this one needed a sentence -- "fully
    connected layers: the shared box and the three trunk arrows labelled in -> out" --
    to undo exactly that ambiguity.)
  * a weight-shared layer is drawn once per lane and tied, the ordinary Siamese
    convention. Drawing it as a single tall box spanning both lanes made it read as a
    merge-then-split, which is why *that* version needed a second sentence, "shared: one
    layer, applied to both".

The parameter counts are still computed -- ``main`` prints them to stdout -- so this
script stays the checkable source for the numbers the caption quotes.

Tool choice: matplotlib. The repo's whole figure pipeline is matplotlib, so this
regenerates with the same interpreter as every other figure and needs no LaTeX or
Inkscape on a co-author's machine; it emits a vector PDF and SVG (editable) next to the
600-dpi PNG the pandoc->docx build embeds.

Regenerate
----------
From a checkout that has the project venv (a bare worktree has none -- point at the main
checkout's ``.venv`` and set ``PYTHONPATH`` to the worktree's ``src``):

    PYTHONPATH=src ./.venv/bin/python scripts/make_architecture_figure.py

    # write straight into the manuscript (a separate, private repo):
    PYTHONPATH=src ./.venv/bin/python scripts/make_architecture_figure.py \
        --out-dir manuscript/bib_2026/figures --stem som_figS02

Outputs ``<stem>.png`` (600 dpi), ``<stem>.pdf`` and ``<stem>.svg``.

The printed size is fixed at 6.500 x 3.477 in because the caption hardcodes
``height="3.476667in"``. Space freed by cutting text goes into larger type, not a
shorter figure; changing ``X_MAX``/``Y_MAX``'s ratio means editing the caption too.
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
    The parameter counts are not drawn any more -- they live in the caption -- but they
    are still derived here and printed by ``main``, so the caption has a source.
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
T_FILL, T_EDGE = "#c6d9c0", "#497b52"  # a trained layer, in either lane or the trunk
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

LANE_A_Y = 64.0
LANE_B_Y = 32.0
TRUNK_Y = 48.0
EUCL_Y = 9.0

REGION_BOT, REGION_TOP = 20.0, 82.0

UNITS_PER_DIM = 18.0 / 128.0  # a 128-wide tensor is 18 units tall
BAR_W = 6.0
INPUT_BAR_H = 18.0  # not to scale: n varies 20-fold. The adjacent "n = 128...2,560"
# label carries that; this used to also carry a drawn break mark, but two white slashes
# across the bar read as an artefact rather than as a break.

BAND_W = 40.0  # the frozen region: 0 .. 40
PROBE_X, PROBE_W = 45.0, 122.0  # the trained region: 45 .. 167

X_PROT, W_PROT = 2.0, 17.0  # 2 .. 19
X_INPUT = 31.0  # 31 .. 37
TAP_X = 42.0  # the Euclidean tap, in the gutter between the two regions
X_SHARED, W_SHARED = 47.0, 17.0  # 47 .. 64
X_CONCAT, W_CONCAT = 80.0, 7.0  # 80 .. 87
W_LAYER, H_LAYER = 17.0, 18.0
X_T1, X_T2, X_T3 = 92.0, 114.0, 136.0  # each .. +17, with a 5-unit arrow between
X_YHAT = 160.0

FS_OP = 7.4  # in -> out, inside a layer box
FS_NOTE = 6.9  # region headers and the small standing notes
FS_BOX = 7.8  # protein names, tensor names
FS_SUB = 6.0  # the activation line under a layer's in -> out; "no activation" is the
# longest string in the figure relative to its box, so this one does not scale with the
# rest -- at 6.4 pt it overruns W_LAYER.


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


def _layer(ax, x, ycenter, label, sub, w=W_LAYER, h=H_LAYER):
    """A trained layer: a box carrying ``in -> out`` and its activation.

    Every trained layer in the figure is drawn by this one function, so "box" means
    "layer" everywhere and no legend has to say which arrows are secretly layers.
    """
    ax.add_patch(
        FancyBboxPatch(
            (x, ycenter - h / 2), w, h,
            boxstyle="round,pad=0,rounding_size=1.6",
            facecolor=T_FILL, edgecolor=T_EDGE, linewidth=0.9, zorder=4,
        )
    )
    ax.text(x + w / 2, ycenter + 1.4, label, ha="center", va="bottom",
            fontsize=FS_OP, color=INK, zorder=6)
    ax.text(x + w / 2, ycenter - 1.8, sub, ha="center", va="top",
            fontsize=FS_SUB, color=MUTED, zorder=6)


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
    # Two words each. That no gradient reaches the pLM, and how big the probe is, are
    # the caption's job.
    ax.add_patch(
        Rectangle((0, REGION_BOT), BAND_W, REGION_TOP - REGION_BOT,
                  facecolor=FROZEN_BAND, edgecolor="none", zorder=0)
    )
    ax.text(BAND_W / 2, REGION_TOP + 1.5, "frozen pLM, mean-pooled",
            ha="center", va="bottom",
            fontsize=FS_NOTE, color=MUTED, style="italic")
    ax.add_patch(
        FancyBboxPatch(
            (PROBE_X, REGION_BOT), PROBE_W, REGION_TOP - REGION_BOT,
            boxstyle="round,pad=0,rounding_size=2",
            facecolor="none", edgecolor="#888888",
            linestyle=(0, (4, 3)), linewidth=0.8, zorder=1,
        )
    )
    ax.text(PROBE_X + PROBE_W / 2, REGION_TOP + 1.5, "trained probe — one per target",
            ha="center", va="bottom",
            fontsize=FS_NOTE, color=MUTED, style="italic")

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
        _arrow(ax, X_PROT + W_PROT, y, X_INPUT, y)
        _bar(ax, X_INPUT, y, INPUT_BAR_H, fill, edge)
        _arrow(ax, X_INPUT + BAR_W, y, X_SHARED, y)

    # x_A under its bar, x_B over its bar, the shared width between them.
    ax.text(X_INPUT + BAR_W / 2, LANE_A_Y - INPUT_BAR_H / 2 - 1.5, "$x_A$",
            ha="center", va="top", fontsize=FS_BOX, color=INK, zorder=6)
    ax.text(X_INPUT + BAR_W / 2, LANE_B_Y + INPUT_BAR_H / 2 + 1.5, "$x_B$",
            ha="center", va="bottom", fontsize=FS_BOX, color=INK, zorder=6)
    # Half a unit left of the bar column's centre: centred on the bars it ran into the
    # Euclidean tap at TAP_X, and then read as one line with the "shared weights" tie
    # beyond it.
    ax.text(X_INPUT + 0.5, TRUNK_Y,
            f"n = {f['n_min']}…{f['n_max']:,}", ha="center", va="center",
            fontsize=FS_NOTE, color=MUTED, zorder=6)

    # ---- the shared projection: one box per lane, tied ---------------------------
    for y in (LANE_A_Y, LANE_B_Y):
        _layer(ax, X_SHARED, y, f"n → {proj}", "ReLU", w=W_SHARED)
    tie_x = X_SHARED + W_SHARED / 2
    for y0, y1 in (
        (LANE_A_Y - H_LAYER / 2, TRUNK_Y + 3.0),
        (TRUNK_Y - 3.0, LANE_B_Y + H_LAYER / 2),
    ):
        ax.plot([tie_x, tie_x], [y0, y1], color=T_EDGE, lw=0.8,
                ls=(0, (1, 1.8)), zorder=3)
    ax.text(tie_x, TRUNK_Y, "shared weights", ha="center", va="center",
            fontsize=FS_NOTE, color=T_EDGE, style="italic", zorder=6)

    # ---- concatenation: the query lane stacked on the target lane ----------------
    _bar(ax, X_CONCAT, TRUNK_Y + half / 2, half, A_FILL, A_EDGE, width=W_CONCAT)
    _bar(ax, X_CONCAT, TRUNK_Y - half / 2, half, B_FILL, B_EDGE, width=W_CONCAT)
    _arrow(ax, X_SHARED + W_SHARED, LANE_A_Y, X_CONCAT, TRUNK_Y + half / 2)
    _arrow(ax, X_SHARED + W_SHARED, LANE_B_Y, X_CONCAT, TRUNK_Y - half / 2)
    # Offset right of the bar's centre so the incoming query-lane arrow clears the
    # label's left edge; keep the offset small, or it stops reading as the bar's label.
    ax.text(X_CONCAT + W_CONCAT / 2 + 2.5, TRUNK_Y + half + 2.0,
            f"concatenate → {concat}", ha="center", va="bottom",
            fontsize=FS_OP, color=INK, zorder=6)

    # ---- the trunk: three boxes, three arrows ------------------------------------
    _arrow(ax, X_CONCAT + W_CONCAT, TRUNK_Y, X_T1, TRUNK_Y)
    _layer(ax, X_T1, TRUNK_Y, f"{concat} → {w1}", "ReLU")
    _arrow(ax, X_T1 + W_LAYER, TRUNK_Y, X_T2, TRUNK_Y)
    _layer(ax, X_T2, TRUNK_Y, f"{w1} → {w2}", "ReLU")
    _arrow(ax, X_T2 + W_LAYER, TRUNK_Y, X_T3, TRUNK_Y)
    _layer(ax, X_T3, TRUNK_Y, f"{w2} → {w3}", "no activation")
    _arrow(ax, X_T3 + W_LAYER, TRUNK_Y, X_YHAT - 1.5, TRUNK_Y)
    ax.text(X_YHAT, TRUNK_Y, "$\\hat{y}$", ha="left", va="center",
            fontsize=10, color=INK, zorder=6)
    # One line, not two. Broken after "TM-score" the list stopped reading as three
    # alternatives and started reading as a stanza; the header carries "one per target",
    # so the "or" only has to separate, not explain.
    ax.text(X_YHAT - 9.0, TRUNK_Y - H_LAYER / 2 - 2.5,
            "PIDE, TM-score or HFSP", ha="center", va="top",
            fontsize=FS_NOTE, color=MUTED, zorder=6)

    # ---- the training-free Euclidean read-out, off the same frozen embeddings ----
    # It forks forward, down the gutter between the frozen band and the trained box,
    # tapping each lane's arrow at a junction dot. The version before this one ran the
    # branch backwards out of the bars and then behind both lane labels.
    ax.plot([TAP_X, TAP_X], [LANE_A_Y, EUCL_Y], color=E_EDGE, lw=0.9,
            ls=(0, (3, 2)), zorder=2)
    for y in (LANE_A_Y, LANE_B_Y):
        ax.plot([TAP_X], [y], marker="o", markersize=2.6, color=E_EDGE, zorder=5)
    _arrow(ax, TAP_X, EUCL_Y, X_SHARED, EUCL_Y, color=E_EDGE, ls=(0, (3, 2)))
    # Left edge under the shared layers, right edge under the second trunk layer, so the
    # box lines up with the thing it is the alternative to.
    ax.add_patch(
        FancyBboxPatch(
            (X_SHARED, EUCL_Y - 6.0), X_T2 + W_LAYER - X_SHARED, 12,
            boxstyle="round,pad=0,rounding_size=2",
            facecolor=E_FILL, edgecolor=E_EDGE, linewidth=0.9, zorder=4,
        )
    )
    ax.text((X_SHARED + X_T2 + W_LAYER) / 2, EUCL_Y + 1.8,
            "Euclidean read-out:   ‖$x_A - x_B$‖$_2$",
            ha="center", va="center", fontsize=FS_BOX, color=INK, zorder=5)
    ax.text((X_SHARED + X_T2 + W_LAYER) / 2, EUCL_Y - 3.2, "no training",
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
    # The figure no longer prints these; the caption does. Derive them here so the
    # caption has a source that fails loudly if models.py changes.
    f = _facts()
    print(
        f"caption: {f['per_dim']}n + {f['fixed']:,} parameters "
        f"({f['p_min']:,} at n = {f['n_min']}, {f['p_max']:,} at n = {f['n_max']:,}); "
        f"targets: {', '.join(TARGETS)}"
    )


if __name__ == "__main__":
    main()
