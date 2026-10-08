#!/usr/bin/env python3
"""Supplementary figure: pre-training moves effective dimensionality by family, not by size.

Reads the participation ratios recorded by the B4 degeneracy gate (freeze/degeneracy_*.json):
PR = (sum lambda)^2 / sum lambda^2 of the centred covariance spectrum over the same 1,000
proteins, for each pretrained model and its randomly initialised twin of identical width.

    uv run python scripts/paper/figures/participation_ratio.py --out <png>

Why the comparison is fair: PR is bounded by embedding width, so absolute PR is not comparable
across families of different width. Every comparison here is between a pretrained model and its
OWN untrained twin, which has the same width, so the ratio is width-controlled.

Colours are the paper's family colours (plm_constants.EMBEDDING_FAMILY_COLOR_MAP), kept for
consistency with Figures 1-3. That palette fails a colour-vision check (ESM-1 green vs ESM-2
orange, dE 5.3 under protanopia), so identity is never carried by colour alone here: each family
also has its own marker, and every point is labelled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from visualization.plm_constants import EMBEDDING_FAMILY_COLOR_MAP, PLM_SIZES  # noqa: E402

FREEZE = REPO / "freeze"

# (key, label, family). Order: by family, then by size -- the order the reader scans.
MODELS = [
    ("ankh_base", "Ankh-base", "Ankh"),
    ("ankh_large", "Ankh-large", "Ankh"),
    ("prott5", "ProtT5", "ProtT5"),
    ("esm1b", "ESM-1b", "ESM-1"),
    ("esm2_8m", "ESM-2 8M", "ESM-2"),
    ("esm2_35m", "ESM-2 35M", "ESM-2"),
    ("esm2_150m", "ESM-2 150M", "ESM-2"),
    ("esm2_650m", "ESM-2 650M", "ESM-2"),
    ("esm2_3b", "ESM-2 3B", "ESM-2"),
    ("esmc_300m", "ESM-C 300M", "ESM-C"),
    ("esmc_600m", "ESM-C 600M", "ESM-C"),
]
MARKER = {"Ankh": "s", "ProtT5": "D", "ESM-1": "^", "ESM-2": "o", "ESM-C": "v"}
INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"


def load() -> list[dict]:
    pre = json.loads((FREEZE / "degeneracy_pretrained.json").read_text())
    ri = json.loads((FREEZE / "degeneracy_random_init.json").read_text())
    rows = []
    for key, label, fam in MODELS:
        twin = ri[f"random_init_{key.replace('prott5', 'prot_t5')}_seed0"]
        p, u = pre[key], twin
        assert p["d"] == u["d"], f"{key}: widths differ, the ratio would not be width-controlled"
        assert p["n"] == u["n"] == 1000, key
        rows.append({"key": key, "label": label, "fam": fam, "d": p["d"],
                     "untrained": u["pr"], "pretrained": p["pr"],
                     "ratio": p["pr"] / u["pr"], "params": PLM_SIZES[key]})
    noise = pre["random_1024"]["pr"]
    return rows, noise


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows, noise = load()

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})

    # --- A: untrained -> pretrained, per model --------------------------------------------
    for i, r in enumerate(rows):
        c, m = EMBEDDING_FAMILY_COLOR_MAP[r["fam"]], MARKER[r["fam"]]
        # An arrowhead on a move shorter than the markers reads as a blob, so arrows are drawn
        # only for at-least-twofold changes; open/filled markers carry direction either way.
        if max(r["ratio"], 1 / r["ratio"]) >= 2:
            a.annotate("", xy=(i, r["pretrained"]), xytext=(i, r["untrained"]),
                       arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, shrinkA=4, shrinkB=4,
                                       mutation_scale=8))
        else:
            a.plot([i, i], [r["untrained"], r["pretrained"]], color=c, lw=1.2, zorder=2)
        a.scatter(i, r["untrained"], s=34, marker=m, facecolor="white", edgecolor=c, linewidth=1.4, zorder=3)
        a.scatter(i, r["pretrained"], s=34, marker=m, facecolor=c, edgecolor=INK, linewidth=0.6, zorder=4)
    a.set_yscale("log")
    a.set_ylim(3, 90)
    a.set_yticks([3, 5, 10, 20, 50])
    a.set_yticklabels(["3", "5", "10", "20", "50"])
    a.set_xticks(range(len(rows)))
    a.set_xticklabels([r["label"] for r in rows], rotation=55, ha="right", fontsize=7.4)
    a.set_ylabel("participation ratio (log)", color=INK, fontsize=9)
    a.set_xlim(-0.6, len(rows) - 0.4)
    style(a)
    a.text(0.02, 0.97, "A", transform=a.transAxes, fontsize=12, fontweight="bold", va="top", color=INK)
    a.text(0.98, 0.97, f"open = untrained twin\nfilled = pretrained\nGaussian 1,024-d: PR {noise:.0f}",
           transform=a.transAxes, ha="right", va="top", fontsize=7, color=MUTED)

    # --- B: fold change against parameter count ------------------------------------------
    b.axhline(1, color=MUTED, lw=0.8, ls="--", zorder=1)
    esm2 = [r for r in rows if r["fam"] == "ESM-2"]
    b.plot([r["params"] for r in esm2], [r["ratio"] for r in esm2],
           color=EMBEDDING_FAMILY_COLOR_MAP["ESM-2"], lw=1.0, alpha=0.6, zorder=2)
    for r in rows:
        c, m = EMBEDDING_FAMILY_COLOR_MAP[r["fam"]], MARKER[r["fam"]]
        b.scatter(r["params"], r["ratio"], s=40, marker=m, facecolor=c, edgecolor=INK, linewidth=0.6, zorder=3)
    lab = {"ankh_base": "Ankh-base", "ankh_large": "Ankh-large", "prott5": "ProtT5", "esm1b": "ESM-1b",
           "esmc_300m": "ESM-C 300M", "esmc_600m": "ESM-C 600M"}
    off = {"ankh_base": (-7, 0), "ankh_large": (0, -11), "prott5": (0, 9), "esm1b": (7, 0),
           "esmc_300m": (-7, 0), "esmc_600m": (7, 0)}
    for r in rows:
        if r["key"] in lab:
            dx, dy = off[r["key"]]
            b.annotate(lab[r["key"]], (r["params"], r["ratio"]), xytext=(dx, dy), textcoords="offset points",
                       fontsize=7, color=INK, ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                       va="center")
    b.text(4e7, 0.43, "ESM-2, 8M → 3B:\n375× the parameters", fontsize=7, color=INK, ha="center", va="center")
    b.set_xscale("log")
    b.set_yscale("log", base=2)
    b.set_ylim(0.2, 6.5)
    b.set_yticks([0.25, 0.5, 1, 2, 4])
    b.set_yticklabels(["¼×", "½×", "1×", "2×", "4×"])
    b.set_xlabel("parameters (log)", color=INK, fontsize=9)
    b.set_ylabel("pretrained ÷ untrained PR", color=INK, fontsize=9)
    style(b)
    b.grid(axis="x", color=GRID, linewidth=0.4)
    b.text(0.02, 0.97, "B", transform=b.transAxes, fontsize=12, fontweight="bold", va="top", color=INK)

    # One legend for both panels: family = colour AND marker.
    handles = [plt.Line2D([], [], ls="", marker=MARKER[f], markersize=6, markerfacecolor=EMBEDDING_FAMILY_COLOR_MAP[f],
                          markeredgecolor=INK, markeredgewidth=0.6, label=f) for f in MARKER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")
    for r in rows:
        print(f"  {r['key']:10} d={r['d']:5}  PR {r['untrained']:6.2f} -> {r['pretrained']:6.2f}  ×{r['ratio']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
