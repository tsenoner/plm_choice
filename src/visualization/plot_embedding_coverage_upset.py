#!/usr/bin/env python3
"""UpSet figure: which proteins each pLM embedding set actually covers.

**Why this figure exists.** The embedding arms do not cover the same proteins, and
``src/shared/datasets.py:99-105`` drops a pair when *either* protein is absent from
that arm's HDF5. The loss is therefore **quadratic** in protein coverage: an arm at
80.3% of the cohort is scored on 64.6% of the pairs its competitors get. Measured on
the published grid, ``esm2_3b`` was trained and evaluated on 558,947 test pairs where
ten other arms got 872,572 -- yet it is reported at rank #10 in a cross-pLM ranking.
A ranking is only a ranking if every row was scored on the same data, so this figure
is the honest disclosure of the input to that claim.

The gaps have three distinct causes, which the figure separates:

* an **interrupted run** -- ``esm2_3b`` (fixable, and being fixed);
* a **default length filter** -- ``--max_seq_len 2000`` skips longer sequences;
* an **architectural ceiling** -- ESM-1b's learned positional embeddings cap at 1022
  tokens (``embedding_generation.py:88``), so ~2.8% of the cohort can *never* be
  embedded by it. ``clean`` inherits exactly this set because CLEAN is built on ESM-1b.

That last one is why topping every arm up cannot produce a uniform test set, and why
restricting all arms to the **intersection** is the only construction that gives every
model the same evaluation data.

Why UpSet rather than a Venn diagram: Venn is unreadable past 3-4 sets and impossible
at 15. UpSet shows the same information as a matrix of intersections and stays legible.

Usage
-----
    # from a pre-computed key-set summary (cheap, preferred)
    plm figures coverage-upset --keysets-json keysets.json --out out/fig_coverage.png

    # or scan the HDF5 files directly, caching a .keys sidecar beside each one
    python -m visualization.plot_embedding_coverage_upset --h5-dir <dir> --out fig.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:  # keep the set maths importable without the repo's plotting stack
    from visualization.plm_constants import (
        EMBEDDING_DISPLAY_NAMES,
        EMBEDDING_FAMILY_COLOR_MAP,
        EMBEDDING_FAMILY_MAP,
    )
except ImportError:  # pragma: no cover - exercised only outside the package
    EMBEDDING_DISPLAY_NAMES, EMBEDDING_FAMILY_COLOR_MAP, EMBEDDING_FAMILY_MAP = {}, {}, {}


# --------------------------------------------------------------------------- #
# Set maths (pure -- pinned by tests/test_coverage_upset.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PatternRow:
    """One column of the UpSet matrix: a distinct membership combination."""

    members: frozenset[str]
    count: int
    complete: bool
    missing: tuple[str, ...]

    @property
    def degree(self) -> int:
        return len(self.members)


def membership_patterns(sets: Mapping[str, Iterable[str]]) -> dict[frozenset[str], int]:
    """Map every distinct membership combination to how many elements have it.

    Iterates the union once and asks each set for membership, which is O(|universe| x
    |sets|) hash lookups rather than materialising 2^n intersections -- at 15 sets the
    latter would be 32,768 combinations, almost all empty.
    """
    materialised = {name: set(values) for name, values in sets.items()}
    universe: set = set()
    for values in materialised.values():
        universe |= values

    counts: Counter = Counter()
    for element in universe:
        combo = frozenset(name for name, values in materialised.items() if element in values)
        if combo:  # an element in no set is not in the union; guard anyway
            counts[combo] += 1
    return dict(counts)


def set_sizes_from_patterns(
    patterns: Mapping[frozenset[str], int], order: Sequence[str]
) -> dict[str, int]:
    """Recover each set's total from the pattern histogram.

    Used as an internal consistency check: these totals must equal the raw key counts,
    otherwise the pattern decomposition lost or double-counted elements.
    """
    sizes = {name: 0 for name in order}
    for combo, n in patterns.items():
        for name in combo:
            if name in sizes:
                sizes[name] += n
    return sizes


def pattern_rows(
    patterns: Mapping[frozenset[str], int],
    order: Sequence[str],
    max_rows: int | None = None,
    return_dropped: bool = False,
):
    """Sort patterns by size (descending) and annotate each one.

    ``max_rows`` truncates for legibility. Truncation is **reported**, never silent:
    pass ``return_dropped=True`` to get the number of patterns left out, and put it in
    the caption. A figure that quietly drops intersections misstates the coverage.
    """
    known = set(order)
    for combo in patterns:
        unknown = combo - known
        if unknown:
            raise ValueError(f"pattern references set(s) missing from `order`: {sorted(unknown)}")

    full = frozenset(order)
    rows = [
        PatternRow(
            members=combo,
            count=n,
            complete=(combo == full),
            missing=tuple(name for name in order if name not in combo),
        )
        for combo, n in patterns.items()
    ]
    # size first; ties broken by higher degree then by name so output is deterministic
    rows.sort(key=lambda r: (-r.count, -r.degree, sorted(r.members)))

    dropped = 0
    if max_rows is not None and len(rows) > max_rows:
        dropped = len(rows) - max_rows
        rows = rows[:max_rows]
    return (rows, dropped) if return_dropped else rows


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_keysets_json(path: Path) -> tuple[list[str], dict[frozenset[str], int]]:
    """Read the compact summary produced by the cluster-side key scan."""
    blob = json.loads(Path(path).read_text())
    models: list[str] = list(blob["models"])
    patterns: dict[frozenset[str], int] = {}
    for mask_str, n in blob["patterns"].items():
        mask = int(mask_str)
        combo = frozenset(m for i, m in enumerate(models) if mask & (1 << i))
        if combo:
            patterns[combo] = patterns.get(combo, 0) + n
    return models, patterns


def load_h5_keysets(h5_dir: Path, use_cache: bool = True) -> dict[str, set]:
    """Read dataset names from every ``.h5`` in a directory, with a sidecar cache.

    Enumerating ~542k names out of one HDF5 group on a network filesystem is dominated
    by scattered metadata reads, so the answer is cached beside the file as
    ``<stem>.keys.txt``. The cache is invalidated on (size, mtime), because a stale key
    list would silently misreport coverage -- exactly the failure this figure documents.
    """
    import h5py

    sets: dict[str, set] = {}
    for h5_path in sorted(Path(h5_dir).glob("*.h5")):
        stat = h5_path.stat()
        sidecar = h5_path.with_suffix(".keys.txt")
        stamp = f"# {stat.st_size} {int(stat.st_mtime)}"
        if use_cache and sidecar.exists():
            lines = sidecar.read_text().splitlines()
            if lines and lines[0] == stamp:
                sets[h5_path.stem] = set(lines[1:])
                continue
        with h5py.File(h5_path, "r") as handle:
            keys = list(handle.keys())
        sets[h5_path.stem] = set(keys)
        try:
            sidecar.write_text("\n".join([stamp, *keys]))
        except OSError:
            pass  # read-only location (e.g. the Zenodo deposit) -- caching is optional
    return sets


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

#: Committed coverage freeze -- the default source, so redrawing needs no cluster access.
DEFAULT_COVERAGE_FREEZE = (
    Path(__file__).resolve().parents[2] / "freeze" / "embedding_key_coverage.json"
)

_COMPLETE = "#2b5d8a"  # all arms present -- the headline bar
_DEFICIENT = "#c1666b"  # at least one arm missing -- the defect this figure discloses
_DOT_ON = "#2f3337"
_DOT_OFF = "#dcdfe3"


def _label(model: str) -> str:
    return EMBEDDING_DISPLAY_NAMES.get(model, model).replace("\n", " ")


def _family_color(model: str) -> str:
    return EMBEDDING_FAMILY_COLOR_MAP.get(EMBEDDING_FAMILY_MAP.get(model, ""), "#8a8f94")


def _fmt(n: int) -> str:
    return f"{n:,}"


def plot_upset(
    models: Sequence[str],
    patterns: Mapping[frozenset[str], int],
    out_path: Path,
    max_rows: int = 20,
    log_scale: bool = True,
    title: str | None = None,
    dpi: int = 300,
) -> Path:
    """Draw the UpSet figure and write it to ``out_path`` (plus a .pdf sibling)."""
    rows, dropped = pattern_rows(patterns, models, max_rows=max_rows, return_dropped=True)
    sizes = set_sizes_from_patterns(patterns, models)
    # Draw sets largest-first so the deficient arms sit together at the bottom.
    order = sorted(models, key=lambda m: (-sizes[m], m))

    n_cols, n_sets = len(rows), len(order)
    # wspace must leave room for the set names, which are drawn on the matrix's left
    # spine and sit in the gap between the two lower panels; at a tight wspace the
    # totals panel clips them to their last few characters.
    fig = plt.figure(figsize=(max(7.5, 0.46 * n_cols + 5.0), 0.26 * n_sets + 3.2))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.35, max(2.6, 0.40 * n_cols)],
        height_ratios=[1.7, 0.26 * n_sets + 0.4],
        wspace=0.30,
        hspace=0.07,
    )
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_matrix = fig.add_subplot(gs[1, 1], sharex=ax_bar)
    ax_sets = fig.add_subplot(gs[1, 0], sharey=ax_matrix)

    x = range(n_cols)

    # --- intersection sizes (magnitude -> one hue; the defect gets the second) ----
    colors = [_COMPLETE if r.complete else _DEFICIENT for r in rows]
    ax_bar.bar(x, [r.count for r in rows], color=colors, width=0.62, zorder=3)
    if log_scale:
        ax_bar.set_yscale("log")
    ax_bar.set_ylabel("Proteins in intersection" + (" (log)" if log_scale else ""))
    # Every bar carries its exact value, so a log axis never hides the magnitude.
    for xi, row in zip(x, rows, strict=True):
        ax_bar.annotate(
            _fmt(row.count),
            (xi, row.count),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            color="#2f3337",
        )
    ax_bar.margins(y=0.28)
    ax_bar.grid(axis="y", color="#eceef0", zorder=0)
    ax_bar.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax_bar.spines[side].set_visible(False)
    ax_bar.tick_params(axis="x", labelbottom=False, length=0)

    # --- membership matrix -------------------------------------------------------
    ypos = {name: i for i, name in enumerate(order)}
    for xi, row in enumerate(rows):
        ax_matrix.scatter(
            [xi] * n_sets,
            range(n_sets),
            s=34,
            color=[_DOT_ON if m in row.members else _DOT_OFF for m in order],
            zorder=3,
        )
        present = sorted(ypos[m] for m in row.members)
        if len(present) > 1:  # the spine that makes a combination readable as one unit
            ax_matrix.plot(
                [xi, xi],
                [present[0], present[-1]],
                color=_DOT_ON,
                lw=1.4,
                zorder=2,
                solid_capstyle="round",
            )
    for i in range(n_sets):  # zebra banding aids row tracking across many columns
        if i % 2 == 0:
            ax_matrix.axhspan(i - 0.5, i + 0.5, color="#f7f8f9", zorder=0)
    ax_matrix.set_yticks(range(n_sets))
    ax_matrix.set_yticklabels([_label(m) for m in order], fontsize=8)
    ax_matrix.set_ylim(n_sets - 0.5, -0.5)
    ax_matrix.set_xticks([])
    ax_matrix.set_xlabel("Membership combination")
    for side in ("top", "right", "bottom", "left"):
        ax_matrix.spines[side].set_visible(False)
    ax_matrix.tick_params(length=0)

    # --- per-model totals (identity -> the paper's family colours) ----------------
    ax_sets.barh(
        range(n_sets),
        [sizes[m] for m in order],
        color=[_family_color(m) for m in order],
        height=0.62,
        zorder=3,
    )
    universe = sum(patterns.values())
    # The x-axis is INVERTED (bars grow leftwards from 0), so "just inside the bar's
    # tip" is a POSITIVE pixel offset with ha="left". The intuitive (-4, "right")
    # places the label off the panel entirely.
    for i, m in enumerate(order):
        ax_sets.annotate(
            _fmt(sizes[m]),
            (sizes[m], i),
            textcoords="offset points",
            xytext=(5, 0),
            ha="left",
            va="center",
            fontsize=7,
            color="#ffffff",
            fontweight="bold",
        )
    ax_sets.set_xlim(universe * 1.02, 0)
    ax_sets.set_xlabel("Proteins in set")
    ax_sets.grid(axis="x", color="#eceef0", zorder=0)
    ax_sets.set_axisbelow(True)
    ax_sets.tick_params(axis="y", labelleft=False, length=0)
    for side in ("top", "right", "left"):
        ax_sets.spines[side].set_visible(False)

    # Identity is never colour-alone: the legend names both categories.
    ax_bar.legend(
        handles=[
            Line2D([], [], marker="s", ls="", ms=8, color=_COMPLETE, label="present in all arms"),
            Line2D([], [], marker="s", ls="", ms=8, color=_DEFICIENT, label="missing from >=1 arm"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )

    if title:
        fig.suptitle(title, fontsize=11, y=0.98)
    if dropped:
        fig.text(
            0.01,
            0.01,
            f"{dropped} further combination(s) omitted for legibility "
            f"({_fmt(sum(r.count for r in pattern_rows(patterns, models)[max_rows:]))} proteins).",
            fontsize=7,
            color="#6b7075",
            ha="left",
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")  # vector for print
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="UpSet figure of protein coverage across pLM embedding sets.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--keysets-json", type=Path, help="Pre-computed key-set summary.")
    src.add_argument("--h5-dir", type=Path, help="Directory of .h5 embedding files.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument(
        "--max-rows", type=int, default=20, help="Max intersection columns to draw (default: 20)."
    )
    parser.add_argument(
        "--linear", action="store_true", help="Linear intersection axis instead of log."
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore/refresh the .keys.txt sidecars when scanning HDF5.",
    )
    args = parser.parse_args(argv)

    if args.h5_dir:
        sets = load_h5_keysets(args.h5_dir, use_cache=not args.no_cache)
        models = sorted(sets)
        patterns = membership_patterns(sets)
    else:
        # Default to the committed freeze so the figure can be redrawn offline.
        # Enumerating 542k HDF5 keys costs ~3 min per file over GPFS and the .h5
        # files are not local, so without this cache every restyle of the plot
        # would need cluster access.
        path = args.keysets_json or DEFAULT_COVERAGE_FREEZE
        if not path.exists():
            print(f"No coverage data at {path}. Pass --h5-dir to build it.", file=sys.stderr)
            return 1
        models, patterns = load_keysets_json(path)
        print(f"source: {path}")

    if not patterns:
        print("No membership patterns found - is the input empty?", file=sys.stderr)
        return 1

    sizes = set_sizes_from_patterns(patterns, models)
    universe = sum(patterns.values())
    complete = patterns.get(frozenset(models), 0)
    print(
        f"sets: {len(models)}   universe: {universe:,}   "
        f"present in all: {complete:,} ({complete / universe * 100:.2f}%)"
    )
    for m in sorted(models, key=lambda k: sizes[k]):
        print(f"  {m:14s} {sizes[m]:>9,}  ({sizes[m] / universe * 100:6.2f}%)")

    out = plot_upset(
        models,
        patterns,
        args.out,
        max_rows=args.max_rows,
        log_scale=not args.linear,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
