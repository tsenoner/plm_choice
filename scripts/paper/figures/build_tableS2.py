#!/usr/bin/env python3
"""Rebuild Table S2 (tbls:performance) VALUES from the 2026-09-18 probe grid.

Reads probe_metrics.csv, keeps dataset=sprot_pre2024_e1_sub10, native Euclidean +
FNN, metric = Pearson R2.  Emits the same pandoc grid table the manuscript already
carries, so the only thing that moves is the numbers and the best/second marking.

Best/second are per (target, read-out column), with the random Gaussian control
excluded from the ranking (it is a control, not a candidate).  A cell with no
trained probe stays "--".

Also parses the CURRENT table out of 90.supplementary.md (read-only) and reports a
cell-by-cell diff, so the change count quoted to the user is measured, not guessed.
"""

from __future__ import annotations

# These paths were absolute to one machine. They are environment variables now, so an
# unset one fails here by name rather than as a FileNotFoundError further down.
import os
import re
from pathlib import Path

import pandas as pd


def _need(var: str) -> str:
    """The value of `var`, or a message naming what to set."""
    try:
        return os.environ[var]
    except KeyError:
        raise SystemExit(f"set {var} before running this script") from None


ARTEFACTS = _need("PAPER_ARTEFACTS")
MANUSCRIPT = _need("MANUSCRIPT_DIR")

METRICS = Path(
    ARTEFACTS,
    "probe_e1_2026-09-18/probe_metrics.csv"
)
SUPP = Path(
    MANUSCRIPT,
    "sections/90.supplementary.md"
)
OUT = Path(
    ARTEFACTS,
    "final_figures_2026-09-21/tableS2.md"
)
# The previously GENERATED table. The manuscript diff below reads 90.supplementary.md,
# which is being hand-edited right now, so it is a moving target; this file is the
# stable reference for "what did the metrics refresh actually move".
PREV = Path(
    ARTEFACTS,
    "final_figures_2026-09-20/tableS2.md"
)

DATASET = "sprot_pre2024_e1_sub10"

# Display label -> arm key in probe_metrics.csv. Order is the table's row order.
ROWS: list[tuple[str, str]] = [
    ("Ankh Base", "ankh_base"),
    ("Ankh Large", "ankh_large"),
    ("CLEAN", "clean"),
    ("ESM-1b", "esm1b"),
    ("ESM2-8M", "esm2_8m"),
    ("ESM2-35M", "esm2_35m"),
    ("ESM2-150M", "esm2_150m"),
    ("ESM2-650M", "esm2_650m"),
    ("ESM2-3B", "esm2_3b"),
    ("ESM3", "esm3_open"),
    ("ESM C-300M", "esmc_300m"),
    ("ESM C-600M", "esmc_600m"),
    ("ProtT5", "prott5"),
    ("ProtTucker", "prottucker"),
    ("Random", "random_1024"),
]
CONTROL = "random_1024"  # excluded from best/second

BLOCKS: list[tuple[str, str]] = [
    ("PIDE", "fident"),
    ("TM-score", "alntmscore"),
    ("HFSP", "hfsp"),
]
COLS: list[tuple[str, str]] = [("Euclidean", "euclidean"), ("FNN", "fnn")]

DASH = "--"
# Full column widths (the span between two "+" in a rule), matched to the table
# already in 90.supplementary.md so the rebuilt block is a drop-in replacement.
# Content width is W - 2: one padding space on each side.
W_PARAM, W_EMB, W_VAL = 11, 12, 24


def load() -> dict[tuple[str, str, str], float]:
    df = pd.read_csv(METRICS)
    df = df[df["dataset"] == DATASET]
    out: dict[tuple[str, str, str], float] = {}
    for _, r in df.iterrows():
        v = r["Pearson_r2"]
        if pd.notna(v):
            out[(r["target"], r["model_type"], r["arm"])] = float(v)
    return out


def mark(txt: str, kind: str) -> str:
    if kind == "best":
        return f"**[{txt}]{{.underline}}**"
    if kind == "second":
        return f"*[{txt}]{{.underline}}*"
    return txt


def build() -> tuple[str, dict[tuple[str, str, str], str]]:
    vals = load()
    cells: dict[tuple[str, str, str], str] = {}

    for _, target in BLOCKS:
        for _, mt in COLS:
            ranked = sorted(
                (
                    (vals[(target, mt, arm)], arm)
                    for _lbl, arm in ROWS
                    if arm != CONTROL and (target, mt, arm) in vals
                ),
                reverse=True,
            )
            best = ranked[0][1] if len(ranked) > 0 else None
            second = ranked[1][1] if len(ranked) > 1 else None
            for _lbl, arm in ROWS:
                v = vals.get((target, mt, arm))
                if v is None:
                    cells[(target, mt, arm)] = DASH
                    continue
                txt = f"{v:.2f}"
                kind = "best" if arm == best else "second" if arm == second else ""
                cells[(target, mt, arm)] = mark(txt, kind)
    return render(cells), cells


def rule(ch: str, first: bool, param_col: bool = True) -> str:
    """Grid-table rule. `first` marks the header separator (=== and alignment :)."""
    if first:
        return f"+:{'=' * (W_PARAM - 2)}:+{'=' * W_EMB}+{'=' * W_VAL}+{'=' * W_VAL}+"
    if param_col:
        return f"+{ch * W_PARAM}+{ch * W_EMB}+{ch * W_VAL}+{ch * W_VAL}+"
    return f"|{' ' * W_PARAM}+{ch * W_EMB}+{ch * W_VAL}+{ch * W_VAL}+"


def row(param: str, emb: str, a: str, b: str) -> str:
    return (
        f"| {param:<{W_PARAM - 2}} "
        f"| {emb:<{W_EMB - 2}} "
        f"| {a:<{W_VAL - 2}} "
        f"| {b:<{W_VAL - 2}} |"
    )


def render(cells: dict[tuple[str, str, str], str]) -> str:
    lines = [rule("-", False)]
    lines.append(row("Parameter", "Embedding", "Euclidean", "FNN"))
    lines.append(rule("=", True))
    for bi, (blabel, target) in enumerate(BLOCKS):
        if bi:
            lines.append(rule("-", False))
        for ri, (lbl, arm) in enumerate(ROWS):
            lines.append(
                row(
                    blabel if ri == 0 else "",
                    lbl,
                    cells[(target, "euclidean", arm)],
                    cells[(target, "fnn", arm)],
                )
            )
            if ri < len(ROWS) - 1:
                lines.append(rule("-", False, param_col=False))
    lines.append(rule("-", False))
    return "\n".join(lines)


# --- diff against what the manuscript currently prints -----------------------
def parse_grid(body: str) -> dict[tuple[str, str, str], str]:
    """Pull Table S2 body cells out of a pandoc grid table."""
    label2arm = dict(ROWS)
    block_of: dict[str, str] = {}
    cur_block: str | None = None
    found: dict[tuple[str, str, str], str] = {}
    seen: set[str] = set()

    for line in body.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != 4:
            continue
        param, emb, a, b = parts
        if emb == "Embedding":
            continue
        if param:
            cur_block = param
            seen.add(param)
        if emb not in label2arm or cur_block is None:
            continue
        target = dict(BLOCKS)[cur_block]
        block_of[cur_block] = target
        arm = label2arm[emb]
        found[(target, "euclidean", arm)] = a
        found[(target, "fnn", arm)] = b
    return found


def parse_current() -> dict[tuple[str, str, str], str]:
    """Table S2 as the supplementary markdown currently prints it."""
    text = SUPP.read_text()
    start = text.index("[]{#tbls:performance}")
    end = text.index("[]{#tbls:dip}")
    return parse_grid(text[start:end])


def parse_prev() -> dict[tuple[str, str, str], str]:
    """Table S2 as the previous run generated it."""
    return parse_grid(PREV.read_text())


def strip_marks(s: str) -> str:
    return re.sub(r"[*\[\]]|\{\.underline\}", "", s).strip()


def diff(new: dict, old: dict, label: str) -> None:
    missing = [k for k in new if k not in old]
    if missing:
        print(f"!! could not parse {label} cells: {missing[:5]}")
        return

    changed_render, changed_value, dash_filled = [], [], []
    for k in new:
        n, o = new[k], old[k]
        if n != o:
            changed_render.append((k, o, n))
            if strip_marks(n) != strip_marks(o):
                changed_value.append((k, o, n))
            if o == DASH:
                dash_filled.append((k, n))

    print(f"\n=== DIFF vs {label} ===")
    print(f"cells whose printed content changed : {len(changed_render)}")
    print(f"  of which the NUMBER changed       : {len(changed_value)}")
    print(f"  of which were '--' and now have a value : {len(dash_filled)}")
    print(f"  marking-only changes (value identical)  : "
          f"{len(changed_render) - len(changed_value)}")
    print("--- every changed cell ---")
    for (target, mt, arm), o, n in sorted(changed_render):
        print(f"  {target:<11} {mt:<10} {arm:<12} {o:<24} -> {n}")


def main() -> None:
    table, new = build()
    OUT.write_text(table + "\n")

    dashes_left = sorted(k for k in new if new[k] == DASH)
    print(f"metrics file : {METRICS}")
    print(f"cells total                : {len(new)}")
    print(f"dash cells remaining       : {len(dashes_left)} -> {dashes_left}")

    diff(new, parse_prev(), "PREVIOUS GENERATED TABLE (2026-09-20)")
    diff(new, parse_current(), "MANUSCRIPT 90.supplementary.md (live, being edited)")

    # best/second marking per (target, read-out), read back off the rendered cells
    print("\n=== best / second per target and read-out ===")
    label_of = {arm: lbl for lbl, arm in ROWS}
    for blabel, target in BLOCKS:
        for clabel, mt in COLS:
            best = [label_of[a] for _l, a in ROWS
                    if new[(target, mt, a)].startswith("**")]
            second = [label_of[a] for _l, a in ROWS
                      if new[(target, mt, a)].startswith("*")
                      and not new[(target, mt, a)].startswith("**")]
            bv = [strip_marks(new[(target, mt, a)]) for _l, a in ROWS
                  if new[(target, mt, a)].startswith("**")]
            sv = [strip_marks(new[(target, mt, a)]) for _l, a in ROWS
                  if new[(target, mt, a)].startswith("*")
                  and not new[(target, mt, a)].startswith("**")]
            print(f"  {blabel:<9} {clabel:<10} best={best[0]:<11}({bv[0]})  "
                  f"second={second[0]:<11}({sv[0]})")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
