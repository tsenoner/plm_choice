#!/usr/bin/env python3
"""Figure 3, side by side: uniform random pairs vs the filtered (alignable) pairs.

Same reducer (fingerprint_full_reduce.py), same 14 arms, same population rule, same
two statistics. The only thing that changes is which pairs are in the table, so every
delta below is attributable to the pair population.

Wasserstein is reported under min-max (the scaling the figure draws) with the /p99
variant alongside, since the min-max divisor is a single extreme pair and is the
part of the estimator most exposed to a change of population.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


# These paths were absolute to one machine. They are environment variables now, so an
# unset one fails here by name rather than as a FileNotFoundError further down.
import os


def _need(var: str) -> str:
    """The value of `var`, or a message naming what to set."""
    try:
        return os.environ[var]
    except KeyError:
        raise SystemExit(f"set {var} before running this script") from None


ARTEFACTS = _need("PAPER_ARTEFACTS")

B = Path(ARTEFACTS)
NEW = B / "final_figures_2026-09-20/fp/out/fingerprint_full.json"
OLD = B / "figures_2026-09-19/fig03/fingerprint_full.json"
OUT = B / "final_figures_2026-09-20/fig03_random_vs_filtered.csv"


def load(p: Path):
    d = json.loads(p.read_text())
    cols = d["columns"]
    return (
        cols,
        np.asarray(d["correlations"], float),
        np.asarray(d["distances"], float),
        np.asarray(d["distances_p99"], float),
        d.get("metadata", {}),
    )


def main() -> None:
    cn, rn, wn, pn, mn = load(NEW)
    co, ro, wo, po, mo = load(OLD)
    assert cn == co, f"arm order differs:\n{cn}\n{co}"

    print(f"pairs used  random={mn['n_pairs_used']:,}   "
          f"filtered={mo['n_pairs_used']:,}")
    print(f"arms        {len(cn)}\n")

    rows = []
    n = len(cn)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "model_a": cn[i], "model_b": cn[j],
                "rho_random": rn[i, j], "rho_filtered": ro[i, j],
                "rho_delta": rn[i, j] - ro[i, j],
                "w_minmax_random": wn[i, j], "w_minmax_filtered": wo[i, j],
                "w_minmax_delta": wn[i, j] - wo[i, j],
                "w_p99_random": pn[i, j], "w_p99_filtered": po[i, j],
                "w_p99_delta": pn[i, j] - po[i, j],
            })
    df = pl.DataFrame(rows)
    df.write_csv(OUT)

    iu = np.triu_indices(n, 1)
    print("=== Spearman rho (lower triangle of the figure) ===")
    print(f"  mean over the 91 cells : {rn[iu].mean():+.3f} (random)  vs "
          f"{ro[iu].mean():+.3f} (filtered)   delta {rn[iu].mean() - ro[iu].mean():+.3f}")
    print(f"  min                    : {rn[iu].min():+.3f}  vs {ro[iu].min():+.3f}")
    print(f"  max                    : {rn[iu].max():+.3f}  vs {ro[iu].max():+.3f}")
    print(f"  cells < 0              : {int((rn[iu] < 0).sum())}  vs "
          f"{int((ro[iu] < 0).sum())}")
    print("\n=== Wasserstein W1, min-max scaled (upper triangle, x100 as drawn) ===")
    print(f"  mean over the 91 cells : {wn[iu].mean() * 100:.1f}  vs "
          f"{wo[iu].mean() * 100:.1f}   delta "
          f"{(wn[iu].mean() - wo[iu].mean()) * 100:+.1f}")
    print(f"  min                    : {wn[iu].min() * 100:.1f}  vs {wo[iu].min() * 100:.1f}")
    print(f"  max                    : {wn[iu].max() * 100:.1f}  vs {wo[iu].max() * 100:.1f}")

    print("\n=== full matrices, cell by cell: random -> filtered ===")
    w = max(len(c) for c in cn)
    print("\n-- Spearman rho --")
    print(" " * (w + 2) + " ".join(f"{c[:9]:>15}" for c in cn))
    for i in range(n):
        cells = []
        for j in range(n):
            cells.append("  .  " if i == j else f"{rn[i, j]:+.2f}->{ro[i, j]:+.2f}")
        print(f"{cn[i]:<{w + 2}}" + " ".join(f"{c:>15}" for c in cells))

    print("\n-- Wasserstein W1 x100 (min-max) --")
    print(" " * (w + 2) + " ".join(f"{c[:9]:>15}" for c in cn))
    for i in range(n):
        cells = []
        for j in range(n):
            cells.append("  .  " if i == j
                         else f"{wn[i, j] * 100:.0f}->{wo[i, j] * 100:.0f}")
        print(f"{cn[i]:<{w + 2}}" + " ".join(f"{c:>15}" for c in cells))

    print(f"\nwrote {OUT}")

    # The claim the sampler's docstring makes, re-measured on the delivered matrices.
    print("\n=== the between-model agreement claim, re-measured ===")
    print(f"  mean rho, alignable pairs : {ro[iu].mean():.3f}")
    print(f"  mean rho, random pairs    : {rn[iu].mean():.3f}")
    ic = cn.index("clean")
    oth = [j for j in range(n) if j != ic]
    print(f"  CLEAN vs the other 13, random   : "
          f"min {rn[ic, oth].min():+.3f}  max {rn[ic, oth].max():+.3f}")
    print(f"  CLEAN vs the other 13, filtered : "
          f"min {ro[ic, oth].min():+.3f}  max {ro[ic, oth].max():+.3f}")


if __name__ == "__main__":
    main()
