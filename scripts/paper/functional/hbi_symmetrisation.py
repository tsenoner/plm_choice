#!/usr/bin/env python3
"""What symmetrising the MMseqs2 hit table costs the homology baseline.

Eligibility is defined "in either direction", so the baseline is too: a query may transfer
from a protein it aligns to *or* that aligns to it. This measures the size of that choice
on the EC cohort — how many hits are reported both ways, how many queries the symmetrised
table adds, and whether the accuracy moves on the queries both schemes can answer. Writes
``hbi_symmetrisation.json``.

    PYTHONPATH=src python side_measurements/hbi_symmetrisation.py
"""

from __future__ import annotations

import json

# These paths were absolute to one machine. They are environment variables now, so an
# unset one fails here by name rather than as a FileNotFoundError further down.
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def _need(var: str) -> str:
    """The value of `var`, or a message naming what to set."""
    try:
        return os.environ[var]
    except KeyError:
        raise SystemExit(f"set {var} before running this script") from None


ARTEFACTS = _need("PAPER_ARTEFACTS")
REPO = _need("REPO")

sys.path.insert(0, f"{REPO}/src")

from evaluation.analysis_io import load_frozen_ids  # noqa: E402
from evaluation.transfer_report import (  # noqa: E402
    ECScorer,
    build_variants,
    hbi_neighbours,
    load_ec_labels,
)

RESULTS = Path(f"{ARTEFACTS}/functional_2026-09-17")
BACKUP = Path(ARTEFACTS)
FREEZE = BACKUP / "c1_strat_2026-09-15/ec_strat_v2_freeze.json"
LABELS = BACKUP / "c1_strat_2026-09-15/ec_labels_v2.tsv"
M8 = RESULTS / "union_allvsall.m8"


def main() -> int:
    ids = list(load_frozen_ids(FREEZE))
    n = len(ids)
    index = {pid: i for i, pid in enumerate(ids)}
    scorer = ECScorer(load_ec_labels(LABELS, ids))
    variants, hits, _ = build_variants(M8, ids)
    variant = {v.name: v for v in variants}["all"]
    nn_sym, _, _, _ = hbi_neighbours(hits, variant, "evalue")

    # The same argmin restricted to the direction MMseqs2 actually reported.
    table = pl.read_csv(
        M8, separator="\t", has_header=False,
        new_columns=["query", "target", "fident", "evalue", "alnlen", "qcov", "tcov"],
        schema_overrides={"query": pl.Utf8, "target": pl.Utf8,
                          "fident": pl.Float64, "evalue": pl.Float64},
    ).filter(pl.col("query").is_in(ids) & pl.col("target").is_in(ids) & (pl.col("query") != pl.col("target")))
    q = np.array([index[x] for x in table["query"].to_list()])
    t = np.array([index[x] for x in table["target"].to_list()])
    ev = table["evalue"].to_numpy()
    fid = table["fident"].to_numpy()
    order = np.lexsort((t, 1.0 - fid, ev, q))
    q_s, t_s = q[order], t[order]
    starts = np.ones(q_s.size, dtype=bool)
    starts[1:] = q_s[1:] != q_s[:-1]
    nn_dir = np.full(n, -1, dtype=np.int64)
    nn_dir[q_s[starts]] = t_s[starts]

    directed_pairs = {(int(a), int(b)) for a, b in zip(q, t, strict=True)}
    unordered = {(min(p), max(p)) for p in directed_pairs}
    both_ways = sum(1 for a, b in unordered if (a, b) in directed_pairs and (b, a) in directed_pairs)

    answerable_dir = nn_dir >= 0
    answerable_sym = nn_sym >= 0
    both = answerable_dir & answerable_sym
    rows = np.flatnonzero(both)
    exact_dir = scorer.pairs(rows, nn_dir[rows])["exact"]
    exact_sym = scorer.pairs(rows, nn_sym[rows])["exact"]
    all_sym = np.flatnonzero(answerable_sym)
    published = pd.read_csv(RESULTS / "ec_v2_transfer" / "summary.csv")
    cell = published[
        (published["arm"] == "hbi_evalue")
        & (published["variant"] == "all")
        & (published["subset"] == "hbi_answerable")
        & (published["score"] == "exact")
        & (published["distance"] == "euclidean")
    ]

    out = {
        "cohort": "ec_v2",
        "n": n,
        "directed_cohort_hit_rows": int(q.size),
        "unordered_cohort_pairs": len(unordered),
        "unordered_pairs_reported_both_ways": both_ways,
        "share_of_unordered_pairs_bidirectional": both_ways / len(unordered),
        "share_of_directed_rows_with_their_reverse": 2 * both_ways / len(directed_pairs),
        "answerable_query_direction_only": int(answerable_dir.sum()),
        "answerable_symmetrised": int(answerable_sym.sum()),
        "queries_whose_neighbour_changes": int((nn_dir[rows] != nn_sym[rows]).sum()),
        "share_whose_neighbour_changes": float((nn_dir[rows] != nn_sym[rows]).mean()),
        "exact_on_both_answerable_query_direction": float(exact_dir.mean()),
        "exact_on_both_answerable_symmetrised": float(exact_sym.mean()),
        "n_both_answerable": int(rows.size),
        "exact_published_symmetrised_all_answerable": float(
            scorer.pairs(all_sym, nn_sym[all_sym])["exact"].mean()
        ),
        "n_all_symmetrised_answerable": int(all_sym.size),
        "published_summary_mean": float(cell["mean"].iloc[0]),
        "published_summary_n": int(cell["n_queries"].iloc[0]),
    }
    path = Path(__file__).with_name("hbi_symmetrisation.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
