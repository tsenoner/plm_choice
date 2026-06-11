"""EC arm: per-pLM correlation of embedding distance vs EC functional distance.

The only module that turns the embeddings dict + EC labels into the two square
matrices the embedding-agnostic stats core consumes (the D12 seam). Writes a per-pair
parquet + a manifest sidecar via the same 3-exit-code CLI contract as recall_fp_report
/ snn_report (the 3rd clone by design — D6; feeds the later analysis_cli extraction).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from evaluation.analysis_io import json_safe, load_embeddings_h5, load_frozen_ids, pairwise_distance_long
from evaluation.ec_hierarchy import ec_distance_matrix_set
from evaluation.label_adapters import parse_ec
from evaluation.stats import (
    correlation_permutation_null,
    correlation_vertex_bca_ci,
    kendall_tau_b,
    spearman_rho,
)

# ── per-pair parquet contract (the EC analogue of SNN_PARQUET_GUARDS) ──────────
# A synthetic single-column ``pair_key`` is the unique key (the barrier's
# unique_columns guard is single-column; encoding (a,b) into one column sidesteps
# any 2-column-key ambiguity while keeping a/b for downstream use).
EC_PER_PAIR_COLUMNS: tuple[str, ...] = ("pair_key", "a", "b", "dist", "ec_dist")
EC_PARQUET_GUARDS: dict[str, tuple[str, ...]] = {
    "required_columns": EC_PER_PAIR_COLUMNS,
    "unique_columns": ("pair_key",),
    "non_null_columns": ("pair_key", "a", "b"),
    "finite_columns": ("dist", "ec_dist"),
}


def ec_dist_histogram(pairs: pd.DataFrame) -> dict[int, int]:
    """Integer-binned counts of ``ec_dist`` (0..4). Fractional values (mean/hausdorff
    aggregations) are floored into their bin."""
    binned = np.floor(pairs["ec_dist"].to_numpy()).astype(int)
    return {b: int(np.count_nonzero(binned == b)) for b in range(0, 5)}


def stratify_by_class(pairs: pd.DataFrame, ec_class: dict[str, str]) -> dict:
    """Within-class vs across-class correlation (class = first EC field).

    A pair is within-class iff both proteins' class labels are known and equal.
    Returns counts + tau_b/rho for each stratum (NaN where a stratum is too small).
    """
    ca = pairs["a"].map(ec_class)
    cb = pairs["b"].map(ec_class)
    known = ca.notna() & cb.notna()
    within = known & (ca == cb)
    across = known & (ca != cb)
    return {
        "n_within": int(within.sum()),
        "n_across": int(across.sum()),
        "tau_b_within": kendall_tau_b(pairs.loc[within, "dist"], pairs.loc[within, "ec_dist"]),
        "tau_b_across": kendall_tau_b(pairs.loc[across, "dist"], pairs.loc[across, "ec_dist"]),
        "rho_within": spearman_rho(pairs.loc[within, "dist"], pairs.loc[within, "ec_dist"]),
        "rho_across": spearman_rho(pairs.loc[across, "dist"], pairs.loc[across, "ec_dist"]),
    }


def stratify_by_superfamily(pairs: pd.DataFrame, superfamily: dict) -> dict:
    """Within- vs across-CATH-superfamily correlation + non-homologous restriction.

    ``superfamily`` maps protein_id -> frozenset of superfamily codes (the multi-domain
    set). A pair is *homologous* (within-superfamily) iff the two sets intersect; the
    non-homologous restriction keeps only the disjoint pairs — isolating function from
    homology (the 92%-confound control). Returns counts + tau_b/rho per stratum.
    """
    def _intersects(a, b):
        sa, sb = superfamily.get(a), superfamily.get(b)
        if not sa or not sb:
            return None  # unknown -> excluded from both strata
        return len(sa & sb) > 0

    rel = [_intersects(a, b) for a, b in zip(pairs["a"], pairs["b"])]
    rel = pd.Series(rel, index=pairs.index)
    within = rel == True  # noqa: E712 (explicit True, not NaN/None)
    across = rel == False  # noqa: E712
    return {
        "n_within_superfamily": int(within.sum()),
        "n_across_superfamily": int(across.sum()),
        "n_nonhomologous": int(across.sum()),
        "tau_b_within_superfamily": kendall_tau_b(pairs.loc[within, "dist"], pairs.loc[within, "ec_dist"]),
        "tau_b_nonhomologous": kendall_tau_b(pairs.loc[across, "dist"], pairs.loc[across, "ec_dist"]),
        "rho_within_superfamily": spearman_rho(pairs.loc[within, "dist"], pairs.loc[within, "ec_dist"]),
        "rho_nonhomologous": spearman_rho(pairs.loc[across, "dist"], pairs.loc[across, "ec_dist"]),
    }
