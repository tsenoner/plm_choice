"""EC-number hierarchical distance + correlation with embedding distance.

EC numbers are 4-field hierarchical: ``class.subclass.subsubclass.serial``
(e.g. ``3.4.21.62``). Two EC numbers agree at level ``L`` if their first ``L``
fields match. The distance is ``4 - L`` where ``L`` is the deepest level of
agreement:

    0  -> all four fields match (same EC)
    1  -> first three fields match
    2  -> first two fields match
    3  -> only the first (class) matches
    4  -> classes differ (no overlap)

Wildcards (``-`` or missing trailing fields) are treated as agreement:
``1.1.1.-`` vs ``1.1.1.5`` returns 0 (the wildcard "matches" 5) — the convention
used in BRENDA enzyme-family analyses.

Note: ``ec_distance`` raises on malformed input by design; skip-and-count of
malformed EC strings belongs in the calling adapter, not here.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr


def _split_ec(ec: str) -> list[str]:
    if not isinstance(ec, str):
        raise TypeError(f"EC must be a string, got {type(ec)!r}")
    fields = ec.strip().split(".")
    if len(fields) > 4:
        raise ValueError(f"EC {ec!r} has more than 4 fields")
    # Pad to 4 with a wildcard.
    while len(fields) < 4:
        fields.append("-")
    return fields


def ec_distance(ec_a: str, ec_b: str) -> int:
    """Integer hierarchical EC distance (0=identical, 4=no overlap).

    Wildcards (``-`` or empty / missing trailing fields) match anything.
    """
    fa = _split_ec(ec_a)
    fb = _split_ec(ec_b)

    matched_depth = 0
    for x, y in zip(fa, fb):
        if x == "-" or y == "-" or x == "" or y == "":
            matched_depth += 1
            continue
        if x == y:
            matched_depth += 1
        else:
            break
    return 4 - matched_depth


def ec_distance_matrix(ec_labels: pd.DataFrame) -> pd.DataFrame:
    """Pairwise EC distance matrix in long form.

    Args:
        ec_labels: DataFrame with columns ``protein_id`` and ``ec_number``.

    Returns:
        Long-form ``[a, b, ec_dist]`` with one row per unordered pair
        (``a < b`` lexicographically).
    """
    if not {"protein_id", "ec_number"}.issubset(ec_labels.columns):
        raise KeyError("ec_labels must have columns protein_id and ec_number")

    ids = ec_labels["protein_id"].tolist()
    ecs = ec_labels["ec_number"].tolist()
    records: list[tuple[str, str, int]] = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
            ec_i = ecs[i] if ids[i] < ids[j] else ecs[j]
            ec_j = ecs[j] if ids[i] < ids[j] else ecs[i]
            records.append((a, b, ec_distance(ec_i, ec_j)))
    return pd.DataFrame(records, columns=["a", "b", "ec_dist"])


def ec_distance_set(
    ec_set_a: "frozenset[str]", ec_set_b: "frozenset[str]", *, agg: str = "min"
) -> float:
    """Set-valued EC distance for multifunctional enzymes.

    Aggregates the cross-product ``{ec_distance(a, b) : a in A, b in B}`` by ``agg``:

    * ``min`` (default) — "share ANY function": 0 if the sets share an EC. The
      analogue of the CATH multi-domain set-intersection rule.
    * ``mean`` — average hierarchical distance over the cross-product.
    * ``hausdorff`` — ``max`` of the two directed set distances (each directed
      distance is the max over one set of the min to the other set).

    ``agg`` is a *recorded* report parameter (it changes the per-pLM number and the
    ranking — see the design spec D7); the report writes it into the manifest and
    reports a sensitivity over all three. Raises ``ValueError`` on an unknown ``agg``
    or an empty set.
    """
    if not ec_set_a or not ec_set_b:
        raise ValueError("ec_distance_set: empty EC set has no defined distance")
    if agg == "min":
        return float(min(ec_distance(a, b) for a in ec_set_a for b in ec_set_b))
    if agg == "mean":
        vals = [ec_distance(a, b) for a in ec_set_a for b in ec_set_b]
        return float(sum(vals) / len(vals))
    if agg == "hausdorff":
        d_ab = max(min(ec_distance(a, b) for b in ec_set_b) for a in ec_set_a)
        d_ba = max(min(ec_distance(a, b) for a in ec_set_a) for b in ec_set_b)
        return float(max(d_ab, d_ba))
    raise ValueError(f"unknown agg={agg!r}; choose min/mean/hausdorff")


def correlate_embedding_distance_with_ec(
    embedding_distances: pd.DataFrame,
    ec_distances: pd.DataFrame,
) -> dict:
    """Spearman rho of embedding distance vs EC distance (ordinal 0-4).

    Args:
        embedding_distances: DataFrame with columns ``[a, b, dist]``.
        ec_distances: DataFrame with columns ``[a, b, ec_dist]``.

    The two tables are inner-joined on ``(a, b)``. EC distance is treated as an
    ordinal variable for the Spearman correlation (higher EC distance -> expect
    higher embedding distance, so positive rho is the prediction).
    """
    merged = embedding_distances.merge(ec_distances, on=["a", "b"], how="inner")
    if merged.empty:
        return {
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
            "n_pairs": 0,
        }
    rho, p = spearmanr(merged["dist"].to_numpy(), merged["ec_dist"].to_numpy())
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n_pairs": int(len(merged)),
    }
