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

from evaluation.analysis_io import (
    _pivot_long_to_matrix,
    json_safe,
    load_embeddings_h5,
    load_frozen_ids,
    pairwise_distance_long,
)
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


class PopulationError(RuntimeError):
    """A pLM is silently missing frozen EC-positive ids and was not flagged capped."""


def _build_matrices(
    embeddings: dict,
    ec_labels: pd.DataFrame,
    expected_ids: list[str],
    *,
    distance: str,
    ec_set_agg: str,
    allow_capped: bool = False,
):
    """D12 seam: embeddings + EC labels -> (ids, dist_matrix, ec_matrix, pairs_df).

    ``ids`` is the intersection of (expected frozen EC-positive ids) ∩ (embeddings) ∩
    (labelled), in the frozen order. Raises :class:`PopulationError` if a frozen id is
    missing from the embeddings and ``allow_capped`` is False (the population-drift
    contract shared with the other arms).
    """
    label_ids = set(ec_labels["protein_id"])
    present = [pid for pid in expected_ids if pid in embeddings and pid in label_ids]
    missing = [pid for pid in expected_ids if pid not in embeddings]
    if missing and not allow_capped:
        raise PopulationError(
            f"{len(missing)} frozen EC-positive id(s) missing from embeddings "
            f"(e.g. {missing[:3]}); pass allow_capped for an arch-capped pLM."
        )
    ids = present
    if len(ids) < 2:
        raise ValueError(f"need >=2 common EC-positive proteins (got {len(ids)})")

    sub_emb = {pid: embeddings[pid] for pid in ids}
    sub_lab = ec_labels[ec_labels["protein_id"].isin(ids)].reset_index(drop=True)

    dist_long = pairwise_distance_long(sub_emb, distance=distance)
    ec_long = ec_distance_matrix_set(sub_lab, agg=ec_set_agg)
    pairs = dist_long.merge(ec_long, on=["a", "b"], how="inner")
    # Both long frames enumerate the same C(N,2) pairs over the same ids, so the inner
    # merge must not drop rows; assert it so a future producer divergence fails loud.
    assert len(pairs) == len(dist_long) == len(ec_long), (
        f"pair-set divergence: dist={len(dist_long)} ec={len(ec_long)} merged={len(pairs)}"
    )

    dist_matrix = _pivot_long_to_matrix(dist_long, ids, "dist")
    ec_matrix = _pivot_long_to_matrix(ec_long, ids, "ec_dist")
    return ids, dist_matrix, ec_matrix, pairs


def _stem(plm: str, representation: str, distance: str) -> str:
    return f"ec_{plm}_{representation}_{distance}"


def ec_correlation_report(
    embeddings: dict,
    ec_labels: pd.DataFrame,
    out_dir: Path | str,
    *,
    plm: str,
    distance: str,
    ec_set_agg: str = "min",
    wildcard_policy: str = "exclude",
    statistic: str = "tau_b",
    representation: str = "raw",
    expected_ec_ids: list[str],
    seed: int = 42,
    n_boot: int = 2000,
    n_perm: int = 1000,
    ci_alpha: float = 0.05,
    allow_capped: bool = False,
    superfamily: dict | None = None,
    overwrite: bool = True,
) -> dict:
    """Score one (plm, distance) EC cell: writes the per-pair parquet, returns the manifest.

    The CLI (``main``) writes the sidecar; this function writes only the parquet (the
    "lenient library, strict barrier" split the other arms use). Computes τ-b + ρ with
    vertex-BCa CIs, the M-permutation null, the ec_dist histogram, class + superfamily
    stratification, and an ``ec_set_agg`` sensitivity over {min, mean, hausdorff}.
    """
    from shared.atomic_io import atomic_write

    ids, dist_matrix, ec_matrix, pairs = _build_matrices(
        embeddings, ec_labels, expected_ec_ids,
        distance=distance, ec_set_agg=ec_set_agg, allow_capped=allow_capped,
    )

    # Per-pair parquet (synthetic single-column unique key).
    pairs = pairs.copy()
    pairs.insert(0, "pair_key", pairs["a"] + "\t" + pairs["b"])
    pairs = pairs[list(EC_PER_PAIR_COLUMNS)]
    out_dir = Path(out_dir)
    pq_path = out_dir / f"{_stem(plm, representation, distance)}.parquet"
    if pq_path.exists() and not overwrite:
        raise FileExistsError(f"{pq_path} exists; pass overwrite=True")
    written_pq = atomic_write(pq_path, lambda p: pairs.to_parquet(p, index=False), mode="replace")

    # Primary statistic + CI.
    lo, hi, point, degenerate, diverged = correlation_vertex_bca_ci(
        dist_matrix, ec_matrix, statistic=statistic, n_boot=n_boot, alpha=ci_alpha, seed=seed)
    rho_lo, rho_hi, rho_point, rho_degen, _ = correlation_vertex_bca_ci(
        dist_matrix, ec_matrix, statistic="spearman", n_boot=n_boot, alpha=ci_alpha, seed=seed)
    null_vals, perm_p = correlation_permutation_null(
        dist_matrix, ec_matrix, statistic=statistic, n_perm=n_perm, seed=seed)

    # ec_set_agg sensitivity: ONLY the EC matrix changes with agg — the embedding
    # distance matrix is identical — so reuse dist_matrix and rebuild only ec_matrix
    # (avoids redoing the cdist 3x and the misleading implication that dist depends on agg).
    iu, ju = np.triu_indices(len(ids), k=1)
    sub_lab_all = ec_labels[ec_labels["protein_id"].isin(ids)].reset_index(drop=True)
    sensitivity = {}
    for agg in ("min", "mean", "hausdorff"):
        ec_long_agg = ec_distance_matrix_set(sub_lab_all, agg=agg)
        em = _pivot_long_to_matrix(ec_long_agg, ids, "ec_dist")
        sensitivity[agg] = kendall_tau_b(dist_matrix[iu, ju], em[iu, ju])

    # Class stratification (first EC field; from the labels).
    # Class label = first field of the lexicographically-smallest EC (a stable convention for multifunctional enzymes spanning >1 class).
    ec_class = {
        row.protein_id: sorted(row.ec_set)[0].split(".")[0]
        for row in ec_labels.itertuples() if row.ec_set
    }
    strat_class = stratify_by_class(pairs, ec_class)
    strat_sf = stratify_by_superfamily(pairs, superfamily) if superfamily else {}

    manifest = {
        "plm": plm,
        "representation": representation,
        "distance": distance,
        "statistic": statistic,
        "ec_set_agg": ec_set_agg,
        "wildcard_policy": wildcard_policy,
        "tau_b": point if statistic == "tau_b" else kendall_tau_b(
            dist_matrix[iu, ju], ec_matrix[iu, ju]),
        "rho": rho_point,
        "ci_lo": lo, "ci_hi": hi, "ci_degenerate": degenerate, "ci_percentile_diverged": diverged,
        "rho_ci_lo": rho_lo, "rho_ci_hi": rho_hi, "rho_ci_degenerate": rho_degen,
        "perm_p_value": perm_p,
        "null_mean": float(np.nanmean(null_vals)), "null_std": float(np.nanstd(null_vals)),
        "ec_dist_histogram": ec_dist_histogram(pairs),
        "sensitivity": sensitivity,
        "stratify_class": strat_class,
        "stratify_superfamily": strat_sf,
        "n_ec_proteins": len(ids),
        "n_pairs": int(len(pairs)),
        "population_n": len(ids),
        "seed": seed, "n_boot": n_boot, "n_perm": n_perm, "ci_alpha": ci_alpha,
        "per_pair_columns": list(EC_PER_PAIR_COLUMNS),
        "path": str(written_pq),
    }
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: score one (plm, distance) EC cell + sidecar. Exit 0 / 1 (drift) / 2 (config)."""
    from shared.atomic_io import atomic_write

    ap = argparse.ArgumentParser(
        prog="ec_report",
        description="Score one pLM's embedding-distance vs EC-distance correlation "
        "against the frozen EC-positive cohort; write a per-pair parquet + sidecar JSON.",
    )
    ap.add_argument("--plm", required=True)
    ap.add_argument("--emb-h5", required=True)
    ap.add_argument("--freeze", required=True, help="EC-positive freeze JSON (its 'ids').")
    ap.add_argument("--ec-tsv", required=True, help="UniProt-style label TSV for parse_ec.")
    ap.add_argument("--ec-col", default=None, help="Structured EC column in the TSV (optional).")
    ap.add_argument(
        "--superfamily-source", default=None,
        help="CATH labels TSV (Gene3D column) for the D9 homology control: enables the "
        "within/across-superfamily + non-homologous strata. Without it those strata are "
        "empty (the metric then cannot separate function from homology — the 92%% confound).",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--distance", required=True, choices=("cosine", "euclidean", "manhattan"))
    ap.add_argument("--statistic", default="tau_b", choices=("tau_b", "spearman"))
    ap.add_argument("--ec-set-agg", default="min", choices=("min", "mean", "hausdorff"))
    ap.add_argument("--wildcard-policy", default="exclude", choices=("exclude", "include"))
    ap.add_argument("--representation", default="raw")
    ap.add_argument("--allow-capped", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--ci-alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    try:
        embeddings = load_embeddings_h5(args.emb_h5)
        expected_ids = load_frozen_ids(args.freeze)
        label_df = pd.read_csv(args.ec_tsv, sep="\t", dtype=str)
        ec_labels = parse_ec(label_df, ec_col=args.ec_col, wildcard_policy=args.wildcard_policy)
        superfamily = None
        if args.superfamily_source:
            from evaluation.label_adapters import load_cath_labels
            cath = load_cath_labels(args.superfamily_source)
            superfamily = dict(zip(cath["protein_id"], cath["superfamily"]))
        manifest = ec_correlation_report(
            embeddings, ec_labels, args.out_dir,
            plm=args.plm, distance=args.distance, statistic=args.statistic,
            ec_set_agg=args.ec_set_agg, wildcard_policy=args.wildcard_policy,
            representation=args.representation, expected_ec_ids=expected_ids,
            seed=args.seed, n_boot=args.n_boot, n_perm=args.n_perm,
            ci_alpha=args.ci_alpha, allow_capped=args.allow_capped,
            superfamily=superfamily, overwrite=True,
        )
    except PopulationError as e:
        print(f"ec_report: POPULATION DRIFT: {e}", file=sys.stderr, flush=True)
        return 1
    except (FileNotFoundError, OSError) as e:
        print(f"ec_report: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, KeyError) as e:
        print(f"ec_report: INPUT ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    sidecar = Path(args.out_dir) / f"{_stem(args.plm, args.representation, args.distance)}.manifest.json"
    written = atomic_write(
        sidecar,
        lambda p: p.write_text(json.dumps(json_safe(manifest), indent=2) + "\n"),
        mode="replace",
    )
    print(f"ec_report: {args.plm} / {args.distance} (n={manifest['n_ec_proteins']}) "
          f"tau_b={manifest['tau_b']} rho={manifest['rho']} "
          f"[{manifest['ci_lo']}, {manifest['ci_hi']}] perm_p={manifest['perm_p_value']} "
          f"-> {written}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
