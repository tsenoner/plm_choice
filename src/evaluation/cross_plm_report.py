"""Cross-pLM agreement analysis step: per-pLM-pair agreement metrics -> parquet + sidecar.

This is the bridge the analysis DAG calls per *pLM pair*. Cross-pLM is a pLM-pair arm: a
cell is a ``(plm_a, plm_b, representation, distance)`` tuple. The bridge owns the glue the
label-free :mod:`evaluation.cross_plm` compute core deliberately does not:

1. **Subset BOTH pLMs to the frozen canonical set.** Either embedding pool may be a superset
   of the analysis population; only the frozen set is scored. The two pLMs are then aligned to
   the **intersection of their covered ids within the frozen set** — ONE shared id order — so
   both distance matrices enumerate identically the same proteins (a paired ρ/W₁/R² is only
   meaningful if both vectors index the same pairs in the same row order).
2. **Assert each population BEFORE scoring (S3).** A silently capped/truncated re-extract on
   *either* side would otherwise compute agreement over a different cohort than its peers. Each
   side is checked via :func:`assert_population`; an architecture-capped pLM (esm1b) passes
   ``allow_capped_*`` and the scored ``n_common`` / ``n_pairs`` is recorded so a capped
   comparison is never folded into a bare cross-pLM mean.
3. **Reject degenerate embeddings** on either side before scoring (non-finite vectors for every
   metric; finite zero-norm vectors additionally under cosine, where they are undefined).
4. **Atomic-write the per-pair parquet** (B7) ``[pair_key, a, b, dist_a, dist_b]`` — the two
   pLMs' aligned distances on each common pair, the raw material a figure/CI re-derives from.

The four symmetric agreement metrics (ρ / R² / W₁-raw / W₁-z) each get a per-cell **vertex
(protein) BCa CI** over BOTH distance matrices. ρ and R² additionally carry a per-cell
**permutation p-value**; **W₁ has NO permutation p by design** (a symmetric label permutation
preserves each matrix's marginal distance distribution, so the null is degenerate — see
:func:`evaluation.cross_plm.cross_plm_permutation_null`). W₁ is reported as a descriptive
distance + CI and its ``perm_p`` is explicitly ``null``; it MUST NOT enter any downstream Holm
family. The CI is a vertex resample (both pLM distance vectors are induced from the same
resampled proteins), so the per-pair Jaccards/distances are NOT i.i.d. — same caveat the
recall-FP / SNN arms attach.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from evaluation.analysis_io import (
    _pivot_long_to_matrix,
    json_safe,
    load_embeddings_h5,
    load_frozen_ids,
    pairwise_distance_long,
)
from evaluation.cross_plm import cross_plm_agreement_ci, cross_plm_permutation_null
from evaluation.population import PopulationError, assert_population
from shared.atomic_io import atomic_write

DISTANCE_CHOICES: tuple[str, ...] = ("cosine", "euclidean", "manhattan")

# The four symmetric agreement metrics scored per cell. ρ and R² carry a permutation p
# (and feed the U7 Holm families); W₁ (raw + z) carries point + CI ONLY (perm_p = null).
CROSS_PLM_METRICS: tuple[str, ...] = ("rho", "r2", "w1_raw", "w1_z")
_PERM_METRICS: frozenset[str] = frozenset({"rho", "r2"})

# The per-pair parquet schema + the fan-in barrier guard semantics, as ONE source of truth
# (the cross-pLM analogue of SNN_PARQUET_GUARDS / EC_PARQUET_GUARDS). A synthetic single-column
# ``pair_key`` is the unique key (the barrier's unique_columns guard is single-column;
# encoding (a,b) into one column sidesteps any 2-column-key ambiguity while keeping a/b).
CROSS_PLM_PER_PAIR_COLUMNS: tuple[str, ...] = ("pair_key", "a", "b", "dist_a", "dist_b")
CROSS_PLM_PARQUET_GUARDS: dict[str, tuple[str, ...]] = {
    "required_columns": CROSS_PLM_PER_PAIR_COLUMNS,
    "unique_columns": ("pair_key",),
    "non_null_columns": ("pair_key", "a", "b"),
    "finite_columns": ("dist_a", "dist_b"),
}

# CI provenance — written into every manifest so a figure caption never presents the interval
# as something it isn't. The agreement CI is a vertex (protein) bootstrap over BOTH pLM
# distance matrices: each bootstrap iteration resamples proteins and both pLMs' distance
# vectors are induced from the SAME resampled proteins. The per-pair distances are therefore
# NOT i.i.d. (pairs sharing a protein are correlated); the interval captures protein-sampling
# variability only.
CI_METHOD = "vertex (protein) BCa bootstrap over both pLM distance matrices"
CI_RESAMPLE_UNIT = "protein (vertex)"
CI_NOTE = (
    "Vertex-level bootstrap: resamples proteins and induces both pLMs' pair-distance vectors "
    "from the same draw. The per-pair distances are NOT i.i.d. (pairs sharing a protein are "
    "correlated), so the interval captures protein-sampling variability only and does not "
    "propagate embedding uncertainty. A cell flagged ci_degenerate is a point, not a 95% "
    "coverage statement (e.g. two identical pLMs: ρ=R²=1, W₁=0). W₁ (raw and z) carries point "
    "+ CI ONLY — it has no permutation p by design and must not enter any Holm family."
)


def _reject_degenerate(embeddings: dict[str, np.ndarray], name: str, distance: str) -> None:
    """Reject embeddings that would yield a meaningless pairwise distance.

    Non-finite (NaN/Inf) vectors are rejected for every metric (a NaN vector produces NaN
    distances that silently corrupt the agreement). A *finite* zero-norm vector additionally
    breaks cosine (``cdist`` cosine of a zero vector is NaN — 0/0), so reject zero-norm
    vectors specifically under cosine; they are valid points under euclidean/manhattan and are
    left alone there. Mirrors :func:`evaluation.snn_report._reject_degenerate` (clone, not
    extract — the D8 convention).
    """
    nonfinite = sorted(k for k, v in embeddings.items() if not np.all(np.isfinite(v)))
    if nonfinite:
        raise ValueError(
            f"{len(nonfinite)} embedding(s) for {name} contain non-finite values "
            f"(NaN/Inf), e.g. {nonfinite[:5]}; refusing to score a degenerate set."
        )
    if distance == "cosine":
        zeronorm = sorted(
            k for k, v in embeddings.items() if float(np.linalg.norm(v)) == 0.0
        )
        if zeronorm:
            raise ValueError(
                f"{len(zeronorm)} zero-norm embedding(s) for {name} under cosine distance, "
                f"e.g. {zeronorm[:5]}; cosine distance is undefined for a zero vector "
                f"(0/0 -> NaN) — refusing to score."
            )


def _artifact_path(out_dir: Path, plm_a: str, plm_b: str, rep: str, distance: str) -> Path:
    return out_dir / f"cross_plm_{plm_a}__{plm_b}_{rep}_{distance}.parquet"


def _agreement_entry(
    mat_a: np.ndarray, mat_b: np.ndarray, metric: str, *,
    n_boot: int, ci_alpha: float, seed: int, n_perm: int,
) -> dict:
    """One metric's manifest entry: point + vertex-BCa CI (+ perm_p for ρ/R² only)."""
    rec = cross_plm_agreement_ci(
        mat_a, mat_b, metric=metric, n_boot=n_boot, alpha=ci_alpha, seed=seed)
    perm_p = None
    if metric in _PERM_METRICS:
        _, perm_p = cross_plm_permutation_null(
            mat_a, mat_b, metric=metric, n_perm=n_perm, seed=seed)
    entry = {
        "point": rec["point"],
        "ci_lo": rec["ci_lo"],
        "ci_hi": rec["ci_hi"],
        "ci_degenerate": rec["degenerate"],
        "percentile_diverged": rec["diverged"],
        "perm_p": perm_p,
    }
    if metric == "r2":  # carry the signed-r CI the R²-CI was mapped from (B1 provenance)
        entry["r_point"] = rec["r_point"]
        entry["r_ci_lo"] = rec["r_ci_lo"]
        entry["r_ci_hi"] = rec["r_ci_hi"]
    return entry


def cross_plm_report(
    embeddings_a: dict[str, np.ndarray],
    embeddings_b: dict[str, np.ndarray],
    out_dir: Path | str,
    *,
    plm_a: str,
    plm_b: str,
    expected_ids: Iterable[str],
    distance: str,
    metrics: Sequence[str] = CROSS_PLM_METRICS,
    representation: str = "raw",
    allow_capped_a: bool = False,
    allow_capped_b: bool = False,
    overwrite: bool = True,
    n_boot: int = 2000,
    n_perm: int = 1000,
    ci_alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Score cross-pLM agreement for one pLM pair and write a per-pair parquet.

    Parameters
    ----------
    embeddings_a, embeddings_b
        ``{protein_id: 1-D np.ndarray}`` for the two pLMs (each may be a superset of the frozen
        set — both are subset to ``expected_ids``, then aligned to their shared id order).
    out_dir
        Directory the per-pair parquet is written into.
    plm_a, plm_b
        Names of the two pLMs — used in population-error messages and the output filename
        (``cross_plm_<plm_a>__<plm_b>_<representation>_<distance>.parquet``).
    expected_ids
        The frozen canonical id set (**required** — pass the committed
        ``canonical_set_<name>.json["ids"]``). Both sides are subset to it and
        population-checked *before* scoring.
    distance
        ``"cosine"``, ``"euclidean"``, or ``"manhattan"`` (**required** — the same metric for
        both pLMs; a distinct cell per distance).
    metrics
        Which agreement metrics to score (default all four: ρ / R² / W₁-raw / W₁-z).
    representation
        Representation axis (``"raw"`` default) — part of the filename so raw/ffn of the same
        pair/distance do not collide.
    allow_capped_a, allow_capped_b
        Forwarded to :func:`assert_population` per side: permit a strict subset of the frozen
        set (an architecture-capped pLM). Each side's present ``n`` is reported
        (``population_n_a`` / ``population_n_b``); ``n_common`` is the scored intersection and
        ``n_pairs`` = C(n_common, 2) the parquet row count.
    overwrite
        If True (**default**), atomic-replace the canonical fixed path in place (B7,
        killed-job-safe). Set False only for ad-hoc never-clobber.
    n_boot, ci_alpha, seed
        Vertex-BCa controls for the per-metric CIs. ``n_perm`` is the per-cell permutation-null
        resample count (ρ/R² only). ``seed`` makes the CIs + perm p byte-reproducible.

    Returns
    -------
    dict
        Manifest with ``plm_a``, ``plm_b``, ``representation``, ``distance``, a ``metrics``
        sub-dict keyed by metric name (each ``{point, ci_lo, ci_hi, ci_degenerate,
        percentile_diverged, perm_p}``; r2 additionally ``r_point/r_ci_lo/r_ci_hi``; W₁
        ``perm_p`` is None), ``population_n_a``, ``population_n_b``, ``n_common``, ``n_pairs``,
        the CI provenance, ``per_pair_columns``, and ``path``. The spec-builder sets the barrier
        ``expected_rows`` from ``n_pairs``.

    Raises
    ------
    evaluation.population.PopulationError
        If either subset pLM population drifts from ``expected_ids`` (and not the matching
        ``allow_capped_*``) — raised before any parquet is written.
    ValueError
        Non-finite/zero-norm embeddings, or fewer than 2 common proteins.
    """
    out_dir = Path(out_dir)
    exp = set(expected_ids)
    embeddings_a = {kk: vv for kk, vv in embeddings_a.items() if kk in exp}
    embeddings_b = {kk: vv for kk, vv in embeddings_b.items() if kk in exp}
    # S3: assert BOTH populations BEFORE scoring so a drifted cell fails loud, not silent.
    assert_population(embeddings_a.keys(), exp, name=plm_a, allow_capped=allow_capped_a)
    assert_population(embeddings_b.keys(), exp, name=plm_b, allow_capped=allow_capped_b)
    _reject_degenerate(embeddings_a, plm_a, distance)
    _reject_degenerate(embeddings_b, plm_b, distance)

    # Align both pLMs to ONE shared id order (intersection within the frozen set), so both
    # distance matrices enumerate identically the same proteins in the same row order.
    shared_ids = sorted(set(embeddings_a) & set(embeddings_b))
    if len(shared_ids) < 2:
        raise ValueError(
            f"need >=2 common proteins for {plm_a} vs {plm_b} (got {len(shared_ids)})"
        )
    sub_a = {pid: embeddings_a[pid] for pid in shared_ids}
    sub_b = {pid: embeddings_b[pid] for pid in shared_ids}

    long_a = pairwise_distance_long(sub_a, distance=distance)
    long_b = pairwise_distance_long(sub_b, distance=distance)
    mat_a = _pivot_long_to_matrix(long_a, shared_ids, "dist")
    mat_b = _pivot_long_to_matrix(long_b, shared_ids, "dist")

    # Per-pair parquet: the two pLMs' aligned distances on each common pair. Both long frames
    # enumerate the same C(n,2) pairs over the same ids, so the inner merge must not drop rows.
    pairs = long_a.merge(long_b, on=["a", "b"], how="inner", suffixes=("_a", "_b"))
    assert len(pairs) == len(long_a) == len(long_b), (
        f"pair-set divergence: a={len(long_a)} b={len(long_b)} merged={len(pairs)}"
    )
    pairs.insert(0, "pair_key", pairs["a"] + "\t" + pairs["b"])
    pairs = pairs[list(CROSS_PLM_PER_PAIR_COLUMNS)]

    metric_records = {
        metric: _agreement_entry(
            mat_a, mat_b, metric, n_boot=n_boot, ci_alpha=ci_alpha, seed=seed, n_perm=n_perm)
        for metric in metrics
    }

    mode = "replace" if overwrite else "timestamp"
    target = _artifact_path(out_dir, plm_a, plm_b, representation, distance)
    written = atomic_write(target, lambda p: pairs.to_parquet(p, index=False), mode=mode)
    return {
        "plm_a": plm_a,
        "plm_b": plm_b,
        "representation": representation,
        "distance": distance,
        "metrics": metric_records,
        "population_n_a": len(embeddings_a),
        "population_n_b": len(embeddings_b),
        "n_common": len(shared_ids),
        "n_pairs": int(len(pairs)),
        "seed": seed,
        "n_boot": n_boot,
        "n_perm": n_perm,
        "ci_alpha": ci_alpha,
        "ci_method": CI_METHOD,
        "ci_resample_unit": CI_RESAMPLE_UNIT,
        "ci_note": CI_NOTE,
        "per_pair_columns": list(CROSS_PLM_PER_PAIR_COLUMNS),
        "path": str(written),
    }


# ── CLI: the analysis-DAG cross-pLM step (4th clone of the report main() envelope) ──
def main(argv: Sequence[str] | None = None) -> int:
    """CLI wrapper: score one (plm_a, plm_b, representation, distance) cell + sidecar.

    Loads both pLM embedding H5s and the committed canonical-set freeze, calls
    :func:`cross_plm_report` (which writes the per-pair parquet), then persists the returned
    manifest as a sidecar JSON — the report itself deliberately does NOT write the sidecar;
    that is this wrapper's job (the barrier spec-builder reads it). Exit codes mirror the other
    DAG mains:

    * ``0`` — scored and wrote the parquet + sidecar.
    * ``1`` — population drift (a pLM silently missing frozen ids, not flagged capped): a
      *data* failure; nothing is written.
    * ``2`` — operator/config fault (missing input file, malformed freeze, too few common
      proteins, non-finite/zero-norm embeddings).
    """
    ap = argparse.ArgumentParser(
        prog="cross_plm_report",
        description="Score cross-pLM agreement (ρ / R² / W₁-raw / W₁-z) for one pLM pair "
        "against the frozen canonical set and write a per-pair parquet + a manifest sidecar.",
    )
    ap.add_argument("--plm-a", required=True, help="First pLM name (used in filenames + manifest).")
    ap.add_argument("--plm-b", required=True, help="Second pLM name.")
    ap.add_argument("--emb-h5-a", required=True, help="Per-protein embedding H5 for pLM A.")
    ap.add_argument("--emb-h5-b", required=True, help="Per-protein embedding H5 for pLM B.")
    ap.add_argument(
        "--freeze", required=True,
        help="Committed canonical-set freeze JSON; its 'ids' are the expected population.",
    )
    ap.add_argument("--out-dir", required=True, help="Directory for the parquet + sidecar.")
    ap.add_argument(
        "--distance", required=True, choices=DISTANCE_CHOICES,
        help="Distance metric (same for both pLMs; a distinct cell per distance).",
    )
    ap.add_argument(
        "--representation", default="raw",
        help="Representation axis (default raw); part of the filename so raw/ffn don't collide.",
    )
    ap.add_argument(
        "--allow-capped-a", action="store_true",
        help="Permit pLM A to be a strict subset of the frozen set (an arch-capped pLM).",
    )
    ap.add_argument(
        "--allow-capped-b", action="store_true",
        help="Permit pLM B to be a strict subset of the frozen set (an arch-capped pLM).",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for the BCa CIs + permutation null (default 42; reproducible).",
    )
    ap.add_argument(
        "--n-boot", type=int, default=2000,
        help="Vertex-BCa bootstrap resamples for each metric CI (default 2000).",
    )
    ap.add_argument(
        "--n-perm", type=int, default=1000,
        help="Permutation-null resamples for the ρ/R² per-cell p-value (default 1000).",
    )
    ap.add_argument(
        "--ci-alpha", type=float, default=0.05,
        help="Two-sided CI coverage error (default 0.05 -> 95%% CI).",
    )
    args = ap.parse_args(argv)

    try:
        embeddings_a = load_embeddings_h5(args.emb_h5_a)
        embeddings_b = load_embeddings_h5(args.emb_h5_b)
        expected_ids = load_frozen_ids(args.freeze)
        manifest = cross_plm_report(
            embeddings_a,
            embeddings_b,
            args.out_dir,
            plm_a=args.plm_a,
            plm_b=args.plm_b,
            expected_ids=expected_ids,
            distance=args.distance,
            representation=args.representation,
            allow_capped_a=args.allow_capped_a,
            allow_capped_b=args.allow_capped_b,
            overwrite=True,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            ci_alpha=args.ci_alpha,
            seed=args.seed,
        )
    except PopulationError as e:
        print(f"cross_plm_report: POPULATION DRIFT: {e}", file=sys.stderr, flush=True)
        return 1
    except (FileNotFoundError, OSError) as e:
        print(f"cross_plm_report: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, KeyError) as e:
        print(f"cross_plm_report: INPUT ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    # The report writes only the parquet; the sidecar manifest is the CLI's job.
    sidecar = (
        Path(args.out_dir)
        / f"cross_plm_{args.plm_a}__{args.plm_b}_{args.representation}_{args.distance}.manifest.json"
    )
    written = atomic_write(
        sidecar,
        lambda p: p.write_text(json.dumps(json_safe(manifest), indent=2) + "\n"),
        mode="replace",
    )
    rho = manifest["metrics"].get("rho", {})
    print(
        f"cross_plm_report: {args.plm_a} vs {args.plm_b} / {args.representation} / "
        f"{args.distance} (n_common={manifest['n_common']}, n_pairs={manifest['n_pairs']}) "
        f"rho={rho.get('point')} [{rho.get('ci_lo')}, {rho.get('ci_hi')}] "
        f"perm_p={rho.get('perm_p')} -> {written}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
