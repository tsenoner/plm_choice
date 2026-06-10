"""Shared-nearest-neighbour analysis step: cross-pLM k-NN agreement -> parquet + sidecar.

This is the bridge the analysis DAG calls per *pLM pair*. SNN is a cross-pLM arm: a
cell is a ``(plm_a, plm_b, representation, distance)`` tuple. The bridge owns the glue
the label-free :func:`evaluation.snn.knn_jaccard_between_plms` deliberately does not:

1. **Subset BOTH pLMs to the frozen canonical set.** Either embedding pool may be a
   superset of the analysis population (prott5/esm3 carry ~1225 keys; only the frozen
   set is scored). Scoring against the full pool would mix the k-NN database; the bridge
   subsets both sides to ``expected_ids`` first.
2. **Assert each population BEFORE scoring (S3).** A silently capped/truncated
   re-extract on *either* side would otherwise compute agreement over a different cohort
   than its peers. Each side is checked via :func:`assert_population`; an
   architecture-capped pLM (e.g. esm1b) passes ``allow_capped_*`` and its per-cell ``n``
   is reported separately so it is never folded into a bare cross-pLM mean.
3. **Reject non-finite embeddings** on either side before scoring (a NaN/Inf vector can
   yield a finite-but-meaningless neighbourhood the parquet finite-guard misses).
4. **Atomic-write the per-query parquet** (B7) so a killed job never leaves a truncated
   artifact the barrier would skip as done.

The emitted per-query parquet (``query``, ``jaccard``, ``k_a``, ``k_b``) is what the B6
barrier's parquet contract validates. The mean-Jaccard CI is a *query-level* BCa
bootstrap (the per-query Jaccards share one k-NN database on each side, so they are NOT
i.i.d. — same caveat the recall-FP arm attaches to its CI).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from evaluation.analysis_io import json_safe, load_embeddings_h5, load_frozen_ids
from evaluation.population import PopulationError, assert_population
from evaluation.snn import knn_jaccard_between_plms
from evaluation.stats import bounded_mean_bca_ci
from shared.atomic_io import atomic_write

DISTANCE_CHOICES: tuple[str, ...] = ("cosine", "euclidean", "manhattan")

# The per-query parquet schema + the fan-in barrier guard semantics, as ONE source of
# truth (the SNN-arm analogue of recall_fp_report.PARQUET_GUARDS). The barrier
# spec-builder arms each cell's ArtifactSpec from here rather than re-deriving the
# column contract, so the "lenient CLI, strict barrier" split cannot drift apart.
SNN_PER_QUERY_COLUMNS: tuple[str, ...] = ("query", "jaccard", "k_a", "k_b")
SNN_PARQUET_GUARDS: dict[str, tuple[str, ...]] = {
    "required_columns": SNN_PER_QUERY_COLUMNS,
    "unique_columns": ("query",),
    "non_null_columns": ("query",),
    "finite_columns": ("jaccard",),
}

# CI provenance — written into every manifest so a figure caption never presents the
# interval as something it isn't. The mean-Jaccard CI is a bootstrap over QUERIES: every
# query is ranked against the same shared k-NN database on each side, so the per-query
# Jaccards are NOT i.i.d. The interval captures query-sampling variability only; it does
# not propagate embedding/DB uncertainty. (Same caveat the recall-FP arm attaches.)
CI_METHOD = "BCa bootstrap, query-level resample"
CI_RESAMPLE_UNIT = "query"
CI_NOTE = (
    "Query-level bootstrap: captures query-sampling variability only. Per-query Jaccards "
    "are NOT i.i.d. (all queries share one k-NN database on each pLM), and the interval "
    "does not propagate embedding/DB uncertainty. A 0-width interval with "
    "ci_degenerate=true marks all-identical per-query Jaccard (e.g. two identical pLMs, "
    "every query 1.0), where the bootstrap is inapplicable — it is a point, not a 95% "
    "coverage statement."
)


def _reject_degenerate(embeddings: dict[str, np.ndarray], name: str, distance: str) -> None:
    """Reject embeddings that would yield a meaningless k-NN neighbourhood.

    Non-finite (NaN/Inf) vectors are rejected for every metric. A *finite* zero-norm
    vector additionally breaks cosine k-NN — sklearn returns distance 1.0 from it to
    every candidate (no error, no NaN), so its own neighbourhood is an arbitrary
    tie-break AND it sits at distance 1.0 from every other query, polluting their
    rankings too. The plain ``isfinite`` guard misses it (0.0 is finite), so reject
    zero-norm vectors specifically under cosine (they are valid points under
    euclidean/manhattan and are left alone there).
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
                f"{len(zeronorm)} zero-norm embedding(s) for {name} under cosine "
                f"distance, e.g. {zeronorm[:5]}; cosine k-NN is undefined for a "
                f"zero vector (distance 1.0 to all) — refusing to score."
            )


def _artifact_path(out_dir: Path, plm_a: str, plm_b: str, rep: str, distance: str) -> Path:
    return out_dir / f"snn_{plm_a}__{plm_b}_{rep}_{distance}.parquet"


def snn_report(
    embeddings_a: dict[str, np.ndarray],
    embeddings_b: dict[str, np.ndarray],
    out_dir: Path | str,
    *,
    plm_a: str,
    plm_b: str,
    expected_ids: Iterable[str],
    distance: str,
    k: int = 10,
    representation: str = "raw",
    allow_capped_a: bool = False,
    allow_capped_b: bool = False,
    overwrite: bool = True,
    n_boot: int = 10_000,
    ci_alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """Score cross-pLM SNN k-NN Jaccard for one pLM pair and write a per-query parquet.

    Parameters
    ----------
    embeddings_a, embeddings_b
        ``{protein_id: 1-D np.ndarray}`` for the two pLMs (each may be a superset of the
        frozen set — both are subset to ``expected_ids`` before scoring).
    out_dir
        Directory the per-query parquet is written into.
    plm_a, plm_b
        Names of the two pLMs — used in the population-error messages and the output
        filename (``snn_<plm_a>__<plm_b>_<representation>_<distance>.parquet``).
    expected_ids
        The frozen canonical id set (**required** — pass the committed
        ``canonical_set_<name>.json["ids"]``). Both sides are subset to it and
        population-checked *before* scoring.
    distance
        ``"cosine"``, ``"euclidean"``, or ``"manhattan"`` (**required** — the same metric
        is used for both pLMs' k-NN). A distinct cell per distance (it is in the filename).
    k
        Neighbours per query (default 10).
    representation
        Representation axis (``"raw"`` default, or e.g. ``"ffn"``) — part of the filename
        so the raw/ffn SNN of the same pair/distance do not collide.
    allow_capped_a, allow_capped_b
        Forwarded to :func:`assert_population` per side: permit a strict subset of the
        frozen set (an architecture-capped pLM). Each side's present ``n`` is reported
        (``population_n_a`` / ``population_n_b``); ``n_common`` is the scored intersection.
    overwrite
        If True (**default**), atomic-replace the canonical fixed path in place (B7,
        killed-job-safe). Set False only for ad-hoc never-clobber.
    n_boot, ci_alpha, seed
        BCa controls for the mean-Jaccard CI (absolute metric → ``B=10_000`` default;
        ``ci_alpha=0.05`` → 95%). ``seed`` makes the CI byte-reproducible.

    Returns
    -------
    dict
        Manifest with ``plm_a``, ``plm_b``, ``representation``, ``distance``, ``k``,
        ``population_n_a``, ``population_n_b``, ``n_common``, ``mean_jaccard``, ``ci_lo``,
        ``ci_hi``, ``ci_degenerate``, the CI provenance, ``per_query_columns``, and
        ``path``. ``(ci_lo, ci_hi)`` is the BCa CI on ``mean_jaccard`` — a query-level
        resample (see ``ci_note``: NOT i.i.d.); ``ci_degenerate`` is True when that pair
        is a point, not a coverage statement (identical pLMs / too few queries). The
        spec-builder sets the barrier ``expected_rows`` from ``n_common``.

    Raises
    ------
    evaluation.population.PopulationError
        If either subset pLM population drifts from ``expected_ids`` (and not the matching
        ``allow_capped_*``) — raised before any parquet is written.
    ValueError
        Non-finite embeddings on either side, or fewer than 2 common proteins (raised by
        :func:`knn_jaccard_between_plms`).
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

    # compute_ci=False: knn_jaccard_between_plms' own B=1000 CI would be discarded — the
    # bridge computes its own degenerate-honest, seed-reproducible CI from per_query below.
    result = knn_jaccard_between_plms(
        embeddings_a, embeddings_b, k=k, distance=distance, compute_ci=False
    )
    per_query = result["per_query"]  # columns: query, jaccard, k_a, k_b
    ci_lo, ci_hi, ci_degenerate = bounded_mean_bca_ci(
        per_query["jaccard"].to_numpy(),
        n_boot=n_boot,
        alpha=ci_alpha,
        rng=np.random.default_rng(seed),
        clip=(0.0, 1.0),
    )

    mode = "replace" if overwrite else "timestamp"
    target = _artifact_path(out_dir, plm_a, plm_b, representation, distance)
    written = atomic_write(
        target, lambda p: per_query.to_parquet(p, index=False), mode=mode
    )
    return {
        "plm_a": plm_a,
        "plm_b": plm_b,
        "representation": representation,
        "distance": distance,
        "k": k,
        "population_n_a": len(embeddings_a),
        "population_n_b": len(embeddings_b),
        "n_common": int(len(per_query)),
        "mean_jaccard": result["mean_jaccard"],
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_degenerate": ci_degenerate,
        "ci_alpha": ci_alpha,
        "n_boot": n_boot,
        "seed": seed,
        "ci_method": CI_METHOD,
        "ci_resample_unit": CI_RESAMPLE_UNIT,
        "ci_note": CI_NOTE,
        "per_query_columns": list(SNN_PER_QUERY_COLUMNS),
        "path": str(written),
    }


# ── CLI: the analysis-DAG SNN step ────────────────────────────────────────────
def main(argv: Sequence[str] | None = None) -> int:
    """CLI wrapper: score one (plm_a, plm_b, representation, distance) SNN cell + sidecar.

    Loads both pLM embedding H5s and the committed canonical-set freeze, calls
    :func:`snn_report` (which writes the per-query parquet), then persists the returned
    manifest as a sidecar JSON — the report itself deliberately does NOT write the
    sidecar; that is this wrapper's job (the barrier spec-builder reads it). Exit codes
    mirror the other DAG mains:

    * ``0`` — scored and wrote the parquet + sidecar.
    * ``1`` — population drift (a pLM silently missing frozen ids, not flagged capped):
      a *data* failure; nothing is written.
    * ``2`` — operator/config fault (missing input file, malformed freeze, too few common
      proteins, non-finite embeddings).
    """
    ap = argparse.ArgumentParser(
        prog="snn_report",
        description="Score cross-pLM SNN k-NN Jaccard for one pLM pair against the frozen "
        "canonical set and write a per-query parquet + a manifest sidecar JSON.",
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
        help="k-NN metric (same for both pLMs; a distinct cell per distance).",
    )
    ap.add_argument("--k", type=int, default=10, help="Neighbours per query (default 10).")
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
        help="RNG seed for the BCa bootstrap CI (default 42; fixes a reproducible interval).",
    )
    ap.add_argument(
        "--n-boot", type=int, default=10_000,
        help="BCa bootstrap resamples for the mean-Jaccard CI (default 10000).",
    )
    ap.add_argument(
        "--ci-alpha", type=float, default=0.05,
        help="Two-sided CI coverage error (default 0.05 -> 95%% CI).",
    )
    args = ap.parse_args(argv)

    # DAG artifacts always replace in place at the barrier's fixed spec path (no
    # timestamped sibling the barrier never checks). The library snn_report(...) still
    # exposes overwrite= for ad-hoc never-clobber use; the CLI does not.
    try:
        embeddings_a = load_embeddings_h5(args.emb_h5_a)
        embeddings_b = load_embeddings_h5(args.emb_h5_b)
        expected_ids = load_frozen_ids(args.freeze)
        manifest = snn_report(
            embeddings_a,
            embeddings_b,
            args.out_dir,
            plm_a=args.plm_a,
            plm_b=args.plm_b,
            expected_ids=expected_ids,
            distance=args.distance,
            k=args.k,
            representation=args.representation,
            allow_capped_a=args.allow_capped_a,
            allow_capped_b=args.allow_capped_b,
            overwrite=True,
            n_boot=args.n_boot,
            ci_alpha=args.ci_alpha,
            seed=args.seed,
        )
    except PopulationError as e:
        print(f"snn_report: POPULATION DRIFT: {e}", file=sys.stderr, flush=True)
        return 1
    except (FileNotFoundError, OSError) as e:
        print(f"snn_report: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, KeyError) as e:
        print(f"snn_report: INPUT ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    # The report writes only the parquet; the sidecar manifest is the CLI's job.
    sidecar = (
        Path(args.out_dir)
        / f"snn_{args.plm_a}__{args.plm_b}_{args.representation}_{args.distance}.manifest.json"
    )
    written = atomic_write(
        sidecar,
        lambda p: p.write_text(json.dumps(json_safe(manifest), indent=2) + "\n"),
        mode="replace",
    )
    print(
        f"snn_report: {args.plm_a} vs {args.plm_b} / {args.representation} / {args.distance} "
        f"(n_common={manifest['n_common']}) mean_jaccard={manifest['mean_jaccard']} "
        f"[{manifest['ci_lo']}, {manifest['ci_hi']}] -> {written}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
