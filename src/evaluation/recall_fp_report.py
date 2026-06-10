"""Recall-at-first-FP analysis step: in-memory result -> barrier-checkable parquet.

This is the bridge the analysis DAG calls per pLM. It owns the glue that the
label-agnostic :func:`evaluation.recall_fp.recall_at_first_fp` and the pure
:func:`evaluation.label_adapters.make_cath_is_positive_fn` deliberately do not:

1. **Subset to the frozen canonical set.** A pLM's embedding pool may be a
   *superset* of the analysis population (prott5/esm3 carry ~1225 keys; only the
   frozen 319 are scored). Scoring against the full pool would mix the retrieval
   database; the bridge subsets to ``expected_ids`` first.
2. **Assert population BEFORE scoring (S3).** After subsetting, the pLM coverage
   is checked against the frozen set via :func:`assert_population` — a silently
   missing protein (truncated re-extract, dropped join) fails the cell loudly
   rather than producing a metric over a different cohort. An architecture-capped
   pLM (e.g. esm1b, 267/319) passes ``allow_capped=True`` and its per-cell ``n``
   is reported separately.
3. **Score each available CATH level** with the set-intersection predicate so
   multi-domain proteins are handled (family is excluded by default — its labels
   are an unmet people-track input).
4. **Atomic-write the per-query parquet** (B7) so a killed job never leaves a
   truncated artifact the barrier would skip as done.

The emitted per-level parquet (the ``per_query`` frame: ``query_id``,
``n_positives``, ``recall``, ``n_ties_at_first_fp``) is what the B6 barrier's
parquet contract validates.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from evaluation.label_adapters import load_cath_labels, make_cath_is_positive_fn
from evaluation.population import PopulationError, assert_population
from evaluation.recall_fp import recall_at_first_fp
from evaluation.stats import bca_bootstrap
from shared.atomic_io import atomic_write

# Phase A scores the two CATH levels Gene3D resolves to; family is deferred (W3).
DEFAULT_LEVELS: tuple[str, ...] = ("fold", "superfamily")

# The per-query parquet schema + the fan-in barrier guard semantics, as ONE source of
# truth. The barrier spec-builder (analysis-DAG wiring) arms each cell's ``ArtifactSpec``
# from here rather than re-deriving the column contract — so the "lenient CLI, strict
# barrier" division of labour cannot drift apart (a scrambled write, an all-NaN ``recall``
# column, or duplicate ``query_id`` is only caught if these guards are actually wired).
PER_QUERY_COLUMNS: tuple[str, ...] = (
    "query_id",
    "n_positives",
    "recall",
    "n_ties_at_first_fp",
)
PARQUET_GUARDS: dict[str, tuple[str, ...]] = {
    "required_columns": PER_QUERY_COLUMNS,
    "unique_columns": ("query_id",),
    "non_null_columns": ("query_id",),
    "finite_columns": ("recall",),
}

# CI provenance — written into every manifest so a figure caption never presents the
# interval as something it isn't. The recall CI is a bootstrap over QUERIES: every query
# is ranked against the same shared retrieval DB, so the per-query recalls are NOT i.i.d.
# (a query is also a DB entry for the others). The interval captures query-sampling
# variability only; it does not propagate embedding/DB uncertainty. (Same caveat the plan
# attaches to the SNN CI.)
CI_METHOD = "BCa bootstrap, query-level resample"
CI_RESAMPLE_UNIT = "query"
CI_NOTE = (
    "Query-level bootstrap: captures query-sampling variability only. Per-query recalls "
    "are NOT i.i.d. (all queries share one retrieval DB), and the interval does not "
    "propagate embedding/DB uncertainty. A 0-width interval with ci_degenerate=true marks "
    "all-identical per-query recall (e.g. perfect retrieval), where the bootstrap is "
    "inapplicable — it is a point, not a 95% coverage statement."
)


def _recall_ci(
    recalls: np.ndarray, *, n_boot: int, alpha: float, rng
) -> tuple[float, float, bool]:
    """BCa CI for the mean of per-query recalls. Returns ``(lo, hi, degenerate)``.

    The mean recall@first-FP is an *absolute* metric (B=10_000 default per plan B3).
    ``degenerate`` is True whenever the returned interval is not a genuine 95% bootstrap
    coverage statement, so the caller can flag it (``ci_degenerate``):

    * fewer than 2 queries → ``(nan, nan, True)`` — no interval is meaningful;
    * all-identical recalls (e.g. perfect retrieval, every query 1.0) → ``(c, c, True)``
      — the bootstrap is inapplicable (every resample is the constant), so this is a
      point, not a coverage interval (scipy would otherwise return NaN);
    * BCa fails to form a finite interval (degenerate jackknife at tiny/odd n) →
      ``(nan, nan, True)`` rather than leaking scipy's NaN/garbage into the manifest.

    Otherwise the BCa interval is clipped to ``[0, 1]`` — the statistic is bounded, and
    BCa can spill past the boundary on skewed data (cf. the ``r2_ci_via_r`` guard).
    """
    recalls = np.asarray(recalls, dtype=float)
    if recalls.size < 2:
        return float("nan"), float("nan"), True
    if float(np.ptp(recalls)) == 0.0:
        c = float(recalls[0])
        return c, c, True
    _, lo, hi = bca_bootstrap(recalls, np.mean, B=n_boot, alpha=alpha, rng=rng)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return float("nan"), float("nan"), True
    return float(min(max(lo, 0.0), 1.0)), float(min(max(hi, 0.0), 1.0)), False


def recall_fp_report(
    embeddings: dict[str, np.ndarray],
    labels: pd.DataFrame,
    out_dir: Path | str,
    *,
    pLM: str,
    expected_ids: Iterable[str],
    distance: str,
    representation: str = "raw",
    levels: Sequence[str] = DEFAULT_LEVELS,
    allow_capped: bool = False,
    overwrite: bool = True,
    n_boot: int = 10_000,
    ci_alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """Score recall-at-first-FP for one pLM and write a parquet per CATH level.

    Parameters
    ----------
    embeddings
        ``{protein_id: 1-D np.ndarray}`` for this pLM (may be a superset of the
        frozen set — it is subset to ``expected_ids`` before scoring).
    labels
        CATH label frame from
        :func:`evaluation.label_adapters.parse_cath_from_gene3d` (frozenset
        ``fold``/``superfamily`` columns).
    out_dir
        Directory the per-level parquet files are written into.
    pLM
        Name of the pLM — used in the population-error message and the output
        filenames (``recall_fp_<pLM>_<representation>_<level>.parquet``).
    expected_ids
        The frozen canonical id set (**required** — pass the committed
        ``canonical_set_<name>.json["ids"]``, do not reconstruct). ``embeddings``
        is subset to it and the result is population-checked *before* scoring, so
        a drifted/truncated cell fails loudly rather than scoring a different
        cohort than its peers. Required by design: the bridge exists so the
        subset+assert is never left to the driver.
    distance
        ``"cosine"``, ``"euclidean"``, or ``"manhattan"`` (**required** — a wrong
        metric silently changes the numbers, so the caller must choose; the
        data-prep pipeline uses euclidean).
    representation
        Representation axis (``"raw"`` default, or e.g. ``"ffn"``) — part of the
        filename so the raw and FFN recall-FP arms of the same pLM/level do not
        collide (plan v3: raw + FFN reps).
    levels
        CATH levels to score (default Topology + Homologous-SF; family is excluded
        — its labels are an unmet people-track input, W3).
    allow_capped
        Forwarded to :func:`assert_population`: permit a strict subset of the
        frozen set (an architecture-capped pLM, e.g. esm1b 267/319) without
        failing. Its per-cell ``n`` is reported (``population_n``) so it is never
        folded into a bare cross-pLM mean.
    overwrite
        If True (**default**), atomic-replace the canonical fixed path in place
        (tmp + ``os.replace`` — killed-job-safe). This is correct for a DAG
        artifact the B6 barrier validates at a *fixed* spec path: a never-clobber
        timestamped sibling would leave the barrier checking the stale file and
        ``needs_rebuild`` unsatisfiable. Set False only for ad-hoc never-clobber.

    n_boot, ci_alpha, seed
        BCa bootstrap CI controls for the per-level mean recall (an absolute metric →
        ``B=10_000`` default per plan B3; ``ci_alpha=0.05`` → 95%). ``seed`` makes the
        CIs byte-reproducible (NEW-2); each level draws an independent stream via
        ``SeedSequence.spawn``.

    Returns
    -------
    dict
        ``{"pLM", "representation", "distance", "population_n", "ci_alpha", "n_boot",
        "seed", "ci_method", "ci_resample_unit", "ci_note", "per_query_columns",
        "levels": {level: {"path", "n_queries_with_positives",
        "n_queries_skipped_no_positives", "n_scored", "mean_recall_1stFP", "ci_lo",
        "ci_hi", "ci_degenerate"}}}``.
        ``population_n`` is the asserted embedding cohort (post-subset);
        ``n_scored`` is the queries actually ranked at that level (cohort ∩
        labelled), which can be smaller when some canonical proteins lack a CATH
        label. ``(ci_lo, ci_hi)`` is the BCa CI on ``mean_recall_1stFP`` — a
        *query-level resample* (see ``ci_note``: NOT i.i.d., DB-uncertainty-blind);
        ``ci_degenerate`` is True when that interval is a point, not a coverage
        statement (perfect/near-degenerate retrieval or too few queries — then the
        CI bounds are the point value or NaN). The spec-builder sets each cell's
        barrier ``expected_rows`` from ``n_queries_with_positives`` (or leaves it
        ``None`` and relies on the barrier's 0-row + unique/non-null/finite guards).
        A level that scores zero queries emits a 0-row parquet (the barrier rejects
        it — intentional) with a NaN ``mean_recall_1stFP`` and NaN CI bounds.

    Raises
    ------
    evaluation.population.PopulationError
        If the subset pLM population drifts from ``expected_ids`` (and not
        ``allow_capped``) — raised before any parquet is written.
    """
    out_dir = Path(out_dir)

    exp = set(expected_ids)
    embeddings = {k: v for k, v in embeddings.items() if k in exp}
    # S3: assert BEFORE scoring so a drifted cell fails loudly, not silently.
    assert_population(embeddings.keys(), exp, name=pLM, allow_capped=allow_capped)
    # Reject a corrupt/degenerate embedding set (NaN/Inf) before scoring. The
    # per-query parquet's finite(recall) guard only catches non-finiteness that
    # propagates into the recall scalar; a NaN/Inf vector can still yield a
    # finite-but-meaningless recall (e.g. under euclidean) the barrier would pass —
    # exactly the B7 "valid-looking but degenerate artifact" hazard, on the input side.
    nonfinite = sorted(k for k, v in embeddings.items() if not np.all(np.isfinite(v)))
    if nonfinite:
        raise ValueError(
            f"{len(nonfinite)} embedding(s) for {pLM} contain non-finite values "
            f"(NaN/Inf), e.g. {nonfinite[:5]}; refusing to score a degenerate set."
        )

    mode = "replace" if overwrite else "timestamp"
    # Independent, seed-reproducible bootstrap stream per level (NEW-2 seed gate):
    # SeedSequence.spawn gives genuinely independent child generators (not consecutive
    # slices of one stream), reproducible given `seed`. seed=None -> fresh randomness.
    level_seeds = np.random.SeedSequence(seed).spawn(len(levels))
    out: dict = {
        "pLM": pLM,
        "representation": representation,
        "distance": distance,
        "population_n": len(embeddings),
        # CI provenance so a re-run is byte-reproducible and the legend is auditable.
        "ci_alpha": ci_alpha,
        "n_boot": n_boot,
        "seed": seed,
        "ci_method": CI_METHOD,
        "ci_resample_unit": CI_RESAMPLE_UNIT,
        "ci_note": CI_NOTE,
        # The per-query parquet schema, surfaced so the barrier spec-builder transcribes
        # the column contract from the artifact's own manifest (see PARQUET_GUARDS).
        "per_query_columns": list(PER_QUERY_COLUMNS),
        "levels": {},
    }
    for level, level_seed in zip(levels, level_seeds):
        is_pos = make_cath_is_positive_fn(labels, level)
        result = recall_at_first_fp(
            embeddings,
            labels,
            distance=distance,
            level=level,
            per_query=True,
            is_positive_fn=is_pos,
        )
        per_query: pd.DataFrame = result["per_query"]
        ci_lo, ci_hi, ci_degenerate = _recall_ci(
            per_query["recall"].to_numpy(),
            n_boot=n_boot,
            alpha=ci_alpha,
            rng=np.random.default_rng(level_seed),
        )
        target = out_dir / f"recall_fp_{pLM}_{representation}_{level}.parquet"
        written = atomic_write(
            target,
            lambda p, df=per_query: df.to_parquet(p, index=False),
            mode=mode,
        )
        out["levels"][level] = {
            "path": str(written),
            "n_queries_with_positives": result["n_queries_with_positives"],
            "n_queries_skipped_no_positives": result[
                "n_queries_skipped_no_positives"
            ],
            "n_scored": (
                result["n_queries_with_positives"]
                + result["n_queries_skipped_no_positives"]
            ),
            "mean_recall_1stFP": result["mean_recall_1stFP"],
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_degenerate": ci_degenerate,
        }
    return out


# ── CLI: the analysis-DAG recall-fp step ──────────────────────────────────────
def _load_embeddings_h5(path: Path | str) -> dict[str, np.ndarray]:
    """Load a per-protein embedding H5 into ``{protein_id: 1-D np.ndarray}``.

    Each dataset is one protein. A 2-D ``(L, D)`` per-residue dataset is mean-pooled
    over residues to a protein-level vector (matching
    :class:`data_preparation.distance_computation`'s loader), so a per-residue H5 is
    accepted as well as the reduced per-protein H5 the extract step writes.
    """
    import h5py

    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arr = np.asarray(f[key][()])  # [()] reads scalar + array datasets alike
            if arr.ndim > 1:
                arr = arr.mean(axis=0)  # (L, D) per-residue -> (D,) protein-level
            out[key] = np.asarray(arr, dtype=np.float32)
    return out


def _json_safe_manifest(manifest: dict) -> dict:
    """Copy ``manifest`` with non-finite ``mean_recall_1stFP`` rendered as ``None``.

    A level that scores zero queries carries ``mean_recall_1stFP = NaN``. ``json.dumps``
    would emit the bare token ``NaN`` — accepted by Python's ``json.loads`` but invalid
    per the JSON spec, so a strict / non-Python reader (the barrier spec-builder) would
    reject the sidecar. Mapping NaN → ``null`` keeps the sidecar standards-valid; the
    contract is "null mean == a 0-query level" (also signalled by ``n_scored == 0``).
    """
    safe = {**manifest, "levels": {}}
    for level, info in manifest["levels"].items():
        info = dict(info)
        for key in ("mean_recall_1stFP", "ci_lo", "ci_hi"):
            v = info.get(key)
            if isinstance(v, float) and not math.isfinite(v):
                info[key] = None
        safe["levels"][level] = info
    return safe


def _load_frozen_ids(freeze_path: Path | str) -> list[str]:
    """Read the committed canonical-set freeze and return its ``ids`` list.

    The freeze (``canonical_set_<name>.json``) is the single source of truth for the
    analysis population — the caller must pass it rather than reconstruct the id set.
    Raises ``ValueError`` if the manifest carries no non-empty ``ids`` list (an
    operator/config fault → CLI exit 2).
    """
    data = json.loads(Path(freeze_path).read_text())
    ids = data.get("ids") if isinstance(data, dict) else None
    if not isinstance(ids, list) or not ids:
        raise ValueError(
            f"freeze {freeze_path} has no non-empty 'ids' list; pass the committed "
            f"canonical_set_<name>.json"
        )
    return ids


def main(argv: Sequence[str] | None = None) -> int:
    """CLI wrapper: score one (pLM, representation) recall-fp cell + write the sidecar.

    Loads the pLM embedding H5, the cath_labels TSV, and the committed canonical-set
    freeze, calls :func:`recall_fp_report` (which writes the per-level parquets), then
    persists the returned manifest as a sidecar JSON — the report itself deliberately
    does NOT write the sidecar; that is this wrapper's job (the barrier spec-builder
    reads it). Exit codes mirror the other DAG mains (:mod:`evaluation.verify_analysis`,
    :mod:`evaluation.analysis_barrier`):

    * ``0`` — scored and wrote the parquets + sidecar.
    * ``1`` — population drift (a pLM silently missing frozen ids, not flagged capped):
      a *data* failure; nothing is written.
    * ``2`` — operator/config fault (missing input file, malformed freeze, bad column).
    """
    ap = argparse.ArgumentParser(
        prog="recall_fp_report",
        description="Score recall-at-first-FP for one pLM against the frozen canonical "
        "set (Topology/SF) and write a per-level parquet + a manifest sidecar JSON.",
    )
    ap.add_argument("--plm", required=True, help="pLM name (used in filenames + manifest).")
    ap.add_argument("--emb-h5", required=True, help="Per-protein embedding H5 for this pLM.")
    ap.add_argument("--cath-tsv", required=True, help="cath_labels TSV (UniProt Gene3D export).")
    ap.add_argument(
        "--freeze",
        required=True,
        help="Committed canonical-set freeze JSON; its 'ids' are the expected population.",
    )
    ap.add_argument("--out-dir", required=True, help="Directory for the parquets + sidecar.")
    ap.add_argument(
        "--distance",
        required=True,
        choices=("cosine", "euclidean", "manhattan"),
        help="Retrieval metric (required — the data-prep pipeline uses euclidean).",
    )
    ap.add_argument(
        "--representation",
        default="raw",
        help="Representation axis (default raw); part of the filenames so raw/ffn don't collide.",
    )
    ap.add_argument(
        "--levels",
        nargs="+",
        choices=("fold", "superfamily", "family"),
        default=list(DEFAULT_LEVELS),
        help=f"CATH levels to score (default {' '.join(DEFAULT_LEVELS)}; family deferred, W3 — "
        "passing it yields a barrier-rejected 0-row parquet, never a fabricated positive).",
    )
    ap.add_argument(
        "--allow-capped",
        action="store_true",
        help="Permit a strict subset of the frozen set (an arch-capped pLM, e.g. esm1b).",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for the BCa bootstrap CIs (default 42; fixes a reproducible interval).",
    )
    ap.add_argument(
        "--n-boot", type=int, default=10_000,
        help="BCa bootstrap resamples for the recall CIs (default 10000; absolute-metric rule).",
    )
    ap.add_argument(
        "--ci-alpha", type=float, default=0.05,
        help="Two-sided CI coverage error (default 0.05 -> 95%% CI).",
    )
    args = ap.parse_args(argv)

    # DAG artifacts always replace in place at the barrier's fixed spec path — a
    # timestamped sibling would orphan the fresh result where the barrier never looks
    # (and desync the sidecar the spec-builder reads). The library recall_fp_report(...)
    # still exposes overwrite= for ad-hoc never-clobber use; the CLI does not.
    overwrite = True
    try:
        embeddings = _load_embeddings_h5(args.emb_h5)
        labels = load_cath_labels(args.cath_tsv)
        expected_ids = _load_frozen_ids(args.freeze)
        manifest = recall_fp_report(
            embeddings,
            labels,
            args.out_dir,
            pLM=args.plm,
            expected_ids=expected_ids,
            distance=args.distance,
            representation=args.representation,
            levels=args.levels,
            allow_capped=args.allow_capped,
            overwrite=overwrite,
            n_boot=args.n_boot,
            ci_alpha=args.ci_alpha,
            seed=args.seed,
        )
    except PopulationError as e:
        print(f"recall_fp_report: POPULATION DRIFT: {e}", file=sys.stderr, flush=True)
        return 1
    except (FileNotFoundError, OSError) as e:
        print(f"recall_fp_report: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, KeyError) as e:
        print(f"recall_fp_report: INPUT ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    # The report writes only the parquets; the sidecar manifest is the CLI's job.
    sidecar = Path(args.out_dir) / f"recall_fp_{args.plm}_{args.representation}.manifest.json"
    written = atomic_write(
        sidecar,
        lambda p: p.write_text(json.dumps(_json_safe_manifest(manifest), indent=2) + "\n"),
        mode="replace" if overwrite else "timestamp",
    )
    print(f"recall_fp_report: {args.plm}/{args.representation} (n={manifest['population_n']}) "
          f"-> {written}", flush=True)
    for level, info in manifest["levels"].items():
        print(
            f"  {level}: n_scored={info['n_scored']} "
            f"mean_recall_1stFP={info['mean_recall_1stFP']} "
            f"[{info['ci_lo']}, {info['ci_hi']}]",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
