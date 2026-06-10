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

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from evaluation.label_adapters import make_cath_is_positive_fn
from evaluation.population import assert_population
from evaluation.recall_fp import recall_at_first_fp
from shared.atomic_io import atomic_write

# Phase A scores the two CATH levels Gene3D resolves to; family is deferred (W3).
DEFAULT_LEVELS: tuple[str, ...] = ("fold", "superfamily")


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

    Returns
    -------
    dict
        ``{"pLM", "representation", "distance", "population_n", "levels":
        {level: {"path", "n_queries_with_positives",
        "n_queries_skipped_no_positives", "n_scored", "mean_recall_1stFP"}}}``.
        ``population_n`` is the asserted embedding cohort (post-subset);
        ``n_scored`` is the queries actually ranked at that level (cohort ∩
        labelled), which can be smaller when some canonical proteins lack a CATH
        label. The spec-builder sets each cell's barrier ``expected_rows`` from
        ``n_queries_with_positives`` (or leaves it ``None`` and relies on the
        barrier's 0-row + unique/non-null/finite guards). A level that scores
        zero queries emits a 0-row parquet (the barrier rejects it — intentional)
        and a NaN ``mean_recall_1stFP``.

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

    mode = "replace" if overwrite else "timestamp"
    out: dict = {
        "pLM": pLM,
        "representation": representation,
        "distance": distance,
        "population_n": len(embeddings),
        "levels": {},
    }
    for level in levels:
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
        }
    return out
