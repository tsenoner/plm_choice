"""AAC-floor analysis step: amino-acid-composition floor → barrier-checkable parquet.

This is the AAC-floor sibling of :mod:`evaluation.recall_fp_report` (Unit 2 of the
AAC-floor arm). It scores the **trivial 20-d amino-acid-composition floor** with the
**same** recall-at-first-FP metric and the **same** CATH set-intersection positive
predicate the pLMs are scored with — the floor a pLM must beat is unfair otherwise.

The AAC vectors are built on the fly from the canonical FASTA (D6) via Unit 1's
:func:`evaluation.aac_floor.build_aac_embeddings`; there is no pre-produced ``aac.h5``.

The one real correctness subtlety is the **C1 capped-cohort fix** (spec §10):

    recall-at-first-FP is a function of the ENTIRE lookup database
    (``recall_fp.py`` runs ``cdist(matrix, matrix)`` over the whole population),
    so a single full-319 AAC cell is NOT comparable to a capped pLM (esm1b, 267)
    whose recall was scored on its 267-protein DB. Comparing them would confound
    DB size with embedding quality — the same asymmetry the SNN arm shipped a fix
    for.

So the AAC floor is produced **once per distinct population**:

  * a full-319 cell (``expected_ids`` = frozen-319, ``population_tag="full319"``)
    for the 14 full pLMs, and
  * a capped cell (``expected_ids`` = esm1b's covered ids, ``population_tag="esm1b"``)
    for esm1b.

``expected_ids`` IS the population to score on; ``population_tag`` distinguishes the
output so the full-cohort AAC cell and a capped-cohort AAC cell never collide. The
producer scores whatever population it is handed — it does NOT need to know about
esm1b; the caller supplies the id set + tag.

**I3 (filename contract):** mirror recall-fp — the distance is separated by ``out_dir``
(the caller passes a per-distance directory), NOT encoded in the filename. The parquet
stem is ``aac_floor_<population_tag>_<level>.parquet`` (population_tag + level in the
name, distance via the directory).

The per-query parquet schema is the SAME as recall-fp; the column contract and the
barrier guards are imported verbatim from :mod:`evaluation.recall_fp_report`
(``PARQUET_GUARDS`` / ``PER_QUERY_COLUMNS`` / ``CI_NOTE`` …) — one source of truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from evaluation.aac_floor import build_aac_embeddings
from evaluation.analysis_io import json_safe as _json_safe
from evaluation.analysis_io import load_frozen_ids as _load_frozen_ids
from evaluation.label_adapters import load_cath_labels, make_cath_is_positive_fn
from evaluation.population import PopulationError, assert_population
from evaluation.recall_fp import recall_at_first_fp
from evaluation.recall_fp_report import (
    CI_METHOD,
    CI_NOTE,
    CI_RESAMPLE_UNIT,
    DEFAULT_LEVELS,
    PER_QUERY_COLUMNS,
    _recall_ci,
)
from shared.atomic_io import atomic_write


def aac_floor_report(
    fasta_path: Path | str,
    labels: pd.DataFrame,
    out_dir: Path | str,
    *,
    expected_ids: Iterable[str],
    distance: str,
    population_tag: str,
    include_other: bool = False,
    levels: Sequence[str] = DEFAULT_LEVELS,
    allow_capped: bool = False,
    overwrite: bool = True,
    n_boot: int = 10_000,
    ci_alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """Score the AAC floor with recall-at-first-FP on ONE population; write per-level parquet.

    Parameters
    ----------
    fasta_path
        The canonical FASTA (D6 — AAC is computed on the fly, no pre-produced H5).
    labels
        CATH label frame from
        :func:`evaluation.label_adapters.load_cath_labels` (frozenset
        ``fold``/``superfamily`` columns) — the SAME frame the pLMs are scored on.
    out_dir
        **Per-distance** directory the per-level parquet files are written into
        (I3 — the distance is separated by the directory, not the filename). The
        caller passes a directory already scoped to ``distance``.
    expected_ids
        **The population to score on** (the C1 fix). For the 14 full pLMs this is
        the frozen canonical-319; for esm1b it is esm1b's covered subset. The AAC
        vectors are built for exactly this set and the recall is computed against
        this whole lookup DB — so a full-319 AAC cell and a capped-267 AAC cell are
        genuinely different numbers (DB size is load-bearing).
    distance
        ``"cosine"``, ``"euclidean"``, or ``"manhattan"`` — recorded in the manifest
        (it is NOT in the filename, I3). Tie density on discrete frequency vectors is
        highest under euclidean → ``n_ties_at_first_fp`` is load-bearing here.
    population_tag
        A short tag (e.g. ``"full319"`` or ``"esm1b"``) that distinguishes this
        population's cell on disk: it is the only varying token in the parquet stem
        ``aac_floor_<population_tag>_<level>.parquet`` and the sidecar
        ``aac_floor_<population_tag>.manifest.json``. The full-cohort and capped-cohort
        AAC cells therefore never collide even in the same per-distance ``out_dir``.
    include_other
        ``False`` (default) → the true 20-d floor; ``True`` → 21-d with a
        non-standard-AA bucket. Recorded in the manifest (D2).
    levels
        CATH levels to score (default Topology + Homologous-SF; family deferred, W3).
    allow_capped
        Forwarded to :func:`assert_population`: permit a strict subset of
        ``expected_ids`` without failing. (Used when ``expected_ids`` itself is an
        upper bound and the FASTA legitimately covers fewer.)
    overwrite
        If True (**default**), atomic-replace the canonical fixed path in place
        (tmp + ``os.replace`` — killed-job-safe), so the B6 barrier validates a
        fixed spec path.
    n_boot, ci_alpha, seed
        BCa bootstrap CI controls for the per-level mean recall (absolute metric →
        ``B=10_000`` default; ``seed`` makes the CIs byte-reproducible; each level
        draws an independent stream via ``SeedSequence.spawn``).

    Returns
    -------
    dict
        Manifest with ``distance``, ``population_tag``, ``include_other``,
        ``population_n``, CI provenance, and ``levels`` (per level: ``path``,
        ``n_queries_with_positives``, ``n_queries_skipped_no_positives``,
        ``n_scored``, ``mean_recall_1stFP``, ``ci_lo``, ``ci_hi``, ``ci_degenerate``,
        and ``n_ties_at_first_fp`` — the floor-quality readout, the total positives
        discarded at the first-FP distance over scored queries).

    Raises
    ------
    ValueError
        From :func:`build_aac_embeddings`: a frozen id absent from the FASTA, or an
        empty population.
    evaluation.population.PopulationError
        If the built AAC population drifts from ``expected_ids`` (and not
        ``allow_capped``) — raised before any parquet is written.
    """
    out_dir = Path(out_dir)
    exp = list(expected_ids)

    # D6: build AAC vectors on the fly from the canonical FASTA, subset to the
    # FASTA-covered slice of `exp` so a capped population (allow_capped) does not
    # ValueError on a legitimately-absent id; assert_population then enforces drift.
    if allow_capped:
        present = _fasta_ids(fasta_path)
        build_ids = [pid for pid in exp if pid in present]
        if not build_ids:
            raise ValueError(
                f"no expected id is present in {Path(fasta_path).name!r}; "
                f"refusing to score an empty AAC population."
            )
    else:
        build_ids = exp
    embeddings = build_aac_embeddings(
        fasta_path, expected_ids=build_ids, include_other=include_other
    )

    # S3 / C1: assert the AAC population against `expected_ids` BEFORE scoring, so a
    # drifted population fails loudly rather than scoring a different cohort than the
    # pLM it will be compared against.
    assert_population(
        embeddings.keys(), set(exp), name=f"aac:{population_tag}", allow_capped=allow_capped
    )

    mode = "replace" if overwrite else "timestamp"
    level_seeds = np.random.SeedSequence(seed).spawn(len(levels))
    out: dict = {
        "floor": "aac",
        "population_tag": population_tag,
        "distance": distance,
        "include_other": include_other,
        "population_n": len(embeddings),
        "ci_alpha": ci_alpha,
        "n_boot": n_boot,
        "seed": seed,
        "ci_method": CI_METHOD,
        "ci_resample_unit": CI_RESAMPLE_UNIT,
        "ci_note": CI_NOTE,
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
        # Floor-quality readout (D7): total positives discarded at the first-FP
        # distance over scored queries — the discrete-frequency tie signal AAC's
        # tie handling was built for. Surfaced, not re-implemented (recall_fp owns it).
        n_ties_summary = (
            int(per_query["n_ties_at_first_fp"].sum()) if len(per_query) else 0
        )
        target = out_dir / f"aac_floor_{population_tag}_{level}.parquet"
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
            "n_ties_at_first_fp": n_ties_summary,
        }
    return out


def _fasta_ids(fasta_path: Path | str) -> set[str]:
    """The set of protein ids present in the FASTA (for the capped-build subset)."""
    from evaluation.canonical_set import parse_fasta

    return {pid for pid, _ in parse_fasta(fasta_path)}


# ── CLI: the analysis-DAG AAC-floor step ──────────────────────────────────────
def main(argv: Sequence[str] | None = None) -> int:
    """CLI wrapper: score one (population, distance) AAC-floor cell + write the sidecar.

    Loads the cath_labels TSV and the committed canonical-set freeze, builds AAC
    vectors from the FASTA, calls :func:`aac_floor_report` (which writes the per-level
    parquets), then persists the returned manifest as a sidecar JSON — the report itself
    deliberately does NOT write the sidecar (the barrier spec-builder reads it). Exit
    codes mirror :mod:`evaluation.recall_fp_report`:

    * ``0`` — scored and wrote the parquets + sidecar.
    * ``1`` — population drift (a built AAC population short of the frozen set, not
      flagged capped): a *data* failure; nothing is written.
    * ``2`` — operator/config fault (missing input file, malformed freeze, a frozen id
      absent from the FASTA, bad column).
    """
    ap = argparse.ArgumentParser(
        prog="aac_floor_report",
        description="Score the 20-d amino-acid-composition floor with recall-at-first-FP "
        "against ONE frozen population (Topology/SF) and write a per-level parquet + a "
        "manifest sidecar JSON. The population (--freeze) and its --population-tag are the "
        "C1 capped-cohort fix: a full-319 cell and a capped-267 cell are scored separately.",
    )
    ap.add_argument(
        "--fasta", required=True, help="Canonical FASTA (AAC is computed on the fly, D6)."
    )
    ap.add_argument("--cath-tsv", required=True, help="cath_labels TSV (UniProt Gene3D export).")
    ap.add_argument(
        "--freeze",
        required=True,
        help="Committed canonical-set freeze JSON; its 'ids' are the population to score on "
        "(C1 — full-319 for full pLMs, esm1b's covered ids for the capped cell).",
    )
    ap.add_argument(
        "--population-tag",
        required=True,
        help="Short tag distinguishing this population's cell on disk (e.g. full319, esm1b) "
        "so the full-cohort and capped-cohort AAC cells never collide.",
    )
    ap.add_argument("--out-dir", required=True, help="Per-distance directory for the parquets + sidecar.")
    ap.add_argument(
        "--distance",
        required=True,
        choices=("cosine", "euclidean", "manhattan"),
        help="Retrieval metric (required; recorded in the manifest, separated by out_dir not "
        "the filename — I3). Tie density on AAC frequencies is highest under euclidean.",
    )
    ap.add_argument(
        "--include-other",
        action="store_true",
        help="Use the 21-d '+other' variant (non-standard AAs in bucket 20); default = 20-d floor.",
    )
    ap.add_argument(
        "--levels",
        nargs="+",
        choices=("fold", "superfamily", "family"),
        default=list(DEFAULT_LEVELS),
        help=f"CATH levels to score (default {' '.join(DEFAULT_LEVELS)}; family deferred, W3).",
    )
    ap.add_argument(
        "--allow-capped",
        action="store_true",
        help="Permit a strict subset of the freeze ids (the population legitimately covers fewer).",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for the BCa bootstrap CIs (default 42; reproducible interval).",
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

    overwrite = True
    try:
        labels = load_cath_labels(args.cath_tsv)
        expected_ids = _load_frozen_ids(args.freeze)
        manifest = aac_floor_report(
            args.fasta,
            labels,
            args.out_dir,
            expected_ids=expected_ids,
            distance=args.distance,
            population_tag=args.population_tag,
            include_other=args.include_other,
            levels=args.levels,
            allow_capped=args.allow_capped,
            overwrite=overwrite,
            n_boot=args.n_boot,
            ci_alpha=args.ci_alpha,
            seed=args.seed,
        )
    except PopulationError as e:
        print(f"aac_floor_report: POPULATION DRIFT: {e}", file=sys.stderr, flush=True)
        return 1
    except (FileNotFoundError, OSError) as e:
        print(f"aac_floor_report: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, KeyError) as e:
        print(f"aac_floor_report: INPUT ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    sidecar = Path(args.out_dir) / f"aac_floor_{args.population_tag}.manifest.json"
    written = atomic_write(
        sidecar,
        lambda p: p.write_text(json.dumps(_json_safe(manifest), indent=2) + "\n"),
        mode="replace" if overwrite else "timestamp",
    )
    print(
        f"aac_floor_report: {args.population_tag} (n={manifest['population_n']}) "
        f"-> {written}",
        flush=True,
    )
    for level, info in manifest["levels"].items():
        print(
            f"  {level}: n_scored={info['n_scored']} "
            f"mean_recall_1stFP={info['mean_recall_1stFP']} "
            f"[{info['ci_lo']}, {info['ci_hi']}] ties={info['n_ties_at_first_fp']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
