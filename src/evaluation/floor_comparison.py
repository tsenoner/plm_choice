"""pLM-vs-AAC-floor one-sided comparison: the paper claim (Unit 4 of the AAC-floor arm).

This is the **net-new science** of the AAC-floor arm (spec §3 Unit 4). No shipped arm
emits a cross-method comparison — recall-fp and the AAC producer each emit only *per-
method* numbers. This module reads the pLM's recall-fp per-query parquet AND the
**population-matched** AAC-floor per-query parquet, and answers the Figure-1 claim:

    "pLM X beats the AAC floor — one-sided, significant (after Holm correction)."

Two correctness pillars (spec §6, §10):

1. **Direction (D4).** The claim is directional: the test is the *one-sided*
   ``paired_wilcoxon(recall_plm, recall_aac, alternative="greater")``. A two-sided test
   would be wrong. The Cliff's-δ effect size travels with it (``paired_wilcoxon`` bundles
   both — M7: it returns ``{statistic, p_value, cliffs_delta}``; there is no ``p_one_sided``
   key on the return, so the manifest remaps ``p_value`` → ``p_one_sided``).

2. **Population-matching (C1, D11).** recall-at-first-FP is a function of the *entire*
   lookup DB, so a capped pLM (esm1b, 267/319) must be compared against an AAC floor
   re-scored on that same 267-protein DB — NOT the full-319 AAC inner-joined down. The
   **caller** supplies which AAC parquet is the population match (full319 AAC for a full
   pLM, esm1b AAC for esm1b); this report just consumes the two paths it is given. The
   inner-join on ``query_id`` then enforces the paired alignment, dropping any query the
   (capped) pLM lacks — never fabricating a row.

**I4 (CRITICAL).** Δ = recall_plm − recall_aac ∈ [−1, 1]. Its CI is computed with
``stats.bca_bootstrap(..., paired=...)`` with **NO (0, 1) clip** — routing it through
``bounded_mean_bca_ci`` (whose default ``clip=(0, 1)``) would truncate a negative lower
bound and silently hide a "floor NOT beaten" result. The pLM/AAC *recall* CIs (computed
upstream in the producers) are unit-range and clipped; the paired *Δ* is signed and is not.

**Multiplicity (D10).** The 15-pLM floor tests for a fixed (distance, level) form a
family → Holm-Bonferroni. The per-cell manifest carries the **raw** one-sided p; the
corrected ``beats_floor`` verdict is a fold-over-pLMs pass (``--apply-holm`` mode, the
self-contained option Ivan chose) that reads all per-pLM sidecars and writes one family
verdict file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from evaluation.analysis_io import json_safe as _json_safe
from evaluation.stats import bca_bootstrap, holm_bonferroni, paired_wilcoxon
from shared.atomic_io import atomic_write

# The comparison per-query parquet schema + its barrier guards — the second, NEW
# per-query contract this arm introduces (the AAC producer reuses recall-fp's verbatim;
# the comparison frame is genuinely different: it carries BOTH methods' recall + their
# signed difference). Mirrors recall_fp_report.PARQUET_GUARDS in shape.
FLOOR_COMPARISON_COLUMNS: tuple[str, ...] = (
    "query_id",
    "recall_plm",
    "recall_aac",
    "delta",
)
FLOOR_PARQUET_GUARDS: dict[str, tuple[str, ...]] = {
    "required_columns": FLOOR_COMPARISON_COLUMNS,
    "unique_columns": ("query_id",),
    "non_null_columns": ("query_id",),
    "finite_columns": ("delta",),
}

# CI provenance for the paired Δ. The paired query resample preserves the within-query
# pairing (the same query's pLM and AAC recalls move together), so it removes the
# common-mode that is SHARED across the two arms for a given query. It does NOT remove
# the shared-retrieval-DB non-i.i.d. structure WITHIN each arm (every query is also a DB
# entry for the others) — record that honestly so a figure caption never overclaims.
CI_METHOD = "BCa bootstrap on paired Δ (recall_plm − recall_aac), query-level resample"
CI_RESAMPLE_UNIT = "query"
CI_NOTE = (
    "Paired query-level bootstrap on Δ = recall_plm − recall_aac. The pairing is kept "
    "(both arms' per-query recalls are resampled by the same query index), so the "
    "interval removes the SHARED-across-arms common mode but NOT the within-arm "
    "shared-retrieval-DB non-i.i.d. structure (each query is also a DB entry for the "
    "others). Treat the interval as query-sampling variability on the paired difference, "
    "not as fully propagating retrieval-DB uncertainty. Δ ∈ [−1, 1] is signed and the "
    "interval is NOT clipped — a negative lower bound is a real 'floor not beaten' signal."
)


def _load_recall(path: Path | str) -> pd.DataFrame:
    """Read a recall-fp-shaped per-query parquet → ``[query_id, recall]`` (deduped check).

    Raises ``FileNotFoundError`` if absent (CLI maps to exit 2), ``ValueError`` if the
    required columns are missing or ``query_id`` is non-unique (a malformed upstream
    artifact must fail loudly, not silently mis-join).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"per-query parquet not found: {p}")
    df = pd.read_parquet(p)
    for col in ("query_id", "recall"):
        if col not in df.columns:
            raise ValueError(f"{p.name}: missing required column {col!r}")
    if df["query_id"].duplicated().any():
        raise ValueError(f"{p.name}: duplicate query_id — cannot pair a non-unique key")
    return df[["query_id", "recall"]]


def floor_comparison_report(
    plm_per_query_path: Path | str,
    aac_per_query_path: Path | str,
    out_dir: Path | str,
    *,
    plm: str,
    distance: str,
    level: str,
    alternative: str = "greater",
    seed: int = 42,
    n_boot: int = 10_000,
    ci_alpha: float = 0.05,
    overwrite: bool = True,
) -> dict:
    """Compare one pLM's recall-fp cell against the population-matched AAC floor.

    Parameters
    ----------
    plm_per_query_path
        The recall-fp per-query parquet for (pLM, distance, level)
        (``recall_fp_<plm>_<rep>_<level>.parquet``; distance is separated by its
        out-dir — I3).
    aac_per_query_path
        The **population-matched** AAC-floor per-query parquet for the matching
        (distance, level). The CALLER picks the right population cell (full319 AAC for a
        full pLM, esm1b AAC for esm1b — the C1 correctness); this report consumes the
        path it is given. The two parquets are inner-joined on ``query_id``, so a capped
        pLM compares only against the AAC scores for its own queries.
    out_dir
        Per-(distance, level) directory the comparison parquet + sidecar land in. The
        caller scopes it to distance (mirroring the producers, I3).
    plm
        pLM name — used in the comparison filename + manifest.
    distance, level
        Recorded in the manifest; ``distance`` is separated by ``out_dir``.
    alternative
        Wilcoxon alternative; ``"greater"`` (default) is the directional floor test
        (pLM recall > AAC recall). Recorded in the manifest.
    seed, n_boot, ci_alpha
        Paired BCa bootstrap controls for the mean-Δ CI. ``seed`` makes the interval
        byte-reproducible.
    overwrite
        Atomic in-place replace (DAG default) vs never-clobber timestamp.

    Returns
    -------
    dict
        Manifest with ``plm, distance, level, n_paired, mean_recall_plm,
        mean_recall_aac, mean_delta, delta_ci_lo, delta_ci_hi, wilcoxon_statistic,
        p_one_sided, cliffs_delta, alternative, seed, n_boot``, the CI provenance, and
        ``comparison_path`` (the per-query parquet written).

    Raises
    ------
    FileNotFoundError
        A per-query parquet is absent (CLI → exit 2).
    ValueError
        Malformed input columns, or the inner-join is EMPTY (no overlapping queries —
        a population mismatch; CLI → exit 1).
    """
    out_dir = Path(out_dir)

    plm_df = _load_recall(plm_per_query_path).rename(columns={"recall": "recall_plm"})
    aac_df = _load_recall(aac_per_query_path).rename(columns={"recall": "recall_aac"})

    # D11: inner-join on query_id — a capped pLM only has its own queries; the join
    # drops any AAC query the pLM lacks (and vice versa). No fabricated rows.
    merged = plm_df.merge(aac_df, on="query_id", how="inner")
    if merged.empty:
        raise ValueError(
            f"no overlapping query_id between {Path(plm_per_query_path).name!r} and "
            f"{Path(aac_per_query_path).name!r} — empty paired set (population mismatch; "
            f"did the caller pass the population-matched AAC cell?)."
        )
    # Deterministic order so the parquet + the paired bootstrap are reproducible.
    merged = merged.sort_values("query_id").reset_index(drop=True)
    merged["delta"] = merged["recall_plm"] - merged["recall_aac"]

    recall_plm = merged["recall_plm"].to_numpy(dtype=float)
    recall_aac = merged["recall_aac"].to_numpy(dtype=float)
    delta = merged["delta"].to_numpy(dtype=float)

    # D4 / M7: one-sided paired Wilcoxon + Cliff's δ, bundled in one return. Remap
    # p_value -> p_one_sided (no p_one_sided key on the return).
    wil = paired_wilcoxon(recall_plm, recall_aac, alternative=alternative)

    # I4: paired BCa CI on mean Δ via bca_bootstrap(paired=...) — NO (0,1) clip. The
    # paired= arg is the SECOND paired array; row indices are resampled once and applied
    # to both, so the pairing is preserved. The statistic receives a (n, 2) array whose
    # columns are (recall_plm, recall_aac) in that order -> mean of the difference.
    if delta.size >= 4 and float(np.ptp(delta)) > 0:
        _, ci_lo, ci_hi = bca_bootstrap(
            recall_plm,
            lambda pair: float(np.mean(pair[:, 0] - pair[:, 1])),
            B=n_boot,
            alpha=ci_alpha,
            paired=recall_aac,
            rng=np.random.default_rng(seed),
        )
    else:
        # Too few queries or a constant Δ -> the bootstrap is inapplicable; report the
        # point as a degenerate interval (NOT a coverage statement) rather than NaN garbage.
        c = float(np.mean(delta))
        ci_lo, ci_hi = c, c

    out: dict = {
        "plm": plm,
        "distance": distance,
        "level": level,
        "n_paired": int(merged.shape[0]),
        "mean_recall_plm": float(np.mean(recall_plm)),
        "mean_recall_aac": float(np.mean(recall_aac)),
        "mean_delta": float(np.mean(delta)),
        "delta_ci_lo": float(ci_lo),
        "delta_ci_hi": float(ci_hi),
        "wilcoxon_statistic": float(wil["statistic"]),
        "p_one_sided": float(wil["p_value"]),
        "cliffs_delta": float(wil["cliffs_delta"]),
        "alternative": alternative,
        "seed": seed,
        "n_boot": n_boot,
        "ci_alpha": ci_alpha,
        "ci_method": CI_METHOD,
        "ci_resample_unit": CI_RESAMPLE_UNIT,
        "ci_note": CI_NOTE,
        "comparison_columns": list(FLOOR_COMPARISON_COLUMNS),
    }

    mode = "replace" if overwrite else "timestamp"
    target = out_dir / f"floor_cmp_{plm}_{level}.parquet"
    frame = merged[list(FLOOR_COMPARISON_COLUMNS)]
    written = atomic_write(
        target,
        lambda p, df=frame: df.to_parquet(p, index=False),
        mode=mode,
    )
    out["comparison_path"] = str(written)
    return out


# ── Holm family driver (D10): fold over the per-pLM sidecars for a fixed cell ──
def apply_holm_family(
    sidecar_dir: Path | str,
    *,
    distance: str,
    level: str,
    alpha: float = 0.05,
    overwrite: bool = True,
) -> dict:
    """Holm-correct the per-pLM one-sided p-values for a fixed (distance, level) family.

    Reads every ``floor_cmp_<plm>_raw.manifest.json`` in ``sidecar_dir`` whose
    ``distance``/``level`` match, applies :func:`stats.holm_bonferroni` across the
    (≤15) raw one-sided p-values, and writes one family-verdict JSON setting
    ``beats_floor`` (bool) per pLM from the corrected p. The per-cell manifests carry
    the RAW p; this is the corrected family pass (option (a), self-contained).

    Returns ``{"distance", "level", "alpha", "n_tests", "verdicts": [...],
    "verdict_path"}`` where each verdict is ``{plm, p_one_sided, p_adj, beats_floor,
    mean_delta, cliffs_delta}``.

    Raises ``ValueError`` if no matching sidecar is found (an empty family is a wiring
    fault, not a silent pass).
    """
    sidecar_dir = Path(sidecar_dir)
    records: list[dict] = []
    for path in sorted(sidecar_dir.glob("floor_cmp_*_raw.manifest.json")):
        m = json.loads(path.read_text())
        if m.get("distance") != distance or m.get("level") != level:
            continue
        records.append(m)
    if not records:
        raise ValueError(
            f"no floor_cmp_*_raw.manifest.json for (distance={distance!r}, level={level!r}) "
            f"in {sidecar_dir} — empty Holm family (wiring fault)."
        )

    p_values = np.array([float(m["p_one_sided"]) for m in records], dtype=float)
    rejected, adjusted = holm_bonferroni(p_values, alpha=alpha)

    verdicts = []
    for m, rej, padj in zip(records, rejected, adjusted):
        verdicts.append(
            {
                "plm": m["plm"],
                "p_one_sided": float(m["p_one_sided"]),
                "p_adj": float(padj),
                "beats_floor": bool(rej),
                "mean_delta": m.get("mean_delta"),
                "cliffs_delta": m.get("cliffs_delta"),
            }
        )
    # Stable, human-friendly order: most-significant first.
    verdicts.sort(key=lambda v: v["p_adj"])

    out: dict = {
        "distance": distance,
        "level": level,
        "alpha": alpha,
        "n_tests": len(records),
        "correction": "holm",
        "verdicts": verdicts,
    }
    mode = "replace" if overwrite else "timestamp"
    target = sidecar_dir / f"floor_family_verdict_{distance}_{level}.json"
    written = atomic_write(
        target,
        lambda p: p.write_text(json.dumps(_json_safe(out), indent=2) + "\n"),
        mode=mode,
    )
    out["verdict_path"] = str(written)
    return out


# ── CLI: per-cell comparison (default) + --apply-holm family mode ─────────────
def main_holm(argv: Sequence[str] | None = None) -> int:
    """``--apply-holm`` mode: fold the per-pLM sidecars into one family verdict.

    * ``0`` — wrote the family verdict.
    * ``2`` — operator fault (missing dir, empty family, malformed sidecar).
    """
    ap = argparse.ArgumentParser(
        prog="floor_comparison --apply-holm",
        description="Holm-correct the per-pLM one-sided p-values for one (distance, level) "
        "family and write the family verdict (beats_floor per pLM from the corrected p).",
    )
    ap.add_argument("--apply-holm", action="store_true", help="Run the Holm family pass.")
    ap.add_argument("--sidecar-dir", required=True, help="Dir of floor_cmp_*_raw.manifest.json.")
    ap.add_argument("--distance", required=True, choices=("cosine", "euclidean", "manhattan"))
    ap.add_argument("--level", required=True, choices=("fold", "superfamily", "family"))
    ap.add_argument("--alpha", type=float, default=0.05, help="Family-wise alpha (default 0.05).")
    args = ap.parse_args(argv)
    try:
        out = apply_holm_family(
            args.sidecar_dir, distance=args.distance, level=args.level, alpha=args.alpha
        )
    except (FileNotFoundError, OSError) as e:
        print(f"floor_comparison: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, KeyError) as e:
        print(f"floor_comparison: INPUT ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    n_beat = sum(1 for v in out["verdicts"] if v["beats_floor"])
    print(
        f"floor_comparison[holm]: {args.distance}/{args.level} "
        f"{n_beat}/{out['n_tests']} pLMs beat the floor -> {out['verdict_path']}",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: one pLM-vs-AAC comparison cell + sidecar (default), or ``--apply-holm`` family.

    Default (per-cell) exit codes:

    * ``0`` — compared and wrote the parquet + sidecar.
    * ``1`` — population mismatch (empty inner-join: the pLM and AAC share no query —
      a *data* failure, e.g. the wrong-population AAC was passed); nothing is written.
    * ``2`` — operator/config fault (missing input parquet, malformed columns).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--apply-holm" in argv:
        return main_holm(argv)

    ap = argparse.ArgumentParser(
        prog="floor_comparison",
        description="One-sided paired pLM-vs-AAC-floor comparison for one (pLM, distance, "
        "level): inner-join the per-query recalls, Δ = pLM − AAC, paired Wilcoxon "
        "(alternative=greater) + Cliff's δ + paired BCa CI on mean Δ. The CALLER must pass "
        "the POPULATION-MATCHED AAC parquet (C1): full319 AAC for a full pLM, esm1b AAC for "
        "esm1b. Run --apply-holm afterward to fold the per-pLM sidecars into the family verdict.",
    )
    ap.add_argument("--apply-holm", action="store_true", help="Switch to the Holm family pass.")
    ap.add_argument("--plm-per-query", required=True, help="Recall-fp per-query parquet for the pLM.")
    ap.add_argument(
        "--aac-per-query", required=True,
        help="POPULATION-MATCHED AAC-floor per-query parquet (caller picks the right cell, C1).",
    )
    ap.add_argument("--out-dir", required=True, help="Per-(distance,level) dir for the parquet + sidecar.")
    ap.add_argument("--plm", required=True, help="pLM name (used in filenames + manifest).")
    ap.add_argument("--distance", required=True, choices=("cosine", "euclidean", "manhattan"))
    ap.add_argument("--level", required=True, choices=("fold", "superfamily", "family"))
    ap.add_argument(
        "--alternative", default="greater", choices=("greater", "less", "two-sided"),
        help="Wilcoxon alternative (default greater = directional floor test).",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for the paired BCa CI (default 42).")
    ap.add_argument("--n-boot", type=int, default=10_000, help="Paired BCa resamples (default 10000).")
    ap.add_argument("--ci-alpha", type=float, default=0.05, help="CI coverage error (default 0.05 -> 95%%).")
    args = ap.parse_args(argv)

    try:
        manifest = floor_comparison_report(
            args.plm_per_query,
            args.aac_per_query,
            args.out_dir,
            plm=args.plm,
            distance=args.distance,
            level=args.level,
            alternative=args.alternative,
            seed=args.seed,
            n_boot=args.n_boot,
            ci_alpha=args.ci_alpha,
        )
    except ValueError as e:
        # An empty inner-join is a population mismatch (data failure) -> exit 1; any
        # other ValueError (malformed columns) is an operator fault -> exit 2.
        msg = str(e)
        if "no overlapping" in msg or "empty paired" in msg:
            print(f"floor_comparison: POPULATION MISMATCH: {e}", file=sys.stderr, flush=True)
            return 1
        print(f"floor_comparison: INPUT ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except (FileNotFoundError, OSError) as e:
        print(f"floor_comparison: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except KeyError as e:
        print(f"floor_comparison: INPUT ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    sidecar = Path(args.out_dir) / f"floor_cmp_{args.plm}_raw.manifest.json"
    written = atomic_write(
        sidecar,
        lambda p: p.write_text(json.dumps(_json_safe(manifest), indent=2) + "\n"),
        mode="replace",
    )
    print(
        f"floor_comparison: {args.plm} {args.distance}/{args.level} "
        f"(n_paired={manifest['n_paired']}) Δ={manifest['mean_delta']:.4f} "
        f"[{manifest['delta_ci_lo']:.4f}, {manifest['delta_ci_hi']:.4f}] "
        f"p={manifest['p_one_sided']:.3g} δ={manifest['cliffs_delta']:.3f} -> {written}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
