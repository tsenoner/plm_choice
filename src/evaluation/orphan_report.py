"""Orphan arm: per-pLM sibling-AUROC of embedding cosine, with a vertex-BCa CI.

The 4th clone of recall_fp_report / snn_report / ec_report (library fn -> per-cell
parquet + manifest sidecar -> 3-exit-code CLI). The arm's *value* ships here: it turns
the embeddings dict + the Bromberg orphan pairs into

* the headline ``siblings_AUROC`` (sklearn roc_auc over the pairs' own sibling column),
* its **vertex-bootstrap BCa CI** (the dyadic-dependence CI, :mod:`orphan_auroc_ci`),
* the naive i.i.d.-pair ``bca_bootstrap`` CI recorded as a FLAGGED, deliberately
  anticonservative comparison,
* the two secondary Spearman ρ (cos vs SNN, cos vs TM).

Reframing (design §11-R2): orphan sequences cap at 400 aa, so truncation cannot have
affected this cohort — this is a **clean re-verification with BCa CIs the stale JSON
never had**, NOT a truncation fix. ProtTucker's recomputed point should land at/near
its published 0.732, not move materially.

Provenance guard (design §11-R1): RELAXED to a defensive WARN. If the H5 carries a
``max_length_cap`` root attr indicating capping, warn loudly; if absent (the orphan
extract never stamped it), proceed normally — do NOT fail-closed. The population/drift
guard (exit 1) is the hard gate; the cap guard is a no-op warning.

The CLI ``main`` writes the sidecar; this library writes only the parquet — the same
"lenient library, strict barrier" split the other arms use.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from evaluation.analysis_io import json_safe, load_embeddings_h5, load_frozen_ids
from evaluation.orphan_auroc_ci import orphan_auroc_vertex_bca_ci
from evaluation.orphan_io import load_orphan_pairs
from evaluation.orphan_score import score_orphan_pairs

# ── per-pair parquet contract (D9; the orphan analogue of EC_PARQUET_GUARDS) ──────────
# pair_key = p1 + "\t" + p2 is the synthetic single-column unique key (the barrier's
# unique_columns guard is single-column; encoding the (p1, p2) pair into one column
# sidesteps any 2-column-key ambiguity while keeping p1/p2 for downstream use).
ORPHAN_PER_PAIR_COLUMNS: tuple[str, ...] = (
    "pair_key", "p1", "p2", "cos", "snn", "tm", "sibling",
)
ORPHAN_PARQUET_GUARDS: dict[str, tuple[str, ...]] = {
    "required_columns": ORPHAN_PER_PAIR_COLUMNS,
    "unique_columns": ("pair_key",),
    "non_null_columns": ("pair_key", "p1", "p2", "sibling"),
    "finite_columns": ("cos", "snn", "tm"),
}

_CI_NOTE = (
    "siblings_AUROC CI is a paired vertex (per-orphan) bootstrap BCa interval — the "
    "Bromberg pairs are dyadic (both endpoints in the orphan set), so pairs sharing an "
    "orphan are correlated and an i.i.d.-pair resample is anticonservative; naive_ci_* "
    "is that i.i.d.-pair CI, recorded as a deliberately-anticonservative comparison only. "
    "Orphan sequences cap at 400 aa so truncation cannot affect this cohort: this is a "
    "clean re-verification with CIs, not a truncation fix (design R2)."
)


class OrphanPopulationError(RuntimeError):
    """A pLM is silently missing frozen orphan ids and was not flagged capped."""


def _stem(plm: str, representation: str, distance: str) -> str:
    return f"orphan_{plm}_{representation}_{distance}"


def _naive_pair_bca_ci(per_pair: pd.DataFrame, *, n_boot: int, alpha: float, seed: int):
    """The naive i.i.d.-pair bootstrap CI for AUROC (the FLAGGED comparison field).

    Resamples ROWS (pairs) i.i.d. — anticonservative because pairs are dyadic. Returns
    ``(lo, hi)`` or ``(nan, nan)`` if the AUROC is degenerate. Skips draws that lose a
    class (one-class roc_auc is undefined), mirroring the vertex CI's NaN policy.
    """
    from sklearn.metrics import roc_auc_score

    cos = per_pair["cos"].to_numpy(dtype=float)
    sib = per_pair["sibling"].to_numpy().astype(bool)
    m = cos.size
    if m < 2 or sib.all() or not sib.any():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sel = rng.integers(0, m, size=m)
        s = sib[sel]
        if s.all() or not s.any():
            continue
        boot.append(roc_auc_score(s, cos[sel]))
    if not boot:
        return float("nan"), float("nan")
    boot = np.asarray(boot)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return lo, hi


def orphan_correlation_report(
    embeddings: dict,
    pairs: pd.DataFrame,
    out_dir: Path | str,
    *,
    plm: str,
    expected_ids: list[str] | None = None,
    representation: str = "raw",
    distance: str = "cosine",
    allow_capped: bool = False,
    seed: int = 42,
    n_boot: int = 2000,
    ci_alpha: float = 0.05,
    overwrite: bool = True,
) -> dict:
    """Score one orphan cell (one pLM, cosine): write the per-pair parquet, return manifest.

    ``pairs`` is the :func:`orphan_io.load_orphan_pairs` frame
    (``[p1, p2, tm, snn, sibling]``). ``expected_ids`` (if given) is the frozen orphan
    cohort: a frozen id missing from ``embeddings`` raises :class:`OrphanPopulationError`
    unless ``allow_capped`` (the population-drift contract shared with the other arms).
    """
    from shared.atomic_io import atomic_write

    if expected_ids is not None:
        missing = [pid for pid in expected_ids if pid not in embeddings]
        if missing and not allow_capped:
            raise OrphanPopulationError(
                f"{len(missing)} frozen orphan id(s) missing from embeddings "
                f"(e.g. {missing[:3]}); pass allow_capped for an arch-capped pLM."
            )

    per_pair, scalars = score_orphan_pairs(embeddings, pairs)

    # Vertex-bootstrap BCa CI (the dyadic-dependence CI) on the sibling AUROC.
    ci = orphan_auroc_vertex_bca_ci(
        per_pair, n_boot=n_boot, alpha=ci_alpha, seed=seed,
    )
    naive_lo, naive_hi = _naive_pair_bca_ci(
        per_pair, n_boot=n_boot, alpha=ci_alpha, seed=seed,
    )

    # Per-pair parquet (synthetic single-column unique key).
    pq_df = per_pair.copy()
    pq_df.insert(0, "pair_key", pq_df["p1"].astype(str) + "\t" + pq_df["p2"].astype(str))
    pq_df = pq_df[list(ORPHAN_PER_PAIR_COLUMNS)]
    out_dir = Path(out_dir)
    pq_path = out_dir / f"{_stem(plm, representation, distance)}.parquet"
    if pq_path.exists() and not overwrite:
        raise FileExistsError(f"{pq_path} exists; pass overwrite=True")
    written_pq = atomic_write(
        pq_path, lambda p: pq_df.to_parquet(p, index=False), mode="replace",
    )

    manifest = {
        "plm": plm,
        "representation": representation,
        "distance": distance,
        "siblings_AUROC": float(ci["point"]),
        "ci_lo": float(ci["ci_lo"]),
        "ci_hi": float(ci["ci_hi"]),
        "ci_degenerate": bool(ci["degenerate"]),
        "percentile_diverged": bool(ci["diverged"]),
        "n_boot_undefined": int(ci["n_boot_undefined"]),
        "naive_ci_lo": float(naive_lo),
        "naive_ci_hi": float(naive_hi),
        "spearman_cos_vs_SNN": float(scalars["spearman_cos_vs_SNN"]),
        "spearman_cos_vs_TM": float(scalars["spearman_cos_vs_TM"]),
        "n_pairs": int(scalars["n_pairs"]),
        "n_pairs_dropped": int(scalars["n_pairs_dropped"]),
        "n_siblings": int(scalars["n_siblings"]),
        "n_proteins": int(scalars["n_proteins"]),
        "population_n": int(scalars["n_proteins"]),
        "seed": seed,
        "n_boot": n_boot,
        "ci_alpha": ci_alpha,
        "ci_note": _CI_NOTE,
        "per_pair_columns": list(ORPHAN_PER_PAIR_COLUMNS),
        "path": str(written_pq),
    }
    return manifest


def _warn_if_capped(emb_h5: Path | str) -> None:
    """Relaxed provenance guard (R1): WARN loudly if the H5 stamps a length cap.

    Reads the HDF5 root attrs; if ``max_length_cap`` is present and not None, the H5 may
    carry truncated embeddings -> emit a loud :class:`UserWarning`. If the marker is
    ABSENT (the orphan extract never stamped it; orphan seqs <=400 aa make truncation
    impossible), proceed silently. NEVER fail-closed.
    """
    import h5py

    try:
        with h5py.File(emb_h5, "r") as f:
            cap = f.attrs.get("max_length_cap", None)
    except OSError:
        return  # the report's own load will surface the real I/O error as exit 2
    if cap is not None:
        warnings.warn(
            f"embedding H5 {emb_h5} carries max_length_cap={cap!r}: the vectors may be "
            f"truncated. Orphan sequences cap at 400 aa so truncation should be "
            f"impossible — investigate the extract before trusting the AUROC.",
            UserWarning,
            stacklevel=2,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: score one orphan cell + sidecar. Exit 0 / 1 (drift) / 2 (config / I/O)."""
    from shared.atomic_io import atomic_write

    ap = argparse.ArgumentParser(
        prog="orphan_report",
        description="Score one pLM's orphan sibling-AUROC (embedding cosine vs the "
        "Bromberg sibling labels) with a vertex-bootstrap BCa CI; write a per-pair "
        "parquet + manifest sidecar.",
    )
    ap.add_argument("--plm", required=True)
    ap.add_argument("--emb-h5", required=True)
    ap.add_argument("--pairs", required=True,
                    help="Bromberg orphan_sibling_score.tsv(.gz) pairs file.")
    ap.add_argument("--freeze", required=True,
                    help="Frozen orphan cohort JSON (its 'ids'); the population gate.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--representation", default="raw")
    ap.add_argument("--allow-capped", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ci-alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    try:
        _warn_if_capped(args.emb_h5)  # relaxed provenance WARN (never gates)
        embeddings = load_embeddings_h5(args.emb_h5)
        expected_ids = load_frozen_ids(args.freeze)
        pairs = load_orphan_pairs(args.pairs)  # strict: self-pairs + malformed rows raise
        manifest = orphan_correlation_report(
            embeddings, pairs, args.out_dir,
            plm=args.plm, representation=args.representation, distance="cosine",
            expected_ids=expected_ids, allow_capped=args.allow_capped,
            seed=args.seed, n_boot=args.n_boot, ci_alpha=args.ci_alpha, overwrite=True,
        )
    except OrphanPopulationError as e:
        print(f"orphan_report: POPULATION DRIFT: {e}", file=sys.stderr, flush=True)
        return 1
    except (FileNotFoundError, OSError) as e:
        print(f"orphan_report: I/O ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, KeyError) as e:
        print(f"orphan_report: INPUT ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    sidecar = Path(args.out_dir) / f"{_stem(args.plm, args.representation, 'cosine')}.manifest.json"
    written = atomic_write(
        sidecar,
        lambda p: p.write_text(json.dumps(json_safe(manifest), indent=2) + "\n"),
        mode="replace",
    )
    print(
        f"orphan_report: {args.plm} (n={manifest['n_proteins']} "
        f"pairs={manifest['n_pairs']} sib={manifest['n_siblings']}) "
        f"AUROC={manifest['siblings_AUROC']} "
        f"[{manifest['ci_lo']}, {manifest['ci_hi']}] -> {written}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
