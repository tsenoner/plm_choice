"""Unit 7 — cross-pLM agreement-matrix assembly + Holm over the 6 families.

The reduce step of the cross-pLM arm: read the per-pair sidecars
:func:`evaluation.cross_plm_report.main` wrote and produce the **Supplementary
agreement-matrix material** —

* the symmetric ``len(plms) × len(plms)`` agreement matrices, one per ``(distance, metric)``
  for all four metrics (ρ / R² / W₁-raw / W₁-z), upper triangle mirrored, diagonal ρ=1 / R²=1 /
  W₁=0 (a pLM vs itself);
* the Holm-Bonferroni–corrected per-cell permutation p-values, applied **per
  ``(distance, metric)`` family** over the ``C(len(plms), 2)`` unordered pLM pairs — **6
  families** ``{ρ, R²} × {euclidean, cosine, manhattan}`` at the real grid. **W₁ is descriptive
  only and enters NO Holm family** (it has no permutation p — see
  :func:`evaluation.cross_plm.cross_plm_permutation_null`).

This is NOT a pLM ranking, a "best pLM", or a significant-Δ table — the "which pLM to choose"
answer lives in the ground-truth arms.

The §9.3 guard order is load-bearing:

1. **Pre-filter size assert** — each family must have EXACTLY ``C(n,2)`` cells *present* before
   any NaN handling. A missing/dead-job cell (``n_present`` < expected) fails loud. This is a
   SEPARATE, PRIOR check from the NaN drop — not a circular ``expected = C(n,2) - drops`` (which
   could not distinguish a missing cell from a degenerate one).
2. **NaN-p filter** — then drop cells whose perm-p is NaN (persisted as JSON ``null`` by
   ``json_safe``, or a literal float NaN); the dropped pairs + count are recorded for provenance.
3. **Holm** — :func:`evaluation.stats.holm_bonferroni` on the surviving p-vector. It RAISES on
   any NaN input, so step 2 is mandatory.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

from evaluation.stats import holm_bonferroni

DEFAULT_DISTANCES: tuple[str, ...] = ("cosine", "euclidean", "manhattan")

# Every metric gets a symmetric agreement matrix; only ρ / R² carry a permutation p and so
# enter the Holm families (W₁-raw / W₁-z are descriptive distances only).
MATRIX_METRICS: tuple[str, ...] = ("rho", "r2", "w1_raw", "w1_z")
HOLM_METRICS: tuple[str, ...] = ("rho", "r2")

# Diagonal of each metric's agreement matrix (a pLM vs itself): perfect rank/linear agreement
# is 1.0; the W₁ between a distribution and itself is 0.0.
_DIAGONAL: dict[str, float] = {"rho": 1.0, "r2": 1.0, "w1_raw": 0.0, "w1_z": 0.0}


def _sidecar_path(sidecar_dir: Path, a: str, b: str, rep: str, dist: str) -> Path:
    return sidecar_dir / f"cross_plm_{a}__{b}_{rep}_{dist}.manifest.json"


def _is_nan_p(p) -> bool:
    """A perm-p is unusable iff it is JSON null (None) or a non-finite float.

    ``p`` is assumed already validated to be ``None`` or a real number (see the entries loop
    in :func:`assemble_agreement_matrices`); a non-numeric value is rejected there as a
    malformed sidecar rather than reaching ``float(p)`` here.
    """
    return p is None or not math.isfinite(float(p))


def assemble_agreement_matrices(
    sidecar_dir: Path | str,
    *,
    plms: Sequence[str],
    representation: str = "raw",
    distances: Sequence[str] = DEFAULT_DISTANCES,
    alpha: float = 0.05,
    expected_n_pairs: int | None = None,
) -> dict:
    """Assemble the symmetric agreement matrices + Holm-corrected families from the sidecars.

    Parameters
    ----------
    sidecar_dir
        Directory holding the ``cross_plm_<a>__<b>_<rep>_<distance>.manifest.json`` sidecars.
    plms
        The pLM names defining the matrix order. The unordered pairs are
        ``itertools.combinations(plms, 2)`` — the SAME (a, b) order the report wrote, so the
        sidecar filenames line up.
    representation
        Representation axis (default ``"raw"``); only sidecars of this rep are read.
    distances
        Distance axis (default cosine + euclidean + manhattan). One Holm family per
        ``(distance, metric)`` for each ρ / R².
    alpha
        Holm rejection level (default 0.05).
    expected_n_pairs
        Override for the per-family expected cell count (default ``C(len(plms), 2)``).

    Returns
    -------
    dict
        ``{plms, representation, distances, alpha, matrices, families}`` where

        * ``matrices[distance][metric]`` is the symmetric ``n × n`` point matrix (nested lists);
        * ``families["<distance>:<metric>"]`` (ρ / R² only) is ``{n_present, n_dropped,
          dropped_pairs, records}``; ``records`` are the SURVIVING cells in pair order, each
          ``{a, b, perm_p, adjusted_p, rejected}``.

    Raises
    ------
    ValueError
        A family with fewer (or more) than the expected number of cells present (the §9.3
        pre-filter size assert — a missing/dead-job cell), or a present sidecar missing a
        metric's ``perm_p``.
    """
    sidecar_dir = Path(sidecar_dir)
    plms = list(plms)
    n = len(plms)
    if n < 2:
        raise ValueError(
            f"need at least 2 pLMs to assemble an agreement matrix (got {n}); "
            f"the unordered-pair grid C({n},2) would be empty."
        )
    idx = {p: i for i, p in enumerate(plms)}
    pairs = list(combinations(plms, 2))
    expected = expected_n_pairs if expected_n_pairs is not None else len(pairs)

    matrices: dict[str, dict[str, list]] = {}
    families: dict[str, dict] = {}

    for dist in distances:
        # Read every present sidecar for this (representation, distance) once.
        present: list[tuple[str, str, dict]] = []
        for a, b in pairs:
            path = _sidecar_path(sidecar_dir, a, b, representation, dist)
            if path.exists():
                present.append((a, b, json.loads(path.read_text())))

        # Symmetric agreement matrices for all four metrics (point values).
        dist_mats: dict[str, list] = {}
        for metric in MATRIX_METRICS:
            mat = np.full((n, n), np.nan)
            np.fill_diagonal(mat, _DIAGONAL[metric])
            for a, b, cell in present:
                v = cell["metrics"][metric]["point"]
                v = float(v) if v is not None else float("nan")
                i, j = idx[a], idx[b]
                mat[i, j] = mat[j, i] = v
            dist_mats[metric] = mat.tolist()
        matrices[dist] = dist_mats

        # Holm families (ρ / R² only).
        for metric in HOLM_METRICS:
            label = f"{dist}:{metric}"
            entries: list[tuple[str, str, object]] = []
            for a, b, cell in present:
                mdict = cell["metrics"].get(metric)
                if not isinstance(mdict, dict) or "perm_p" not in mdict:
                    raise ValueError(
                        f"family {label}: cell {a}__{b} has no '{metric}' perm_p (malformed "
                        f"sidecar)."
                    )
                perm_p = mdict["perm_p"]
                # perm_p must be a real number or JSON null; a string/list/dict is a corrupt
                # sidecar — fail loud here rather than throw an untyped float() error downstream.
                if perm_p is not None and (
                    isinstance(perm_p, bool) or not isinstance(perm_p, (int, float))
                ):
                    raise ValueError(
                        f"family {label}: cell {a}__{b} '{metric}' perm_p must be a number or "
                        f"null, got {perm_p!r} (malformed sidecar)."
                    )
                entries.append((a, b, perm_p))

            # §9.3 step 1 — pre-filter size assert (separate from + prior to the NaN drop).
            if len(entries) != expected:
                raise ValueError(
                    f"family {label}: {len(entries)} cell(s) present, expected {expected} "
                    f"(C({n},2)); a missing/dead-job cell — refusing to Holm-correct a short "
                    f"family (the §9.3 pre-filter size assert)."
                )

            # §9.3 step 2 — NaN-p filter (record what was dropped, then Holm the survivors).
            surviving: list[tuple[str, str, float]] = []
            dropped_pairs: list[list[str]] = []
            for a, b, p in entries:
                if _is_nan_p(p):
                    dropped_pairs.append([a, b])
                else:
                    surviving.append((a, b, float(p)))

            # §9.3 step 3 — Holm over the surviving p-vector (in pair order).
            if surviving:
                ps = np.array([p for _, _, p in surviving], dtype=float)
                rejected, adjusted = holm_bonferroni(ps, alpha=alpha)
            else:
                rejected, adjusted = np.array([], dtype=bool), np.array([], dtype=float)

            records = [
                {"a": a, "b": b, "perm_p": p,
                 "adjusted_p": float(adj), "rejected": bool(rej)}
                for (a, b, p), adj, rej in zip(surviving, adjusted, rejected)
            ]
            families[label] = {
                "n_present": len(entries),
                "n_dropped": len(dropped_pairs),
                "dropped_pairs": dropped_pairs,
                "records": records,
            }

    return {
        "plms": plms,
        "representation": representation,
        "distances": list(distances),
        "alpha": alpha,
        "matrices": matrices,
        "families": families,
    }
