#!/usr/bin/env python3
"""Figure 3 (the fingerprint matrix), reduced from the full corrected pair cohort.

``pairwise_embedding_comparison.py`` computes the two triangles of Figure 3 by
holding every arm's per-pair distance column in one dataframe and calling
``scipy.stats.spearmanr`` / ``wasserstein_distance`` per pair of arms.  That is fine
for the 10% subset (5.9M pairs) and impossible for the corrected cohort: 75,849,972
pairs x 14 arms is ~11.6 GB of parquet on the cluster, and 91 calls to
``spearmanr`` would each re-rank 74.5M values.

This script does the same two statistics once, on the cluster, and writes a
~100 KB JSON the laptop can draw from.  Three things differ from the shipped
estimator, all of them deliberate:

*  **One pair set for every cell.**  The shipped code intersects the two arms'
   non-null rows *per pair*, so a cell involving CLEAN (whose 1,022-residue context
   drops 958,116 pairs) was computed on a different population from its neighbours,
   and the matrix was not a comparison between comparable numbers.  Here a single
   mask -- finite in all 14 arms, and not an identical-sequence pair -- defines the
   population, and every cell is computed on exactly those rows.

*  **Ranks once per arm, not once per cell.**  With one shared pair set the rank
   vector of an arm does not depend on which other arm it is compared to, so
   Spearman's rho is the Pearson correlation of 14 precomputed mid-rank vectors.
   Ties are resolved to average ranks, which matters: the distances are stored
   rounded to four decimals, so an arm like ankh_base has only ~12k distinct
   values over 74.5M pairs and "ignore ties" would be a different statistic.

*  **Wasserstein from order statistics.**  Because every arm is reduced on the same
   rows, the two empirical distributions have equal n, and
   ``W1 = mean(|sort(a) - sort(b)|)`` is exact -- no binning, no subsampling.

Identical-sequence pairs (the same protein deposited under two accessions) are
excluded by SEQUENCE, reusing the mask ``ridge_identical_pairs.py`` wrote.  The
``distance == 0`` rule is wrong here: the float16-stored embeddings put those pairs
at 0.0004--0.0056 rather than 0.0, so it would miss most of them in most arms.

Both normalisations of W are reported.  ``minmax`` is what the published figure and
its caption used; ``p99`` divides by each arm's own 99th percentile instead, which is
the scale Figure 2 switched to after the min-max divisor turned out to be a single
pair out of 75.5M that moves by up to 13.7% under resampling.  Reporting both is the
only way to say whether Figure 3's numbers depend on that choice.

    python scripts/fingerprint_full_reduce.py \
        --dist-root $DSS/ridge_full \
        --identical-root $DSS/ridge_identical \
        --out-dir $DSS/fingerprint_full
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import polars as pl

SPLITS = ("train", "val", "test")

#: The 14 pLMs of Figure 3.  ``random_1024`` is the i.i.d. Gaussian control and is
#: excluded by design -- it is not a protein language model and its distances are a
#: concentration-of-measure artefact, not a map of anything.
ARMS = (
    "ankh_base",
    "ankh_large",
    "clean",
    "esm1b",
    "esm2_8m",
    "esm2_35m",
    "esm2_150m",
    "esm2_650m",
    "esm2_3b",
    "esm3_open",
    "esmc_300m",
    "esmc_600m",
    "prott5",
    "prottucker",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def load_arm(dist_root: Path, arm: str, splits: tuple[str, ...]) -> np.ndarray:
    """One arm's distance column, concatenated across splits in split order."""
    col = f"dist_{arm}"
    parts = []
    for split in splits:
        path = dist_root / split / f"{col}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        v = pl.read_parquet(path, columns=[col]).to_series().to_numpy()
        parts.append(np.asarray(v, dtype=np.float64))
    return np.concatenate(parts)


def load_identical(identical_root: Path, splits: tuple[str, ...]) -> np.ndarray:
    """Positional identical-sequence mask, same split order as the distances."""
    return np.concatenate(
        [
            pl.read_parquet(identical_root / f"{split}_identical.parquet")
            .to_series()
            .to_numpy()
            for split in splits
        ]
    )


def ranks_and_sorted(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average ranks (float64, in input order) and the sorted values, from one sort.

    Average -- not ordinal -- ranks, because the four-decimal storage makes ties the
    rule rather than the exception.  Returning the sorted copy as well means the
    Wasserstein side costs no extra sort.
    """
    order = np.argsort(x, kind="stable")
    xs = x[order]
    n = xs.size

    change = np.empty(n, dtype=bool)
    change[0] = True
    np.not_equal(xs[1:], xs[:-1], out=change[1:])
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n
    # 1-based average rank of each tie run: (first + last)/2 with 1-based indices.
    mid = (starts + ends - 1) / 2.0 + 1.0
    ranks_sorted = np.repeat(mid, ends - starts)

    r = np.empty(n, dtype=np.float64)
    r[order] = ranks_sorted
    return r, xs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dist-root", required=True, type=Path)
    ap.add_argument("--identical-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    args = ap.parse_args()

    splits = tuple(args.splits)
    arms = list(args.arms)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # --- Load every arm, then define ONE population ---------------------------
    raw: dict[str, np.ndarray] = {}
    n_rows = None
    for arm in arms:
        t0 = time.time()
        raw[arm] = load_arm(args.dist_root, arm, splits)
        if n_rows is None:
            n_rows = raw[arm].size
        elif raw[arm].size != n_rows:
            raise ValueError(
                f"{arm} has {raw[arm].size:,} rows, expected {n_rows:,} -- the arms are "
                "not row-aligned, so no positional mask or per-pair statistic is valid"
            )
        log(f"loaded {arm}: {raw[arm].size:,} rows in {time.time() - t0:.1f}s "
            f"rss={rss_gb():.2f} GiB")

    identical = load_identical(args.identical_root, splits)
    if identical.size != n_rows:
        raise ValueError(
            f"identical mask has {identical.size:,} rows, distances have {n_rows:,}"
        )

    keep = ~identical
    n_identical = int(identical.sum())
    per_arm_nan = {}
    for arm in arms:
        finite = np.isfinite(raw[arm])
        per_arm_nan[arm] = int((~finite).sum())
        keep &= finite
    n_used = int(keep.sum())
    log(f"rows={n_rows:,} identical_sequence={n_identical:,} "
        f"common_valid={n_used:,} ({n_used / n_rows:.4%})")
    log(f"per-arm non-finite: {per_arm_nan}")

    # --- Per-arm reduction ----------------------------------------------------
    z: dict[str, np.ndarray] = {}          # unit-norm centred ranks -> rho by dot product
    s_minmax: dict[str, np.ndarray] = {}   # sorted, min-max normalised  -> W1
    s_p99: dict[str, np.ndarray] = {}      # sorted, /p99 normalised     -> W1 robustness
    stats_per_arm: dict[str, dict] = {}

    for arm in arms:
        t0 = time.time()
        x = raw.pop(arm)[keep]
        r, xs = ranks_and_sorted(x)

        zz = r - r.mean()
        nrm = float(np.linalg.norm(zz))
        z[arm] = zz / nrm

        lo, hi = float(xs[0]), float(xs[-1])
        p99 = float(xs[min(int(round(0.99 * (n_used - 1))), n_used - 1)])
        if hi <= lo:
            raise ValueError(f"{arm}: degenerate range [{lo}, {hi}]")
        s_minmax[arm] = ((xs - lo) / (hi - lo)).astype(np.float32)
        s_p99[arm] = (xs / p99).astype(np.float32)

        q25 = float(xs[int(round(0.25 * (n_used - 1)))])
        med = float(xs[int(round(0.50 * (n_used - 1)))])
        q75 = float(xs[int(round(0.75 * (n_used - 1)))])
        stats_per_arm[arm] = {
            "n": n_used,
            "n_nonfinite_before_mask": per_arm_nan[arm],
            "min": lo,
            "max": hi,
            "p25": q25,
            "median": med,
            "p75": q75,
            "p99": p99,
            "n_distinct": int(starts_count(xs)),
        }
        del x, r, xs, zz
        log(f"reduced {arm} in {time.time() - t0:.1f}s "
            f"min={lo:.6g} max={hi:.6g} p99={p99:.6g} rss={rss_gb():.2f} GiB")

    # --- The two triangles ----------------------------------------------------
    n = len(arms)
    rho = np.eye(n)
    w_mm = np.zeros((n, n))
    w_99 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = arms[i], arms[j]
            rho[i, j] = rho[j, i] = float(np.dot(z[a], z[b]))
            w_mm[i, j] = w_mm[j, i] = float(
                np.mean(np.abs(s_minmax[a] - s_minmax[b]), dtype=np.float64)
            )
            w_99[i, j] = w_99[j, i] = float(
                np.mean(np.abs(s_p99[a] - s_p99[b]), dtype=np.float64)
            )
        log(f"row {i + 1}/{n} ({arms[i]}) done rss={rss_gb():.2f} GiB")

    meta = {
        "source": str(args.dist_root),
        "identical_root": str(args.identical_root),
        "splits": list(splits),
        "arms": arms,
        "n_pairs_total": int(n_rows),
        "n_identical_sequence": n_identical,
        "n_pairs_used": n_used,
        "population": "rows finite in ALL arms AND not an identical-sequence pair",
        "rho": "Spearman, average ranks, one shared pair set for every cell",
        "wasserstein_normalisation": "minmax (drawn); p99 reported alongside",
        "per_arm": stats_per_arm,
        "seconds": round(time.time() - t_start, 1),
    }

    out = {
        "columns": arms,
        "correlations": rho.tolist(),
        "distances": w_mm.tolist(),
        "distances_p99": w_99.tolist(),
        "metadata": meta,
    }
    (args.out_dir / "fingerprint_full.json").write_text(json.dumps(out))
    log(f"wrote {args.out_dir / 'fingerprint_full.json'}")

    # The two files the shipped plotter's cache loader understands, so the figure can
    # also be drawn by the normal --visualizations combined path.
    (args.out_dir / "correlation_data.json").write_text(
        json.dumps({"correlations": rho.tolist(),
                    "ci_lower": rho.tolist(), "ci_upper": rho.tolist(),
                    "columns": arms, "metadata": meta})
    )
    (args.out_dir / "wasserstein_data.json").write_text(
        json.dumps({"distances": w_mm.tolist(), "columns": arms, "metadata": meta})
    )

    rows = ["model_a,model_b,spearman_rho,wasserstein_minmax,wasserstein_p99,n_pairs"]
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                f"{arms[i]},{arms[j]},{rho[i, j]:.6f},{w_mm[i, j]:.6f},"
                f"{w_99[i, j]:.6f},{n_used}"
            )
    (args.out_dir / "fingerprint_full_values.csv").write_text("\n".join(rows) + "\n")
    log(f"wrote {args.out_dir / 'fingerprint_full_values.csv'}  "
        f"total {time.time() - t_start:.1f}s peak_rss={rss_gb():.2f} GiB")
    return 0


def starts_count(xs: np.ndarray) -> int:
    """Number of distinct values in an already-sorted array."""
    if xs.size == 0:
        return 0
    return 1 + int(np.count_nonzero(xs[1:] != xs[:-1]))


if __name__ == "__main__":
    raise SystemExit(main())
