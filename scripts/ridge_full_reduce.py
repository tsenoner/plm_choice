#!/usr/bin/env python3
"""M-8: reduce the full-cohort pair distances to something a laptop can draw.

``ridge_distances_full.sbatch`` wrote one parquet per arm per split -- 9.7 GB for
train alone, 75,849,972 pairs over 15 arms.  Pulling that home to make a figure is
absurd, and sampling it to make a figure throws away the reason we computed all of
it.  This script streams each arm on the cluster and writes two small things:

  * a **fine histogram on the raw distance scale** (50,000 bins).  Every
    normalisation we are choosing between -- min-max, division by a percentile,
    division by the median -- is a positive affine map, so the same histogram
    re-plots under any of them by rescaling the bin edges.  The density the figure
    draws therefore comes from *all* 75.8M pairs, not from a subsample.
  * **exact quantiles**, computed on the values rather than read off a KDE.  The
    shipped plotter estimates q25/q75 by ``np.searchsorted`` on the cumulative sum
    of a 500-point KDE grid (``plot_ridge_distributions``), which is accurate to
    about one grid cell.  Quoting a quartile in a manuscript off a smoothed curve
    when the exact value costs one ``np.percentile`` is indefensible.

It also computes Hartigan's dip on the full cohort.  The dip is a vertical
sup-distance between the ECDF and the closest unimodal CDF, so it is invariant
under any increasing affine map of x -- the same number for raw, min-max and
percentile-scaled data.  That invariance is checked here rather than assumed.

Identical-sequence pairs -- the same protein deposited under two accessions -- are
excluded from every statistic, because they are a property of Swiss-Prot's redundancy
and not of any embedding space.  With ``--identical-root`` they are identified from the
sequences (see ``ridge_identical_pairs.py``), so every arm drops the same rows; the
fallback rule ``distance == 0`` is circular and, in the float16-stored arms, misses
about 9% of them because they land at ~0.001 rather than 0.0.

    python scripts/ridge_full_reduce.py \
        --dist-root $DSS/ridge_full --out-dir $DSS/ridge_full_summary --arm esm1b
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

#: Quantile grid.  Dense in both tails: the tails are what min-max normalisation
#: anchors on, so the caption has to be able to say how far out the anchor sits.
QUANTILES = (
    0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 25.0, 30.0,
    40.0, 50.0, 60.0, 70.0, 75.0, 80.0, 90.0, 95.0, 97.5, 99.0, 99.5, 99.9,
    99.99, 100.0,
)

N_BINS = 50_000
SUBSAMPLE = 500_000
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def load_arm(dist_root: Path, arm: str, splits: tuple[str, ...]) -> tuple[np.ndarray, dict]:
    """Concatenate one arm's distance column across splits, in split order."""
    col = f"dist_{arm}"
    parts: list[np.ndarray] = []
    per_split: dict[str, int] = {}
    for split in splits:
        path = dist_root / split / f"dist_{arm}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        t0 = time.time()
        v = pl.read_parquet(path, columns=[col]).to_series().to_numpy()
        per_split[split] = int(v.size)
        parts.append(v.astype(np.float64, copy=False))
        log(f"  {split}: {v.size:,} rows in {time.time() - t0:.1f}s rss={rss_gb():.2f} GiB")
    return np.concatenate(parts), per_split


def dip_of(x: np.ndarray) -> tuple[float, float]:
    from diptest import diptest as _diptest

    d, p = _diptest(x)
    return float(d), float(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dist-root",
        type=Path,
        help="Cluster layout: <root>/<split>/dist_<arm>.parquet, concatenated over --splits.",
    )
    src.add_argument(
        "--pairs-parquet",
        type=Path,
        help="One flat table carrying dist_<arm> columns -- the 10%% subset. Same estimator, "
        "so subset and full cohort are compared without a second implementation.",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--arm", required=True)
    ap.add_argument(
        "--identical-root",
        type=Path,
        default=None,
        help="Directory of <split>_identical.parquet from ridge_identical_pairs.py. "
        "With it, identical-sequence pairs are excluded by SEQUENCE, the same rows for "
        "every arm; without it, the fallback is distance == 0, which is circular and "
        "misses the float16 arms' identical pairs (they land at ~0.001, not 0.0).",
    )
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--bins", type=int, default=N_BINS)
    ap.add_argument("--subsample", type=int, default=SUBSAMPLE)
    ap.add_argument(
        "--dip-max-n",
        type=int,
        default=200_000_000,
        help="Skip the full-cohort dip above this n (the C routine sorts a float64 copy).",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    if args.dist_root is not None:
        log(f"arm={args.arm} splits={args.splits}")
        x, per_split = load_arm(args.dist_root, args.arm, tuple(args.splits))
    else:
        log(f"arm={args.arm} source={args.pairs_parquet}")
        x = (
            pl.read_parquet(args.pairs_parquet, columns=[f"dist_{args.arm}"])
            .to_series()
            .to_numpy()
            .astype(np.float64, copy=False)
        )
        per_split = {args.pairs_parquet.stem: int(x.size)}
    n_rows = int(x.size)

    # Identical-sequence pairs: the same protein under two accessions.  They are the
    # spike at zero and they pin the min-max minimum.  Defined from the SEQUENCES when
    # the mask is available, so every arm drops the same rows and the definition does
    # not depend on the quantity being plotted.
    identical = None
    if args.identical_root is not None:
        if args.dist_root is not None:
            # Positional mask: the distance parquets were written row-for-row from the
            # same pair tables, in the same split order.
            identical = np.concatenate([
                pl.read_parquet(args.identical_root / f"{split}_identical.parquet")
                .to_series()
                .to_numpy()
                for split in args.splits
            ])
            if identical.size != n_rows:
                raise ValueError(
                    f"identical mask has {identical.size:,} rows, distances have {n_rows:,}"
                )
        else:
            # The 10% subset is a row subset of train, so a positional mask does not
            # apply; mark its rows by accession pair instead.
            keyed = pl.concat([
                pl.read_parquet(p)
                for p in sorted(args.identical_root.glob("*_identical_keyed.parquet"))
            ]).with_columns(pl.lit(True).alias("_identical"))
            sub = pl.read_parquet(args.pairs_parquet, columns=["query", "target"])
            identical = (
                sub.join(keyed, on=["query", "target"], how="left")
                .select(pl.col("_identical").fill_null(False))
                .to_series()
                .to_numpy()
            )
            if identical.size != n_rows:
                raise ValueError(
                    f"keyed identical join produced {identical.size:,} rows, "
                    f"distances have {n_rows:,} -- the join duplicated rows"
                )

    finite = np.isfinite(x)
    n_nan = int((~finite).sum())

    if identical is not None:
        n_identical = int((identical & finite).sum())
        keep = finite & ~identical
        # How many of those the naive distance==0 rule would have missed, per arm.
        n_zero_after = int((x[keep] == 0.0).sum())
        x = x[keep]
        n_zero = n_identical
        log(f"  rows={n_rows:,} nan={n_nan:,} identical_sequence={n_identical:,} "
            f"(exact-zero survivors after the mask: {n_zero_after:,})")
        x = x[x > 0.0] if n_zero_after else x
    else:
        x = x[finite]
        zero = x == 0.0
        n_zero = int(zero.sum())
        n_zero_after = 0
        x = x[~zero]
        log(f"  rows={n_rows:,} nan={n_nan:,} zero={n_zero:,}")
    n_valid = n_rows - n_nan
    n_used = int(x.size)
    log(f"  used={n_used:,}")

    q = np.percentile(x, QUANTILES)
    qd = {f"p{p:g}": float(v) for p, v in zip(QUANTILES, q, strict=True)}
    q25, med, q75 = qd["p25"], qd["p50"], qd["p75"]
    iqr = q75 - q25
    qcd = iqr / (q75 + q25) if (q75 + q25) > 0 else float("nan")

    x_min, x_max = float(x.min()), float(x.max())
    mean, std = float(x.mean()), float(x.std())
    log(f"  q25={q25:.6g} med={med:.6g} q75={q75:.6g} IQR={iqr:.6g} QCD={qcd:.4f}")
    log(f"  min={x_min:.6g} max={x_max:.6g} p99={qd['p99']:.6g} max/p99={x_max / qd['p99']:.2f}")

    # Fine histogram on the RAW scale.  Re-plottable under any affine rescaling.
    hist, edges = np.histogram(x, bins=args.bins, range=(0.0, x_max))
    # A second histogram in log10, for the log-axis candidate.  x is already > 0.
    lo = np.log10(max(x_min, np.nextafter(0.0, 1.0)))
    hist_log, edges_log = np.histogram(np.log10(x), bins=args.bins // 10,
                                       range=(lo, np.log10(x_max)))

    rng = np.random.default_rng(SEED)
    sub = x if n_used <= args.subsample else rng.choice(x, args.subsample, replace=False)

    # --- Hartigan's dip ------------------------------------------------------
    dip: dict[str, object] = {}
    if n_used <= args.dip_max_n:
        t0 = time.time()
        d_full, p_full = dip_of(x)
        dip["full"] = {"n": n_used, "dip": d_full, "p_value": p_full,
                       "seconds": round(time.time() - t0, 1)}
        log(f"  dip(full n={n_used:,}) = {d_full:.6f} p={p_full:.4g} "
            f"({time.time() - t0:.1f}s)")
        # Affine invariance check: min-max and /p99 must give the same dip.
        d_mm, _ = dip_of((x - x_min) / (x_max - x_min))
        d_p99, _ = dip_of(x / qd["p99"])
        dip["affine_invariance_check"] = {
            "dip_raw": d_full, "dip_minmax": d_mm, "dip_over_p99": d_p99,
            "max_abs_diff": max(abs(d_full - d_mm), abs(d_full - d_p99)),
        }
    else:
        dip["full"] = None

    # Replicates at fixed n, because the dip statistic shrinks like n^-1/2 for a
    # unimodal sample: a dip computed on 75.8M pairs is not comparable with the
    # published one computed on ~6M, and neither is comparable with diptest's
    # p-value table, which stops at n = 72,000.
    for n_rep, reps in ((100_000, 25), (1_000_000, 5)):
        if n_rep > n_used:
            log(f"  dip(n={n_rep:,}): skipped, only {n_used:,} pairs available")
            continue
        ds, ps = [], []
        for _ in range(reps):
            s = rng.choice(x, n_rep, replace=False)
            d, p = dip_of(s)
            ds.append(d)
            ps.append(p)
        dip[f"n{n_rep}"] = {
            "n": n_rep, "reps": reps,
            "dip_mean": float(np.mean(ds)), "dip_sd": float(np.std(ds, ddof=1)),
            "p_mean": float(np.mean(ps)), "p_min": float(np.min(ps)),
            "p_max": float(np.max(ps)),
        }
        log(f"  dip(n={n_rep:,} x{reps}) = {np.mean(ds):.6f} +/- {np.std(ds, ddof=1):.6f} "
            f"p in [{np.min(ps):.3g}, {np.max(ps):.3g}]")

    summary = {
        "arm": args.arm,
        "source": str(args.dist_root or args.pairs_parquet),
        "splits": list(args.splits) if args.dist_root is not None else [],
        "n_rows": n_rows,
        "n_per_split": per_split,
        "n_nan": n_nan,
        "n_valid": n_valid,
        "n_zero_identical": n_zero,
        "n_zero_after_mask": n_zero_after,
        "identical_excluded_by": "sequence" if args.identical_root else "distance==0",
        "n_used": n_used,
        "min": x_min,
        "max": x_max,
        "mean": mean,
        "std": std,
        "q25": q25,
        "median": med,
        "q75": q75,
        "iqr": iqr,
        "qcd": qcd,
        "quantiles": qd,
        "hist": {"bins": int(args.bins), "lo": 0.0, "hi": x_max},
        "hist_log10": {"bins": int(args.bins // 10), "lo": float(lo),
                       "hi": float(np.log10(x_max))},
        "subsample_n": int(sub.size),
        "subsample_seed": SEED,
        "dip": dip,
        "seconds": round(time.time() - t_start, 1),
        "peak_rss_gb": round(rss_gb(), 2),
    }
    (args.out_dir / f"{args.arm}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.out_dir / f"{args.arm}.npz",
        hist=hist.astype(np.int64),
        edges=edges.astype(np.float64),
        hist_log10=hist_log.astype(np.int64),
        edges_log10=edges_log.astype(np.float64),
        subsample=sub.astype(np.float32),
        quantile_levels=np.asarray(QUANTILES, dtype=np.float64),
        quantile_values=q.astype(np.float64),
    )
    log(f"wrote {args.out_dir / (args.arm + '.json')} and .npz "
        f"in {time.time() - t_start:.1f}s peak_rss={rss_gb():.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
