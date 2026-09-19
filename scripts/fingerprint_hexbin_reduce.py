#!/usr/bin/env python3
"""Reduce the full corrected pair cohort to the supplementary fingerprint figure.

The supplementary "density fingerprint" figure is a lower-triangular grid of joint
distance distributions, one panel per pLM pair. Drawing it needs only a 50x50 count
matrix per pair, so the 75.8-million-pair cohort is reduced here -- on the cluster,
where the per-arm distance files live -- to exactly the JSON that
``visualization/pairwise_embedding_comparison.py`` already consumes as its hexbin
cache. Nothing downstream has to know the figure was not computed in one process.

Three things this does that a naive rerun of the in-repo path would not:

1. **Identical-sequence pairs are excluded from a sequence mask, not from a distance.**
   Swiss-Prot holds one sequence under two accessions 325,019 times. Dropping them on
   ``distance == 0`` is circular, and it also misses them in the arms stored as
   float16 (ProtT5, ProtTucker, the random control), whose identical pairs land at
   0.0004-0.0056 rather than at zero. The mask is read from the precomputed,
   row-aligned ``<split>_identical.parquet``.

2. **Every panel is drawn over one common pair set** -- the rows finite in *all*
   arms. CLEAN and ESM-1b exist only for the <=1022-residue cohort, so a per-panel
   mask would silently give those rows a different denominator in 27 of the 91
   panels. With a common set, each arm's global min/max *are* the per-panel extremes,
   which is what makes the one-pass binning below exactly equal to
   ``np.histogram2d(x, y, bins=50)`` run per panel.

3. **Binning is done once per arm, not once per panel.** Each arm's values are
   digitised to a bin index by ``searchsorted``, exactly as ``histogramdd`` does it;
   a panel is then a ``bincount`` over the paired indices. ``--verify-pairs``
   re-derives a sample of panels with ``np.histogram2d`` itself -- in both its
   explicit-edge and its ``bins=50`` form -- and fails the run on any disagreement.

    python scripts/fingerprint_hexbin_reduce.py \
        --dist-dir $DSS/ridge_full --identical-dir $DSS/ridge_identical \
        --out $DSS/fingerprint_hexbin
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from shared.embedding_names import is_iid_random_baseline
from visualization.plm_constants import EMBEDDING_FAMILY_MAP, PLM_SIZES

#: Check the pair keys agree between arms on this many rows at each end of a split.
ALIGNMENT_PROBE_ROWS = 50_000


def discover_arms(dist_dir: Path, split: str) -> list[str]:
    """Arms present for ``split``, ordered and filtered exactly as the plotter does.

    The order is the figure's row/column order, so it has to come from the same
    (family, size, name) key ``pairwise_embedding_comparison`` sorts by -- otherwise
    the axis labels and the panels disagree.
    """
    arms = [p.name[len("dist_") : -len(".parquet")] for p in dist_dir.glob("dist_*.parquet")]
    arms = [a for a in arms if not is_iid_random_baseline(a) and a.lower() != "prostt5"]
    if not arms:
        raise SystemExit(f"no usable dist_*.parquet in {dist_dir}")
    return sorted(
        arms,
        key=lambda a: (EMBEDDING_FAMILY_MAP.get(a.lower(), "Unknown"), PLM_SIZES.get(a.lower(), 0), a.lower()),
    )


def check_alignment(dist_dir: Path, arms: list[str], n_rows: int) -> int:
    """Confirm every arm's file holds the same pairs in the same order.

    The columns are read positionally afterwards, so this is the assumption the whole
    reduction rests on. A silent misalignment would pair protein A's distance in one
    arm with protein B's in another and still produce a plausible-looking figure.
    """
    heights = {}
    for arm in arms:
        heights[arm] = pl.scan_parquet(dist_dir / f"dist_{arm}.parquet").select(pl.len()).collect().item()
    if len(set(heights.values())) != 1:
        raise SystemExit(f"{dist_dir}: arms disagree on row count: {heights}")
    n = next(iter(heights.values()))

    ref_arm = arms[0]
    probe = min(n_rows, n)
    ref_head = pl.read_parquet(dist_dir / f"dist_{ref_arm}.parquet", columns=["query", "target"], n_rows=probe)
    for arm in arms[1:]:
        head = pl.read_parquet(dist_dir / f"dist_{arm}.parquet", columns=["query", "target"], n_rows=probe)
        if not head.equals(ref_head):
            raise SystemExit(f"{dist_dir}/dist_{arm}.parquet: pair order differs from dist_{ref_arm}.parquet")
    return n


def load_split(dist_dir: Path, arms: list[str], identical_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One split's per-arm distances plus its identical-sequence mask."""
    n = check_alignment(dist_dir, arms, ALIGNMENT_PROBE_ROWS)

    identical = pl.read_parquet(identical_path)["identical"].to_numpy()
    if identical.size != n:
        raise SystemExit(
            f"{identical_path}: {identical.size:,} mask rows but {n:,} pair rows in {dist_dir} -- not row-aligned"
        )

    out: dict[str, np.ndarray] = {}
    for arm in arms:
        col = f"dist_{arm}"
        # float64, i.e. exactly what the in-repo path reads out of polars. float32
        # would halve the memory but it also moves values that sit on a bin edge,
        # and it makes numpy build float32 bin edges in the auto-bin path -- so the
        # reduction would no longer be the same histogram the repo would compute.
        out[arm] = pl.read_parquet(dist_dir / f"dist_{arm}.parquet", columns=[col])[col].to_numpy().astype(np.float64)
        print(f"    {arm}: {out[arm].size:,} values", flush=True)
    return out, identical


def digitise(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per value, binned exactly as ``np.histogram2d`` bins.

    ``histogramdd`` -- which ``histogram2d`` is -- always assigns bins by
    ``searchsorted`` against the edge array, then pulls values sitting exactly on the
    rightmost edge back into the last bin. Reproducing that literally is not
    pedantry: the obvious ``(v - lo) / (hi - lo) * nbins`` shortcut disagrees with it
    on values that land near an edge in floating point, and ``--verify-pairs`` caught
    exactly that on the esm2_35m x esmc_300m panel.
    """
    if edges[-1] <= edges[0]:
        raise SystemExit(f"degenerate range [{edges[0]}, {edges[-1]}] -- every value identical?")
    idx = np.searchsorted(edges, values, side="right") - 1
    idx[values == edges[-1]] -= 1
    return idx.astype(np.int32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist-dir", required=True, type=Path, help="directory holding <split>/dist_<arm>.parquet")
    ap.add_argument("--identical-dir", required=True, type=Path, help="directory holding <split>_identical.parquet")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--gridsize", type=int, default=50)
    ap.add_argument("--verify-pairs", type=int, default=3, help="panels to re-derive with np.histogram2d (0 to skip)")
    ap.add_argument(
        "--p99-limit",
        type=float,
        default=0.0,
        help=(
            "Cap each arm's axis at this multiple of its 99th percentile, accumulating the "
            "tail in the edge bin. 0 (the default) reproduces the published min-max binning. "
            "1.2 is the limit the ridge figure uses for the same distances."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    arms = discover_arms(args.dist_dir / args.splits[0], args.splits[0])
    print(f"{len(arms)} arms: {', '.join(arms)}", flush=True)

    per_split_counts: dict[str, dict[str, int]] = {}
    chunks: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
    identical_chunks: list[np.ndarray] = []
    for split in args.splits:
        print(f"  reading {split}", flush=True)
        dist, identical = load_split(
            args.dist_dir / split, arms, args.identical_dir / f"{split}_identical.parquet"
        )
        finite = np.ones(identical.size, dtype=bool)
        for arm in arms:
            finite &= np.isfinite(dist[arm])
        per_split_counts[split] = {
            "n_pairs": int(identical.size),
            "n_identical_sequence": int(identical.sum()),
            "n_missing_in_some_arm": int((~finite).sum()),
            "n_kept": int((finite & ~identical).sum()),
        }
        for arm in arms:
            chunks[arm].append(dist[arm])
        identical_chunks.append(identical)
        del dist

    values = {arm: np.concatenate(chunks[arm]) for arm in arms}
    del chunks
    identical = np.concatenate(identical_chunks)
    del identical_chunks

    keep = ~identical
    for arm in arms:
        keep &= np.isfinite(values[arm])
    n_total, n_keep = identical.size, int(keep.sum())
    print(f"{n_total:,} pairs -> {n_keep:,} kept ({100 * n_keep / n_total:.3f}%)", flush=True)

    values = {arm: values[arm][keep] for arm in arms}
    del identical, keep

    G = args.gridsize
    ranges: dict[str, tuple[float, float]] = {}
    edges: dict[str, np.ndarray] = {}
    index: dict[str, np.ndarray] = {}
    clipped: dict[str, int] = {}
    for arm in arms:
        lo, hi = float(values[arm].min()), float(values[arm].max())
        n_clipped = 0
        if args.p99_limit:
            # Min-max binning is not robust to a heavy right tail, and the corrected
            # cohort has one: ESM-1b runs to 31.36 with a 99th percentile of 5.22, so
            # 99% of its pairs land in 8 of 50 bins and its whole row and column of the
            # grid collapse into a strip. 1.2 x p99 is the same axis limit the ridge
            # figure already uses for these distances. Values past it are accumulated in
            # the edge bin rather than dropped, so every panel keeps the identical pair
            # set and the exclusion stays a statable count instead of a hidden one.
            limit = float(np.percentile(values[arm], 99.0)) * args.p99_limit
            if limit < hi:
                n_clipped = int((values[arm] > limit).sum())
                hi = limit
        ranges[arm] = (lo, hi)
        clipped[arm] = n_clipped
        # linspace over the arm's own extremes is what histogram2d builds for bins=G,
        # and because every panel is drawn over one common pair set these extremes are
        # each panel's extremes too.
        edges[arm] = np.linspace(lo, hi, G + 1)
        index[arm] = digitise(np.minimum(values[arm], hi) if n_clipped else values[arm], edges[arm])
        note = f"  ({n_clipped:,} above, {100 * n_clipped / n_keep:.3f}%, binned at the edge)" if n_clipped else ""
        print(f"  {arm}: [{lo:.4f}, {hi:.4f}]{note}", flush=True)
    hexbin: dict[str, object] = {"metadata": {"dist_cols": [f"dist_{a}" for a in arms], "gridsize": G, "max_count": 0}}
    csv_rows: list[dict[str, object]] = []
    max_count = 0

    pairs = list(itertools.combinations(arms, 2))
    for k, (a, b) in enumerate(pairs, 1):
        counts = np.bincount(index[a].astype(np.int64) * G + index[b], minlength=G * G).reshape(G, G)
        max_count = max(max_count, int(counts.max()))
        # Both orderings: the plotter looks a panel up by (x-axis arm, y-axis arm).
        hexbin[f"dist_{a}_vs_dist_{b}"] = {
            "counts": counts.tolist(),
            "xedges": edges[a].tolist(),
            "yedges": edges[b].tolist(),
        }
        hexbin[f"dist_{b}_vs_dist_{a}"] = {
            "counts": counts.T.tolist(),
            "xedges": edges[b].tolist(),
            "yedges": edges[a].tolist(),
        }
        ia, ib = np.nonzero(counts)
        ca = 0.5 * (edges[a][:-1] + edges[a][1:])
        cb = 0.5 * (edges[b][:-1] + edges[b][1:])
        for x, y in zip(ia, ib, strict=True):
            csv_rows.append(
                {
                    "arm_x": a,
                    "arm_y": b,
                    "bin_x": int(x),
                    "bin_y": int(y),
                    "centre_x": float(ca[x]),
                    "centre_y": float(cb[y]),
                    "count": int(counts[x, y]),
                }
            )
        if k % 10 == 0 or k == len(pairs):
            print(f"  panel {k}/{len(pairs)}", flush=True)

    hexbin["metadata"]["max_count"] = max_count

    if args.verify_pairs and not any(clipped.values()):
        rng = np.random.default_rng(args.seed)
        for a, b in [pairs[i] for i in rng.choice(len(pairs), min(args.verify_pairs, len(pairs)), replace=False)]:
            got = np.array(hexbin[f"dist_{a}_vs_dist_{b}"]["counts"])
            # Both forms: explicit edges pins the binning, bins=G pins the edges
            # themselves as the ones histogram2d would have chosen unaided.
            for label, ref in (
                ("explicit edges", np.histogram2d(values[a], values[b], bins=[edges[a], edges[b]])[0]),
                ("bins=G", np.histogram2d(values[a], values[b], bins=G)[0]),
            ):
                if not np.array_equal(ref.astype(np.int64), got):
                    raise SystemExit(
                        f"verification failed for {a} vs {b} ({label}): "
                        f"{int(np.abs(ref - got).sum()):,} counts misplaced"
                    )
            print(f"  verified {a} vs {b} against np.histogram2d", flush=True)

    (args.out / "hexbin_data.json").write_text(json.dumps(hexbin))
    pl.DataFrame(csv_rows).write_csv(args.out / "fingerprint_hexbin_values.csv")
    pl.DataFrame(
        [
            {
                "arm": a,
                "axis_min": ranges[a][0],
                "axis_max": ranges[a][1],
                "n_pairs": n_keep,
                "n_above_axis_max": clipped[a],
                "pct_above_axis_max": 100.0 * clipped[a] / n_keep,
            }
            for a in arms
        ]
    ).write_csv(args.out / "fingerprint_hexbin_ranges.csv")
    summary = {
        "arms": arms,
        "gridsize": G,
        "splits": args.splits,
        "per_split": per_split_counts,
        "n_pairs_total": n_total,
        "n_pairs_used": n_keep,
        "n_panels": len(pairs),
        "max_count": max_count,
        "p99_limit": args.p99_limit,
        "n_clipped_per_arm": clipped,
        "identical_source": str(args.identical_dir),
        "dist_source": str(args.dist_dir),
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "fingerprint_hexbin_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
