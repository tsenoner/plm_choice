#!/usr/bin/env python3
"""E7/M-8: recompute per-pair embedding distances for the ridge figure.

Reproduces the metric of ``src/data_preparation/distance_computation.py`` exactly:

  * metric   : Euclidean (L2) on protein-level embedding vectors
  * 2-D H5   : ``np.mean(emb, axis=0)`` -- for the (1, D) arms (prott5/prottucker)
               that is identical to a flatten
  * missing  : a pair whose query or target lacks an embedding gets NaN
  * rounding : ``np.round(d, decimals=4)``

The reference implementation loops pair-by-pair in Python.  This one loads each
arm's embedding matrix into RAM once and evaluates the pairs vectorised in
chunks; the arithmetic is the same L2 norm, accumulated in float64.

No normalisation happens here.  The min-max scaling the ridge figure shows is
applied later, per column, by ``normalize_distribution`` in the plotting code.

Writes one parquet per arm (query, target, dist_<arm>) into --out-dir.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import h5py
import numpy as np
import polars as pl


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_rss_gb() -> float:
    # Linux ru_maxrss is in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def h5_keys(h5_path: Path) -> list[str] | None:
    """Dataset names from the ``<stem>.keys.txt`` sidecar, if it is current.

    Same contract as ``shared.h5_keys.load_h5_keysets``: the first line stamps
    ``# <size> <int(mtime)>`` and a mismatch means recount.  Reusing it matters --
    that module measured 60-160 s per file to enumerate ~542k names out of a
    GPFS-hosted HDF5 group.
    """
    sidecar = h5_path.with_suffix(".keys.txt")
    if not sidecar.exists():
        return None
    st = h5_path.stat()
    lines = sidecar.read_text().splitlines()
    if not lines or lines[0] != f"# {st.st_size} {int(st.st_mtime)}":
        log(f"  sidecar {sidecar.name} is stale, falling back to h5py key scan")
        return None
    return lines[1:]


def load_arm(
    h5_path: Path, wanted: list[str], core_max_bytes: float
) -> tuple[list[str], np.ndarray]:
    """Load protein-level embeddings for `wanted` ids into one float32 matrix.

    Reads with the ``core`` driver when the file fits in the job's memory: these
    are superblock-v0 symbol-table groups, so ~324k scattered per-dataset reads
    across a multi-GB span cost far more than one sequential slurp (the same
    measurement that motivates shared/h5_keys.py).
    """
    t0 = time.time()
    wanted_set = set(wanted)
    size = h5_path.stat().st_size
    kwargs = (
        {"driver": "core", "backing_store": False} if size < core_max_bytes else {}
    )
    log(f"  opening {h5_path.name} ({size / 1024**3:.2f} GiB) driver={kwargs or 'sec2'}")

    with h5py.File(h5_path, "r", **kwargs) as f:
        keys = h5_keys(h5_path)
        present = sorted(wanted_set.intersection(keys if keys is not None else f.keys()))
        if not present:
            raise RuntimeError(f"no requested protein found in {h5_path}")
        t_keys = time.time()

        dim = int(f[present[0]].shape[-1])
        ndim_seen: set[int] = set()
        mat = np.empty((len(present), dim), dtype=np.float32)
        for i, pid in enumerate(present):
            emb = f[pid][:]
            ndim_seen.add(emb.ndim)
            if emb.ndim > 1:
                # Matches the reference loader: collapse a (1, D) or (L, D)
                # dataset to protein level by averaging over axis 0.
                emb = np.mean(emb, axis=0)
            mat[i] = emb

    log(
        f"  loaded {len(present):,} vectors dim={dim} ndim_in_file={sorted(ndim_seen)} "
        f"({mat.nbytes / 1024**3:.2f} GiB) keys={t_keys - t0:.1f}s "
        f"total={time.time() - t0:.1f}s peak_rss={peak_rss_gb():.2f} GiB"
    )
    return present, mat


def pair_distances(
    pairs: pl.DataFrame, present: list[str], mat: np.ndarray, chunk: int
) -> np.ndarray:
    """Vectorised L2 between the embedding of each pair's query and target.

    Ids are mapped to row indices with ``replace_strict``, not a join: it is
    elementwise, so the result is pinned to the input row order by construction.
    A left join would *probably* preserve that order too, but "probably" silently
    mis-pairs 5.9M distances if it ever does not, and the parquet this writes is
    attached to the pair table positionally.
    """
    lut = dict(zip(present, range(len(present))))
    qi = (
        pairs["query"]
        .replace_strict(lut, default=-1, return_dtype=pl.Int64)
        .to_numpy()
    )
    ti = (
        pairs["target"]
        .replace_strict(lut, default=-1, return_dtype=pl.Int64)
        .to_numpy()
    )
    del lut

    n = qi.size
    valid = (qi >= 0) & (ti >= 0)
    log(f"  pairs valid {int(valid.sum()):,}/{n:,} (NaN {n - int(valid.sum()):,})")

    out = np.full(n, np.nan, dtype=np.float64)
    vidx = np.flatnonzero(valid)
    for s in range(0, vidx.size, chunk):
        sel = vidx[s : s + chunk]
        diff = mat[qi[sel]] - mat[ti[sel]]
        # float64 accumulation of the squared differences, then sqrt -> the same
        # value np.linalg.norm(a - b) returns for these vectors.
        out[sel] = np.sqrt(np.einsum("ij,ij->i", diff, diff, dtype=np.float64))
        del diff
    return np.round(out, decimals=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--emb-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument(
        "--core-max-gb",
        type=float,
        default=8.0,
        help="Read an .h5 smaller than this with the in-RAM core driver.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pl.read_parquet(args.pairs, columns=["query", "target"])
    wanted = (
        pl.concat(
            [pairs.select(pl.col("query").alias("id")), pairs.select(pl.col("target").alias("id"))]
        )
        .unique()["id"]
        .to_list()
    )
    log(f"pairs={len(pairs):,} unique proteins={len(wanted):,}")

    arms = args.arms or sorted(p.stem for p in args.emb_dir.glob("*.h5"))
    log(f"arms ({len(arms)}): {arms}")

    manifest_path = args.out_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"pairs": str(args.pairs), "n_pairs": len(pairs), "arms": {}}
    )

    for arm in arms:
        out_path = args.out_dir / f"dist_{arm}.parquet"
        if out_path.exists() and arm in manifest["arms"]:
            log(f"{arm}: exists, skipping")
            continue
        log(f"{arm}: start")
        t0 = time.time()
        present, mat = load_arm(
            args.emb_dir / f"{arm}.h5", wanted, args.core_max_gb * 1024**3
        )
        d = pair_distances(pairs, present, mat, args.chunk)
        dim = int(mat.shape[1])
        del present, mat
        gc.collect()

        pairs.select("query", "target").with_columns(
            pl.Series(f"dist_{arm}", d)
        ).write_parquet(out_path, compression="zstd")

        finite = np.isfinite(d)
        manifest["arms"][arm] = {
            "dim": dim,
            "n_valid": int(finite.sum()),
            "n_nan": int((~finite).sum()),
            "min": float(np.nanmin(d)),
            "max": float(np.nanmax(d)),
            "median": float(np.nanmedian(d)),
            "seconds": round(time.time() - t0, 1),
            "peak_rss_gb_after": round(peak_rss_gb(), 2),
        }
        manifest["peak_rss_gb"] = round(peak_rss_gb(), 2)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log(f"{arm}: done in {time.time() - t0:.1f}s peak_rss={peak_rss_gb():.2f} GiB")
        del d
        gc.collect()

    log(f"ALL DONE peak_rss={peak_rss_gb():.2f} GiB")


if __name__ == "__main__":
    main()
