#!/usr/bin/env python3
"""Re-project one PCA arm from its native embeddings, on the shared cohort.

Why this exists. The PCA deposit was built on 2025-08-01 from the native embeddings
as they stood that day, and on that day esm2_3b.h5 held 435,298 of the cohort's
proteins -- an interrupted embedding run, diagnosed and repaired in
scripts/lrz/complete_esm2_3b.sbatch. That repair rewrote the NATIVE file, which is
now complete at 540,881. Nothing rewrote the projection, so the PCA arm still carries
the truncated 435,298 and is missing 103,899 cohort proteins outright. It therefore
cannot be cut to the cohort at all:

    esm2_3b.h5 is missing 103,899 cohort proteins (e.g. ['A0A023FF81', ...]);
    it cannot be cut to the cohort.

Note what this means for the PCA panel as published: ESM-2 3B's point there was fitted
and scored on 80% of the proteins every other point used, which is a second and
undisclosed reason it sits apart from its family.

The fit is done on the cohort itself, so the basis and the output describe the same
proteins. The other arms keep their original projection -- their fit sets differ from
the cohort by at most 2.9% of proteins drawn from the same distribution, which does
not move a 128-dimensional principal subspace, whereas a missing fifth of the
proteins is not a basis question at all.

    python scripts/paper/rebuild_pca_arm.py \
        --native-dir <embeddings_cohort2k> --arm esm2_3b.h5 --out <file.h5>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from _cohort import COHORT_SIZE, compute_cohort
from sklearn.decomposition import PCA


def load_cohort_matrix(path: Path, order: list[str], report_every: int = 100_000) -> np.ndarray:
    """Read the arm into one preallocated array, in cohort order."""
    with h5py.File(path, "r") as h:
        dim = h[order[0]].shape[-1]
        out = np.empty((len(order), dim), dtype=np.float32)
        for i, pid in enumerate(order):
            out[i] = np.asarray(h[pid]).reshape(-1)
            if report_every and i and i % report_every == 0:
                print(f"    read {i:,}/{len(order):,}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--native-dir", required=True, type=Path, help="native embeddings_cohort2k dir")
    ap.add_argument("--arm", required=True, help="file name of the arm, e.g. esm2_3b.h5")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--components", type=int, default=128, help="must match the other PCA arms")
    ap.add_argument("--expect", type=int, default=COHORT_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cohort = compute_cohort(args.native_dir, expect=args.expect)
    order = sorted(cohort)

    src = args.native_dir / args.arm
    if not src.exists():
        raise SystemExit(f"no such native arm: {src}")

    print(f"reading {src.name} on the cohort", flush=True)
    x = load_cohort_matrix(src, order)
    print(f"  {x.shape[0]:,} x {x.shape[1]} float32 ({x.nbytes / 1e9:.1f} GB)", flush=True)

    if not np.all(np.isfinite(x)):
        raise SystemExit(
            f"{src.name} holds non-finite values on the cohort. The original PCA script "
            "replaced those with zeros silently; refusing to repeat that here."
        )

    # Randomized SVD, because only the leading 128 of up to 2,560 components are kept and
    # a full LAPACK decomposition of a 526,871 x 2,560 matrix buys nothing but hours.
    print(f"fitting PCA({args.components}) ...", flush=True)
    pca = PCA(n_components=args.components, svd_solver="randomized", random_state=args.seed)
    y = pca.fit_transform(x).astype(np.float32)
    kept = float(pca.explained_variance_ratio_.sum())
    print(f"  {args.components} components retain {kept:.4f} of the variance", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".part")
    with h5py.File(tmp, "w") as fout:
        for i, pid in enumerate(order):
            fout.create_dataset(pid, data=y[i])
    tmp.rename(args.out)
    print(f"wrote {len(order):,} x {args.components} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
