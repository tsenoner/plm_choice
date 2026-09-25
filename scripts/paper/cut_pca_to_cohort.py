#!/usr/bin/env python3
"""Cut the PCA embedding arms down to the shared protein cohort.

Why. The exclusion freeze only SUBTRACTS: it names the proteins that at least one native
arm is missing, so every native arm filters to exactly ``intersection_all`` (526,871) and
the runtime guard stays quiet. The PCA arms were projected from a fuller deposit and hold
1,357 proteins that no native arm has -- ids the freeze therefore never names -- so they
filter to 528,228 and every PCA cell logged

    cohort WARNING [ankh_base]: 528,228 proteins after filtering, but the freeze defines a
    526,871-protein cohort (+1,357). This arm is NOT on the shared cohort

A probe trained on those arms is scored on a different pair population from its native
twin, which is exactly the comparison Figure S3 exists to make. This rewrites each arm
with the surplus removed, so the two grids sit on the same proteins.

    python scripts/paper/cut_pca_to_cohort.py --native <dir> --pca <dir> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from _cohort import COHORT_SIZE, compute_cohort


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--native", required=True, type=Path, help="native embeddings_cohort2k dir")
    ap.add_argument("--pca", required=True, type=Path, help="PCA embeddings dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expect", type=int, default=COHORT_SIZE, help="cohort the freeze defines")
    args = ap.parse_args(argv)

    # The cohort is what EVERY native arm holds -- computed rather than assumed, so a
    # changed deposit cannot silently move it.
    cohort = compute_cohort(args.native, expect=args.expect)

    args.out.mkdir(parents=True, exist_ok=True)
    order = sorted(cohort)
    for src in sorted(args.pca.glob("*.h5")):
        dst = args.out / src.name
        if dst.exists():
            print(f"  {src.name}: exists, skipping")
            continue
        with h5py.File(src, "r") as fin:
            have = set(fin.keys())
            missing = cohort - have
            if missing:
                raise SystemExit(
                    f"{src.name} is missing {len(missing):,} cohort proteins "
                    f"(e.g. {sorted(missing)[:3]}); it cannot be cut to the cohort."
                )
            tmp = dst.with_suffix(".part")
            with h5py.File(tmp, "w") as fout:
                for pid in order:
                    fout.create_dataset(pid, data=np.asarray(fin[pid]))
            tmp.rename(dst)
        print(f"  {src.name}: {len(have):,} -> {len(order):,}  (dropped {len(have) - len(order):,})")

    print(f"wrote {len(list(args.out.glob('*.h5')))} arms to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
