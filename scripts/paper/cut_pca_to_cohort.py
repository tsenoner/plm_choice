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


def arm_keys(path: Path) -> set[str]:
    with h5py.File(path, "r") as h:
        return set(h.keys())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--native", required=True, type=Path, help="native embeddings_cohort2k dir")
    ap.add_argument("--pca", required=True, type=Path, help="PCA embeddings dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expect", type=int, default=526871, help="cohort size the freeze defines")
    args = ap.parse_args(argv)

    native = sorted(args.native.glob("*.h5"))
    if not native:
        raise SystemExit(f"no native arms in {args.native}")
    # The cohort is what EVERY native arm holds -- computed here rather than assumed, so a
    # changed deposit cannot silently move it.
    cohort: set[str] | None = None
    for p in native:
        k = arm_keys(p)
        cohort = k if cohort is None else (cohort & k)
        print(f"  {p.name:22} {len(k):>8,}  running intersection {len(cohort):>8,}")
    assert cohort is not None
    print(f"cohort = {len(cohort):,} proteins over {len(native)} native arms")
    if len(cohort) != args.expect:
        raise SystemExit(
            f"cohort is {len(cohort):,}, expected {args.expect:,}. Refusing to cut against a "
            "cohort that does not match the freeze -- fix the inputs, not this number."
        )

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
