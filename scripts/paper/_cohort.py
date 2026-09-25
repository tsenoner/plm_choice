"""The shared protein cohort, computed from the native arms rather than assumed.

Both the PCA cut and the PCA rebuild have to agree on which proteins the cohort
contains, and getting that wrong is silent: the probe still trains, it is just
scored on a different pair population from the arm it is compared against. So the
definition lives in one place and carries its own guard.
"""

from __future__ import annotations

from pathlib import Path

import h5py

# What every native arm holds after the exclusion freeze. The freeze only SUBTRACTS,
# so it can enforce this number but cannot discover it -- see cut_pca_to_cohort.py.
COHORT_SIZE = 526_871


def arm_keys(path: Path) -> set[str]:
    with h5py.File(path, "r") as h:
        return set(h.keys())


def compute_cohort(native_dir: Path, expect: int = COHORT_SIZE, verbose: bool = True) -> set[str]:
    """Intersect every native arm, and refuse to return anything but `expect`."""
    native = sorted(native_dir.glob("*.h5"))
    if not native:
        raise SystemExit(f"no native arms in {native_dir}")

    cohort: set[str] | None = None
    for p in native:
        k = arm_keys(p)
        cohort = k if cohort is None else (cohort & k)
        if verbose:
            print(f"  {p.name:22} {len(k):>8,}  running intersection {len(cohort):>8,}")
    assert cohort is not None

    if verbose:
        print(f"cohort = {len(cohort):,} proteins over {len(native)} native arms")
    if len(cohort) != expect:
        raise SystemExit(
            f"cohort is {len(cohort):,}, expected {expect:,}. Refusing to proceed against a "
            "cohort that does not match the freeze -- fix the inputs, not this number."
        )
    return cohort
