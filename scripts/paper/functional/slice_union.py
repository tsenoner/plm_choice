"""Slice every embedding arm down to the EC-v2 + GO-MF union cohort.

Modelled on ~/c1_strat/slice.py (the EC-only slicer): default HDF5 driver, write
to <dst>.part and os.replace() so an interrupted job never leaves a truncated
file that the "skip completed arms" check would trust, and exit non-zero the
moment an arm is missing a cohort id (that would mean the union cohort was not
pre-filtered against the embedding cohort, which must be fixed upstream rather
than silently tolerated here).

Covers 15 pretrained arms -> <arm>.h5 and 11 random-init arms -> randinit_<model>.h5
(prost_t5 is deliberately excluded: ProstT5 is out of the study).
"""

import os
import sys
import time

import h5py

HERE = os.path.expanduser("~/c1_strat/slices_union")
IDS_TXT = f"{HERE}/union_ids.txt"

DSS = (
    "/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2"
    "/plm_choice_data/data/processed"
)
PRETRAINED_DIR = f"{DSS}/sprot_pre2024/embeddings_cohort2k"
RANDINIT_DIR = f"{DSS}/sprot_pre2024_subset/embeddings_random_init"

PRETRAINED_ARMS = [
    "ankh_base",
    "ankh_large",
    "clean",
    "esm1b",
    "esm2_150m",
    "esm2_35m",
    "esm2_3b",
    "esm2_650m",
    "esm2_8m",
    "esm3_open",
    "esmc_300m",
    "esmc_600m",
    "prott5",
    "prottucker",
    "random_1024",
]
# prost_t5 is present on disk but excluded on purpose -- ProstT5 is out of the study.
RANDINIT_MODELS = [
    "ankh_base",
    "ankh_large",
    "esm1b",
    "esm2_150m",
    "esm2_35m",
    "esm2_3b",
    "esm2_650m",
    "esm2_8m",
    "esmc_300m",
    "esmc_600m",
    "prot_t5",
]


def load_ids(path: str) -> list[str]:
    """Return the union cohort ids in file order, rejecting duplicates."""
    ids = [line.strip() for line in open(path) if line.strip()]
    if len(set(ids)) != len(ids):
        sys.exit(f"{path}: {len(ids) - len(set(ids))} duplicate ids")
    return ids


def slice_arm(src: str, dst: str, ids: list[str]) -> None:
    """Copy exactly `ids` from `src` into a fresh `dst`, preserving shape+dtype.

    Shape is preserved rather than flattened so that these slices match the
    existing EC slices bit for bit (prott5/prottucker/prot_t5 keep their (1, D)
    layout); flattening stays a consumer-side concern.
    """
    part = dst + ".part"
    try:
        # Default driver, not core: the core driver would pull the whole multi-GB
        # arm into RAM, and one pass over ~23k datasets is cheap either way.
        with h5py.File(src, "r") as fin, h5py.File(part, "w") as fo:
            missing = [p for p in ids if p not in fin]
            if missing:
                sys.exit(
                    f"{os.path.basename(src)}: {len(missing)} cohort ids absent "
                    f"(first: {missing[:3]}) -- PRE-FILTER VIOLATED"
                )
            for p in ids:
                fo.create_dataset(p, data=fin[p][:], dtype=fin[p].dtype)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    os.replace(part, dst)


def main() -> None:
    os.makedirs(HERE, exist_ok=True)
    ids = load_ids(IDS_TXT)
    print(f"union cohort: {len(ids)} ids", flush=True)

    jobs: list[tuple[str, str]] = [
        (f"{PRETRAINED_DIR}/{a}.h5", f"{HERE}/{a}.h5") for a in PRETRAINED_ARMS
    ]
    jobs += [
        (f"{RANDINIT_DIR}/random_init_{m}_seed0.h5", f"{HERE}/randinit_{m}.h5")
        for m in RANDINIT_MODELS
    ]

    absent = [src for src, _ in jobs if not os.path.exists(src)]
    if absent:
        sys.exit(f"{len(absent)} source arms not found: {absent}")

    for src, dst in jobs:
        name = os.path.basename(dst)[:-3]
        if os.path.exists(dst):
            print(f"  {name:22s} already complete, skipped", flush=True)
            continue
        t0 = time.time()
        slice_arm(src, dst, ids)
        print(
            f"  {name:22s} {os.path.getsize(dst) / 1e6:8.1f} MB "
            f"{time.time() - t0:6.1f} s",
            flush=True,
        )

    print(f"all {len(jobs)} arms sliced", flush=True)


if __name__ == "__main__":
    main()
