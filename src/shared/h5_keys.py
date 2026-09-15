"""Read the dataset names out of the embedding HDF5 files.

Lives in ``shared`` because two unrelated consumers need the same answer: the
coverage UpSet figure (``visualization.plot_embedding_coverage_upset``) and the
cohort exclusion freeze (``shared.protein_cohort``). Keeping one implementation
also keeps ONE copy of the measured core-driver rationale below, which was
previously restated in prose in a figure module and an sbatch heredoc.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path

#: Largest file to read with the in-RAM ``core`` driver. Sized against the LRZ login
#: node's 4 GiB per-user cgroup, where a 3.53 GB file peaked at 3.70 GB RSS and a
#: 4.62 GB one was SIGKILLed at 4.18 GB. It is a *parameter* of where you are running,
#: not a property of the data -- a batch job with ``--mem=32G`` should raise it.
CORE_DRIVER_MAX_BYTES = 4.0e9


def load_h5_keysets(
    h5_dir: Path | str,
    use_cache: bool = True,
    core_driver_max_bytes: float = CORE_DRIVER_MAX_BYTES,
) -> dict[str, set]:
    """Read dataset names from every ``.h5`` in a directory, with a sidecar cache.

    Enumerating ~542k names out of one HDF5 group on a network filesystem is dominated
    by scattered metadata reads, so the answer is cached beside the file as
    ``<stem>.keys.txt``. The cache is invalidated on (size, mtime), because a stale key
    list would silently misreport coverage -- exactly the failure this data documents.

    Cold reads use the ``core`` driver, which slurps the file in one sequential pass
    instead of chasing scattered metadata. These are superblock-v0 symbol-table groups:
    ~0.52 metadata reads per key at 356 B each, and on GPFS an identical 356 B read
    costs 3.0 us confined to an 8.4 MB span versus 175 us across a 4.62 GB span. Reading
    the whole 4.6 GB sequentially takes 2.76 s at 1.7 GB/s while the scattered
    equivalent takes 60-100 s at ~95% iowait. Measured end to end: esm1b.h5
    161.7 s -> 2.4 s (66.8x), esm2_650m.h5 106.8 s -> 1.85 s (57.8x), key sets
    byte-identical. Never run this concurrently -- P>=2 was observed to SIGKILL.
    """
    import h5py

    sets: dict[str, set] = {}
    for h5_path in sorted(Path(h5_dir).glob("*.h5")):
        stat = h5_path.stat()
        sidecar = h5_path.with_suffix(".keys.txt")
        stamp = f"# {stat.st_size} {int(stat.st_mtime)}"
        if use_cache and sidecar.exists():
            lines = iter(sidecar.read_text().splitlines())
            if next(lines, None) == stamp:
                sets[h5_path.stem] = set(lines)  # consume the iterator, don't copy 542k
                continue
        kwargs = (
            {"driver": "core", "backing_store": False}
            if stat.st_size < core_driver_max_bytes
            else {}
        )
        with h5py.File(h5_path, "r", **kwargs) as handle:
            keys = list(handle.keys())
        sets[h5_path.stem] = set(keys)
        try:
            sidecar.write_text("\n".join(chain((stamp,), keys)))
        except OSError:
            pass  # read-only location (e.g. the Zenodo deposit) -- caching is optional
    return sets
