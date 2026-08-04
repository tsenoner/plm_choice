"""Test-suite wide settings.

Matplotlib must be non-interactive before any test imports pyplot: several
modules under test draw figures, and on a machine with a GUI backend selected
that either opens windows or fails outright on a headless runner. CI sets
MPLBACKEND=Agg, which meant a green CI could hide a locally broken test.
"""

import gc

import matplotlib
import pytest

matplotlib.use("Agg", force=True)


@pytest.fixture
def read_h5():
    """Open an HDF5 file for reading that this process just finished writing.

    Plain ``h5py.File(path, "r")`` raises ``BlockingIOError`` here, and the file
    is fine — a separate process reads it without complaint. HDF5's default file
    close degree is *weak*, so the underlying file stays open until every object
    belonging to it is freed; when the writer's frame ends up in a reference
    cycle, that happens at the next collection rather than at close. Forcing a
    collection first releases it.

    Only tests need this. Production writes one file per process and reads it
    back elsewhere. The other available fix, ``locking=False``, is deliberately
    NOT used: these files live on GPFS and are written by concurrent Slurm array
    tasks, where turning HDF5 locking off risks real corruption to avoid a
    test-only annoyance.
    """
    import h5py

    opened = []

    def _open(path):
        gc.collect()
        handle = h5py.File(path, "r")
        opened.append(handle)
        return handle

    yield _open

    for handle in opened:
        handle.close()
