"""Test-suite wide settings.

Matplotlib must be non-interactive before any test imports pyplot: several
modules under test draw figures, and on a machine with a GUI backend selected
that either opens windows or fails outright on a headless runner. CI sets
MPLBACKEND=Agg, which meant a green CI could hide a locally broken test.
"""

import contextlib
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

    with contextlib.ExitStack() as stack:

        def _open(path):
            gc.collect()
            return stack.enter_context(h5py.File(path, "r"))

        yield _open


def tiny_esm_config(**overrides):
    """An ESM config small enough to build offline in a fraction of a second.

    Shared so the "how to build a network-free ESM" knowledge — the rotary
    position type and the pad id, both of which the batching code depends on —
    is stated once rather than in each test module.
    """
    from transformers.models.esm.configuration_esm import EsmConfig

    kwargs = dict(
        vocab_size=33,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        position_embedding_type="rotary",
        pad_token_id=1,
    )
    kwargs.update(overrides)
    return EsmConfig(**kwargs)


def tiny_esm(seed=0, **overrides):
    """An untrained tiny ESM in eval mode, built without touching the network."""
    import torch
    from transformers import AutoModel

    torch.manual_seed(seed)
    return AutoModel.from_config(tiny_esm_config(**overrides)).eval()
