"""Preloading embeddings must change speed, never numbers."""

from __future__ import annotations

import h5py
import numpy as np
import polars as pl
import pytest

from shared.datasets import H5PyDataset


@pytest.fixture
def pair_fixture(tmp_path):
    """Three proteins with distinctive embeddings, and the pairs between them."""
    rng = np.random.default_rng(0)
    ids = ["P1", "P2", "P3"]
    vectors = {pid: rng.normal(size=8).astype(np.float32) for pid in ids}
    h5_path = tmp_path / "arm.h5"
    with h5py.File(h5_path, "w") as hdf:
        # One arm ships (1, D) rather than (D,); the loader flattens either.
        hdf.create_dataset("P1", data=vectors["P1"])
        hdf.create_dataset("P2", data=vectors["P2"].reshape(1, -1))
        hdf.create_dataset("P3", data=vectors["P3"])
    frame = pl.DataFrame(
        {"query": ["P1", "P2", "P3"], "target": ["P2", "P3", "P1"], "fident": [0.1, 0.5, 0.9]}
    )
    return frame, str(h5_path), vectors


def test_preloaded_and_lazy_return_identical_values(pair_fixture):
    frame, h5_path, _ = pair_fixture
    lazy = H5PyDataset(frame, h5_path, "fident", preload=False)
    eager = H5PyDataset(frame, h5_path, "fident", preload=True)
    for i in range(len(frame)):
        q_lazy, t_lazy, v_lazy = lazy[i]
        q_eager, t_eager, v_eager = eager[i]
        np.testing.assert_array_equal(q_lazy, q_eager)
        np.testing.assert_array_equal(t_lazy, t_eager)
        assert v_lazy == v_eager


def test_preloaded_dataset_reads_the_file_once(pair_fixture):
    """The point of preloading: no open file handle is kept per worker."""
    frame, h5_path, _ = pair_fixture
    eager = H5PyDataset(frame, h5_path, "fident", preload=True)
    _ = eager[0]
    assert eager.file is None


def test_env_var_enables_preloading(pair_fixture, monkeypatch):
    frame, h5_path, _ = pair_fixture
    monkeypatch.setenv("PLM_PRELOAD_EMBEDDINGS", "1")
    assert H5PyDataset(frame, h5_path, "fident")._store is not None
    monkeypatch.setenv("PLM_PRELOAD_EMBEDDINGS", "0")
    assert H5PyDataset(frame, h5_path, "fident")._store is None


def test_preloading_flattens_and_casts(pair_fixture):
    frame, h5_path, vectors = pair_fixture
    eager = H5PyDataset(frame, h5_path, "fident", preload=True)
    q, _, _ = eager[1]  # P2 is the (1, D) one
    assert q.shape == (8,)
    assert q.dtype == np.float32
    np.testing.assert_array_equal(q, vectors["P2"])
