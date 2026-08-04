"""The streaming all-vs-all histogram must not depend on --batch-size.

``scripts/all_vs_all.py`` computes its histogram in row batches. Each batch row
``j`` (global ``i+j``) was compared against every target from ``i`` onward with
only the diagonal masked, so intra-batch pairs were counted twice while
cross-batch pairs were counted once. The ``histogram * 2`` at the end then
assumed a clean upper triangle, making the totals batch-size dependent: on a
40-protein set the histogram summed to 1720 / 1920 / 3120 for batch sizes
5 / 10 / 40, against a true 1560.

That is exactly the class of bug that produced a 500-protein result which looked
like a full-cohort one, so it gets a test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "all_vs_all.py"


@pytest.fixture(scope="module")
def all_vs_all():
    """scripts/ is not a package, so load the module from its path."""
    spec = importlib.util.spec_from_file_location("all_vs_all", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def embeddings():
    rng = np.random.default_rng(0)
    return rng.normal(size=(40, 8)).astype(np.float32)


BATCH_SIZES = [1, 5, 7, 10, 13, 39, 40, 100]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_total_count_is_every_unordered_pair_twice(all_vs_all, embeddings, batch_size):
    n = len(embeddings)
    expected = n * (n - 1)  # each unordered pair counted in both directions
    histogram, _ = all_vs_all.compute_histogram_batched(
        embeddings, n_bins=2000, min_val=0.0, max_val=12.0, batch_size=batch_size
    )
    assert histogram.sum() == expected


def test_histogram_is_identical_across_batch_sizes(all_vs_all, embeddings):
    histograms = [
        all_vs_all.compute_histogram_batched(
            embeddings, n_bins=2000, min_val=0.0, max_val=12.0, batch_size=bs
        )[0]
        for bs in BATCH_SIZES
    ]
    for bs, histogram in zip(BATCH_SIZES[1:], histograms[1:]):
        assert np.array_equal(histogram, histograms[0]), (
            f"batch_size={bs} produced a different histogram — --batch-size is "
            f"documented as a memory knob and must not change the result"
        )


def test_matches_an_independent_scipy_computation(all_vs_all, embeddings):
    """Bit-exact against scipy's pdist, not merely self-consistent."""
    histogram, edges = all_vs_all.compute_histogram_batched(
        embeddings, n_bins=2000, min_val=0.0, max_val=12.0, batch_size=7
    )
    reference, _ = np.histogram(pdist(embeddings, "euclidean"), bins=edges)
    assert np.array_equal(histogram, reference * 2)


def test_self_distances_are_excluded(all_vs_all):
    """A set of identical vectors has only zero distances, none of them self."""
    embeddings = np.ones((6, 4), dtype=np.float32)
    histogram, _ = all_vs_all.compute_histogram_batched(
        embeddings, n_bins=10, min_val=0.0, max_val=1.0, batch_size=2
    )
    # 6*5 = 30 ordered off-diagonal pairs; the 6 self-pairs must not appear.
    assert histogram.sum() == 30


def test_max_embeddings_cap_is_opt_in(all_vs_all):
    """The cap that caused the original damage must default to off."""
    import inspect

    signature = inspect.signature(all_vs_all.load_embeddings)
    assert signature.parameters["max_embeddings"].default is None
