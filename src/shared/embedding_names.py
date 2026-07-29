"""Naming conventions for embedding sets, shared by the producers and the figures.

Two independent exclusion filters — one in the all-vs-all cache builder, one in the
pairwise figures — each dropped every embedding whose name starts with ``random``.
That is right for the i.i.d. ``random_<dim>`` noise floor and wrong for the
``random_init_*`` untrained-architecture baselines (reviewer R1.9): they are a
different control, and dropping them silently deletes the arm from the figures.

The two filters are one predicate here so that a fix lands once instead of needing
to be found in every copy.
"""

from __future__ import annotations

#: Filename/key prefix for untrained-architecture runs, written by
#: ``embedding_generation.py --random_init`` as ``random_init_<model_key>.h5``.
RANDOM_INIT_PREFIX = "random_init"


def is_iid_random_baseline(embedding_name: str) -> bool:
    """True for the i.i.d. random-vector floor, False for untrained architectures.

    ``embedding_name`` is the lowercase HDF5 stem with any ``dist_`` prefix already
    stripped. ``random_1024`` is the floor; ``random_init_esm2_650m`` is a real model
    architecture with untrained weights and belongs in the figures.
    """
    name = embedding_name.lower()
    return name.startswith("random") and not name.startswith(RANDOM_INIT_PREFIX)


__all__ = ["RANDOM_INIT_PREFIX", "is_iid_random_baseline"]
