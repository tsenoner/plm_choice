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

import re

#: Filename/key prefix for untrained-architecture runs, written by
#: ``embedding_generation.py --random_init`` as
#: ``random_init_<model_key>_seed<N>.h5`` (see :func:`random_init_stem`).
RANDOM_INIT_PREFIX = "random_init"

_RANDOM_INIT_STEM_RE = re.compile(rf"^{RANDOM_INIT_PREFIX}_(?P<model>.+)_seed(?P<seed>\d+)$")


def random_init_stem(model_key: str, seed: int) -> str:
    """HDF5 stem for one untrained-architecture run.

    The seed is part of the name because D-6 reports this arm as mean±sd over
    seeds 0/1/2. Without it all three seeds resolve to one path, the writer skips
    every already-present protein, and the run exits 0 having produced a single
    seed's vectors — published as ``sd = 0.000``, a fabricated error bar.

    Producer and figures both go through here so the format is stated once.
    """
    return f"{RANDOM_INIT_PREFIX}_{model_key.replace('/', '_')}_seed{int(seed)}"


def parse_random_init_stem(stem: str) -> tuple[str, int] | None:
    """Inverse of :func:`random_init_stem`; ``None`` if ``stem`` is not one.

    Grouping the seeds back together is the whole point of putting the seed in
    the name — without a parser, ``random_init_esm2_650m_seed0`` cannot be
    matched to its pretrained twin ``esm2_650m`` for the mean±sd.
    """
    match = _RANDOM_INIT_STEM_RE.match(stem.lower())
    if match is None:
        return None
    return match.group("model"), int(match.group("seed"))


def is_iid_random_baseline(embedding_name: str) -> bool:
    """True for the i.i.d. random-vector floor, False for untrained architectures.

    ``embedding_name`` is the lowercase HDF5 stem with any ``dist_`` prefix already
    stripped. ``random_1024`` is the floor; ``random_init_esm2_650m`` is a real model
    architecture with untrained weights and belongs in the figures.
    """
    name = embedding_name.lower()
    return name.startswith("random") and not name.startswith(RANDOM_INIT_PREFIX)


__all__ = [
    "RANDOM_INIT_PREFIX",
    "is_iid_random_baseline",
    "parse_random_init_stem",
    "random_init_stem",
]
