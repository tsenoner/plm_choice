"""Shared pLM metadata used by every figure: sizes, families, colours, labels.

These tables were previously duplicated verbatim between
``create_performance_summary_plots.py`` and ``pairwise_embedding_comparison.py``.
Keeping one copy is what makes a family colour or a display label consistent
across panels — a divergence here shows up as the same model drawn in two
different colours in two figures of the same paper.

Keys are the lowercase HDF5 file stem of each embedding set.
"""

from __future__ import annotations

#: Parameter count per pLM. ``random_1024`` is the untrained floor, hence 0.
PLM_SIZES: dict[str, int] = {
    "prott5": 1_500_000_000,
    "prottucker": 1_500_000_000,
    "prostt5": 1_500_000_000,
    "clean": 650_000_000,
    "esm1b": 650_000_000,
    "esm2_8m": 8_000_000,
    "esm2_35m": 35_000_000,
    "esm2_150m": 150_000_000,
    "esm2_650m": 650_000_000,
    "esm2_3b": 3_000_000_000,
    "esmc_300m": 300_000_000,
    "esmc_600m": 600_000_000,
    "esm3_open": 1_400_000_000,
    "ankh_base": 450_000_000,
    "ankh_large": 1_150_000_000,
    "random_1024": 0,
}

#: Which model family each embedding set belongs to.
EMBEDDING_FAMILY_MAP: dict[str, str] = {
    "prott5": "ProtT5",
    "prottucker": "ProtT5",
    "prostt5": "ProtT5",
    "clean": "ESM-1",
    "esm1b": "ESM-1",
    "esm2_8m": "ESM-2",
    "esm2_35m": "ESM-2",
    "esm2_150m": "ESM-2",
    "esm2_650m": "ESM-2",
    "esm2_3b": "ESM-2",
    "esmc_300m": "ESM-C",
    "esmc_600m": "ESM-C",
    "esm3_open": "ESM-3",
    "ankh_base": "Ankh",
    "ankh_large": "Ankh",
    "random_1024": "Random",
}

#: One colour per family — the paper's visual key.
EMBEDDING_FAMILY_COLOR_MAP: dict[str, str] = {
    "ProtT5": "#ff1493",
    "ESM-1": "#4daf4a",
    "ESM-2": "#ff7f00",
    "ESM-C": "#1f77b4",
    "ESM-3": "#984ea3",
    "Ankh": "#ffd700",
    "Random": "#808080",
}

#: Family colour projected onto each individual embedding set.
EMBEDDING_COLOR_MAP: dict[str, str] = {
    embedding: EMBEDDING_FAMILY_COLOR_MAP.get(family, "#808080")
    for embedding, family in EMBEDDING_FAMILY_MAP.items()
}

#: Two-line axis labels, sized for tick text.
EMBEDDING_DISPLAY_NAMES: dict[str, str] = {
    "ankh_base": "Ankh\nBase",
    "ankh_large": "Ankh\nLarge",
    "clean": "CLEAN",
    "esm1b": "ESM\n1b",
    "esm2_8m": "ESM2\n8M",
    "esm2_35m": "ESM2\n35M",
    "esm2_150m": "ESM2\n150M",
    "esm2_650m": "ESM2\n650M",
    "esm2_3b": "ESM2\n3B",
    "esm3_open": "ESM3",
    "esmc_300m": "ESM C\n300M",
    "esmc_600m": "ESM C\n600M",
    "prostt5": "Prost\nT5",
    "prott5": "Prot\nT5",
    "prottucker": "Prot\nTucker",
    "random_1024": "Random",
}

#: Marker per probe architecture.
MODEL_MARKER_MAP: dict[str, str] = {
    "fnn": "o",  # Circle
    "linear": "s",  # Square
    "linear_distance": "^",  # Triangle up
    "euclidean": "X",  # X
}

__all__ = [
    "PLM_SIZES",
    "EMBEDDING_FAMILY_MAP",
    "EMBEDDING_FAMILY_COLOR_MAP",
    "EMBEDDING_COLOR_MAP",
    "EMBEDDING_DISPLAY_NAMES",
    "MODEL_MARKER_MAP",
]
