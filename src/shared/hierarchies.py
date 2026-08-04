"""Structural-hierarchy level names shared by the producers and the figures.

``ecod_homology_pairs.py`` writes the ``ecod_<level>_same`` columns and
``create_retrieval_plots.py`` labels the axes that display them. When each owned its
own copy of the level list they disagreed on casing (``t_group`` vs ``T_group``), so a
level produced by the pipeline fell through the plotter's lookup and was rendered as
the raw column name. One table, one casing — the lowercase form, because that is what
the parquet columns actually carry.
"""

from __future__ import annotations

#: ECOD hierarchy levels, coarsest to finest.
ECOD_LEVELS: list[str] = ["arch", "x_group", "h_group", "t_group", "f_group"]

#: Human-readable labels for plot titles, legends and axes.
ECOD_LEVEL_LABELS: dict[str, str] = {
    "arch": "Architecture",
    "x_group": "X-group",
    "h_group": "H-group",
    "t_group": "T-group",
    "f_group": "F-group",
}

#: SCOP levels, coarsest to finest — the column names used by the SCOP parquets.
SCOP_LEVELS: list[str] = ["fold_id", "sf_id", "fa_id"]

#: Human-readable labels for the SCOP levels.
SCOP_LEVEL_LABELS: dict[str, str] = {
    "fold_id": "Fold",
    "sf_id": "Superfamily",
    "fa_id": "Family",
}

#: Every level label a figure might have to render, whichever hierarchy it came from.
LEVEL_LABELS: dict[str, str] = {**ECOD_LEVEL_LABELS, **SCOP_LEVEL_LABELS}

__all__ = [
    "ECOD_LEVELS",
    "ECOD_LEVEL_LABELS",
    "SCOP_LEVELS",
    "SCOP_LEVEL_LABELS",
    "LEVEL_LABELS",
]
