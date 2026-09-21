"""The filtering figure must never certify a panel it did not draw.

``filtering_thresholds.csv`` exists so a supplementary caption can be checked
against the run that produced the PNGs instead of against memory. That only
works if the CSV and the PNGs come from the same run, and there is now one way
for them not to: ``--reuse-plots`` deliberately keeps panels that already exist
on disk. Writing the CSV anyway would compute it from today's data and set it
down beside a figure drawn from older data -- the exact failure that let a
July-2025 funnel outlive both the HFSP correction and the deduplication, only
this time with a file asserting the stale numbers were current.

The second test pins the other half of the same promise: ``_threshold_stats``
says "the CSV and the red text on the PNG can never drift apart", and they only
cannot if both divide by the same denominator. The CSV counts finite values; the
annotation used to divide by ``len(data)``, which includes nulls, so the two
disagreed for any metric column carrying them -- and the CSV emits an
``n_non_finite`` field precisely because the author expected some.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl

from data_preparation.merge_datasets import ProteinAnalysisPipeline


def _pipe() -> ProteinAnalysisPipeline:
    """``_create_distribution_plots`` touches no state; bypass the on-disk __init__."""
    return object.__new__(ProteinAnalysisPipeline)


def _frames(n: int = 200) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(0)
    mmseqs = pl.DataFrame(
        {
            "qcov": rng.uniform(0.5, 1.0, n),
            "tcov": rng.uniform(0.5, 1.0, n),
            "fident": rng.uniform(0.0, 1.0, n),
            "hfsp": rng.uniform(-40.0, 90.0, n),
        }
    )
    foldcomp = pl.DataFrame({"avg_plddt": rng.uniform(40.0, 99.0, n)})
    foldseek = pl.DataFrame(
        {
            "min_cov": rng.uniform(0.5, 1.0, n),
            "alntmscore": rng.uniform(0.0, 1.0, n),
        }
    )
    return mmseqs, foldcomp, foldseek


def test_reused_panels_suppress_the_threshold_csv(tmp_path):
    """A run that redraws writes the CSV; a run that reuses a panel must not."""
    plots_dir = tmp_path / "plots"
    mmseqs, foldcomp, foldseek = _frames()
    stats_csv = plots_dir / "filtering_thresholds.csv"

    _pipe()._create_distribution_plots(mmseqs, foldcomp, foldseek, plots_dir)
    assert stats_csv.exists(), "a full redraw must write the numbers it drew"

    # The panels are now on disk. Delete only the CSV, so its reappearance can
    # only come from this second, reusing run.
    stats_csv.unlink()
    _pipe()._create_distribution_plots(
        mmseqs, foldcomp, foldseek, plots_dir, reuse=True
    )
    assert not stats_csv.exists(), (
        "filtering_thresholds.csv was written next to panels this run did not "
        "draw -- it would certify a stale figure with current numbers"
    )


def test_annotation_and_csv_share_one_denominator(monkeypatch, tmp_path):
    """The red "N (x%) < t" text must be the percentage the CSV reports."""
    data = np.concatenate([np.linspace(0.0, 1.0, 100), np.full(25, np.nan)])
    threshold = 0.3

    annotations: list[str] = []
    real_text = matplotlib.pyplot.text
    monkeypatch.setattr(
        matplotlib.pyplot,
        "text",
        lambda *a, **k: (annotations.append(a[2]), real_text(*a, **k))[1],
    )
    ProteinAnalysisPipeline._create_violin_plot(
        data, threshold, "T", (0.0, 1.0), tmp_path / "p.png"
    )

    stats = ProteinAnalysisPipeline._threshold_stats(data, threshold, "T", "A", "c", "u")
    assert stats["n_non_finite"] == 25, "fixture must actually carry nulls"
    assert len(annotations) == 1
    assert f"({stats['pct_below_threshold']:.1f}%)" in annotations[0], (
        f"panel says {annotations[0]!r}, CSV says "
        f"{stats['pct_below_threshold']:.1f}% -- the two denominators differ"
    )
