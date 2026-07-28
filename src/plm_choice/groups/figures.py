"""Regenerate the manuscript figures from computed metrics.

These read the parquet/CSV artifacts written by plm train and
plm evaluate and write PNG/PDF panels. Nothing here recomputes embeddings,
so figures are cheap to redraw once the metrics exist.
"""

from __future__ import annotations

import typer

from plm_choice.bridge import PASSTHROUGH_CONTEXT, run_module_main, run_repo_script

app = typer.Typer(help=__doc__, no_args_is_help=True)

_SUMMARY = "Cross-model summaries"
_PAIRWISE = "Embedding-space comparison"
_SINGLE = "Single-run diagnostics"


def _cmd(name: str, *, panel: str, help_: str):
    def decorator(fn):
        return app.command(
            name,
            help=help_,
            rich_help_panel=panel,
            context_settings=PASSTHROUGH_CONTEXT,
            add_help_option=False,
        )(fn)

    return decorator


@_cmd(
    "summary",
    panel=_SUMMARY,
    help_="Performance-vs-model-size summary panels across every run.",
)
def summary(ctx: typer.Context) -> None:
    run_module_main(
        "visualization.create_performance_summary_plots",
        ctx.args,
        prog="plm figures summary",
    )


@_cmd(
    "grid",
    panel=_SUMMARY,
    help_="Evaluation grid: model type x embedding x target, one cell per run.",
)
def grid(ctx: typer.Context) -> None:
    run_module_main(
        "visualization.create_evaluation_grid_plots", ctx.args, prog="plm figures grid"
    )


@_cmd(
    "rank",
    panel=_SUMMARY,
    help_="Rank pLMs by performance from a parsed-metrics CSV.",
)
def rank(ctx: typer.Context) -> None:
    run_repo_script(
        "scripts/rank_plms_by_performance.py", ctx.args, prog="plm figures rank"
    )


@_cmd(
    "quartiles",
    panel=_SUMMARY,
    help_="Correlate model performance against embedding-distance quartiles.",
)
def quartiles(ctx: typer.Context) -> None:
    run_repo_script(
        "scripts/analyze_performance_vs_quartiles.py",
        ctx.args,
        prog="plm figures quartiles",
    )


@_cmd(
    "pairwise",
    panel=_PAIRWISE,
    help_="Full pairwise embedding-space comparison (hexbin, ridge, Wasserstein).",
)
def pairwise(ctx: typer.Context) -> None:
    run_module_main(
        "visualization.pairwise_embedding_comparison",
        ctx.args,
        prog="plm figures pairwise",
    )


@_cmd(
    "comparison",
    panel=_PAIRWISE,
    help_="Curated subset of the pairwise panels used in the manuscript.",
)
def comparison(ctx: typer.Context) -> None:
    run_module_main(
        "visualization.create_embedding_comparison_plots",
        ctx.args,
        prog="plm figures comparison",
    )


@_cmd(
    "ecdf",
    panel=_SINGLE,
    help_="ECDF of predicted vs true values from an inference .npz.",
)
def ecdf(ctx: typer.Context) -> None:
    run_module_main("visualization.plot_ecdf", ctx.args, prog="plm figures ecdf")


@_cmd(
    "compare-runs",
    panel=_SINGLE,
    help_="Diff the metrics of two training runs side by side.",
)
def compare_runs(ctx: typer.Context) -> None:
    run_repo_script("scripts/compare_runs.py", ctx.args, prog="plm figures compare-runs")
