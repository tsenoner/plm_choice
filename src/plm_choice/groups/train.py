"""Fit the probes that read a target off an embedding pair.

run trains a single probe; sweep drives the full grid of
model type x embedding x target and can chain evaluation afterwards.

Model types: fnn (feed-forward), linear (on concatenated embeddings),
linear_distance (on the embedding difference) and euclidean
(training-free distance baseline).
"""

from __future__ import annotations

import typer

from plm_choice.bridge import PASSTHROUGH_CONTEXT, run_module_main

app = typer.Typer(help=__doc__, no_args_is_help=True)


def _cmd(name: str, *, help_: str):
    def decorator(fn):
        return app.command(
            name,
            help=help_,
            context_settings=PASSTHROUGH_CONTEXT,
            add_help_option=False,
        )(fn)

    return decorator


@_cmd("run", help_="Train one probe (one model type, one embedding, one target).")
def run(ctx: typer.Context) -> None:
    run_module_main("training.train", ctx.args, prog="plm train run")


@_cmd(
    "sweep",
    help_="Train the full grid of model types x embeddings x targets; --evaluate_after_train to chain evaluation.",
)
def sweep(ctx: typer.Context) -> None:
    run_module_main("training.run_experiments", ctx.args, prog="plm train sweep")
