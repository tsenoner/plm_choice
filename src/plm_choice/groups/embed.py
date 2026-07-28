"""Turn protein sequences into embedding matrices.

generate runs a pLM over a FASTA and streams per-protein vectors to HDF5.
random builds the untrained-baseline matrix with the same shape and ids.
pca reduces an existing set of embeddings.

The 15 embedding files used in the paper are published on Zenodo
(concept DOI 10.5281/zenodo.17469267) — fetching them is usually cheaper than
regenerating them.
"""

from __future__ import annotations

import typer

from plm_choice.bridge import PASSTHROUGH_CONTEXT, run_module_main, run_repo_script

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


@_cmd(
    "generate",
    help_="Embed a FASTA with one pLM (positional: FASTA_FILE MODEL_KEY) into HDF5.",
)
def generate(ctx: typer.Context) -> None:
    run_module_main(
        "data_preparation.embeddings.embedding_generation",
        ctx.args,
        prog="plm embed generate",
    )


@_cmd(
    "random",
    help_="Build a random-vector matrix matching a template HDF5 (untrained floor).",
)
def random(ctx: typer.Context) -> None:
    run_module_main(
        "data_preparation.embeddings.random_embeddings",
        ctx.args,
        prog="plm embed random",
    )


@_cmd("pca", help_="PCA-reduce a directory of embedding HDF5 files.")
def pca(ctx: typer.Context) -> None:
    run_repo_script("scripts/reduce_embeddings_pca.py", ctx.args, prog="plm embed pca")
