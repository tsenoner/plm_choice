"""Root Typer application for the ``plm`` command.

Design notes
------------
* Command bodies import their heavy dependencies (torch, lightning, h5py)
  *inside* the body. Keeping the module top level free of them is what makes
  ``plm --help`` respond in milliseconds instead of seconds.
* ``help_option_names`` is declared once here; Click copies it into every child
  context, so ``-h`` works at every level without repeating the setting.
* Exit codes follow the analysis-DAG convention already used by the wrapped
  modules: ``0`` success, ``1`` a data-level failure, ``2`` an operator or
  config fault. Typer's own usage errors also exit ``2``, which is the same
  class of problem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from plm_choice import __version__
from plm_choice.groups import data as data_group
from plm_choice.groups import embed as embed_group
from plm_choice.groups import evaluate as evaluate_group
from plm_choice.groups import figures as figures_group
from plm_choice.groups import train as train_group

app = typer.Typer(
    name="plm",
    help=(
        "Which pLM to choose? — benchmark protein language model embeddings "
        "against sequence, structure and function similarity.\n\n"
        "Run [bold]plm stages[/bold] to see the analysis pipeline in order, or "
        "[bold]plm doctor[/bold] to check whether a fresh clone has what it needs."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(data_group.app, name="data", rich_help_panel="Pipeline")
app.add_typer(embed_group.app, name="embed", rich_help_panel="Pipeline")
app.add_typer(train_group.app, name="train", rich_help_panel="Pipeline")
app.add_typer(evaluate_group.app, name="evaluate", rich_help_panel="Pipeline")
app.add_typer(figures_group.app, name="figures", rich_help_panel="Pipeline")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"plm {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Which pLM to choose? — analysis toolkit."""


# ── Ordered description of the analysis DAG ───────────────────────────────────
# (stage label, command, one-line purpose)
_STAGES: tuple[tuple[str, str, str], ...] = (
    ("1. Cohort", "plm data novel-2024", "derive the novel-2024 protein cohort"),
    ("2. Embed", "plm embed generate", "run each pLM over the cohort FASTA -> HDF5"),
    ("", "plm embed random", "the untrained random-vector floor"),
    ("3. Pairs", "plm data merge", "merge MMseqs2 + Foldseek hits into a pair table"),
    ("", "plm data split", "split the pair table into train/val/test"),
    ("4. Probes", "plm train sweep", "train the model-type x embedding x target grid"),
    ("5. Metrics", "plm evaluate run-many", "evaluate every trained run"),
    ("", "plm evaluate recall-fp", "retrieval read-out (recall to first false positive)"),
    ("", "plm evaluate ec", "functional axis: EC hierarchical distance"),
    ("", "plm evaluate aac-floor", "amino-acid-composition floor"),
    ("6. Gate", "plm evaluate spec-merge", "combine the per-family barrier specs"),
    ("", "plm evaluate barrier", "check the artifacts against the spec"),
    ("7. Figures", "plm figures summary", "performance-vs-size panels"),
    ("", "plm figures pairwise", "embedding-space comparison panels"),
)


@app.command("stages", rich_help_panel="Orientation")
def stages() -> None:
    """Print the analysis pipeline in dependency order."""
    width = max(len(cmd) for _, cmd, _ in _STAGES)
    for label, cmd, purpose in _STAGES:
        prefix = f"{label:<11}" if label else " " * 11
        typer.echo(f"{prefix}{cmd:<{width}}  {purpose}")


@app.command("doctor", rich_help_panel="Orientation")
def doctor(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", help="Root of the bulk data tree."),
    ] = Path("data"),
) -> None:
    """Report whether this checkout has the inputs needed to reproduce results.

    Exits 0 when the environment looks usable, 1 when something required is
    missing. Nothing is downloaded or modified.
    """
    import importlib.util
    import shutil

    ok = True

    typer.echo(f"plm {__version__}")

    # Python packages the analysis needs at import time.
    for mod in ("numpy", "pandas", "polars", "h5py", "matplotlib", "scipy", "torch"):
        found = importlib.util.find_spec(mod) is not None
        typer.echo(f"  [{'ok ' if found else 'MISS'}] python package {mod}")
        ok &= found

    # First-party packages — proves the src-layout install actually resolved.
    for mod in ("evaluation", "training", "visualization", "data_preparation", "shared"):
        found = importlib.util.find_spec(mod) is not None
        typer.echo(f"  [{'ok ' if found else 'MISS'}] analysis package {mod}")
        ok &= found

    # External tools; absence is a warning, not a failure — most steps run without them.
    for tool in ("mmseqs", "foldseek"):
        typer.echo(
            f"  [{'ok ' if shutil.which(tool) else 'warn'}] external tool {tool}"
            f"{'' if shutil.which(tool) else '  (only needed to rebuild pair tables)'}"
        )

    # Data roots.
    for label, path in (
        ("embeddings/datasets", data_dir),
        ("frozen manifests", Path("freeze")),
    ):
        present = path.is_dir()
        typer.echo(f"  [{'ok ' if present else 'warn'}] {label}: {path}")

    embeddings = sorted(data_dir.rglob("*.h5")) if data_dir.is_dir() else []
    typer.echo(f"  [{'ok ' if embeddings else 'warn'}] {len(embeddings)} .h5 embedding file(s) found")
    if not embeddings:
        typer.echo(
            "        fetch them from Zenodo (concept DOI 10.5281/zenodo.17469267)"
        )

    if not ok:
        typer.echo("\nSomething required is missing — try `uv sync --locked`.", err=True)
        raise typer.Exit(code=1)
    typer.echo("\nEnvironment looks usable.")


if __name__ == "__main__":  # pragma: no cover
    app()
