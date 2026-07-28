"""Build the protein cohorts and their pair tables.

Pipeline order: fetch/derive a cohort -> merge the homology-search hits
into one pair table -> split it into train/val/test -> optionally
subset for fast iteration. canonical-set and ec-freeze produce the
frozen manifests the evaluation steps are checked against.

Options are forwarded verbatim to each underlying script; run
plm data <command> --help to see them.
"""

from __future__ import annotations

import typer

from plm_choice.bridge import (
    PASSTHROUGH_CONTEXT,
    run_argv_main,
    run_module_main,
    run_path_main,
    run_repo_script,
    repo_root,
)

app = typer.Typer(help=__doc__, no_args_is_help=True)

_COHORT = "Cohort assembly"
_FREEZE = "Frozen manifests"
_NOVEL = "2024-novel cohort"
_DIST = "Pairwise distances"

_NOVEL_DIR = "src/data_preparation/2024_new_proteins"


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


# ── Cohort assembly ───────────────────────────────────────────────────────────


@_cmd(
    "merge",
    panel=_COHORT,
    help_="Merge MMseqs2 + Foldseek hits into one protein-pair table (fident/HFSP/TM).",
)
def merge(ctx: typer.Context) -> None:
    run_module_main("data_preparation.merge_datasets", ctx.args, prog="plm data merge")


@_cmd(
    "split",
    panel=_COHORT,
    help_="Split a merged pair table into train/val/test parquet files.",
)
def split(ctx: typer.Context) -> None:
    run_module_main("data_preparation.split_dataset", ctx.args, prog="plm data split")


@_cmd(
    "subset",
    panel=_COHORT,
    help_="Draw a smaller subset of an existing split for fast iteration.",
)
def subset(ctx: typer.Context) -> None:
    run_repo_script("scripts/create_subset_datasets.py", ctx.args, prog="plm data subset")


@_cmd(
    "explore-subset",
    panel=_COHORT,
    help_="Plot and summarise how a subset's distributions compare to the full set.",
)
def explore_subset(ctx: typer.Context) -> None:
    run_repo_script(
        "scripts/explore_subset_distribution.py", ctx.args, prog="plm data explore-subset"
    )


# ── Frozen manifests ──────────────────────────────────────────────────────────


@_cmd(
    "canonical-set",
    panel=_FREEZE,
    help_="Freeze a canonical protein set (sha256 + ids + pair index + coverage).",
)
def canonical_set(ctx: typer.Context) -> None:
    from evaluation.canonical_set import main

    run_argv_main(main, ctx.args)


@_cmd(
    "ec-freeze",
    panel=_FREEZE,
    help_="Derive the EC-label freeze consumed by `plm evaluate ec`.",
)
def ec_freeze(ctx: typer.Context) -> None:
    run_repo_script("scripts/derive_ec_freeze.py", ctx.args, prog="plm data ec-freeze")


# ── Pairwise distances ────────────────────────────────────────────────────────


@_cmd(
    "distances",
    panel=_DIST,
    help_="Compute embedding distances for the pairs listed in a dataset.",
)
def distances(ctx: typer.Context) -> None:
    run_module_main(
        "data_preparation.distance_computation", ctx.args, prog="plm data distances"
    )


@_cmd(
    "all-vs-all",
    panel=_DIST,
    help_="Exact N x N distance table across every embedding file, plus viz caches.",
)
def all_vs_all(ctx: typer.Context) -> None:
    run_module_main(
        "data_preparation.all_vs_all_distance_computation",
        ctx.args,
        prog="plm data all-vs-all",
    )


# ── 2024-novel cohort ─────────────────────────────────────────────────────────
# These live in a directory whose name starts with a digit and has no
# __init__.py, so they are reachable only as file paths, never as modules.


@_cmd(
    "uniref-index",
    panel=_NOVEL,
    help_="Index a UniRef50 XML dump into SQLite (source-checkout only).",
)
def uniref_index(ctx: typer.Context) -> None:
    run_path_main(
        repo_root() / _NOVEL_DIR / "extract_uniref_to_sqlite.py",
        ctx.args,
        prog="plm data uniref-index",
    )


@_cmd(
    "novel-2024",
    panel=_NOVEL,
    help_="Identify proteins new in 2024 and dissimilar to the earlier release.",
)
def novel_2024(ctx: typer.Context) -> None:
    run_path_main(
        repo_root() / _NOVEL_DIR / "identify_novel_dissimilar_proteins.py",
        ctx.args,
        prog="plm data novel-2024",
    )


@_cmd(
    "plddt",
    panel=_NOVEL,
    help_="Extract per-model pLDDT from a ColabFold output tree.",
)
def plddt(ctx: typer.Context) -> None:
    run_repo_script(
        "scripts/extract_plddt_from_colabfold.py", ctx.args, prog="plm data plddt"
    )


@_cmd(
    "best-pdbs",
    panel=_NOVEL,
    help_="Pick the highest-ranked ColabFold PDB per protein.",
)
def best_pdbs(ctx: typer.Context) -> None:
    run_repo_script(
        "scripts/get_best_colabfold_pdbs.py", ctx.args, prog="plm data best-pdbs"
    )
