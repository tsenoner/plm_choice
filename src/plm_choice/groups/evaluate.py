"""Run the analysis-DAG steps that produce reviewer-facing numbers.

Every command here is a *passthrough*: Typer contributes the command name and
hands the remaining argv straight to the module's own argparse. No flag is
re-declared, so these commands cannot drift from the modules they expose, and
the main(argv) -> int contract that fifteen test files assert is untouched.

Use plm evaluate <command> --help to see a module's real options.
"""

from __future__ import annotations

import typer

from plm_choice.bridge import PASSTHROUGH_CONTEXT, run_argv_main

app = typer.Typer(
    help=__doc__,
    no_args_is_help=True,
)

_REPORTS = "Analysis steps (reports)"
_SPECS = "Fan-in barrier specs"
_BARRIER = "Barrier & verification"
_PROBES = "Trained-probe inference"


def _passthrough(name: str, *, panel: str, help_: str):
    """Declare a command that forwards raw argv to a module's argparse."""

    def decorator(fn):
        return app.command(
            name,
            help=help_,
            rich_help_panel=panel,
            context_settings=PASSTHROUGH_CONTEXT,
            add_help_option=False,
        )(fn)

    return decorator


# ── Analysis steps ────────────────────────────────────────────────────────────


@_passthrough(
    "ec",
    panel=_REPORTS,
    help_="EC hierarchical-distance vs embedding-distance correlation.",
)
def ec(ctx: typer.Context) -> None:
    from evaluation.ec_report import main

    run_argv_main(main, ctx.args)


@_passthrough(
    "recall-fp",
    panel=_REPORTS,
    help_="Recall-to-first-false-positive and AUROC at SCOP family/superfamily/fold.",
)
def recall_fp(ctx: typer.Context) -> None:
    from evaluation.recall_fp_report import main

    run_argv_main(main, ctx.args)


@_passthrough(
    "aac-floor",
    panel=_REPORTS,
    help_="Amino-acid-composition floor: how much signal a composition probe recovers.",
)
def aac_floor(ctx: typer.Context) -> None:
    from evaluation.aac_floor_report import main

    run_argv_main(main, ctx.args)


@_passthrough(
    "orphan",
    panel=_REPORTS,
    help_="Sibling-AUROC of embedding cosine on an orphan-family set (vertex-BCa CI).",
)
def orphan(ctx: typer.Context) -> None:
    from evaluation.orphan_report import main

    run_argv_main(main, ctx.args)


@_passthrough(
    "snn",
    panel=_REPORTS,
    help_="Shared-nearest-neighbour agreement between two pLM embedding spaces.",
)
def snn(ctx: typer.Context) -> None:
    from evaluation.snn_report import main

    run_argv_main(main, ctx.args)


@_passthrough(
    "cross-plm",
    panel=_REPORTS,
    help_="Cross-pLM agreement: W1 distance and permutation test between two spaces.",
)
def cross_plm(ctx: typer.Context) -> None:
    from evaluation.cross_plm_report import main

    run_argv_main(main, ctx.args)


# NOTE: evaluation/pdb_tm_bias.py is deliberately absent from this group. It is a
# library (pdb_bias_report / paired_tm_delta / r2_pLM_distance_vs_tm) with no
# `main` and no `__main__` block, so there is nothing to wrap yet. Give it a
# `main(argv) -> int` in the DAG dialect and it can be added here unchanged.


@_passthrough(
    "floor-comparison",
    panel=_REPORTS,
    help_="Compare a model against its floor; pass --apply-holm for the Holm correction.",
)
def floor_comparison(ctx: typer.Context) -> None:
    from evaluation.floor_comparison import main

    run_argv_main(main, ctx.args)


# ── Fan-in barrier specs ──────────────────────────────────────────────────────


@_passthrough(
    "spec-recall-fp",
    panel=_SPECS,
    help_="Build the recall-fp family's fan-in barrier spec.",
)
def spec_recall_fp(ctx: typer.Context) -> None:
    from evaluation.recall_fp_barrier_spec import main

    run_argv_main(main, ctx.args)


@_passthrough("spec-ec", panel=_SPECS, help_="Build the EC family's barrier spec.")
def spec_ec(ctx: typer.Context) -> None:
    from evaluation.ec_barrier_spec import main

    run_argv_main(main, ctx.args)


@_passthrough(
    "spec-aac-floor", panel=_SPECS, help_="Build the AAC-floor family's barrier spec."
)
def spec_aac_floor(ctx: typer.Context) -> None:
    from evaluation.aac_floor_barrier_spec import main

    run_argv_main(main, ctx.args)


@_passthrough(
    "spec-orphan", panel=_SPECS, help_="Build the orphan family's barrier spec."
)
def spec_orphan(ctx: typer.Context) -> None:
    from evaluation.orphan_barrier_spec import main

    run_argv_main(main, ctx.args)


@_passthrough("spec-snn", panel=_SPECS, help_="Build the SNN family's barrier spec.")
def spec_snn(ctx: typer.Context) -> None:
    from evaluation.snn_barrier_spec import main

    run_argv_main(main, ctx.args)


@_passthrough(
    "spec-cross-plm", panel=_SPECS, help_="Build the cross-pLM family's barrier spec."
)
def spec_cross_plm(ctx: typer.Context) -> None:
    from evaluation.cross_plm_barrier_spec import main

    run_argv_main(main, ctx.args)


@_passthrough(
    "spec-merge", panel=_SPECS, help_="Merge several family specs into one barrier spec."
)
def spec_merge(ctx: typer.Context) -> None:
    from evaluation.spec_merge import main

    run_argv_main(main, ctx.args)


# ── Barrier & verification ────────────────────────────────────────────────────


@_passthrough(
    "barrier",
    panel=_BARRIER,
    help_="Check a barrier spec against the produced artifacts (the analysis gate).",
)
def barrier(ctx: typer.Context) -> None:
    from evaluation.analysis_barrier import main

    run_argv_main(main, ctx.args)


@_passthrough(
    "verify",
    panel=_BARRIER,
    help_="Verify a frozen canonical set against its manifest (sha256 + ids).",
)
def verify(ctx: typer.Context) -> None:
    from evaluation.verify_analysis import main

    run_argv_main(main, ctx.args)


# ── Trained-probe inference ───────────────────────────────────────────────────


@_passthrough(
    "run",
    panel=_PROBES,
    help_="Evaluate one trained probe run directory (metrics + plots).",
)
def run(ctx: typer.Context) -> None:
    from plm_choice.bridge import run_module_main

    run_module_main("evaluation.evaluate", ctx.args, prog="plm evaluate run")


@_passthrough(
    "run-many",
    panel=_PROBES,
    help_="Fan `plm evaluate run` out over every run directory under a root.",
)
def run_many(ctx: typer.Context) -> None:
    from plm_choice.bridge import run_module_main

    run_module_main("evaluation.evaluate_multiple", ctx.args, prog="plm evaluate run-many")


@_passthrough(
    "infer-pairs",
    panel=_PROBES,
    help_="Predict a pair file with one trained probe; writes <pairs>_<plm>_pred.npz.",
)
def infer_pairs(ctx: typer.Context) -> None:
    from plm_choice.bridge import run_module_main

    run_module_main("evaluation.infer_pairs", ctx.args, prog="plm evaluate infer-pairs")


@_passthrough(
    "infer-batch",
    panel=_PROBES,
    help_="Fan `infer-pairs` out over several embeddings, skipping finished outputs.",
)
def infer_batch(ctx: typer.Context) -> None:
    from plm_choice.bridge import run_module_main

    run_module_main("evaluation.batch_inferer", ctx.args, prog="plm evaluate infer-batch")
