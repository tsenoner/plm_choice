"""Tests for the `plm` Typer CLI.

The CLI is a thin front-end: it must expose every analysis entry point and
forward exit codes faithfully, without changing any wrapped module's behaviour.
These tests pin exactly that contract.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from plm_choice.cli import app

runner = CliRunner()


def _command_names(typer_app) -> list[str]:
    """Registered command names on a Typer app (sub-apps excluded)."""
    return [c.name or c.callback.__name__ for c in typer_app.registered_commands]


def _group(name: str):
    for group in app.registered_groups:
        if group.name == name:
            return group.typer_instance
    raise AssertionError(f"no such command group: {name}")


GROUPS = ["data", "embed", "train", "evaluate", "figures"]


def test_root_help_lists_every_group():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in GROUPS:
        assert group in result.output


def test_short_help_flag_is_wired():
    """-h is declared once on the root and must reach the leaves."""
    assert runner.invoke(app, ["-h"]).exit_code == 0
    assert runner.invoke(app, ["evaluate", "-h"]).exit_code == 0


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip().startswith("plm ")


def test_no_args_shows_help():
    # Typer's no_args_is_help exits 2 (a usage fault), matching the analysis-DAG
    # convention where 2 means "operator/config fault".
    result = runner.invoke(app, [])
    assert result.exit_code == 2


def test_unknown_command_is_a_usage_error():
    assert runner.invoke(app, ["not-a-group"]).exit_code == 2


@pytest.mark.parametrize("group", GROUPS)
def test_group_help_renders(group):
    result = runner.invoke(app, [group, "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.parametrize("group", GROUPS)
def test_every_command_has_a_help_string(group):
    """A command with no help renders as a blank row in the group listing."""
    for command in _group(group).registered_commands:
        assert command.help, f"{group} {command.name} has no help text"


def test_stages_lists_the_pipeline_in_order():
    result = runner.invoke(app, ["stages"])
    assert result.exit_code == 0
    # The DAG order is the point of this command: cohort before embeddings
    # before pairs before probes before figures.
    order = ["plm data novel-2024", "plm embed generate", "plm data merge",
             "plm train sweep", "plm evaluate barrier", "plm figures summary"]
    positions = [result.output.index(step) for step in order]
    assert positions == sorted(positions)


def test_doctor_reports_first_party_packages():
    result = runner.invoke(app, ["doctor"])
    # Exit 0 or 1 are both legitimate (1 = something missing); what must hold is
    # that it checked the first-party packages the src-layout install provides.
    assert result.exit_code in (0, 1)
    for package in ("evaluation", "training", "visualization", "data_preparation", "shared"):
        assert package in result.output


# ── The wrapped-module contract ───────────────────────────────────────────────
#
# Fifteen test files assert `main(argv) -> int` on the evaluation modules. The
# CLI must expose them without re-declaring a single flag, so that it cannot
# drift from the module it wraps.

PASSTHROUGH_COMMANDS = [
    ("evaluate", "ec", "evaluation.ec_report"),
    ("evaluate", "recall-fp", "evaluation.recall_fp_report"),
    ("evaluate", "aac-floor", "evaluation.aac_floor_report"),
    ("evaluate", "orphan", "evaluation.orphan_report"),
    ("evaluate", "snn", "evaluation.snn_report"),
    ("evaluate", "cross-plm", "evaluation.cross_plm_report"),
    ("evaluate", "floor-comparison", "evaluation.floor_comparison"),
    ("evaluate", "spec-recall-fp", "evaluation.recall_fp_barrier_spec"),
    ("evaluate", "spec-ec", "evaluation.ec_barrier_spec"),
    ("evaluate", "spec-aac-floor", "evaluation.aac_floor_barrier_spec"),
    ("evaluate", "spec-orphan", "evaluation.orphan_barrier_spec"),
    ("evaluate", "spec-snn", "evaluation.snn_barrier_spec"),
    ("evaluate", "spec-cross-plm", "evaluation.cross_plm_barrier_spec"),
    ("evaluate", "spec-merge", "evaluation.spec_merge"),
    ("evaluate", "barrier", "evaluation.analysis_barrier"),
    ("evaluate", "verify", "evaluation.verify_analysis"),
    ("data", "canonical-set", "evaluation.canonical_set"),
]


@pytest.mark.parametrize("group,command,module", PASSTHROUGH_COMMANDS)
def test_passthrough_target_exposes_argv_main(group, command, module):
    """Each passthrough target must still be a `main(argv) -> int`."""
    import importlib
    import inspect

    main = importlib.import_module(module).main
    params = list(inspect.signature(main).parameters)
    assert params and params[0] == "argv", f"{module}.main lost its argv parameter"


@pytest.mark.parametrize("group,command,module", PASSTHROUGH_COMMANDS)
def test_passthrough_command_is_registered(group, command, module):
    assert command in _command_names(_group(group))


def test_passthrough_forwards_argparse_usage_error_as_exit_2():
    """Missing required args must surface argparse's own error and exit 2."""
    result = runner.invoke(app, ["data", "canonical-set"])
    assert result.exit_code == 2


def test_passthrough_forwards_module_help_not_typer_help():
    """`--help` must reach the wrapped argparse, not be eaten by Typer."""
    result = runner.invoke(app, ["evaluate", "ec", "--help"])
    assert result.exit_code == 0
    # argparse renders its prog name; Typer would have rendered "plm evaluate ec".
    assert "ec_report" in result.output


def test_run_argv_main_translates_exit_code():
    """Typer discards a returned int, so the bridge must raise typer.Exit."""
    import typer

    from plm_choice.bridge import run_argv_main

    for code in (0, 1, 2, 7):
        with pytest.raises(typer.Exit) as excinfo:
            run_argv_main(lambda argv, _c=code: _c, [])
        assert excinfo.value.exit_code == code


def test_run_argv_main_treats_none_as_success():
    import typer

    from plm_choice.bridge import run_argv_main

    with pytest.raises(typer.Exit) as excinfo:
        run_argv_main(lambda argv: None, [])
    assert excinfo.value.exit_code == 0


# ── runpy-bridged legacy modules ──────────────────────────────────────────────

RUNPY_COMMANDS = [
    ("data", "merge", "data_preparation.merge_datasets"),
    ("data", "split", "data_preparation.split_dataset"),
    ("data", "distances", "data_preparation.distance_computation"),
    ("embed", "generate", "data_preparation.embeddings.embedding_generation"),
    ("embed", "random", "data_preparation.embeddings.random_embeddings"),
    ("train", "run", "training.train"),
    ("train", "sweep", "training.run_experiments"),
    ("figures", "summary", "visualization.create_performance_summary_plots"),
    ("figures", "grid", "visualization.create_evaluation_grid_plots"),
    ("figures", "pairwise", "visualization.pairwise_embedding_comparison"),
    ("figures", "comparison", "visualization.create_embedding_comparison_plots"),
    ("figures", "ecdf", "visualization.plot_ecdf"),
    ("figures", "retrieval", "visualization.create_retrieval_plots"),
    ("data", "go-similarity", "data_preparation.go_semantic_similarity"),
    ("data", "ec-distance", "data_preparation.brenda_hfsp_validation"),
    ("data", "pdb-tmscore", "data_preparation.pdb_tmscore"),
    ("data", "ecod-pairs", "data_preparation.ecod_homology_pairs"),
    ("data", "organisms", "data_preparation.organism_landscape"),
    ("data", "merge-columns", "data_preparation.merge_parquet_columns"),
    ("evaluate", "classification", "evaluation.classification_eval"),
    ("evaluate", "overtraining", "evaluation.overtraining_analysis"),
    ("evaluate", "run", "evaluation.evaluate"),
    ("evaluate", "run-many", "evaluation.evaluate_multiple"),
    ("evaluate", "infer-pairs", "evaluation.infer_pairs"),
    ("evaluate", "infer-batch", "evaluation.batch_inferer"),
]


@pytest.mark.parametrize("group,command,module", RUNPY_COMMANDS)
def test_runpy_command_is_registered(group, command, module):
    assert command in _command_names(_group(group))


@pytest.mark.parametrize("group,command,module", RUNPY_COMMANDS)
def test_runpy_target_module_is_importable(group, command, module):
    """The dotted path the bridge will hand to runpy must actually resolve."""
    import importlib.util

    assert importlib.util.find_spec(module) is not None, f"{module} is not importable"


def test_bridge_normalises_string_exit_payload(capsys):
    """sys.exit("message") must become exit 1 with the message on stderr."""
    from plm_choice.bridge import _exit_code

    assert _exit_code(None) == 0
    assert _exit_code(0) == 0
    assert _exit_code(3) == 3
    assert _exit_code("boom") == 1
    assert "boom" in capsys.readouterr().err
