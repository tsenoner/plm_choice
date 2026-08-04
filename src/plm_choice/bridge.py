"""Bridges that expose the existing argparse entry points as Typer commands.

The repository has two CLI dialects, and neither is rewritten here:

``main(argv) -> int``
    The analysis-DAG modules (``evaluation/*_report.py``, ``*_barrier_spec.py``,
    ``canonical_set``, ``analysis_barrier``, ``spec_merge``, ``verify_analysis``,
    ``floor_comparison``). Fifteen test files assert the integer return of these
    functions, so they must keep working byte-for-byte. :func:`run_argv_main`
    wraps them.

``main()`` / ``main(Namespace)`` / bare ``if __name__ == "__main__"``
    The older data-prep, training and visualization modules — there is no
    callable that accepts ``argv``. :func:`run_module_main` re-executes them
    through :mod:`runpy` with a synthesised ``sys.argv`` instead, so the module
    itself needs no edit. This matters because several of these modules
    (``merge_datasets``, ``split_dataset``, ``run_experiments``, ``train``) are
    on the critical path of an in-flight data rebuild.

Two details are load-bearing:

* Typer **discards** a value returned from a command body (``typer/core.py``
  deliberately does not ``ctx.exit(rv)``), so an exit code has to be raised as
  ``typer.Exit``. Returning the int would silently always exit 0.
* Commands that forward raw argv are declared with ``allow_extra_args``,
  ``ignore_unknown_options`` and ``add_help_option=False`` so that the wrapped
  module's own argparse owns both ``--help`` and its usage errors.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path

import typer

#: ``context_settings`` for a command that forwards raw argv to argparse.
#: ``add_help_option=False`` is passed separately as a command kwarg.
PASSTHROUGH_CONTEXT: dict[str, bool] = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


@contextmanager
def _patched_argv(prog: str, argv: Sequence[str]):
    """Temporarily present ``[prog, *argv]`` as ``sys.argv``."""
    original = sys.argv
    sys.argv = [prog, *argv]
    try:
        yield
    finally:
        sys.argv = original


def run_argv_main(
    main: Callable[[Sequence[str] | None], int], argv: Sequence[str]
) -> None:
    """Run an ``main(argv) -> int`` entry point and honour its exit code.

    The wrapped modules follow a documented contract: ``0`` success, ``1`` a
    data-level failure, ``2`` an operator/config fault. Typer also uses ``2``
    for its own usage errors, which is the same class of problem, so the codes
    are forwarded unchanged rather than remapped — remapping would break the
    exit codes the DAG and the test-suite pin.
    """
    raise typer.Exit(code=int(main(list(argv)) or 0))


def run_module_main(dotted: str, argv: Sequence[str], *, prog: str) -> None:
    """Execute ``python -m <dotted> <argv>`` in-process, forwarding its exit code.

    Used for modules whose argparse lives inside ``main()`` or inside the
    ``if __name__ == "__main__"`` block, i.e. where there is nothing to hand an
    ``argv`` list to.
    """
    with _patched_argv(prog, argv):
        try:
            runpy.run_module(dotted, run_name="__main__", alter_sys=True)
        except SystemExit as exc:  # argparse --help and usage errors land here
            raise typer.Exit(code=_exit_code(exc.code)) from None
    raise typer.Exit(code=0)


def _exit_code(code: object) -> int:
    """Normalise ``SystemExit.code`` (which may be ``None``, an int, or a message)."""
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    # A string payload means the module called sys.exit("message").
    print(code, file=sys.stderr)
    return 1


def repo_root() -> Path:
    """Best-effort path to the source checkout.

    Only used by commands that shell out to files which are not shipped in the
    wheel (``scripts/``). Returns the directory containing ``pyproject.toml`` if
    one can be found by walking up from this file, else the current directory.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def run_repo_script(relative: str, argv: Sequence[str], *, prog: str) -> None:
    """Run a helper under ``scripts/`` from the source checkout.

    These are analysis one-offs that are intentionally not packaged into the
    wheel; they are only reachable when running from a clone.
    """
    script = repo_root() / relative
    if not script.is_file():
        raise typer.BadParameter(
            f"{relative} is only available in a source checkout (looked in "
            f"{repo_root()}). Clone the repository to use this command."
        )
    with _patched_argv(prog, argv):
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            raise typer.Exit(code=_exit_code(exc.code)) from None
    raise typer.Exit(code=0)
