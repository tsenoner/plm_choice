"""Command-line front-end for the "Which pLM to choose?" analysis code.

This package deliberately contains *only* the CLI. The analysis itself lives in
the sibling top-level packages ``data_preparation``, ``evaluation``, ``shared``,
``training`` and ``visualization``, and the CLI wraps those modules without
editing them (see :mod:`plm_choice.bridge` for how).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
