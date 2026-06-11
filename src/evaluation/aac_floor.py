"""AAC-vector bridge for the AAC-floor arm (Unit 1).

Spec: docs/superpowers/specs/2026-06-11-aac-floor-design.md §3 Unit 1

Provides a single entry point, :func:`build_aac_embeddings`, that converts a FASTA
file into the ``{pid: (20,) float32}`` (or ``(21,)`` with ``include_other=True``)
dict that :func:`evaluation.recall_fp.recall_at_first_fp` consumes.

Design decisions:
- **D2**: 20-d is the headline floor (``include_other=False``); ``include_other=True``
  gives 21-d with a non-standard-AA bucket.
- **D6**: AAC is computed on the fly from the canonical FASTA — no pre-produced H5.
- **I2 (fan-fix)**: :func:`evaluation.canonical_set.parse_fasta` returns
  ``list[tuple[str, str]]``, NOT a dict. This bridge does ``dict(parse_fasta(path))``
  before calling ``extract_aac``.
- Missing frozen ids → ``ValueError`` (data fault, not a silent drop).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from data_preparation.aac import extract_aac
from evaluation.canonical_set import parse_fasta


def build_aac_embeddings(
    fasta_path: Path | str,
    *,
    expected_ids: list[str],
    include_other: bool = False,
) -> dict[str, np.ndarray]:
    """Build AAC frequency vectors for a set of expected protein ids.

    Parses ``fasta_path``, subsets to ``expected_ids``, and runs the AAC
    featurizer.  Returns ``{pid: ndarray}`` in the exact shape that
    :func:`evaluation.recall_fp.recall_at_first_fp` consumes:

    - ``include_other=False`` (default) → ``(20,)`` float32 per protein.
      Frequencies sum to ≤ 1 when non-standard AAs are present (they are dropped).
    - ``include_other=True`` → ``(21,)`` float32; non-standard AAs go into bucket 20.
      Frequencies always sum to 1.

    Parameters
    ----------
    fasta_path:
        Path to the FASTA file (canonical source).
    expected_ids:
        The frozen id set every analysis arm must score (e.g. from
        :func:`evaluation.analysis_io.load_frozen_ids`).  Every id in
        ``expected_ids`` must be present in the FASTA.
    include_other:
        If ``True``, collapse non-standard AAs into a 21st bucket so
        frequencies sum to exactly 1.  Default ``False`` (20-d, true floor).

    Returns
    -------
    dict[str, np.ndarray]
        ``{pid: frequency_vector}`` for every id in ``expected_ids``.

    Raises
    ------
    ValueError
        If any ``expected_id`` is absent from the FASTA (data fault).
    ValueError
        If ``expected_ids`` is empty, or if no matching proteins remain after
        subsetting (empty-after-subset guard).
    """
    # parse_fasta returns list[tuple[str, str]] (I2 fan-fix: NOT a dict)
    fasta_list = parse_fasta(fasta_path)
    fasta_dict: dict[str, str] = dict(fasta_list)

    expected = list(expected_ids)
    if not expected:
        raise ValueError(
            "expected_ids is empty; at least one protein id is required."
        )

    # Check for missing frozen ids — any absence is a data fault.
    missing = sorted(pid for pid in expected if pid not in fasta_dict)
    if missing:
        raise ValueError(
            f"{len(missing)} expected id(s) absent from FASTA "
            f"{Path(fasta_path).name!r}: {missing}"
        )

    # Subset to only expected ids (drop any extra records in the FASTA).
    subset_dict: dict[str, str] = {pid: fasta_dict[pid] for pid in expected}

    # extract_aac: normalize=True (frequency), reduce=True (per-protein vector)
    return extract_aac(subset_dict, normalize=True, include_other=include_other, reduce=True)
