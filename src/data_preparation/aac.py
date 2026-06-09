"""Amino-acid composition (AAC) baseline.

20-d (or 21-d with an ``other`` bucket) per-protein composition vector — the
absolute lower bound for any embedding-based metric, required as a floor in
Figure 1 of the pLM-choice paper.

Standard AAs (alphabetical): ACDEFGHIKLMNPQRSTVWY (20).
Non-standard / ambiguous: BJOUXZ — collapsed into a 21st "other" bucket when
``include_other=True``, otherwise silently dropped (per-protein frequencies then
sum to <=1 instead of =1).
"""

from __future__ import annotations

import numpy as np

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
_STD_INDEX = {aa: i for i, aa in enumerate(STANDARD_AA)}


def extract_aac(
    fasta_dict: dict[str, str],
    normalize: bool = True,
    include_other: bool = True,
    reduce: bool = True,
) -> dict[str, np.ndarray]:
    """Compute amino-acid composition per protein.

    Args:
        fasta_dict: ``{protein_id: amino_acid_sequence}``.
        normalize: If True, divide counts by sequence length so the vector is a
            frequency distribution. If False, return raw counts.
        include_other: If True, collapse non-standard AAs (BJOUXZ + anything
            else) into a 21st "other" bucket so frequencies sum to exactly 1.
            If False, drop them — frequencies then sum to <=1.
        reduce: If True (default), return per-protein ``(D,)`` vector. If False,
            return per-residue ``(L, D)`` one-hot encoding.

    Returns:
        ``{protein_id: np.ndarray}``:
            - reduce=True:  shape ``(20,)`` or ``(21,)`` float32.
            - reduce=False: shape ``(L, 20)`` or ``(L, 21)`` float32 (one-hot
              per position).
    """
    n_dim = 21 if include_other else 20
    out: dict[str, np.ndarray] = {}

    for pid, seq in fasta_dict.items():
        length = len(seq)
        if reduce:
            vec = np.zeros(n_dim, dtype=np.float32)
            for aa in seq.upper():
                idx = _STD_INDEX.get(aa)
                if idx is not None:
                    vec[idx] += 1.0
                elif include_other:
                    vec[20] += 1.0
                # else: silently drop
            if normalize and length > 0:
                vec /= float(length)
            out[pid] = vec
        else:
            onehot = np.zeros((length, n_dim), dtype=np.float32)
            for j, aa in enumerate(seq.upper()):
                idx = _STD_INDEX.get(aa)
                if idx is not None:
                    onehot[j, idx] = 1.0
                elif include_other:
                    onehot[j, 20] = 1.0
            out[pid] = onehot

    return out
