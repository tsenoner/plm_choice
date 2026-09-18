"""All-pairs GO Molecular Function similarity matrices for the functional-transfer readout.

The transfer readout (:mod:`evaluation.transfer_report`) scores every query against
EVERY eligible neighbour, not only the nearest one: the chance baseline is the exact
mean over all eligible neighbours and the oracle is the best one by label. It therefore
needs two dense protein x protein matrices over the GO cohort:

* **propagated-set F1** -- each protein's annotated MF terms closed upward over
  ``is_a`` + ``part_of`` (the CAFA protein-centric convention), root excluded. The F1 of
  a transferred set N against the true set Q is ``2|Q & N| / (|Q| + |N|)``, which is
  symmetric, so one matrix serves both directions.
* **Wang (2007) best-match average** of the UNpropagated annotated sets, exactly as
  :meth:`data_preparation.go_semantic_similarity.WangSimilarity.protein_similarity_bma`
  defines it.

Why vectorised: a ~7k-protein cohort has ~25M unordered pairs, and the scalar BMA is a
Python double loop over term pairs per protein pair -- hours per run. The matrix form
below is exact (not an approximation), so it is validated against the scalar reference
rather than trusted: :func:`max_abs_diff_vs_scalar` is run by the tests on >=200 random
pairs and by the report CLI on the real cohort.

Matrix form of Wang BMA. With ``S[t, a]`` the Wang S-value of ancestor ``a`` for term
``t`` (``S[t, t] = 1``), ``M`` its support and ``SV = S.sum(1)``::

    sim(t, u)   = (S @ M.T + M @ S.T)[t, u] / (SV[t] + SV[u])
    A[i, u]     = max over t in T_i of sim(t, u)
    G           = A @ B.T              # B = protein x term indicator of annotated sets
    BMA(i, j)   = (G[i, j] + G[j, i]) / (|T_i| + |T_j|)

``G[i, j]`` is the sum over j's terms of their best match in i, so ``G[j, i]`` is the
forward half of the BMA and ``G[i, j]`` the backward half.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from data_preparation.go_semantic_similarity import EDGE_WEIGHTS, GOTerm, WangSimilarity

#: Root of the Molecular Function aspect. Every MF term has it as an ancestor, so it
#: carries no information as a scored term: it is removed from annotated sets and from
#: propagated sets. It still contributes to Wang S-values (the Wang definition sums over
#: all shared ancestors, root included), which is what the scalar reference does.
MF_ROOT = "GO:0003674"
MF_NAMESPACE = "molecular_function"
#: "protein binding": uninformative, IPI-driven and the single most common MF term; the
#: sensitivity variant of the GO arm drops it.
PROTEIN_BINDING = "GO:0005515"


# --------------------------------------------------------------------------- #
#                              ontology helpers
# --------------------------------------------------------------------------- #


def parse_alt_ids(obo_path: Path | str) -> dict[str, str]:
    """``alt_id -> primary id`` for every non-obsolete ``[Term]`` in an OBO file.

    ``parse_obo`` keys terms by their primary id only, but annotations in a 2024
    Swiss-Prot release can name a term by a secondary id that a later ontology merged
    away. Without this map such an annotation would look like an unknown term and be
    dropped, silently shrinking that protein's label set.
    """
    alt: dict[str, str] = {}
    in_term = False
    primary = ""
    pending: list[str] = []
    obsolete = False

    def _flush() -> None:
        if in_term and primary and not obsolete:
            for a in pending:
                alt[a] = primary

    with open(obo_path) as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("[") and line.endswith("]"):
                _flush()
                in_term = line == "[Term]"
                primary, pending, obsolete = "", [], False
            elif in_term and line.startswith("id: "):
                primary = line[4:].strip()
            elif in_term and line.startswith("alt_id: "):
                pending.append(line[8:].strip())
            elif in_term and line.startswith("is_obsolete: true"):
                obsolete = True
    _flush()
    return alt


def clean_mf_annotations(
    raw: Mapping[str, Iterable[str]],
    go_terms: Mapping[str, GOTerm],
    alt_ids: Mapping[str, str] | None = None,
    *,
    drop_terms: Iterable[str] = (),
) -> tuple[dict[str, frozenset[str]], dict[str, int]]:
    """Reduce raw per-protein GO ids to scoreable MF term sets.

    Steps, each counted in the returned dict: map secondary ids to primary ids; drop ids
    absent from the ontology (obsolete terms); drop non-MF terms; drop the MF root; drop
    ``drop_terms`` (the protein-binding sensitivity). Proteins left with no term are
    omitted from the result -- the caller decides whether that is fatal (a frozen id
    without labels) or expected (the sensitivity variant removes them by design).
    """
    alt_ids = alt_ids or {}
    drop = frozenset(drop_terms)
    counts = {"alt_id_mapped": 0, "not_in_ontology": 0, "not_mf": 0, "root": 0, "dropped_term": 0}
    out: dict[str, frozenset[str]] = {}
    for pid, terms in raw.items():
        kept: set[str] = set()
        for term in terms:
            if term not in go_terms and term in alt_ids:
                term = alt_ids[term]
                counts["alt_id_mapped"] += 1
            if term not in go_terms:
                counts["not_in_ontology"] += 1
            elif go_terms[term].namespace != MF_NAMESPACE:
                counts["not_mf"] += 1
            elif term == MF_ROOT:
                counts["root"] += 1
            elif term in drop:
                counts["dropped_term"] += 1
            else:
                kept.add(term)
        if kept:
            out[pid] = frozenset(kept)
    return out, counts


def wang_s_values(
    go_terms: Mapping[str, GOTerm], term_ids: Iterable[str]
) -> dict[str, dict[str, float]]:
    """Wang S-values ``{term: {ancestor: S_term(ancestor)}}`` for ``term_ids``.

    Computed by memoised recursion, ``S_t(t) = 1`` and
    ``S_t(a) = max over parents p of t of w(t, p) * S_p(a)``, which equals the
    max-over-paths product the scalar reference computes by depth-first search. It is
    deliberately a second, independent formulation: validating the matrix BMA against
    the scalar code would prove little if both read the same S-values.
    """
    memo: dict[str, dict[str, float]] = {}

    def _s(term: str) -> dict[str, float]:
        cached = memo.get(term)
        if cached is not None:
            return cached
        values = {term: 1.0}
        node = go_terms.get(term)
        if node is not None:
            for parent, relation in node.parents:
                weight = EDGE_WEIGHTS[relation]
                for anc, s in _s(parent).items():
                    v = weight * s
                    if v > values.get(anc, 0.0):
                        values[anc] = v
        memo[term] = values
        return values

    return {t: _s(t) for t in term_ids}


def propagate_mf(
    protein_terms: Sequence[frozenset[str]], go_terms: Mapping[str, GOTerm]
) -> list[frozenset[str]]:
    """Close each annotated set upward over ``is_a`` + ``part_of`` within MF, root excluded."""
    vocab = sorted(set().union(*protein_terms))
    ancestors = {
        t: frozenset(
            a for a in s if a != MF_ROOT and a in go_terms and go_terms[a].namespace == MF_NAMESPACE
        )
        for t, s in wang_s_values(go_terms, vocab).items()
    }
    return [frozenset().union(*(ancestors[t] for t in terms)) for terms in protein_terms]


# --------------------------------------------------------------------------- #
#                              set-overlap matrices
# --------------------------------------------------------------------------- #


def set_indicator(
    sets: Sequence[Iterable[str]], vocab: Sequence[str] | None = None
) -> tuple[sp.csr_matrix, list[str]]:
    """Sparse ``len(sets) x |vocab|`` 0/1 matrix (float64) and the column vocabulary."""
    sets = [frozenset(s) for s in sets]
    vocab = sorted(set().union(*sets)) if vocab is None else list(vocab)
    col = {v: k for k, v in enumerate(vocab)}
    indptr = np.zeros(len(sets) + 1, dtype=np.int64)
    indices: list[int] = []
    for i, s in enumerate(sets):
        indices.extend(sorted(col[v] for v in s))
        indptr[i + 1] = len(indices)
    data = np.ones(len(indices), dtype=np.float64)
    mat = sp.csr_matrix(
        (data, np.asarray(indices, dtype=np.int64), indptr), shape=(len(sets), len(vocab))
    )
    return mat, vocab


def shared_counts(sets: Sequence[Iterable[str]]) -> np.ndarray:
    """Dense ``n x n`` matrix of ``|set_i & set_j|`` (float64, exact for these sizes).

    Sparse times dense rather than sparse times sparse: nearly every pair of MF-propagated
    sets shares a term, so the product is dense and a sparse result would only cost more.
    """
    x, _ = set_indicator(sets)
    return np.asarray(x @ x.T.toarray())


def propagated_f1_matrix(
    protein_terms: Sequence[frozenset[str]], go_terms: Mapping[str, GOTerm]
) -> np.ndarray:
    """Protein-centric F1 between propagated MF sets, for every ordered pair.

    Raises ``ValueError`` if a protein's propagated set is empty (an annotation of the
    root only, which :func:`clean_mf_annotations` already removes).
    """
    propagated = propagate_mf(protein_terms, go_terms)
    sizes = np.array([len(s) for s in propagated], dtype=np.float64)
    if (sizes == 0).any():
        raise ValueError(f"{int((sizes == 0).sum())} protein(s) have an empty propagated MF set")
    shared = shared_counts(propagated)
    return 2.0 * shared / (sizes[:, None] + sizes[None, :])


# --------------------------------------------------------------------------- #
#                                Wang BMA
# --------------------------------------------------------------------------- #


def wang_term_similarity_matrix(
    terms: Sequence[str], go_terms: Mapping[str, GOTerm]
) -> np.ndarray:
    """Dense ``|terms| x |terms|`` Wang term similarity, ``(S M^T + M S^T) / (SV_t + SV_u)``.

    The diagonal is set to exactly 1.0: ``sim(t, t) = 1`` by definition, and the matrix
    sums can land one ulp off it, which would make an identical-set transfer score
    fractionally below the maximum the oracle tests for.
    """
    s_values = wang_s_values(go_terms, terms)
    ancestors = sorted(set().union(*(s.keys() for s in s_values.values())))
    col = {a: k for k, a in enumerate(ancestors)}
    rows, cols, vals = [], [], []
    for r, t in enumerate(terms):
        for anc, v in s_values[t].items():
            rows.append(r)
            cols.append(col[anc])
            vals.append(v)
    s_mat = sp.csr_matrix((vals, (rows, cols)), shape=(len(terms), len(ancestors)))
    support = s_mat.copy()
    support.data[:] = 1.0
    sv = np.asarray(s_mat.sum(axis=1)).ravel()
    cross = np.asarray(s_mat @ support.T.toarray())  # sum_a S_t(a) * [a in anc(u)]
    sim = (cross + cross.T) / (sv[:, None] + sv[None, :])
    np.fill_diagonal(sim, 1.0)
    return sim


def wang_bma_matrix(
    protein_terms: Sequence[frozenset[str]], go_terms: Mapping[str, GOTerm]
) -> np.ndarray:
    """Wang best-match-average similarity for every ordered protein pair (dense, symmetric).

    Only the per-protein row maximum ``A`` is a Python loop (one iteration per protein,
    not per pair); everything else is matrix algebra.
    """
    if any(not t for t in protein_terms):
        raise ValueError("every protein needs >=1 annotated term for Wang BMA")
    indicator, vocab = set_indicator(protein_terms)
    sim = wang_term_similarity_matrix(vocab, go_terms)
    pos = {t: k for k, t in enumerate(vocab)}
    best = np.empty((len(protein_terms), len(vocab)), dtype=np.float64)
    for i, terms in enumerate(protein_terms):
        np.max(sim[[pos[t] for t in terms]], axis=0, out=best[i])
    del sim
    # (B @ A.T)[j, i] = sum over u in T_j of max_{t in T_i} sim(t, u) = G[i, j].
    g = np.asarray(indicator @ np.ascontiguousarray(best.T)).T
    del best
    sizes = np.asarray(indicator.sum(axis=1)).ravel()
    return (g + g.T) / (sizes[:, None] + sizes[None, :])


def random_ordered_pairs(n: int, n_pairs: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, cols)`` of ``n_pairs`` random ordered ``i != j`` index pairs.

    Falls back to EVERY ordered pair when the cohort is too small to supply ``n_pairs``,
    so a tiny fixture still gets an exhaustive check rather than a sample with repeats.
    Shared by the in-module scalar validation and by the report CLI, which runs the same
    check on the real cohort: one sampling convention, so the two agree on what "200
    random pairs" means.
    """
    if n < 2:
        raise ValueError("need >=2 proteins to compare pairs")
    if n * (n - 1) <= n_pairs:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    else:
        rng = np.random.default_rng(seed)
        pairs = []
        while len(pairs) < n_pairs:
            i, j = (int(v) for v in rng.integers(0, n, size=2))
            if i != j:
                pairs.append((i, j))
    rows = np.array([p[0] for p in pairs], dtype=np.int64)
    cols = np.array([p[1] for p in pairs], dtype=np.int64)
    return rows, cols


def scalar_bma_values(
    protein_terms: Sequence[frozenset[str]],
    rows: np.ndarray,
    cols: np.ndarray,
    go_terms: Mapping[str, GOTerm],
) -> np.ndarray:
    """Reference BMA for the given index pairs, via ``WangSimilarity.protein_similarity_bma``.

    The slow, obviously-correct definition. It exists to be disagreed with: any deviation
    from the matrix form is a bug in the matrix form, never a rounding allowance.
    """
    wang = WangSimilarity(dict(go_terms))
    return np.array(
        [
            wang.protein_similarity_bma(set(protein_terms[i]), set(protein_terms[j]))
            for i, j in zip(rows, cols, strict=True)
        ],
        dtype=np.float64,
    )


def max_abs_diff_vs_scalar(
    protein_terms: Sequence[frozenset[str]],
    bma: np.ndarray,
    go_terms: Mapping[str, GOTerm],
    *,
    n_pairs: int = 200,
    seed: int = 42,
) -> tuple[float, int]:
    """Largest ``|bma[i, j] - scalar BMA(i, j)|`` over ``n_pairs`` random ``i != j`` pairs.

    The scalar reference is ``WangSimilarity.protein_similarity_bma``. Returns the max
    difference and the number of pairs checked (all ordered pairs if fewer exist).
    """
    rows, cols = random_ordered_pairs(len(protein_terms), n_pairs, seed)
    reference = scalar_bma_values(protein_terms, rows, cols, go_terms)
    worst = float(np.max(np.abs(np.asarray(bma)[rows, cols] - reference)))
    return worst, int(rows.size)
