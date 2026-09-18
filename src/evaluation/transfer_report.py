"""Leave-one-out 1-NN annotation transfer — the primary EC / GO-MF functional readout.

**Why this replaces the correlation readout.** The EC arm so far reports Kendall tau-b
between embedding distance and EC hierarchical distance over ALL pairs, and lands at
0.00-0.06 for every pLM. On the same embeddings, nearest-neighbour EC transfer is
53-81% correct. Both numbers are right: 99.8% of pairs are different-EC pairs whose
distance ordering carries no functional information, so tau-b is dominated by noise the
question does not care about. The field-standard readout (goPredSim / embedding-based
annotation transfer, Littmann et al. 2021) asks the question the benchmark is actually
about — *if I hand this embedding an unannotated protein and copy its nearest
neighbour's label, how often am I right?* — and that is what this module computes. tau-b
survives as a secondary number (``tau_b.csv``, GO only; the EC arm keeps ``ec_report``).

**Why the identity control is the point.** Reviewer R2.1 objects that our functional
ground truth is a sequence-identity proxy. A 1-NN transfer score alone would inherit the
objection, so every cell is computed under three neighbour-eligibility variants: all
other cohort proteins; only neighbours with no MMseqs2 hit at fident >= 0.30 in either
direction; only neighbours with no MMseqs2 hit at all (E <= 1e-3, either direction). The
gap between the first and the last is the part of the readout that sequence identity
cannot explain.

**The comparison the manuscript leads with.** The question a reader actually has is not
"is this pLM better than chance" but "is this pLM better than searching the sequence".
So the same transfer also runs with the neighbour chosen by MMseqs2 instead of by
embedding distance — homology-based inference (HBI), the arms ``hbi_evalue`` (best hit by
lowest E-value) and ``hbi_fident`` (best hit by highest fractional identity). They are
scored by the identical code on the identical queries, so a pLM minus ``hbi_evalue`` is a
paired per-query difference with a bootstrap CI like any other arm pair. HBI has one
structural weakness a mean over all queries would hide: where MMseqs2 finds no hit it
cannot answer at all. Every cell is therefore reported on three query subsets —
``all_queries`` (the variant's queries; embedding arms only, because HBI has no answer for
some of them), ``hbi_answerable`` (the queries HBI can answer, where the comparison is
like-for-like) and ``no_hit`` (the queries MMseqs2 leaves without a single cohort hit,
where only an embedding can answer at all). Under the identity variants the eligibility
rule applies to HBI too: it transfers from the best hit that is still *allowed*, so the
identity control is a stratification of both readouts rather than a gate on one of them.

**Baselines, not just a number.** Each cell also carries the exact chance expectation
(the mean of the score row over *eligible* neighbours, averaged over queries — an
expectation, not a sample) and the oracle (the mean over queries of the best score any
eligible neighbour could give). ``random_1024`` — i.i.d. noise — must land on chance;
that is the validity check for the whole pipeline.

Outputs in ``--out-dir``: ``summary.csv``, ``paired_differences.csv``,
``per_query.parquet``, ``per_query_baseline.parquet``, ``tau_b.csv`` (GO) and
``manifest.json``. In ``per_query.parquet`` the ``neighbour_distance`` column always holds
the quantity that was minimised to pick that neighbour: the embedding distance for a pLM
arm, the E-value for ``hbi_evalue``, ``1 - fident`` for ``hbi_fident`` (so an ``hbi_evalue``
0.0 is an E-value underflow, not a zero distance — the units are recorded in the
manifest). For the same reason ``summary.csv`` carries ``baseline_scope``: chance and
oracle on an HBI row are taken over that query's MMseqs2 hits, the only neighbours a
sequence search could have transferred from.

Usage::

    PYTHONPATH=src python -m evaluation.transfer_report \\
        --labels-kind go --freeze go_mf_cohort_freeze.json \\
        --labels go_mf_labels.tsv --go-obo data/reference/go/go-basic.obo \\
        --emb-dir slices/ --identity-m8 union_allvall.m8 --out-dir results/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from data_preparation.go_semantic_similarity import GOTerm, parse_obo
from evaluation.analysis_io import json_safe, load_frozen_ids
from evaluation.go_similarity_matrix import (
    PROTEIN_BINDING,
    clean_mf_annotations,
    parse_alt_ids,
    propagate_mf,
    random_ordered_pairs,
    scalar_bma_values,
    set_indicator,
    wang_term_similarity_matrix,
)
from evaluation.label_adapters import parse_ec
from evaluation.stats import kendall_tau_b

# ── input contracts ───────────────────────────────────────────────────────────
#: Header-less column order of the MMseqs2 easy-search table this module reads, i.e.
#: ``--format-output query,target,fident,evalue,alnlen,qcov,tcov``.
M8_COLUMNS: tuple[str, ...] = ("query", "target", "fident", "evalue", "alnlen", "qcov", "tcov")
#: UniProt export column names of the EC label TSV (the EC v2 cohort's ``ec_labels_v2.tsv``).
EC_ID_COLUMN, EC_LABEL_COLUMN = "Entry", "EC number"
#: Column names of the GO label TSV written by ``data_preparation.export_go_annotations``.
GO_ID_COLUMN, GO_LABEL_COLUMN = "protein_id", "GO_term"

#: EC transfer scores, deepest first. A protein may carry several EC numbers; a level
#: counts as matched if ANY pair of the two proteins' EC numbers agrees to that depth —
#: the same "share any function" rule as ``ec_hierarchy.ec_distance_set(agg="min")``.
EC_SCORES: tuple[str, ...] = ("exact", "share3", "share2", "class")
#: GO transfer scores: protein-centric F1 of the propagated term sets, and Wang BMA of
#: the unpropagated annotated sets.
GO_SCORES: tuple[str, ...] = ("f1", "wang_bma")

VARIANT_ALL = "all"

#: The homology baseline's two neighbour criteria, as ``arm label -> minimised quantity``.
#: ``evalue`` is the headline one (it is what a sequence search ranks by); ``fident`` is
#: the second variant the design asks for, and ``1 - fident`` is stored so that "smaller is
#: better" holds for every arm's ``neighbour_distance`` column.
HBI_ARMS: dict[str, str] = {"hbi_evalue": "evalue", "hbi_fident": "fident"}

#: Query subsets every cell is reported on (see the module docstring).
SUBSET_ALL = "all_queries"
SUBSET_HBI = "hbi_answerable"
SUBSET_NO_HIT = "no_hit"
SUBSET_DESCRIPTIONS: dict[str, str] = {
    SUBSET_ALL: "every query the variant leaves with an eligible neighbour (embedding arms only)",
    SUBSET_HBI: "queries with >=1 eligible MMseqs2 hit, i.e. the ones HBI can answer",
    SUBSET_NO_HIT: "queries with no MMseqs2 hit to any cohort protein (HBI cannot answer)",
}


class TransferInputError(RuntimeError):
    """An input is missing, malformed, or inconsistent with the freeze — never silent."""


def _log(message: str) -> None:
    """Progress to stderr; stdout stays clean for the one-line result summary."""
    print(f"[{time.strftime('%H:%M:%S')}] transfer_report: {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
#                                   inputs
# --------------------------------------------------------------------------- #


def sha256_file(path: Path | str, *, chunk: int = 1 << 22) -> str:
    """Streaming SHA-256 of a file, so a multi-GB embedding slice costs no extra RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def arm_name(path: Path | str) -> str:
    """Report label for an embedding slice.

    ``random_init_<model>_seed0.h5`` becomes ``randinit_<model>``: the random-init arms
    are the untrained controls and must be readable as such next to their trained twin in
    every table, and the ``_seed0`` suffix is noise (there is one seed).
    """
    stem = Path(path).stem
    if stem.startswith("random_init_"):
        stem = "randinit_" + stem[len("random_init_") :]
        if stem.endswith("_seed0"):
            stem = stem[: -len("_seed0")]
    return stem


def discover_arms(emb_dir: Path | str) -> list[Path]:
    """Every ``.h5`` in ``emb_dir``, sorted by report label (stable table row order)."""
    paths = sorted(Path(emb_dir).glob("*.h5"), key=arm_name)
    if not paths:
        raise TransferInputError(f"no .h5 embedding slices in {emb_dir}")
    return paths


def load_arm_matrix(h5_path: Path | str, ids: Sequence[str], required: Iterable[str]) -> np.ndarray:
    """``(len(ids), D)`` float64 matrix in ``ids`` order, flattened and checked.

    ``required`` is the frozen id set the slice must cover in full: a slice that silently
    lost proteins would change the cohort per arm and make the arms incomparable, so a
    missing id raises instead of shrinking the population. ProtT5/ProtTucker store
    ``(1, D)``, so every dataset is flattened. Non-finite values raise — a NaN would
    propagate into every distance of that row and quietly win every argmin.
    """
    import h5py

    with h5py.File(h5_path, "r") as handle:
        present = set(handle.keys())
        missing = [pid for pid in required if pid not in present]
        if missing:
            raise TransferInputError(
                f"{h5_path}: {len(missing)} frozen id(s) missing from the slice "
                f"(e.g. {missing[:5]})"
            )
        first = np.asarray(handle[ids[0]][()]).ravel()
        out = np.empty((len(ids), first.size), dtype=np.float64)
        for row, pid in enumerate(ids):
            vec = np.asarray(handle[pid][()]).ravel()
            if vec.size != first.size:
                raise TransferInputError(
                    f"{h5_path}: {pid} has dimension {vec.size}, expected {first.size}"
                )
            out[row] = vec
    if not np.isfinite(out).all():
        raise TransferInputError(f"{h5_path}: embedding matrix has non-finite values")
    return out


def load_ec_labels(path: Path | str, ids: Sequence[str]) -> list[frozenset[str]]:
    """One frozenset of fully-specified EC numbers per frozen id, in ``ids`` order."""
    frame = pd.read_csv(path, sep="\t", dtype=str)
    for column in (EC_ID_COLUMN, EC_LABEL_COLUMN):
        if column not in frame.columns:
            raise TransferInputError(f"{path}: EC label TSV needs a {column!r} column")
    parsed = parse_ec(
        frame, ec_col=EC_LABEL_COLUMN, id_col=EC_ID_COLUMN, wildcard_policy="exclude"
    )
    lookup = dict(zip(parsed["protein_id"], parsed["ec_set"], strict=True))
    missing = [pid for pid in ids if pid not in lookup]
    if missing:
        raise TransferInputError(
            f"{path}: {len(missing)} frozen id(s) have no fully-specified EC "
            f"(e.g. {missing[:5]})"
        )
    return [lookup[pid] for pid in ids]


def read_go_annotations(path: Path | str) -> dict[str, set[str]]:
    """``protein_id -> {GO term}`` from the 2+-column export TSV (extra columns ignored)."""
    frame = pl.read_csv(path, separator="\t", has_header=True, infer_schema_length=0)
    for column in (GO_ID_COLUMN, GO_LABEL_COLUMN):
        if column not in frame.columns:
            raise TransferInputError(f"{path}: GO label TSV needs a {column!r} column")
    raw: dict[str, set[str]] = defaultdict(set)
    for pid, term in zip(frame[GO_ID_COLUMN], frame[GO_LABEL_COLUMN], strict=True):
        raw[pid].add(term)
    return dict(raw)


def load_go_labels(
    path: Path | str,
    ids: Sequence[str],
    go_terms: Mapping[str, GOTerm],
    alt_ids: Mapping[str, str],
    *,
    drop_protein_binding: bool,
) -> tuple[list[str], list[frozenset[str]], dict[str, int]]:
    """Scoreable MF term sets for the frozen ids; returns ``(kept_ids, term_sets, counts)``.

    Without ``drop_protein_binding`` every frozen id must keep >=1 scoreable MF term — the
    cohort builder already guarantees it, so a protein that loses all of them here means
    the labels and the freeze disagree and the run must stop. With the flag, proteins left
    termless are the *expected* casualties of the sensitivity variant: they are removed
    from the cohort (as queries and as neighbours alike) and counted.
    """
    raw = read_go_annotations(path)
    missing = [pid for pid in ids if pid not in raw]
    if missing:
        raise TransferInputError(
            f"{path}: {len(missing)} frozen id(s) have no GO annotation (e.g. {missing[:5]})"
        )
    drop_terms = (PROTEIN_BINDING,) if drop_protein_binding else ()
    cleaned, counts = clean_mf_annotations(
        {pid: raw[pid] for pid in ids}, go_terms, alt_ids, drop_terms=drop_terms
    )
    kept = [pid for pid in ids if pid in cleaned]
    counts["proteins_dropped"] = len(ids) - len(kept)
    if len(kept) != len(ids) and not drop_protein_binding:
        lost = [pid for pid in ids if pid not in cleaned]
        raise TransferInputError(
            f"{path}: {len(lost)} frozen id(s) have no scoreable MF term after cleaning "
            f"(e.g. {lost[:5]}); the labels and the freeze disagree"
        )
    return kept, [cleaned[pid] for pid in kept], counts


# --------------------------------------------------------------------------- #
#                            neighbour eligibility
# --------------------------------------------------------------------------- #


class EligibilityVariant:
    """Which neighbours a query may transfer from, as a row-block boolean mask.

    ``excluded`` is a sparse symmetric relation over cohort positions (an MMseqs2 hit that
    disqualifies the neighbour); ``None`` means the unrestricted variant. The diagonal is
    always ineligible — this is a leave-one-out readout, so a protein can never be its own
    neighbour.
    """

    def __init__(self, name: str, description: str, excluded: sp.csr_matrix | None, n: int):
        self.name = name
        self.description = description
        self.n = n
        self._excluded = excluded.tocsr() if excluded is not None else None
        self.n_excluded_pairs = int(self._excluded.nnz) if self._excluded is not None else 0

    def block(self, lo: int, hi: int) -> np.ndarray:
        """``(hi - lo, n)`` boolean: True where that neighbour may be transferred from."""
        if self._excluded is None:
            mask = np.ones((hi - lo, self.n), dtype=bool)
        else:
            mask = self._excluded[lo:hi].toarray() == 0
        mask[np.arange(hi - lo), np.arange(lo, hi)] = False
        return mask

    def pair_mask(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Eligibility of an explicit ordered-pair list — the sparse form of ``block``.

        The homology baseline scans a hit list, not a dense row block, and it must obey the
        same eligibility rule as the embedding arms or the two readouts would be answering
        different questions under the identity variants.
        """
        ok = rows != cols
        if self._excluded is not None:
            ok &= np.asarray(self._excluded[rows, cols]).ravel() == 0
        return ok


def _symmetric_relation(rows: np.ndarray, cols: np.ndarray, n: int) -> sp.csr_matrix:
    """Sparse symmetric 0/1 relation from one directed edge list ("in either direction")."""
    both_rows = np.concatenate([rows, cols])
    both_cols = np.concatenate([cols, rows])
    data = np.ones(both_rows.size, dtype=np.int8)
    mat = sp.coo_matrix((data, (both_rows, both_cols)), shape=(n, n)).tocsr()
    mat.data[:] = 1
    return mat


class IdentityHits:
    """The cohort's MMseqs2 hits as position arrays, symmetrised ("in either direction").

    One table serves both halves of the identity question: an eligibility variant EXCLUDES
    a neighbour because it has a hit, the homology baseline TRANSFERS FROM its best hit, so
    the two must agree on what a hit is down to the row. MMseqs2 reports an alignment under
    one query, but both proteins are queries of this cohort, so every directed hit is stored
    in both directions and a query's candidate neighbours are all proteins it aligns to
    *or* that align to it.
    """

    def __init__(
        self, rows: np.ndarray, cols: np.ndarray, fident: np.ndarray, evalue: np.ndarray, n: int
    ):
        self.n = n
        self.rows = np.concatenate([rows, cols])
        self.cols = np.concatenate([cols, rows])
        self.fident = np.concatenate([fident, fident])
        self.evalue = np.concatenate([evalue, evalue])
        #: True where the protein has >=1 hit to another cohort protein, in either
        #: direction — the complement is the ``no_hit`` subset, where HBI has no answer.
        self.has_hit = np.zeros(n, dtype=bool)
        self.has_hit[self.rows] = True

    @property
    def n_directed(self) -> int:
        """Hits after symmetrisation (each reported alignment counts twice)."""
        return int(self.rows.size)


def hbi_neighbours(
    hits: IdentityHits, variant: EligibilityVariant, criterion: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(nn, value, tied, primary_tied)`` — the best *eligible* MMseqs2 hit of every protein.

    ``nn[i] = -1`` and ``value[i] = inf`` mean the search leaves ``i`` unanswerable under
    this variant. ``value`` is the quantity minimised (the E-value, or ``1 - fident``), so
    it reads like the embedding arms' ``neighbour_distance``. Ties on the primary criterion
    are broken by the secondary one and then by lowest cohort index — the same
    deterministic rule as the embedding ``argmin``.

    Two tie counts, because they are not the same question and the embedding arms only
    have one. ``tied`` is the comparable one: the pick is still ambiguous after the full
    tie-break chain, i.e. it depends on the id order, exactly as an embedding ``argmin``
    tie does. ``primary_tied`` is the criterion alone being ambiguous, which is common and
    mostly harmless — MMseqs2 rounds ``fident``, so thousands of queries have several
    candidates at the same reported identity that the E-value then separates.
    """
    if criterion not in ("evalue", "fident"):
        raise TransferInputError(f"unknown HBI criterion {criterion!r}")
    eligible = variant.pair_mask(hits.rows, hits.cols)
    rows, cols = hits.rows[eligible], hits.cols[eligible]
    evalue, fident = hits.evalue[eligible], hits.fident[eligible]
    primary, secondary = (evalue, 1.0 - fident) if criterion == "evalue" else (1.0 - fident, evalue)

    nn = np.full(hits.n, -1, dtype=np.int64)
    value = np.full(hits.n, np.inf)
    tied = np.zeros(hits.n, dtype=bool)
    primary_tied = np.zeros(hits.n, dtype=bool)
    if rows.size == 0:
        return nn, value, tied, primary_tied

    # lexsort's LAST key is the primary one: group by query, then best criterion, then the
    # tie-break chain. The first row of each group is that query's neighbour.
    order = np.lexsort((cols, secondary, primary, rows))
    rows, cols, primary, secondary = rows[order], cols[order], primary[order], secondary[order]
    starts = np.ones(rows.size, dtype=bool)
    starts[1:] = rows[1:] != rows[:-1]
    first = np.flatnonzero(starts)
    group = np.cumsum(starts) - 1
    nn[rows[first]] = cols[first]
    value[rows[first]] = primary[first]
    at_best_primary = primary == primary[first][group]
    n_primary_tied = np.bincount(group, weights=at_best_primary.astype(np.float64))
    primary_tied[rows[first]] = n_primary_tied > 1
    # Ambiguous only if the SECONDARY criterion does not separate the co-leaders either;
    # what is left is then decided by the id order, like an embedding argmin tie.
    ambiguous = at_best_primary & (secondary == secondary[first][group])
    tied[rows[first]] = np.bincount(group, weights=ambiguous.astype(np.float64)) > 1
    return nn, value, tied, primary_tied


def build_variants(
    m8_path: Path | str | None,
    ids: Sequence[str],
    *,
    fident_max: float = 0.30,
    evalue_max: float = 1e-3,
) -> tuple[list[EligibilityVariant], IdentityHits | None, dict]:
    """The three neighbour-eligibility variants of the identity control, the hit table the
    homology baseline transfers from, and the hit counts.

    The MMseqs2 table is an all-vs-all search over the UNION of both cohorts, so most of
    its rows concern proteins outside this cohort and are dropped here rather than in the
    search (one search serves the EC and the GO arm). Self hits are dropped: a protein is
    never its own neighbour anyway, and keeping them would exclude every query from every
    identity variant.
    """
    n = len(ids)
    variants = [
        EligibilityVariant(VARIANT_ALL, "every other cohort protein", None, n),
    ]
    stats: dict = {"m8": None if m8_path is None else str(m8_path)}
    if m8_path is None:
        return variants, None, stats

    table = pl.read_csv(
        m8_path,
        separator="\t",
        has_header=False,
        new_columns=list(M8_COLUMNS),
        schema_overrides={
            "query": pl.Utf8,
            "target": pl.Utf8,
            # Pinned, not inferred: polars types a column from its first rows, and an m8
            # whose first hits happen to print an integral E-value ("0") would then abort
            # the run on the first "1e-30" further down.
            "fident": pl.Float64,
            "evalue": pl.Float64,
        },
    )
    stats["n_hits_total"] = int(table.height)
    # A join, not a Python id->index loop: the union search can run to tens of millions of
    # rows and it both restricts to this cohort and maps to positions in one pass.
    index = pl.DataFrame({"id": list(ids), "idx": np.arange(n, dtype=np.int64)})
    table = (
        table.join(index.rename({"id": "query", "idx": "qi"}), on="query", how="inner")
        .join(index.rename({"id": "target", "idx": "ti"}), on="target", how="inner")
        .filter(pl.col("qi") != pl.col("ti"))
    )
    stats["n_hits_in_cohort"] = int(table.height)
    rows = table["qi"].to_numpy()
    cols = table["ti"].to_numpy()
    fident = table["fident"].to_numpy()
    evalue = table["evalue"].to_numpy()
    hits = IdentityHits(rows, cols, fident, evalue, n)
    stats["n_proteins_with_a_hit"] = int(hits.has_hit.sum())
    stats["n_proteins_without_a_hit"] = int(n - hits.has_hit.sum())

    ident_hit = fident >= fident_max
    any_hit = evalue <= evalue_max
    stats["n_hits_fident_ge_threshold"] = int(ident_hit.sum())
    stats["n_hits_evalue_le_threshold"] = int(any_hit.sum())

    ident = _symmetric_relation(rows[ident_hit], cols[ident_hit], n)
    hit = _symmetric_relation(rows[any_hit], cols[any_hit], n)
    variants.append(
        EligibilityVariant(
            f"fident_lt_{fident_max:g}",
            f"no MMseqs2 hit with fident >= {fident_max:g} in either direction",
            ident,
            n,
        )
    )
    variants.append(
        EligibilityVariant(
            f"no_hit_evalue_{evalue_max:g}",
            f"no MMseqs2 hit with E-value <= {evalue_max:g} in either direction",
            hit,
            n,
        )
    )
    stats["n_excluded_pairs"] = {v.name: v.n_excluded_pairs for v in variants}
    return variants, hits, stats


# --------------------------------------------------------------------------- #
#                                label scorers
# --------------------------------------------------------------------------- #


class ECScorer:
    """0/1 EC transfer scores at four hierarchy depths, for every ordered pair.

    Each depth is a set-intersection test on EC prefixes: proteins ``i`` and ``j`` match at
    depth ``L`` iff some EC of ``i`` and some EC of ``j`` share their first ``L`` fields.
    Expressed as a protein x prefix indicator ``P_L``, that is ``(P_L P_L^T) > 0`` — one
    sparse product per depth instead of a Python double loop over 24M pairs. EC v2 labels
    are fully specified, so there are no wildcards to reason about.
    """

    names = EC_SCORES
    _DEPTH = {"exact": 4, "share3": 3, "share2": 2, "class": 1}

    def __init__(self, ec_sets: Sequence[frozenset[str]]):
        if any(not s for s in ec_sets):
            raise TransferInputError("every protein needs >=1 EC number")
        self._ind: dict[str, sp.csr_matrix] = {}
        for name, depth in self._DEPTH.items():
            prefixes = [
                frozenset(".".join(ec.split(".")[:depth]) for ec in terms) for terms in ec_sets
            ]
            self._ind[name] = set_indicator(prefixes)[0]

    def block(self, lo: int, hi: int) -> dict[str, np.ndarray]:
        return {
            name: np.asarray((ind[lo:hi] @ ind.T).toarray() > 0, dtype=np.float64)
            for name, ind in self._ind.items()
        }

    def pairs(self, rows: np.ndarray, cols: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for name, ind in self._ind.items():
            shared = np.asarray(ind[rows].multiply(ind[cols]).sum(axis=1)).ravel()
            out[name] = np.asarray(shared > 0, dtype=np.float64)
        return out


class GOScorer:
    """GO-MF transfer scores: propagated-set F1 and Wang best-match-average.

    Both are computed from sparse indicators so that a row block costs one sparse-dense
    product rather than a Python loop over term pairs:

    * ``f1[i, j] = 2 |P_i & P_j| / (|P_i| + |P_j|)`` with ``P`` the ancestor-closed MF sets
      (is_a + part_of, root excluded) — the CAFA protein-centric convention. It is
      symmetric, so one matrix covers both transfer directions.
    * ``wang_bma`` follows ``go_similarity_matrix``: with ``best[i, u] = max_{t in T_i}
      sim(t, u)`` and ``B`` the annotated-set indicator, ``G = best B^T`` and
      ``BMA(i, j) = (G[i, j] + G[j, i]) / (|T_i| + |T_j|)``. ``G[i, j]`` sums j's terms'
      best match in i, so the two halves of the BMA are ``G`` and its transpose.

    ``best`` is the only dense array (n x |vocab| float64); everything else is sparse.

    ``drop_from_propagated`` is the protein-binding sensitivity: the terms are removed
    from the ancestor closure, not only from the annotations, because that is the only
    place the removal can change ``f1`` (see ``go_similarity_matrix.propagate_mf``). It
    deliberately does NOT touch ``wang_bma``, which is defined on the unpropagated
    annotated sets and whose Wang S-values sum over all shared ancestors by definition.
    ``propagated_drop`` records how many propagated sets actually lost the term, so a
    sensitivity that changed nothing can never be reported as one that did.
    """

    names = GO_SCORES

    def __init__(
        self,
        term_sets: Sequence[frozenset[str]],
        go_terms: Mapping[str, GOTerm],
        *,
        drop_from_propagated: Sequence[str] = (),
    ):
        if any(not s for s in term_sets):
            raise TransferInputError("every protein needs >=1 MF term")
        self._b, vocab = set_indicator(term_sets)
        self._sizes = np.asarray(self._b.sum(axis=1)).ravel()
        sim = wang_term_similarity_matrix(vocab, go_terms)
        col = {term: k for k, term in enumerate(vocab)}
        self._best = np.empty((len(term_sets), len(vocab)), dtype=np.float64)
        for i, terms in enumerate(term_sets):
            np.max(sim[[col[t] for t in terms]], axis=0, out=self._best[i])
        del sim
        propagated = propagate_mf(term_sets, go_terms)
        self.propagated_drop: dict[str, int] = {
            term: sum(1 for s in propagated if term in s) for term in drop_from_propagated
        }
        if drop_from_propagated:
            propagated = propagate_mf(term_sets, go_terms, drop_terms=drop_from_propagated)
        self._p, _ = set_indicator(propagated)
        self._psizes = np.asarray(self._p.sum(axis=1)).ravel()
        if (self._psizes == 0).any():
            raise TransferInputError("a protein's propagated MF set is empty")

    def bma_block(self, lo: int, hi: int) -> np.ndarray:
        forward = np.asarray(self._b @ self._best[lo:hi].T).T  # G[i, j]
        backward = np.asarray(self._b[lo:hi] @ self._best.T)  # G[j, i]
        return (forward + backward) / (self._sizes[lo:hi, None] + self._sizes[None, :])

    def f1_block(self, lo: int, hi: int) -> np.ndarray:
        shared = (self._p[lo:hi] @ self._p.T).toarray()
        return 2.0 * shared / (self._psizes[lo:hi, None] + self._psizes[None, :])

    def block(self, lo: int, hi: int) -> dict[str, np.ndarray]:
        return {"f1": self.f1_block(lo, hi), "wang_bma": self.bma_block(lo, hi)}

    def pairs(
        self, rows: np.ndarray, cols: np.ndarray, *, chunk: int = 4096
    ) -> dict[str, np.ndarray]:
        f1 = np.empty(rows.size, dtype=np.float64)
        bma = np.empty(rows.size, dtype=np.float64)
        for start in range(0, rows.size, chunk):
            r = rows[start : start + chunk]
            c = cols[start : start + chunk]
            forward = np.asarray(self._b[c].multiply(self._best[r]).sum(axis=1)).ravel()
            backward = np.asarray(self._b[r].multiply(self._best[c]).sum(axis=1)).ravel()
            bma[start : start + r.size] = (forward + backward) / (self._sizes[r] + self._sizes[c])
            shared = np.asarray(self._p[r].multiply(self._p[c]).sum(axis=1)).ravel()
            f1[start : start + r.size] = 2.0 * shared / (self._psizes[r] + self._psizes[c])
        return {"f1": f1, "wang_bma": bma}


# --------------------------------------------------------------------------- #
#                          distances and nearest neighbours
# --------------------------------------------------------------------------- #


class BlockDistances:
    """Row blocks of the dense ``n x n`` embedding-distance matrix, in float64.

    Euclidean uses the squared-norm identity with the tiny negatives clipped (they are
    float cancellation, not real distances); cosine is ``1 - cos`` on unit-normalised rows,
    clipped to ``[0, 2]``. Blocked because the full matrix for a 20k cohort is 3.2 GB and
    nothing downstream needs more than one row at a time.
    """

    def __init__(self, x: np.ndarray, distance: str):
        if distance == "euclidean":
            self._x = x
            self._sq = np.einsum("ij,ij->i", x, x)
        elif distance == "cosine":
            norms = np.linalg.norm(x, axis=1)
            if not np.all(norms > 0):
                raise TransferInputError(
                    f"{int((norms == 0).sum())} zero-norm embedding(s): cosine is undefined"
                )
            self._x = x / norms[:, None]
        else:
            raise TransferInputError(f"unknown distance {distance!r}")
        self.distance = distance
        self.n = x.shape[0]

    def block(self, lo: int, hi: int) -> np.ndarray:
        # numpy 2.x reports spurious "divide by zero / overflow / invalid value in
        # matmul" here from uninitialised SIMD tail lanes (gh-27509), which on a 26-arm
        # run buries the stderr log in false alarms. The inputs were checked finite when
        # the slice was loaded, and the result is checked again below — so this silences
        # the false alarm without weakening the guard that a real NaN must stop the run.
        with np.errstate(all="ignore"):
            gram = self._x[lo:hi] @ self._x.T
            if self.distance == "euclidean":
                out = self._sq[lo:hi, None] + self._sq[None, :] - 2.0 * gram
                np.maximum(out, 0.0, out=out)
                np.sqrt(out, out=out)
            else:
                out = 1.0 - gram
                np.clip(out, 0.0, 2.0, out=out)
        if not np.isfinite(out).all():
            raise TransferInputError(f"non-finite {self.distance} distances in rows {lo}:{hi}")
        return out


def nearest_neighbour_block(
    dist: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(nn, best, tied, dropped)`` for one row block.

    Ineligible neighbours become ``+inf``, so ``argmin`` returns the lowest-index minimum —
    the documented tie-break — and a row with no eligible neighbour has ``best = inf`` and
    is reported dropped rather than silently assigned neighbour 0. ``tied`` flags rows whose
    minimum is attained more than once: a tie means the reported score depends on the id
    order, so the count belongs in the summary.
    """
    masked = np.where(eligible, dist, np.inf)
    nn = masked.argmin(axis=1)
    best = masked[np.arange(masked.shape[0]), nn]
    dropped = ~np.isfinite(best)
    tied = np.count_nonzero(masked == best[:, None], axis=1) > 1
    tied[dropped] = False
    return nn, best, tied, dropped


# --------------------------------------------------------------------------- #
#                            baselines and bootstrap
# --------------------------------------------------------------------------- #


def variant_baselines(
    scorer: ECScorer | GOScorer,
    variants: Sequence[EligibilityVariant],
    n: int,
    *,
    block_size: int = 1024,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-query chance, oracle and eligible-neighbour count for every (variant, score).

    Both baselines are exact, not sampled: ``chance[i]`` is the mean of the score row over
    the eligible neighbours (what a uniformly random eligible neighbour scores in
    expectation) and ``oracle[i]`` the best score any eligible neighbour offers. For the
    0/1 EC scores the oracle mean is exactly the fraction of queries that HAVE an eligible
    neighbour at that depth. Neither depends on the embedding, so this runs once for the
    whole report rather than once per arm — the single streaming pass over the score matrix.
    """
    out = {
        v.name: {
            **{f"chance_{s}": np.zeros(n) for s in scorer.names},
            **{f"oracle_{s}": np.zeros(n) for s in scorer.names},
            "n_eligible": np.zeros(n, dtype=np.int64),
        }
        for v in variants
    }
    for lo in range(0, n, block_size):
        hi = min(lo + block_size, n)
        scores = scorer.block(lo, hi)
        for variant in variants:
            eligible = variant.block(lo, hi)
            counts = np.count_nonzero(eligible, axis=1)
            out[variant.name]["n_eligible"][lo:hi] = counts
            safe = np.maximum(counts, 1)
            for name, matrix in scores.items():
                out[variant.name][f"chance_{name}"][lo:hi] = (
                    np.sum(matrix, axis=1, where=eligible) / safe
                )
                out[variant.name][f"oracle_{name}"][lo:hi] = np.max(
                    matrix, axis=1, where=eligible, initial=-np.inf
                )
        _log(f"baselines: rows {hi}/{n}")
    for variant in variants:
        empty = out[variant.name]["n_eligible"] == 0
        for name in scorer.names:
            out[variant.name][f"chance_{name}"][empty] = np.nan
            out[variant.name][f"oracle_{name}"][empty] = np.nan
    return out


def hbi_baselines(
    scorer: ECScorer | GOScorer,
    hits: IdentityHits,
    variant: EligibilityVariant,
    n: int,
) -> dict[str, np.ndarray]:
    """Per-query chance and oracle over the MMseqs2 HIT LIST — the homology arm's own ceiling.

    The cohort oracle is not a ceiling homology search could ever reach: it may only
    transfer from a protein its own search returned. Printing the cohort oracle on an HBI
    row would therefore overstate how far the search is from *its* best possible answer, so
    the HBI rows carry these baselines instead (``baseline_scope = mmseqs_hits``) and the
    embedding rows keep the cohort ones (``baseline_scope = cohort``). ``chance`` is the mean
    over the query's eligible hits — what picking a random hit instead of the best one would
    score — and ``oracle`` the best label match anywhere in that hit list.

    The hit table is symmetrised, so a pair reported in both directions appears twice; it is
    deduplicated here, otherwise a bidirectional hit would count twice in the mean.
    """
    eligible = variant.pair_mask(hits.rows, hits.cols)
    rows, cols = hits.rows[eligible], hits.cols[eligible]
    out: dict[str, np.ndarray] = {
        **{f"chance_{s}": np.full(n, np.nan) for s in scorer.names},
        **{f"oracle_{s}": np.full(n, np.nan) for s in scorer.names},
        "n_eligible": np.zeros(n, dtype=np.int64),
    }
    if rows.size == 0:
        return out
    # np.unique sorts, and rows * n + cols is monotonic in (row, col), so the survivors come
    # back grouped by query — which is what lets the group statistics be reduceat, not a loop.
    keep = np.unique(rows.astype(np.int64) * n + cols.astype(np.int64), return_index=True)[1]
    rows, cols = rows[keep], cols[keep]
    counts = np.bincount(rows, minlength=n)
    out["n_eligible"] = counts.astype(np.int64)
    present = np.flatnonzero(counts)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])[present]
    scores = scorer.pairs(rows, cols)
    for name, values in scores.items():
        out[f"chance_{name}"][present] = np.add.reduceat(values, offsets) / counts[present]
        out[f"oracle_{name}"][present] = np.maximum.reduceat(values, offsets)
    return out


def bootstrap_weights(n_queries: int, n_boot: int, seed: int) -> np.ndarray:
    """``(n_boot, n_queries)`` resample multiplicities for the query bootstrap.

    A resampled mean is ``(w @ v) / n_queries``, so storing multiplicities instead of index
    lists turns the whole bootstrap of every arm into one BLAS product against the same
    weights — which is also what makes the arm differences *paired*: every arm is evaluated
    on the identical resamples.
    """
    rng = np.random.default_rng(seed)
    weights = np.empty((n_boot, n_queries), dtype=np.float64)
    for b in range(n_boot):
        weights[b] = np.bincount(rng.integers(0, n_queries, size=n_queries), minlength=n_queries)
    return weights


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided percentile interval along the last axis."""
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=-1)
    return lo, hi


# --------------------------------------------------------------------------- #
#                                   tau-b
# --------------------------------------------------------------------------- #


def upper_triangle(block_fn, n: int, *, block_size: int = 1024, mask: np.ndarray | None = None) -> np.ndarray:
    """Flatten the strict upper triangle of a blocked symmetric matrix into a 1-D array.

    ``mask`` (a boolean over the triangle, in the same order) subsamples pairs for the
    secondary tau-b, which otherwise runs over ~200M pairs per arm.
    """
    total = n * (n - 1) // 2
    keep = total if mask is None else int(np.count_nonzero(mask))
    out = np.empty(keep, dtype=np.float64)
    read = write = 0
    for lo in range(0, n, block_size):
        hi = min(lo + block_size, n)
        matrix = block_fn(lo, hi)
        for i in range(lo, hi):
            row = matrix[i - lo, i + 1 :]
            if mask is None:
                out[write : write + row.size] = row
                write += row.size
            else:
                sel = mask[read : read + row.size]
                taken = np.count_nonzero(sel)
                out[write : write + taken] = row[sel]
                write += taken
            read += row.size
    return out


def pair_subsample_mask(n_pairs: int, max_pairs: int, seed: int) -> np.ndarray | None:
    """Boolean mask keeping ~``max_pairs`` of ``n_pairs`` (``None`` = keep everything)."""
    if max_pairs <= 0 or n_pairs <= max_pairs:
        return None
    rng = np.random.default_rng(seed)
    return rng.random(n_pairs) < (max_pairs / n_pairs)


# --------------------------------------------------------------------------- #
#                                   driver
# --------------------------------------------------------------------------- #


def _versions() -> dict:
    import h5py
    import pyarrow
    import scipy

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "polars": pl.__version__,
        "pyarrow": pyarrow.__version__,
        "h5py": h5py.__version__,
    }
    try:
        versions["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        versions["git_commit"] = None
    return versions


def run_transfer_report(
    *,
    labels_kind: str,
    freeze: Path | str,
    labels: Path | str,
    emb_dir: Path | str,
    out_dir: Path | str,
    go_obo: Path | str | None = None,
    arms: Sequence[str] | None = None,
    distances: Sequence[str] = ("euclidean", "cosine"),
    identity_m8: Path | str | None = None,
    drop_protein_binding: bool = False,
    fident_max: float = 0.30,
    evalue_max: float = 1e-3,
    n_boot: int = 2000,
    seed: int = 42,
    block_size: int = 1024,
    tau: bool = True,
    tau_max_pairs: int = 0,
    wang_check_pairs: int = 200,
    hash_arms: bool = True,
) -> dict:
    """Score every (arm, distance, variant, subset, score) cell — the embedding arms and the
    MMseqs2 homology baseline alike — and write the report. Returns the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_ids = load_frozen_ids(freeze)
    go_terms: dict[str, GOTerm] = {}
    label_counts: dict[str, int] = {}
    if labels_kind == "ec":
        if drop_protein_binding:
            raise TransferInputError("--drop-protein-binding is a GO-only sensitivity")
        ids = list(frozen_ids)
        scorer: ECScorer | GOScorer = ECScorer(load_ec_labels(labels, ids))
    elif labels_kind == "go":
        if go_obo is None:
            raise TransferInputError("--go-obo is required for --labels-kind go")
        go_terms = parse_obo(Path(go_obo))
        ids, term_sets, label_counts = load_go_labels(
            labels, frozen_ids, go_terms, parse_alt_ids(go_obo),
            drop_protein_binding=drop_protein_binding,
        )
        scorer = GOScorer(
            term_sets,
            go_terms,
            drop_from_propagated=(PROTEIN_BINDING,) if drop_protein_binding else (),
        )
    else:
        raise TransferInputError(f"unknown labels_kind {labels_kind!r}")
    n = len(ids)
    if n < 3:
        raise TransferInputError(f"need >=3 cohort proteins (got {n})")
    _log(f"cohort n={n} ({labels_kind}), scores={list(scorer.names)}")

    # The matrix Wang BMA is only admissible if it is the same number as the scalar
    # definition, so the report re-checks it on the real cohort and records the result.
    wang_check: dict = {}
    if labels_kind == "go" and wang_check_pairs > 0:
        rows, cols = random_ordered_pairs(n, wang_check_pairs, seed)
        ours = scorer.pairs(rows, cols)["wang_bma"]
        reference = scalar_bma_values(term_sets, rows, cols, go_terms)
        worst = float(np.max(np.abs(ours - reference)))
        wang_check = {"n_pairs": int(rows.size), "max_abs_diff_vs_scalar": worst}
        if worst >= 1e-9:
            raise TransferInputError(
                f"vectorised Wang BMA differs from the scalar reference by {worst:.3g}"
            )
        _log(f"Wang BMA validated against the scalar reference on {rows.size} pairs ({worst:.2g})")

    variants, hits, identity_stats = build_variants(
        identity_m8, ids, fident_max=fident_max, evalue_max=evalue_max
    )
    _log(f"variants: {[v.name for v in variants]}")

    baselines = variant_baselines(scorer, variants, n, block_size=block_size)

    arm_paths = discover_arms(emb_dir)
    if arms is not None:
        wanted = set(arms)
        arm_paths = [p for p in arm_paths if arm_name(p) in wanted]
        unknown = wanted - {arm_name(p) for p in arm_paths}
        if unknown:
            raise TransferInputError(f"requested arm(s) not in {emb_dir}: {sorted(unknown)}")
    arm_labels = [arm_name(p) for p in arm_paths]
    hbi_labels = list(HBI_ARMS) if hits is not None else []
    report_arms = arm_labels + hbi_labels

    # Kept queries depend only on the variant (eligibility is embedding-independent), so
    # the bootstrap weights — and therefore the pairing between arms — are fixed per variant.
    kept: dict[str, np.ndarray] = {
        v.name: np.flatnonzero(baselines[v.name]["n_eligible"] > 0) for v in variants
    }

    # HBI's candidates are the variant-eligible part of the hit table, so *which* queries it
    # can answer depends on the variant but not on the criterion. The embedding arms are
    # then reported on those same subsets, which is the only way the two readouts can be
    # differenced per query.
    hbi_picks: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
    hbi_baseline: dict[str, dict[str, np.ndarray]] = {}
    subset_masks: dict[str, list[tuple[str, np.ndarray]]] = {}
    query_sets: dict[tuple[str, str], np.ndarray] = {}
    for variant in variants:
        rows = kept[variant.name]
        masks = [(SUBSET_ALL, np.ones(rows.size, dtype=bool))]
        if hits is not None:
            answerable: np.ndarray | None = None
            for hbi_arm, criterion in HBI_ARMS.items():
                pick = hbi_neighbours(hits, variant, criterion)
                hbi_picks[(variant.name, hbi_arm)] = pick
                reach = pick[0] >= 0
                if answerable is None:
                    answerable = reach
                elif not np.array_equal(answerable, reach):
                    raise TransferInputError(
                        f"{variant.name}: the two HBI criteria disagree on which queries "
                        "are answerable, which they cannot — same candidate set"
                    )
            hbi_baseline[variant.name] = hbi_baselines(scorer, hits, variant, n)
            masks.append((SUBSET_HBI, answerable[rows]))
            # The no-hit queries have nothing to exclude, so this subset is numerically
            # identical under all three variants; publishing it once stops the same cell
            # being read as three measurements (and triple-counted in any group-by).
            if variant.name == VARIANT_ALL:
                masks.append((SUBSET_NO_HIT, ~hits.has_hit[rows]))
        subset_masks[variant.name] = [(name, mask) for name, mask in masks if mask.any()]
        for name, mask in subset_masks[variant.name]:
            query_sets[(variant.name, name)] = rows[mask]
        _log(
            f"variant {variant.name}: "
            + ", ".join(f"{name} n={int(mask.sum())}" for name, mask in masks)
        )

    functional_distance: np.ndarray | None = None
    tau_mask: np.ndarray | None = None
    if labels_kind == "go" and tau:
        n_pairs = n * (n - 1) // 2
        tau_mask = pair_subsample_mask(n_pairs, tau_max_pairs, seed)
        _log(f"tau-b: building 1 - Wang BMA over {n_pairs} pairs"
             f"{'' if tau_mask is None else f' (subsampled to {int(tau_mask.sum())})'}")
        functional_distance = 1.0 - upper_triangle(
            scorer.bma_block, n, block_size=block_size, mask=tau_mask
        )

    per_query_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    tau_rows: list[dict] = []
    # {(distance, variant, subset, score): {arm: per-query score vector}}
    collected: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = defaultdict(dict)
    subset_names = [SUBSET_ALL, SUBSET_HBI, SUBSET_NO_HIT]

    def emit(
        arm: str,
        distance: str,
        variant_name: str,
        rows: np.ndarray,
        cols: np.ndarray,
        neighbour_distance: np.ndarray,
        tied: np.ndarray,
        subsets: Sequence[tuple[str, np.ndarray]],
        primary_tied: np.ndarray | None = None,
    ) -> None:
        """Score one arm's chosen neighbours and file them under each query subset.

        ``rows`` are cohort positions in increasing order and ``subsets`` selects from them,
        so an embedding arm's slice and the HBI arm's own rows line up query for query —
        which is what makes the later bootstrap of their difference paired.

        ``tied`` means the same thing for every arm — the pick depends on the id order —
        so the column is comparable across arms. ``primary_tied`` is the wider HBI-only
        count (the criterion alone was ambiguous); for an embedding arm the two coincide.
        """
        scores = scorer.pairs(rows, cols)
        primary_tied = tied if primary_tied is None else primary_tied
        frame = pd.DataFrame(
            {
                "arm": pd.Categorical([arm] * rows.size, categories=report_arms),
                "distance": pd.Categorical([distance] * rows.size, categories=list(distances)),
                "variant": pd.Categorical(
                    [variant_name] * rows.size, categories=[v.name for v in variants]
                ),
                "query_id": pd.Categorical.from_codes(rows, categories=ids),
                "neighbour_id": pd.Categorical.from_codes(cols, categories=ids),
                "neighbour_distance": neighbour_distance,
                "tied": tied,
                "primary_tied": primary_tied,
            }
        )
        for name, values in scores.items():
            frame[f"score_{name}"] = values
        per_query_frames.append(frame)
        for subset, mask in subsets:
            summary_rows.append(
                {
                    "arm": arm,
                    "distance": distance,
                    "variant": variant_name,
                    "subset": subset,
                    "n_queries": int(np.count_nonzero(mask)),
                    # The design's quantity: queries the VARIANT left without any eligible
                    # neighbour. It is a property of the variant, not of the subset — the
                    # queries a subset leaves out are counted separately, because a methods
                    # section that confused the two would state a false number.
                    "n_dropped": int(n - kept[variant_name].size),
                    "n_not_in_subset": int(kept[variant_name].size - np.count_nonzero(mask)),
                    "n_ties": int(np.count_nonzero(tied[mask])),
                    "n_primary_ties": int(np.count_nonzero(primary_tied[mask])),
                }
            )
            for name, values in scores.items():
                collected[(distance, variant_name, subset, name)][arm] = values[mask]

    for arm_path, arm in zip(arm_paths, arm_labels, strict=True):
        _log(f"arm {arm}: loading {arm_path.name}")
        matrix = load_arm_matrix(arm_path, ids, frozen_ids)
        for distance in distances:
            blocks = BlockDistances(matrix, distance)
            best = np.zeros((len(variants), n))
            tied = np.zeros((len(variants), n), dtype=bool)
            dropped = np.zeros((len(variants), n), dtype=bool)
            nn_all = np.zeros((len(variants), n), dtype=np.int64)
            for lo in range(0, n, block_size):
                hi = min(lo + block_size, n)
                dist_block = blocks.block(lo, hi)
                for vi, variant in enumerate(variants):
                    nn, b, t, d = nearest_neighbour_block(dist_block, variant.block(lo, hi))
                    nn_all[vi, lo:hi] = nn
                    best[vi, lo:hi] = b
                    tied[vi, lo:hi] = t
                    dropped[vi, lo:hi] = d
            for vi, variant in enumerate(variants):
                rows = kept[variant.name]
                if rows.size == 0:
                    raise TransferInputError(f"variant {variant.name} left no query with a neighbour")
                if dropped[vi, rows].any():
                    raise TransferInputError(
                        f"{arm}/{distance}/{variant.name}: eligibility and drop bookkeeping "
                        f"disagree on {int(dropped[vi, rows].sum())} quer(ies)"
                    )
                emit(
                    arm,
                    distance,
                    variant.name,
                    rows,
                    nn_all[vi, rows],
                    best[vi, rows],
                    tied[vi, rows],
                    subset_masks[variant.name],
                )
            if functional_distance is not None:
                embedding_distance = upper_triangle(
                    blocks.block, n, block_size=block_size, mask=tau_mask
                )
                tau_rows.append(
                    {
                        "arm": arm,
                        "distance": distance,
                        "tau_b": kendall_tau_b(embedding_distance, functional_distance),
                        "n_pairs": int(embedding_distance.size),
                        "subsampled": tau_mask is not None,
                    }
                )
                del embedding_distance
                _log(f"arm {arm}/{distance}: tau_b={tau_rows[-1]['tau_b']:.4f}")
            del blocks
        del matrix

    # The homology baseline does not depend on the embedding distance, but its rows are
    # repeated under each distance so that every paired difference against an embedding arm
    # is formed inside one cell of the table rather than across two.
    for variant in variants:
        rows = query_sets.get((variant.name, SUBSET_HBI))
        if rows is None:
            continue
        for hbi_arm in hbi_labels:
            nn, value, tied_hbi, primary_tied_hbi = hbi_picks[(variant.name, hbi_arm)]
            for distance in distances:
                emit(
                    hbi_arm,
                    distance,
                    variant.name,
                    rows,
                    nn[rows],
                    value[rows],
                    tied_hbi[rows],
                    [(SUBSET_HBI, np.ones(rows.size, dtype=bool))],
                    primary_tied=primary_tied_hbi[rows],
                )
        _log(f"variant {variant.name}: HBI scored on {rows.size} answerable queries")

    boot = bootstrap_means(
        collected,
        {key: int(rows.size) for key, rows in query_sets.items()},
        n_boot=n_boot,
        seed=seed,
    )
    summary = _finish_summary(
        summary_rows,
        collected,
        boot,
        baselines,
        query_sets,
        scorer.names,
        hbi_baseline=hbi_baseline,
        hbi_arms=set(hbi_labels),
    )
    paired = _paired_differences(collected, boot, report_arms)

    per_query = pd.concat(per_query_frames, ignore_index=True)
    baseline_frame = _baseline_frame(
        baselines, kept, ids, scorer.names, subset_masks=subset_masks
    )

    # Staged, then moved into place at the very end: the report is either all there or
    # untouched. The out-dir is reused across runs, so a crash after the first write would
    # otherwise leave fresh CSVs beside a stale manifest — complete-looking and wrong.
    staging = out_dir / f".staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        summary.to_csv(staging / "summary.csv", index=False)
        paired.to_csv(staging / "paired_differences.csv", index=False)
        per_query.to_parquet(staging / "per_query.parquet", index=False)
        baseline_frame.to_parquet(staging / "per_query_baseline.parquet", index=False)
        # Only when it was actually computed: a header-only tau_b.csv next to a manifest
        # that lists it reads as "tau-b ran and found nothing", which is a different claim
        # from "this run skipped tau-b".
        if labels_kind == "go" and tau:
            pd.DataFrame(
                tau_rows, columns=["arm", "distance", "tau_b", "n_pairs", "subsampled"]
            ).to_csv(staging / "tau_b.csv", index=False)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    inputs = {"freeze": str(freeze), "labels": str(labels)}
    hashed = {str(freeze): sha256_file(freeze), str(labels): sha256_file(labels)}
    for key, path in (("go_obo", go_obo), ("identity_m8", identity_m8)):
        if path is not None:
            inputs[key] = str(path)
            hashed[str(path)] = sha256_file(path)
    if hash_arms:
        for path in arm_paths:
            hashed[str(path)] = sha256_file(path)

    manifest = {
        "labels_kind": labels_kind,
        "inputs": inputs,
        "sha256": hashed,
        "emb_dir": str(emb_dir),
        "arms": arm_labels,
        "hbi_arms": hbi_labels,
        "distances": list(distances),
        "variants": [
            {
                "name": v.name,
                "description": v.description,
                "n_excluded_ordered_pairs": v.n_excluded_pairs,
                "n_queries": int(kept[v.name].size),
                "n_dropped": int(n - kept[v.name].size),
                "median_eligible_neighbours": float(
                    np.median(baselines[v.name]["n_eligible"][kept[v.name]])
                ),
                "subsets": {
                    name: int(query_sets[(v.name, name)].size)
                    for name in subset_names
                    if (v.name, name) in query_sets
                },
            }
            for v in variants
        ],
        "subsets": {
            name: SUBSET_DESCRIPTIONS[name]
            for name in subset_names
            if any((v.name, name) in query_sets for v in variants)
        },
        "subset_note": f"{SUBSET_NO_HIT} is published under variant {VARIANT_ALL!r} only: "
        "a query with no MMseqs2 hit has nothing for an identity variant to exclude, so the "
        "cell would be identical under all three",
        "hbi": {
            "criteria": dict(HBI_ARMS),
            "note": "an HBI row is identical under every distance; it is repeated so that "
            "each paired difference against an embedding arm sits in one cell",
            "neighbour_distance_units": {
                "<embedding arm>": "the named distance (euclidean, or 1 - cosine)",
                "hbi_evalue": "the MMseqs2 E-value of the chosen hit (0.0 means underflow, "
                "not a zero distance)",
                "hbi_fident": "1 - fident of the chosen hit",
            },
            "baseline_scope": "summary.csv chance/oracle on an hbi_* row are computed over "
            "that query's eligible MMseqs2 hits, not over the whole cohort: the cohort "
            "oracle is not a ceiling a sequence search could reach",
            "n_queries_without_any_cohort_hit": (
                0 if hits is None else int((~hits.has_hit).sum())
            ),
            "n_hits_symmetrised": 0 if hits is None else hits.n_directed,
        },
        "identity_search": identity_stats,
        "n_frozen": len(frozen_ids),
        "n_cohort": n,
        "label_cleaning": label_counts,
        "drop_protein_binding": drop_protein_binding,
        "propagated_cleaning": {
            "dropped_from_propagated_sets": getattr(scorer, "propagated_drop", {}),
            "note": "f1 is computed on the propagated sets, so a term is only really "
            "dropped if it is dropped from the closure; wang_bma is defined on the "
            "unpropagated annotated sets and is unaffected by this sensitivity",
        },
        "wang_bma_check": wang_check,
        "parameters": {
            "fident_max": fident_max,
            "evalue_max": evalue_max,
            "n_boot": n_boot,
            "seed": seed,
            "block_size": block_size,
            "tau_max_pairs": tau_max_pairs,
            "ci": "95% percentile bootstrap over queries",
        },
        "outputs": {
            "summary": str(out_dir / "summary.csv"),
            "paired_differences": str(out_dir / "paired_differences.csv"),
            "per_query": str(out_dir / "per_query.parquet"),
            "per_query_baseline": str(out_dir / "per_query_baseline.parquet"),
            **({"tau_b": str(out_dir / "tau_b.csv")} if labels_kind == "go" and tau else {}),
        },
        "versions": _versions(),
    }
    try:
        (staging / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2) + "\n")
        for path in sorted(staging.iterdir()):
            os.replace(path, out_dir / path.name)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rmdir()
    return manifest


def _finish_summary(
    rows: list[dict],
    collected: Mapping[tuple[str, str, str, str], Mapping[str, np.ndarray]],
    boot: Mapping[tuple[str, str, str, str], Mapping[str, np.ndarray]],
    baselines: Mapping[str, Mapping[str, np.ndarray]],
    query_sets: Mapping[tuple[str, str], np.ndarray],
    score_names: Sequence[str],
    *,
    hbi_baseline: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    hbi_arms: set[str] | None = None,
) -> pd.DataFrame:
    """One summary row per (arm, distance, variant, subset, score) with its bootstrap CI.

    chance and oracle are averaged over the SUBSET's queries, not the variant's, so a cell
    is always compared against the baseline of the queries it was actually scored on — the
    ``no_hit`` queries are not a random sample of the cohort and their chance level differs.

    They are also scoped to the arm's own candidate set (``baseline_scope``): an embedding
    arm chooses among every eligible cohort protein, the homology arms only among their
    MMseqs2 hits, and the cohort oracle printed on an HBI row would be a ceiling that arm
    could not reach even in principle.
    """
    hbi_baseline = hbi_baseline or {}
    hbi_arms = hbi_arms or set()
    out: list[dict] = []
    for row in rows:
        variant, subset = row["variant"], row["subset"]
        queries = query_sets[(variant, subset)]
        is_hbi = row["arm"] in hbi_arms and variant in hbi_baseline
        source = hbi_baseline[variant] if is_hbi else baselines[variant]
        for score in score_names:
            key = (row["distance"], variant, subset, score)
            samples = boot[key][row["arm"]]
            lo, hi = percentile_ci(samples)
            values = collected[key][row["arm"]]
            out.append(
                {
                    **row,
                    "score": score,
                    "mean": float(values.mean()),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "chance": float(np.mean(source[f"chance_{score}"][queries])),
                    "oracle": float(np.mean(source[f"oracle_{score}"][queries])),
                    "baseline_scope": "mmseqs_hits" if is_hbi else "cohort",
                }
            )
    columns = [
        "arm", "distance", "variant", "subset", "score", "n_queries", "n_dropped",
        "n_not_in_subset", "mean", "ci_lo", "ci_hi", "chance", "oracle", "baseline_scope",
        "n_ties", "n_primary_ties",
    ]
    return pd.DataFrame(out)[columns].sort_values(["distance", "variant", "subset", "score", "arm"])


def bootstrap_means(
    collected: Mapping[tuple[str, str, str, str], Mapping[str, np.ndarray]],
    n_queries: Mapping[tuple[str, str], int],
    *,
    n_boot: int,
    seed: int,
) -> dict[tuple[str, str, str, str], dict[str, np.ndarray]]:
    """``(distance, variant, subset, score) -> arm -> (n_boot,)`` resampled means.

    One weight matrix per (variant, subset) — the unit on which queries are resampled —
    reused by every arm/distance/score of that unit and then freed (it is
    ``n_boot x n_queries`` floats, 320 MB for the GO cohort). Reuse is not an optimisation
    here: it is what makes the arm differences, including pLM minus HBI, paired.
    """
    out: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = {}
    for (variant, subset), size in n_queries.items():
        weights = bootstrap_weights(size, n_boot, seed)
        for key, per_arm in collected.items():
            if key[1] != variant or key[2] != subset:
                continue
            arms = list(per_arm)
            # numpy 2.x emits spurious "divide by zero / overflow / invalid value in
            # matmul" on small products (uninitialised SIMD tail lanes, gh-27509). The
            # operands here are finite scores and integer multiplicities, so the only
            # honest response is to silence the false alarm and then CHECK the result —
            # a real NaN must still stop the run rather than reach a published CI.
            with np.errstate(all="ignore"):
                means = (np.stack([per_arm[a] for a in arms]) @ weights.T) / size
            if not np.isfinite(means).all():
                raise TransferInputError(f"non-finite bootstrap means for {key}")
            out[key] = dict(zip(arms, means, strict=True))
        del weights
    return out


def _paired_differences(
    collected: Mapping[tuple[str, str, str, str], Mapping[str, np.ndarray]],
    boot: Mapping[tuple[str, str, str, str], Mapping[str, np.ndarray]],
    arm_labels: Sequence[str],
) -> pd.DataFrame:
    """Every arm pair's mean difference with a paired-bootstrap CI.

    Paired because both arms are averaged over the SAME resample of queries, so the
    difference's interval removes the query-sampling variance the two arms share — the
    only way a 1-point gap between two pLMs can be called significant on 7k queries. The
    HBI arms sort last in ``arm_labels``, so a pLM-vs-HBI row reads ``mean_diff = pLM -
    homology search``: positive means the embedding wins.
    """
    rows: list[dict] = []
    for (distance, variant, subset, score), per_arm in collected.items():
        arms = [a for a in arm_labels if a in per_arm]
        for i, arm_a in enumerate(arms):
            for arm_b in arms[i + 1 :]:
                key = (distance, variant, subset, score)
                diff = boot[key][arm_a] - boot[key][arm_b]
                lo, hi = percentile_ci(diff)
                rows.append(
                    {
                        "distance": distance,
                        "variant": variant,
                        "subset": subset,
                        "score": score,
                        "arm_a": arm_a,
                        "arm_b": arm_b,
                        "mean_a": float(per_arm[arm_a].mean()),
                        "mean_b": float(per_arm[arm_b].mean()),
                        "mean_diff": float(per_arm[arm_a].mean() - per_arm[arm_b].mean()),
                        "ci_lo": float(lo),
                        "ci_hi": float(hi),
                        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
                    }
                )
    return pd.DataFrame(rows)


def _baseline_frame(
    baselines: Mapping[str, Mapping[str, np.ndarray]],
    kept: Mapping[str, np.ndarray],
    ids: Sequence[str],
    score_names: Sequence[str],
    *,
    subset_masks: Mapping[str, Sequence[tuple[str, np.ndarray]]],
) -> pd.DataFrame:
    """Per-query chance/oracle plus the subset flags, kept out of ``per_query.parquet``
    because they are embedding-independent: repeating them per arm would multiply the file
    by 28. The flags are what lets a reader re-derive any subset mean from
    ``per_query.parquet`` instead of trusting ``summary.csv``."""
    # Every variant gets every column, even where that subset is empty for it, so the file
    # has one schema rather than one per variant.
    all_subsets = sorted({name for masks in subset_masks.values() for name, _ in masks})
    frames = []
    for variant, rows in kept.items():
        data = {
            "variant": pd.Categorical([variant] * rows.size, categories=list(kept)),
            "query_id": pd.Categorical.from_codes(rows, categories=list(ids)),
            "n_eligible": baselines[variant]["n_eligible"][rows],
        }
        for score in score_names:
            data[f"chance_{score}"] = baselines[variant][f"chance_{score}"][rows]
            data[f"oracle_{score}"] = baselines[variant][f"oracle_{score}"][rows]
        present = dict(subset_masks[variant])
        for subset in all_subsets:
            data[f"in_{subset}"] = present.get(subset, np.zeros(rows.size, dtype=bool))
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transfer_report",
        description="Leave-one-out 1-NN annotation transfer (EC or GO-MF) against the "
        "MMseqs2 homology baseline, with the sequence-identity control, exact chance/oracle "
        "baselines and paired-bootstrap arm differences.",
    )
    parser.add_argument("--labels-kind", required=True, choices=("ec", "go"))
    parser.add_argument("--freeze", required=True, help="Cohort freeze JSON (its 'ids').")
    parser.add_argument(
        "--labels", required=True,
        help="EC: TSV with 'Entry' + ';'-joined 'EC number'. GO: TSV with protein_id + GO_term.",
    )
    parser.add_argument("--go-obo", default=None, help="go-basic.obo (required for --labels-kind go).")
    parser.add_argument("--emb-dir", required=True, help="Directory of per-arm .h5 slices.")
    parser.add_argument(
        "--arms", nargs="+", default=None,
        help="Arm labels to score (default: every .h5 in --emb-dir; random_init_<m>_seed0 "
        "is labelled randinit_<m>).",
    )
    parser.add_argument("--distances", nargs="+", default=["euclidean", "cosine"],
                        choices=("euclidean", "cosine"))
    parser.add_argument(
        "--identity-m8", default=None,
        help="MMseqs2 all-vs-all table, header-less columns "
        "query,target,fident,evalue,alnlen,qcov,tcov. Without it only the unrestricted "
        "variant is scored, the identity control is NOT answered and there is no homology "
        "(hbi_evalue / hbi_fident) baseline to compare the pLMs against.",
    )
    parser.add_argument("--fident-max", type=float, default=0.30)
    parser.add_argument("--evalue-max", type=float, default=1e-3)
    parser.add_argument(
        "--drop-protein-binding", action="store_true",
        help="GO sensitivity: drop GO:0005515 from every annotation AND from every "
        "propagated set, and remove the proteins left with no term (all counted in the "
        "manifest). The propagated half is the half that bites: UniProt exports the "
        "specific descendants (GO:0042802, GO:0042803), never the generic term, so "
        "dropping it from the annotations alone provably changes no score. wang_bma is "
        "defined on the unpropagated sets and does not move under this flag.",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--no-tau", action="store_true", help="Skip the secondary GO tau-b.")
    parser.add_argument(
        "--tau-max-pairs", type=int, default=0,
        help="0 (default) uses every cohort pair, as the design requires. tau-b holds the "
        "whole pair vector in memory, so this is what sets the job's --mem: measured peak "
        "RSS was 2.9 GB at n=7,000 (17 s per arm, both distances) and 7.9 GB at n=14,000 "
        "(78 s per arm) on the real ontology. A cap subsamples pairs reproducibly (recorded "
        "in tau_b.csv) if the cohort outgrows the node.",
    )
    parser.add_argument("--wang-check-pairs", type=int, default=200)
    parser.add_argument("--no-hash-arms", action="store_true",
                        help="Skip hashing the embedding slices (they dominate the manifest cost).")
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Exit 0 on success, 2 on any input/consistency fault (never a partial report)."""
    args = build_parser().parse_args(argv)
    try:
        manifest = run_transfer_report(
            labels_kind=args.labels_kind,
            freeze=args.freeze,
            labels=args.labels,
            go_obo=args.go_obo,
            emb_dir=args.emb_dir,
            arms=args.arms,
            distances=tuple(args.distances),
            identity_m8=args.identity_m8,
            drop_protein_binding=args.drop_protein_binding,
            fident_max=args.fident_max,
            evalue_max=args.evalue_max,
            n_boot=args.n_boot,
            seed=args.seed,
            block_size=args.block_size,
            tau=not args.no_tau,
            tau_max_pairs=args.tau_max_pairs,
            wang_check_pairs=args.wang_check_pairs,
            hash_arms=not args.no_hash_arms,
            out_dir=args.out_dir,
        )
    except (TransferInputError, FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"transfer_report: INPUT ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        f"transfer_report: {manifest['labels_kind']} n={manifest['n_cohort']} "
        f"arms={len(manifest['arms'])} variants={len(manifest['variants'])} "
        f"-> {manifest['outputs']['summary']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
