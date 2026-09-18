"""The vectorised GO-MF matrices must equal the scalar definitions they replace.

``go_similarity_matrix`` exists because the scalar Wang BMA is far too slow for all
~25M pairs of the GO cohort. A matrix rewrite is only admissible if it is the SAME
number, so the central test here compares it with
``WangSimilarity.protein_similarity_bma`` on >=200 random protein pairs over a random
DAG with both ``is_a`` and ``part_of`` edges and multiple paths to an ancestor (the case
where the max-over-paths S-value matters), and again on the real ontology when it is
available.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from data_preparation.go_semantic_similarity import GOTerm, WangSimilarity, parse_obo
from evaluation.go_similarity_matrix import (
    MF_ROOT,
    PROTEIN_BINDING,
    clean_mf_annotations,
    max_abs_diff_vs_scalar,
    parse_alt_ids,
    propagate_mf,
    propagated_f1_matrix,
    wang_bma_matrix,
    wang_s_values,
    wang_term_similarity_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _term(tid: str, parents=(), namespace: str = "molecular_function") -> GOTerm:
    t = GOTerm(id=tid, name=tid, namespace=namespace)
    t.parents = list(parents)
    return t


def _random_mf_dag(n_terms: int = 60, seed: int = 0) -> dict[str, GOTerm]:
    """Random DAG under the MF root: 1-3 parents per term, ~25% of edges part_of."""
    rng = np.random.default_rng(seed)
    terms = {MF_ROOT: _term(MF_ROOT)}
    ids = [MF_ROOT]
    for k in range(1, n_terms):
        tid = f"GO:{k:07d}"
        n_par = int(rng.integers(1, min(3, len(ids)) + 1))
        parents = rng.choice(len(ids), size=n_par, replace=False)
        rels = ["part_of" if rng.random() < 0.25 else "is_a" for _ in parents]
        terms[tid] = _term(tid, [(ids[p], r) for p, r in zip(parents, rels, strict=True)])
        ids.append(tid)
    return terms


def _random_protein_sets(terms: dict[str, GOTerm], n: int, seed: int) -> list[frozenset[str]]:
    rng = np.random.default_rng(seed)
    pool = sorted(t for t in terms if t != MF_ROOT)
    return [
        frozenset(rng.choice(pool, size=int(rng.integers(1, 5)), replace=False).tolist())
        for _ in range(n)
    ]


# ── S-values and term similarity ─────────────────────────────────────────────


def test_s_values_equal_scalar_depth_first_search():
    terms = _random_mf_dag()
    scalar = WangSimilarity(terms)
    ours = wang_s_values(terms, terms)
    for tid in terms:
        ref = scalar._compute_s_values(tid)
        assert ours[tid].keys() == ref.keys()
        for anc, v in ref.items():
            assert ours[tid][anc] == pytest.approx(v, abs=1e-12)


def test_s_value_takes_max_over_paths():
    # C reaches R via A (is_a, is_a: 0.8*0.8=0.64) and via B (part_of, is_a: 0.6*0.8=0.48).
    terms = {
        MF_ROOT: _term(MF_ROOT),
        "GO:A": _term("GO:A", [(MF_ROOT, "is_a")]),
        "GO:B": _term("GO:B", [(MF_ROOT, "is_a")]),
        "GO:C": _term("GO:C", [("GO:A", "is_a"), ("GO:B", "part_of")]),
    }
    s = wang_s_values(terms, ["GO:C"])["GO:C"]
    assert s == pytest.approx({"GO:C": 1.0, "GO:A": 0.8, "GO:B": 0.6, MF_ROOT: 0.64})


def test_term_similarity_matrix_equals_scalar_for_every_term_pair():
    terms = _random_mf_dag(n_terms=40, seed=3)
    vocab = sorted(terms)
    sim = wang_term_similarity_matrix(vocab, terms)
    scalar = WangSimilarity(terms)
    ref = np.array([[scalar.term_similarity(a, b) for b in vocab] for a in vocab])
    assert np.max(np.abs(sim - ref)) < 1e-12
    assert np.all(np.diag(sim) == 1.0)


# ── design item 11: vectorised BMA == scalar BMA on >=200 random pairs ───────


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorised_bma_equals_scalar_on_random_pairs(seed):
    terms = _random_mf_dag(n_terms=80, seed=seed)
    proteins = _random_protein_sets(terms, n=60, seed=seed + 100)
    bma = wang_bma_matrix(proteins, terms)
    worst, n_checked = max_abs_diff_vs_scalar(proteins, bma, terms, n_pairs=250, seed=seed)
    assert n_checked >= 200
    assert worst < 1e-9


def test_bma_matrix_is_symmetric_with_unit_diagonal():
    terms = _random_mf_dag(n_terms=50, seed=7)
    proteins = _random_protein_sets(terms, n=25, seed=8)
    proteins.append(proteins[0])  # an identical set elsewhere in the cohort
    bma = wang_bma_matrix(proteins, terms)
    assert np.allclose(bma, bma.T, atol=1e-15)
    assert np.allclose(np.diag(bma), 1.0, atol=1e-15)
    assert bma[0, len(proteins) - 1] == pytest.approx(1.0, abs=1e-15)


def _main_checkout() -> Path | None:
    """This repo's primary checkout, when the tests are running from a git worktree.

    ``data/`` is gitignored and lives only in the main checkout, so in a worktree the
    real-ontology check below would find no OBO and skip — quietly deleting the only test
    that validates the vectorised Wang BMA against the scalar reference on the real 38k-term
    ontology. A worktree's ``.git`` is a file pointing at ``<main>/.git/worktrees/<name>``.
    """
    dotgit = REPO_ROOT / ".git"
    if not dotgit.is_file():
        return None
    text = dotgit.read_text().strip()
    if not text.startswith("gitdir:"):
        return None
    for parent in Path(text.split(":", 1)[1].strip()).parents:
        if parent.name == ".git":
            return parent.parent
    return None


def _real_obo() -> Path | None:
    main = _main_checkout()
    candidates = [
        os.environ.get("PLM_CHOICE_GO_OBO"),
        REPO_ROOT / "data/reference/go/go-basic.obo",
        None if main is None else main / "data/reference/go/go-basic.obo",
    ]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


@pytest.mark.skipif(_real_obo() is None, reason="go-basic.obo not available (set PLM_CHOICE_GO_OBO)")
def test_vectorised_bma_equals_scalar_on_real_ontology():
    go_terms = parse_obo(_real_obo())
    mf = sorted(t for t, v in go_terms.items() if v.namespace == "molecular_function" and t != MF_ROOT)
    rng = np.random.default_rng(42)
    proteins = [
        frozenset(rng.choice(mf, size=int(rng.integers(1, 6)), replace=False).tolist())
        for _ in range(80)
    ]
    proteins.append(frozenset({PROTEIN_BINDING}))
    proteins.append(frozenset({PROTEIN_BINDING, "GO:0003824"}))
    bma = wang_bma_matrix(proteins, go_terms)
    worst, n_checked = max_abs_diff_vs_scalar(proteins, bma, go_terms, n_pairs=300, seed=42)
    assert n_checked >= 200
    assert worst < 1e-9


# ── propagated-set F1 ─────────────────────────────────────────────────────────


def _small_ontology() -> dict[str, GOTerm]:
    """R <- A <- B ; R <- C <- D ; D part_of A ; plus a BP term X above nothing in MF."""
    return {
        MF_ROOT: _term(MF_ROOT),
        "GO:A": _term("GO:A", [(MF_ROOT, "is_a")]),
        "GO:B": _term("GO:B", [("GO:A", "is_a")]),
        "GO:C": _term("GO:C", [(MF_ROOT, "is_a")]),
        "GO:D": _term("GO:D", [("GO:C", "is_a"), ("GO:A", "part_of")]),
        "GO:X": _term("GO:X", [], namespace="biological_process"),
    }


def test_propagation_follows_is_a_and_part_of_and_excludes_root():
    terms = _small_ontology()
    p1, p2, p3 = propagate_mf(
        [frozenset({"GO:B"}), frozenset({"GO:D"}), frozenset({"GO:A", "GO:C"})], terms
    )
    assert p1 == {"GO:B", "GO:A"}
    assert p2 == {"GO:D", "GO:C", "GO:A"}  # A only via part_of
    assert p3 == {"GO:A", "GO:C"}


def test_propagation_can_drop_an_inherited_ancestor():
    """The protein-binding sensitivity. GO:A is never annotated on p1 — it is inherited
    from GO:B — so dropping it from the ANNOTATIONS cannot remove it from the propagated
    set, which is where the F1 lives. ``drop_terms`` removes it after the closure."""
    terms = _small_ontology()
    sets = [frozenset({"GO:B"}), frozenset({"GO:D"})]
    assert all("GO:A" in s for s in propagate_mf(sets, terms))
    dropped = propagate_mf(sets, terms, drop_terms=("GO:A",))
    assert dropped == [frozenset({"GO:B"}), frozenset({"GO:D", "GO:C"})]
    assert MF_ROOT not in set().union(*dropped)  # the root stays excluded either way


def test_propagated_f1_hand_computed():
    terms = _small_ontology()
    f1 = propagated_f1_matrix(
        [frozenset({"GO:B"}), frozenset({"GO:D"}), frozenset({"GO:A", "GO:C"})], terms
    )
    expected = np.array(
        [
            [1.0, 2 * 1 / (2 + 3), 2 * 1 / (2 + 2)],
            [2 * 1 / (2 + 3), 1.0, 2 * 2 / (3 + 2)],
            [2 * 1 / (2 + 2), 2 * 2 / (3 + 2), 1.0],
        ]
    )
    assert np.allclose(f1, expected, atol=1e-15)


def test_propagated_f1_equals_precision_recall_definition():
    """F1 of transferring N onto true Q is 2PR/(P+R) with P=|Q&N|/|N|, R=|Q&N|/|Q|."""
    terms = _random_mf_dag(n_terms=50, seed=11)
    proteins = _random_protein_sets(terms, n=15, seed=12)
    f1 = propagated_f1_matrix(proteins, terms)
    prop = propagate_mf(proteins, terms)
    for i, q in enumerate(prop):
        for j, n in enumerate(prop):
            inter = len(q & n)
            if inter == 0:
                assert f1[i, j] == 0.0
                continue
            p, r = inter / len(n), inter / len(q)
            assert f1[i, j] == pytest.approx(2 * p * r / (p + r), abs=1e-12)


# ── label cleaning ────────────────────────────────────────────────────────────


def test_clean_mf_annotations_counts_every_drop():
    terms = _small_ontology()
    terms[PROTEIN_BINDING] = _term(PROTEIN_BINDING, [(MF_ROOT, "is_a")])
    raw = {
        "P1": {"GO:OLD", "GO:X", MF_ROOT},          # alt id -> B; BP term; root
        "P2": {PROTEIN_BINDING},                     # only protein binding
        "P3": {"GO:GONE", "GO:C", PROTEIN_BINDING},  # obsolete + real + binding
    }
    cleaned, counts = clean_mf_annotations(
        raw, terms, {"GO:OLD": "GO:B"}, drop_terms={PROTEIN_BINDING}
    )
    assert cleaned == {"P1": frozenset({"GO:B"}), "P3": frozenset({"GO:C"})}
    assert counts == {
        "alt_id_mapped": 1, "not_in_ontology": 1, "not_mf": 1, "root": 1, "dropped_term": 2,
    }


def test_parse_alt_ids_ignores_obsolete_terms(tmp_path):
    obo = tmp_path / "mini.obo"
    obo.write_text(
        "format-version: 1.2\n\n"
        "[Term]\nid: GO:0000001\nalt_id: GO:0000009\nalt_id: GO:0000008\n"
        "namespace: molecular_function\n\n"
        "[Term]\nid: GO:0000002\nalt_id: GO:0000007\nis_obsolete: true\n\n"
        "[Typedef]\nid: part_of\n"
    )
    assert parse_alt_ids(obo) == {"GO:0000009": "GO:0000001", "GO:0000008": "GO:0000001"}
