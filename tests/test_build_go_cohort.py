"""The GO-MF cohort definition must be reproducible from its rules, not from a run.

Every published GO number is a mean over the 20,073 proteins ``scripts/build_go_cohort.py``
selects, so the rules that pick them are part of the result. The script's own determinism
was checked by re-running it end to end (byte-identical freeze), but a rerun cannot say
*which* rule a future edit broke. These tests pin the four choices the design left to the
script — the quartile tie, the superfamily assignment, the cap's order-independence, and
the FASTA/m8 plumbing the identity control is built from — on inputs small enough to check
by hand.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build_go_cohort.py"


@pytest.fixture(scope="module")
def cohort():
    """scripts/ is not a package, so load the module from its path."""
    spec = importlib.util.spec_from_file_location("build_go_cohort", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── superfamily assignment ────────────────────────────────────────────────────


def test_a_multi_superfamily_protein_takes_the_lexicographically_smallest_id(cohort):
    """String order, not numeric: 3.40.50.10140 sorts before 3.40.50.720, and the answer
    must not depend on the order the cross-references appear in the dump."""
    ids = ["3.40.50.720", "3.40.50.10140", "1.10.10.10"]
    assert cohort.assign_superfamily(ids) == "1.10.10.10"
    assert cohort.assign_superfamily(reversed(ids)) == "1.10.10.10"
    assert cohort.assign_superfamily(["3.40.50.720", "3.40.50.10140"]) == "3.40.50.10140"


def test_a_protein_without_a_superfamily_is_an_error_not_a_silent_bucket(cohort):
    with pytest.raises(ValueError, match="no Gene3D id"):
        cohort.assign_superfamily([])


# ── length quartiles ──────────────────────────────────────────────────────────


def test_length_quartile_ties_fall_in_the_lower_bin(cohort):
    """Documented rule: quartile = number of edges STRICTLY below the length, so a protein
    exactly on an edge belongs to the bin below it."""
    edges = np.array([10.0, 20.0, 30.0])
    got = [cohort.length_quartile(x, edges) for x in (5, 10, 11, 20, 21, 30, 31)]
    assert got == [0, 0, 1, 1, 2, 2, 3]


def test_length_quartile_edges_are_the_three_quartiles(cohort):
    edges = cohort.length_quartile_edges(range(1, 101))
    assert edges.tolist() == pytest.approx(np.quantile(np.arange(1.0, 101.0), [0.25, 0.5, 0.75]))
    assert len(edges) == 3


# ── the cap ───────────────────────────────────────────────────────────────────


def test_stratified_cap_keeps_small_cells_whole_and_caps_the_rest(cohort):
    cells = {("A", 0): [f"p{i:02d}" for i in range(10)], ("B", 1): ["q0", "q1"]}
    ids, n_capped = cohort.stratified_cap(cells, cap=4, seed=42)
    assert n_capped == 1
    assert len(ids) == 4 + 2
    assert {"q0", "q1"} <= set(ids)
    assert ids == sorted(ids)


def test_stratified_cap_depends_only_on_the_contents_and_the_seed(cohort):
    """Cell and member order come from dict insertion and file order, which are not part
    of the experiment: the drawn cohort must not move when they change."""
    members = [f"p{i:02d}" for i in range(30)]
    forward = {("A", 0): members, ("B", 0): ["x", "y"]}
    reversed_order = {("B", 0): ["y", "x"], ("A", 0): list(reversed(members))}
    assert cohort.stratified_cap(forward, cap=7, seed=42) == cohort.stratified_cap(
        reversed_order, cap=7, seed=42
    )
    other_seed = cohort.stratified_cap(forward, cap=7, seed=7)[0]
    assert other_seed != cohort.stratified_cap(forward, cap=7, seed=42)[0]


def test_content_sha256_is_order_independent(cohort):
    """The freeze's content hash is shared.protein_cohort.content_hash — the one every other
    freeze in the repo records — so it must hash the SET, not the order it arrived in."""
    assert cohort.content_hash(["b", "a"]) == cohort.content_hash(["a", "b"])
    assert cohort.content_hash(["a"]) != cohort.content_hash(["a", "b"])


# ── the union FASTA and the hit table behind the identity control ─────────────


def _write_fasta(tmp_path: Path, seqs: dict[str, str], line_width: int = 4) -> tuple[Path, Path]:
    fasta, fai = tmp_path / "u.fasta", tmp_path / "u.fasta.fai"
    offset = 0
    index = []
    with open(fasta, "w") as handle:
        for pid, seq in seqs.items():
            header = f">{pid}\n"
            handle.write(header)
            offset += len(header)
            wrapped = [seq[i : i + line_width] for i in range(0, len(seq), line_width)]
            handle.write("\n".join(wrapped) + "\n")
            index.append(f"{pid}\t{len(seq)}\t{offset}\t{line_width}\t{line_width + 1}")
            offset += len(seq) + len(wrapped)
    fai.write_text("\n".join(index) + "\n")
    return fasta, fai


def test_iter_fasta_records_reads_the_exact_sequence_by_offset(cohort, tmp_path):
    seqs = {"P1": "MKVLA", "P2": "AC", "P3": "MKVLAAAAAAAA"}
    fasta, fai = _write_fasta(tmp_path, seqs)
    assert dict(cohort.iter_fasta_records(fasta, fai, ["P3", "P1"])) == {
        "P3": seqs["P3"],
        "P1": seqs["P1"],
    }


def test_an_id_missing_from_the_index_is_an_error(cohort, tmp_path):
    """A union FASTA silently missing a protein would drop it from every identity variant."""
    fasta, fai = _write_fasta(tmp_path, {"P1": "MKVLA"})
    with pytest.raises(KeyError, match="P9"):
        list(cohort.iter_fasta_records(fasta, fai, ["P9"]))


def test_drop_self_hits_removes_only_self_rows_and_counts_them(cohort, tmp_path):
    raw = tmp_path / "raw.m8"
    raw.write_text(
        "A\tA\t1.0\t0.0\t10\t1.0\t1.0\n"
        "A\tB\t0.5\t1e-20\t10\t0.9\t0.9\n"
        "B\tA\t0.5\t1e-19\t10\t0.9\t0.9\n"
        "C\tC\t1.0\t0.0\t10\t1.0\t1.0\n"
    )
    out = tmp_path / "clean.m8"
    stats = cohort.drop_self_hits(raw, out)
    assert stats == {
        "hits_raw": 4,
        "self_hits_dropped": 2,
        "hits": 2,
        "queries_with_non_self_hit": 2,
    }
    assert [line.split("\t")[:2] for line in out.read_text().splitlines()] == [["A", "B"], ["B", "A"]]
