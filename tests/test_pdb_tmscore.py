"""
Fixture tests for the experimental-TM pipeline (task B6 / reviewer R2.2).

Why these exist at all: the defect the 2026-09-18 rewrite fixed was a SILENTLY
EMPTY parser -- ``load_sifts_mapping`` indexed ``parts[2]`` of a two-column file,
every row raised IndexError into a bare ``except: continue``, and the job logged
"0 pairs" and exited 0. Nothing in the suite noticed, because nothing in the
suite called the module. A green suite would look exactly the same if the
rewrite had reintroduced it. So every hand-rolled parser here gets one fixture:
one SIFTS line, one entries.idx line, one mmCIF stanza, one -outfmt 2 line.
"""

import gzip

import pytest

from data_preparation.pdb_tmscore import (
    MAX_PDB_RESSEQ,
    _auth_seq_key,
    _split_cif_line,
    _write_ca_pdb,
    extract_chain_ca_pdb,
    load_entry_metadata,
    load_entry_pdb_map,
    load_sifts_chain_mapping,
    run_usalign,
    select_representative_chain,
)

# --------------------------------------------------------------------------- #
#                              REFERENCE TABLES
# --------------------------------------------------------------------------- #

# Residue-level SIFTS: PDB CHAIN SP_PRIMARY RES_BEG RES_END PDB_BEG PDB_END
# SP_BEG SP_END. Two header lines (a '#' date comment and the column names).
_CHAIN_SIFTS = (
    "# 2026/09/01 - SIFTS\n"
    "PDB\tCHAIN\tSP_PRIMARY\tRES_BEG\tRES_END\tPDB_BEG\tPDB_END\tSP_BEG\tSP_END\n"
    "1abc\tA\tP12345\t1\t100\t1\t100\t10\t109\n"
    "1abc\tB\tP12345\t1\t50\t201\t250\t10\t59\n"
    "2xyz\tA\tQ99999\t1\t80\t1\t80\t1\t80\n"
)

# Entry-level SIFTS: SP_PRIMARY, PDB (semicolon-joined). One '#' date line, then
# the header.
_ENTRY_SIFTS = (
    "# 2026/09/01 - SIFTS\n"
    "SP_PRIMARY\tPDB\n"
    "P12345\t1abc;3def\n"
    "Q99999\t2xyz\n"
    "P00000\t9zzz\n"
)


def test_load_sifts_chain_mapping_keeps_chain_and_author_ranges(tmp_path):
    p = tmp_path / "pdb_chain_uniprot.tsv"
    p.write_text(_CHAIN_SIFTS)
    out = load_sifts_chain_mapping(p, {"P12345"})
    # Chain-level, not entry-level: the two chains of 1abc stay separate keys.
    assert set(out["P12345"]) == {("1abc", "A"), ("1abc", "B")}
    # (sp_beg, sp_end, pdb_beg, pdb_end) -- author labels stay strings.
    assert out["P12345"][("1abc", "B")] == [(10, 59, "201", "250")]
    assert "Q99999" not in out  # not a target


def test_load_sifts_chain_mapping_reads_gzip(tmp_path):
    p = tmp_path / "pdb_chain_uniprot.tsv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(_CHAIN_SIFTS)
    assert set(load_sifts_chain_mapping(p, {"Q99999"})["Q99999"]) == {("2xyz", "A")}


def test_load_entry_pdb_map_splits_the_semicolon_list(tmp_path):
    p = tmp_path / "uniprot_pdb.tsv"
    p.write_text(_ENTRY_SIFTS)
    out = load_entry_pdb_map(p, {"P12345", "Q99999"})
    # The original bug used the whole "1abc;3def" string as one pdb_id.
    assert out["P12345"] == {"1abc", "3def"}
    assert out["Q99999"] == {"2xyz"}
    assert "P00000" not in out


# entries.idx: IDCODE, HEADER, ACCESSION DATE, COMPOUND, SOURCE, AUTHOR LIST,
# RESOLUTION, EXPERIMENT TYPE. An EMPTY experiment type means X-ray.
def _entries_idx(*rows: str) -> str:
    return "IDCODE\tHEADER\tDATE\tCOMPOUND\tSOURCE\tAUTHORS\tRESOLUTION\tEXPERIMENT TYPE\n" \
           "------\t------\t----\t--------\t------\t-------\t----------\t---------------\n" \
           + "".join(rows)


def test_load_entry_metadata_empty_experiment_type_is_xray(tmp_path):
    p = tmp_path / "entries.idx"
    p.write_text(_entries_idx("1ABC\th\td\tc\ts\ta\t1.80\t\n"))
    assert load_entry_metadata(p)["1abc"] == ("X-RAY DIFFRACTION", 1.80)


def test_load_entry_metadata_nmr_has_no_resolution(tmp_path):
    p = tmp_path / "entries.idx"
    p.write_text(_entries_idx("2XYZ\th\td\tc\ts\ta\tNOT\tSOLUTION NMR\n"))
    assert load_entry_metadata(p)["2xyz"] == ("SOLUTION NMR", None)


def test_load_entry_metadata_zero_resolution_becomes_none(tmp_path):
    p = tmp_path / "entries.idx"
    p.write_text(_entries_idx("3DEF\th\td\tc\ts\ta\t0.00\tSOLUTION NMR\n"))
    assert load_entry_metadata(p)["3def"] == ("SOLUTION NMR", None)


def test_load_entry_metadata_warns_when_rows_are_short(tmp_path, caplog):
    """A row without the trailing tab loses its (empty) EXPERIMENT TYPE.

    Every X-ray entry would then silently lose its metadata, fail the method
    allowlist, and narrow the cohort with no error anywhere. It must be loud.
    """
    p = tmp_path / "entries.idx"
    p.write_text(_entries_idx("1ABC\th\td\tc\ts\ta\t1.80\n"))
    with caplog.at_level("WARNING"):
        meta = load_entry_metadata(p)
    assert meta == {}
    assert "fewer than 8 tab-separated fields" in caplog.text


# --------------------------------------------------------------------------- #
#                         REPRESENTATIVE CHAIN CHOICE
# --------------------------------------------------------------------------- #

_SEG = [(1, 100, "1", "100")]  # 100 residues of SIFTS coverage


def test_prefers_xray_over_nmr_regardless_of_file_order():
    cands = {("1nmr", "A"): _SEG, ("1xry", "A"): _SEG}
    meta = {"1nmr": ("SOLUTION NMR", None), "1xry": ("X-RAY DIFFRACTION", 2.5)}
    pick, why = select_representative_chain(
        "P1", cands, {"1nmr", "1xry"}, meta, resolution_cutoff=3.0
    )
    assert (pick["pdb_id"], why) == ("1xry", "")


def test_prefers_best_resolution_within_a_method():
    cands = {("1low", "A"): _SEG, ("1hig", "A"): _SEG}
    meta = {"1low": ("X-RAY DIFFRACTION", 2.9), "1hig": ("X-RAY DIFFRACTION", 1.2)}
    pick, _ = select_representative_chain(
        "P1", cands, {"1low", "1hig"}, meta, resolution_cutoff=3.0
    )
    assert pick["pdb_id"] == "1hig"


def test_resolution_cutoff_is_applied():
    """--resolution_cutoff was accepted and ignored before 2026-09-18."""
    cands = {("1bad", "A"): _SEG}
    meta = {"1bad": ("X-RAY DIFFRACTION", 3.5)}
    pick, why = select_representative_chain(
        "P1", cands, {"1bad"}, meta, resolution_cutoff=3.0
    )
    assert (pick, why) == (None, "no_candidate_after_filters")


def test_nmr_is_exempt_from_the_resolution_cutoff():
    cands = {("1nmr", "A"): _SEG}
    meta = {"1nmr": ("SOLUTION NMR", None)}
    pick, _ = select_representative_chain(
        "P1", cands, {"1nmr"}, meta, resolution_cutoff=1.0
    )
    assert pick["pdb_id"] == "1nmr"


def test_method_allowlist_is_applied():
    """EXPERIMENTAL_METHODS was defined and never referenced before the rewrite."""
    cands = {("1ecr", "A"): _SEG}
    meta = {"1ecr": ("ELECTRON CRYSTALLOGRAPHY", 3.0)}
    pick, why = select_representative_chain(
        "P1", cands, {"1ecr"}, meta, resolution_cutoff=3.0
    )
    assert (pick, why) == (None, "no_candidate_after_filters")


def test_entry_must_also_be_in_the_entry_level_map():
    """F1: the entry-level file defines the pair set, so it gates the chains."""
    cands = {("1abc", "A"): _SEG}
    meta = {"1abc": ("X-RAY DIFFRACTION", 1.5)}
    pick, why = select_representative_chain("P1", cands, set(), meta, 3.0)
    assert (pick, why) == (None, "no_candidate_after_filters")


def test_coverage_guard_beats_a_higher_resolution_fragment():
    """A 1.0 A 30-residue fragment must not beat a 2.5 A full-length chain."""
    cands = {("1frg", "A"): [(1, 30, "1", "30")], ("1ful", "A"): _SEG}
    meta = {"1frg": ("X-RAY DIFFRACTION", 1.0), "1ful": ("X-RAY DIFFRACTION", 2.5)}
    pick, _ = select_representative_chain(
        "P1", cands, {"1frg", "1ful"}, meta, resolution_cutoff=3.0, coverage_guard=0.90
    )
    assert pick["pdb_id"] == "1ful"
    # ... but a fragment inside the guard is allowed to win on resolution.
    cands[("1frg", "A")] = [(1, 95, "1", "95")]
    pick, _ = select_representative_chain(
        "P1", cands, {"1frg", "1ful"}, meta, resolution_cutoff=3.0, coverage_guard=0.90
    )
    assert pick["pdb_id"] == "1frg"


def test_tie_break_is_deterministic_and_not_file_order():
    """select_best_structure used to return entries[0], i.e. SIFTS file order."""
    meta = {"1bbb": ("X-RAY DIFFRACTION", 2.0), "1aaa": ("X-RAY DIFFRACTION", 2.0)}
    forward = {("1bbb", "B"): _SEG, ("1aaa", "A"): _SEG}
    reverse = dict(reversed(list(forward.items())))
    keys = {"1aaa", "1bbb"}
    a, _ = select_representative_chain("P1", forward, keys, meta, 3.0)
    b, _ = select_representative_chain("P1", reverse, keys, meta, 3.0)
    assert (a["pdb_id"], a["chain"]) == (b["pdb_id"], b["chain"]) == ("1aaa", "A")


# --------------------------------------------------------------------------- #
#                                mmCIF PARSING
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,expected",
    [
        ("27", (27, "")),
        ("27A", (27, "A")),
        ("-3", (-3, "")),
        ("+5", (5, "")),
        (" 12B ", (12, "B")),
        ("", None),
        (".", None),
        ("?", None),
        ("None", None),
        ("abc", None),
        ("-", None),
    ],
)
def test_auth_seq_key(label, expected):
    assert _auth_seq_key(label) == expected


def test_split_cif_line_respects_quoting():
    line = "ATOM 1 C 'CA X' . MSE A 1 1.0 2.0 3.0"
    assert _split_cif_line(line)[3] == "CA X"


_CIF_HEADER = """data_TEST
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""


def _atom(i, *, grp="ATOM", name="CA", alt=".", comp="ALA", chain="A", seq=1, model="1"):
    return (
        f"{grp} {i} C {name} {alt} {comp} A {seq} "
        f"{seq:.3f} 0.000 0.000 1.00 10.00 {seq} {chain} {model}\n"
    )


def _cif(tmp_path, atom_lines, name="x.cif.gz"):
    p = tmp_path / name
    with gzip.open(p, "wt") as fh:
        fh.write(_CIF_HEADER + "".join(atom_lines) + "#\n")
    return p


def _atom_lines(pdb_text):
    return [ln for ln in pdb_text.splitlines() if ln.startswith("ATOM")]


def test_extract_writes_fixed_width_pdb_not_raw_cif_rows(tmp_path):
    """The old implementation copied mmCIF _atom_site rows into a file named .pdb."""
    cif = _cif(tmp_path, [_atom(i, seq=i) for i in range(1, 31)])
    out = tmp_path / "o.pdb"
    n, fallback = extract_chain_ca_pdb(cif, "A", [(1, 30, "1", "30")], out)
    assert (n, fallback) == (30, False)
    lines = _atom_lines(out.read_text())
    assert len(lines) == 30
    first = lines[0]
    assert first[:6] == "ATOM  "
    assert first[12:16] == " CA "  # PDB columns 13-16
    assert first[17:20] == "ALA"  # resName
    assert first[21] == "A"  # chainID, always relabelled
    assert int(first[22:26]) == 1  # resSeq, renumbered 1..N
    assert float(first[30:38]) == pytest.approx(1.0)  # x
    assert out.read_text().rstrip().endswith("END")


def test_extract_restricts_to_the_sifts_author_range(tmp_path):
    cif = _cif(tmp_path, [_atom(i, seq=i) for i in range(1, 31)])
    out = tmp_path / "o.pdb"
    n, fallback = extract_chain_ca_pdb(cif, "A", [(5, 20, "5", "20")], out)
    assert (n, fallback) == (16, False)
    assert len(_atom_lines(out.read_text())) == 16


def test_extract_keeps_only_the_first_model(tmp_path):
    """An NMR ensemble otherwise stacks every model into one 'chain'."""
    cif = _cif(
        tmp_path,
        [_atom(i, seq=i, model="1") for i in range(1, 21)]
        + [_atom(i, seq=i, model="2") for i in range(1, 21)],
    )
    n, _ = extract_chain_ca_pdb(cif, "A", [(1, 20, "1", "20")], tmp_path / "o.pdb")
    assert n == 20


def test_extract_keeps_one_altloc(tmp_path):
    cif = _cif(
        tmp_path,
        [_atom(i, seq=i, alt="A") for i in range(1, 11)]
        + [_atom(i, seq=i, alt="B") for i in range(1, 11)],
    )
    n, _ = extract_chain_ca_pdb(cif, "A", [(1, 10, "1", "10")], tmp_path / "o.pdb")
    assert n == 10


def test_extract_keeps_mse_hetatm_and_drops_other_hetatm(tmp_path):
    """Selenomethionine is a genuine residue of the chain, deposited as HETATM."""
    cif = _cif(
        tmp_path,
        [_atom(i, seq=i) for i in range(1, 11)]
        + [_atom(11, grp="HETATM", comp="MSE", seq=11)]
        + [_atom(12, grp="HETATM", comp="HOH", seq=12)],
    )
    out = tmp_path / "o.pdb"
    n, _ = extract_chain_ca_pdb(cif, "A", [(1, 12, "1", "12")], out)
    assert n == 11
    assert [ln[17:20] for ln in _atom_lines(out.read_text())][-1] == "MSE"


def test_extract_ignores_other_chains(tmp_path):
    cif = _cif(
        tmp_path,
        [_atom(i, seq=i, chain="A") for i in range(1, 11)]
        + [_atom(i, seq=i, chain="B") for i in range(1, 26)],
    )
    n, _ = extract_chain_ca_pdb(cif, "B", [(1, 25, "1", "25")], tmp_path / "o.pdb")
    assert n == 25


def test_extract_falls_back_to_the_whole_chain_when_ranges_match_nothing(tmp_path):
    """Renumbered entries: keep the chain rather than drop the protein."""
    cif = _cif(tmp_path, [_atom(i, seq=i) for i in range(1, 31)])
    out = tmp_path / "o.pdb"
    n, fallback = extract_chain_ca_pdb(cif, "A", [(1, 5, "900", "920")], out)
    assert (n, fallback) == (30, True)
    assert len(_atom_lines(out.read_text())) == 30


def test_extract_returns_zero_for_a_chain_that_is_not_there(tmp_path):
    cif = _cif(tmp_path, [_atom(i, seq=i) for i in range(1, 11)])
    out = tmp_path / "o.pdb"
    assert extract_chain_ca_pdb(cif, "Z", [(1, 10, "1", "10")], out) == (0, False)
    assert not out.exists()


def test_write_ca_pdb_truncates_loudly_and_reports_what_it_wrote(tmp_path, caplog):
    """resSeq is four columns wide; a silent cut would misreport the length."""
    rows = [("ALA", 1.0, 2.0, 3.0, 1.0, 0.0, "C")] * (MAX_PDB_RESSEQ + 51)
    out = tmp_path / "o.pdb"
    with caplog.at_level("WARNING"):
        n = _write_ca_pdb(rows, out)
    assert n == MAX_PDB_RESSEQ
    assert len(_atom_lines(out.read_text())) == MAX_PDB_RESSEQ
    assert "writing only the first" in caplog.text


# --------------------------------------------------------------------------- #
#                              US-ALIGN PARSING
# --------------------------------------------------------------------------- #

# -outfmt 2: PDBchain1 PDBchain2 TM1 TM2 RMSD ID1 ID2 IDali L1 L2 Lali
_OUTFMT2 = (
    "#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali\n"
    "a.pdb\tb.pdb\t0.8000\t0.6000\t2.00\t0.50\t0.40\t0.30\t100\t150\t90\n"
)


class _Proc:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


def test_run_usalign_derives_the_three_normalisations(tmp_path, monkeypatch):
    import data_preparation.pdb_tmscore as mod

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Proc(_OUTFMT2))
    res = run_usalign(tmp_path / "a.pdb", tmp_path / "b.pdb", "USalign")
    assert res["tm1"] == pytest.approx(0.80)
    assert res["tm2"] == pytest.approx(0.60)
    # min -> normalised by the LONGER chain; max -> by the SHORTER one.
    assert res["tmscore_exp"] == pytest.approx(0.60)
    assert res["tmscore_exp_short"] == pytest.approx(0.80)
    # length-weighted mean, (TM1*L1 + TM2*L2) / (L1 + L2)
    assert res["tmscore_exp_avg"] == pytest.approx((0.8 * 100 + 0.6 * 150) / 250)
    assert (res["len1"], res["len2"], res["len_ali"]) == (100, 150, 90)
    assert res["rmsd"] == pytest.approx(2.0)
    assert res["seqid_ali"] == pytest.approx(0.30)


def test_run_usalign_returns_none_on_failure(tmp_path, monkeypatch):
    import data_preparation.pdb_tmscore as mod

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Proc("", returncode=1))
    assert run_usalign(tmp_path / "a.pdb", tmp_path / "b.pdb", "USalign") is None


def test_run_usalign_returns_none_when_the_binary_is_missing(tmp_path, monkeypatch):
    import data_preparation.pdb_tmscore as mod

    def _boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(mod.subprocess, "run", _boom)
    assert run_usalign(tmp_path / "a.pdb", tmp_path / "b.pdb", "nope") is None
