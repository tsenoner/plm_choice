import pandas as pd
import pytest

from evaluation.label_adapters import parse_ec, parse_ec_from_protein_names


def _df():
    return pd.DataFrame(
        {
            "Entry": ["P1", "P2", "P3", "P4", "P5"],
            "Protein names": [
                "Alcohol dehydrogenase (EC 1.1.1.1) (EC 1.1.1.71)",  # multifunctional
                "Some kinase (EC 2.7.11.1)",
                "Partial enzyme (EC 3.4.21.-)",   # wildcard -> excluded by default
                "Hypothetical protein",            # no EC -> omitted
                "Preliminary (EC 1.1.1.n1)",       # preliminary -> included
            ],
        }
    )


def test_parse_ec_returns_frozenset_per_protein():
    out = parse_ec(_df())
    assert list(out.columns) == ["protein_id", "ec_set"]
    p1 = out[out["protein_id"] == "P1"].iloc[0]["ec_set"]
    assert p1 == frozenset({"1.1.1.1", "1.1.1.71"})
    assert all(isinstance(s, frozenset) for s in out["ec_set"])


def test_wildcard_excluded_by_default():
    out = parse_ec(_df())
    # P3 had only a wildcard EC -> no valid EC -> protein omitted entirely.
    assert "P3" not in set(out["protein_id"])


def test_wildcard_included_when_policy_include():
    out = parse_ec(_df(), wildcard_policy="include")
    p3 = out[out["protein_id"] == "P3"].iloc[0]["ec_set"]
    assert p3 == frozenset({"3.4.21.-"})


def test_preliminary_n1_included():
    out = parse_ec(_df())
    p5 = out[out["protein_id"] == "P5"].iloc[0]["ec_set"]
    assert p5 == frozenset({"1.1.1.n1"})


def test_protein_with_no_ec_omitted():
    out = parse_ec(_df())
    assert "P4" not in set(out["protein_id"])


def test_structured_ec_col_preferred_over_name_regex():
    df = pd.DataFrame(
        {
            "Entry": ["P1"],
            "Protein names": ["wrong (EC 9.9.9.9)"],
            "EC number": ["1.1.1.1; 2.7.11.1"],  # structured, ';'-separated
        }
    )
    out = parse_ec(df, ec_col="EC number")
    assert out.iloc[0]["ec_set"] == frozenset({"1.1.1.1", "2.7.11.1"})


def test_class_only_ec_excluded_by_default_via_structured_col():
    # The spec names class-only "1.-.-.-" explicitly. It cannot reach the name-regex
    # (which requires 4 numeric/n fields), so it is tested via the structured ec_col.
    df = pd.DataFrame({"Entry": ["P1", "P2"], "EC number": ["1.-.-.-", "1.1.1.1"]})
    out_excl = parse_ec(df, ec_col="EC number")  # default exclude
    assert "P1" not in set(out_excl["protein_id"])        # class-only dropped
    assert out_excl[out_excl["protein_id"] == "P2"].iloc[0]["ec_set"] == frozenset({"1.1.1.1"})
    out_incl = parse_ec(df, ec_col="EC number", wildcard_policy="include")
    assert out_incl[out_incl["protein_id"] == "P1"].iloc[0]["ec_set"] == frozenset({"1.-.-.-"})


def test_scalar_parser_untouched():
    # The pre-existing scalar parser still returns [protein_id, ec_number] one-per-row.
    out = parse_ec_from_protein_names(_df())
    assert list(out.columns) == ["protein_id", "ec_number"]
    assert out[out["protein_id"] == "P1"].iloc[0]["ec_number"] == "1.1.1.1"
