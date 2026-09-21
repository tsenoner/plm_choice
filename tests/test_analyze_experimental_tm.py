"""
Fixture tests for the B6 / R2.2 analysis (src/evaluation/analyze_experimental_tm.py).

This is the only code path in the repo that produces the experimental-TM numbers
quoted in the manuscript, so the two things that could move one of them silently
get a test: which side of the signed difference is "predicted", and whether the
CATH stratification would notice being handed the wrong file.
"""

import gzip

import numpy as np
import polars as pl
import pytest

from evaluation.analyze_experimental_tm import cluster_bootstrap_r, corr_block, load_cath

# cath-b-newest-all: '<domain> <version> <C.A.T.H> <boundaries>'. Field 3 is the
# full four-level homologous-superfamily code.
_CATH_B = (
    "1abcA01 v4_3_0 1.10.8.10 5-114:A\n"
    "1abcB01 v4_3_0 3.40.50.300 2-99:B\n"
    "2xyzA01 v4_3_0 1.10.8.10 1-80:A\n"
)

# cath-domain-list.txt: field 3 is the ARCHITECTURE digit, not a C.A.T.H code.
# It splits and indexes exactly the same way, which is the whole problem.
_CATH_DOMAIN_LIST = (
    "1abcA01     1    10     8    10     1     1     1     1     1   114  2.000\n"
    "1abcB01     3    40    50   300     1     1     1     1     1    99  1.800\n"
)


def _gz(tmp_path, text, name="cath.gz"):
    p = tmp_path / name
    with gzip.open(p, "wt") as fh:
        fh.write(text)
    return p


def test_load_cath_reads_superfamily_codes_for_wanted_chains(tmp_path):
    p = _gz(tmp_path, _CATH_B)
    out = load_cath(p, {("1abc", "A"), ("2xyz", "A")})
    assert out == {("1abc", "A"): {"1.10.8.10"}, ("2xyz", "A"): {"1.10.8.10"}}
    # Chains nobody asked about are not carried.
    assert ("1abc", "B") not in out


def test_load_cath_refuses_the_domain_list_file(tmp_path):
    """cath-domain-list.txt parses happily and stratifies meaninglessly.

    Field 3 is "1" (the architecture digit), so every pair would still get a
    same/different label and the CATH block of stats.json would be nonsense with
    no error anywhere. The shape check is the only thing standing in the way.
    """
    p = _gz(tmp_path, _CATH_DOMAIN_LIST)
    with pytest.raises(SystemExit, match="cath-b-newest-all"):
        load_cath(p, {("1abc", "A")})


def test_corr_block_signs_the_difference_as_predicted_minus_experimental():
    """corr_block(x=experimental, y=predicted); d = y - x.

    Swapping the two arguments flips the sign of every reported gap and of
    frac_pred_higher without changing r, so nothing else in the output would
    reveal the mistake.
    """
    exp = np.linspace(0.2, 0.8, 50)
    pred = exp + 0.10  # the predicted structure scores HIGHER throughout
    blk = corr_block(exp, pred)
    assert blk["mean_signed_diff"] == pytest.approx(0.10)
    assert blk["median_signed_diff"] == pytest.approx(0.10)
    assert blk["frac_pred_higher"] == pytest.approx(1.0)
    assert blk["mean_exp"] == pytest.approx(exp.mean())
    assert blk["mean_pred"] == pytest.approx(pred.mean())
    assert blk["pearson_r"] == pytest.approx(1.0)


def test_corr_block_reports_only_n_below_the_floor():
    blk = corr_block(np.arange(5.0), np.arange(5.0))
    assert blk == {"n": 5}


def _pair_frame(n_prot=60, seed=0):
    """All-vs-all pairs over n_prot proteins with a correlated pred column."""
    rng = np.random.default_rng(seed)
    prots = [f"P{i:03d}" for i in range(n_prot)]
    q, t = [], []
    for i in range(n_prot):
        for j in range(i + 1, n_prot):
            q.append(prots[i])
            t.append(prots[j])
    x = rng.uniform(0, 1, len(q))
    y = 0.8 * x + 0.2 * rng.uniform(0, 1, len(q))
    return pl.DataFrame({"query": q, "target": t, "exp": x, "pred": y})


def test_cluster_bootstrap_ci_brackets_the_point_estimate():
    """Resampling PROTEINS, not pairs: pairs sharing a protein are dependent."""
    df = _pair_frame()
    lo, hi = cluster_bootstrap_r(df, "exp", "pred", n_boot=200, seed=0)
    r = corr_block(df["exp"].to_numpy(), df["pred"].to_numpy())["pearson_r"]
    assert lo < r < hi
    assert 0.0 < lo < hi < 1.0


def test_cluster_bootstrap_is_reproducible_for_a_given_seed():
    df = _pair_frame()
    a = cluster_bootstrap_r(df, "exp", "pred", n_boot=200, seed=7)
    b = cluster_bootstrap_r(df, "exp", "pred", n_boot=200, seed=7)
    assert a == b
    assert not np.isnan(a).any()


def test_cluster_bootstrap_returns_nan_on_a_frame_too_small_to_resample():
    """Documented, not accidental: fewer than 50 usable resamples -> (nan, nan).

    The guards inside are absolute counts (>= 50 pair-slots per draw, >= 50
    surviving draws), so a handful of pairs yields no interval rather than a
    falsely tight one. The real cohort is ~10k pairs and never hits this.
    """
    df = _pair_frame(n_prot=5)
    lo, hi = cluster_bootstrap_r(df, "exp", "pred", n_boot=200, seed=0)
    assert np.isnan(lo) and np.isnan(hi)
