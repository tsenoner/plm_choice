import numpy as np
import pytest
from scipy.stats import kendalltau

from evaluation.stats import kendall_tau_b


def test_matches_scipy_variant_b():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = x + rng.normal(size=50) * 0.5
    assert kendall_tau_b(x, y) == pytest.approx(kendalltau(x, y, variant="b").statistic)


def test_perfect_monotone_is_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert kendall_tau_b(x, y) == pytest.approx(1.0)


def test_constant_margin_returns_nan():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(kendall_tau_b(x, y))


from evaluation.stats import _induced_pair_values


def test_induced_pairs_full_index_is_all_upper_triangle():
    dist = np.array([[0.0, 1.0, 2.0],
                     [1.0, 0.0, 3.0],
                     [2.0, 3.0, 0.0]])
    ec = np.array([[0.0, 4.0, 4.0],
                   [4.0, 0.0, 1.0],
                   [4.0, 1.0, 0.0]])
    d, e = _induced_pair_values(dist, ec, np.array([0, 1, 2]))
    assert sorted(d.tolist()) == [1.0, 2.0, 3.0]
    assert sorted(e.tolist()) == [1.0, 4.0, 4.0]


def test_induced_pairs_drops_self_pairs_and_keeps_multiplicity():
    dist = np.array([[0.0, 1.0], [1.0, 0.0]])
    ec = np.array([[0.0, 4.0], [4.0, 0.0]])
    # resample picks index 0 twice and index 1 once: positions (0,1),(0,2),(1,2)
    # idx = [0, 0, 1]: pair(pos0,pos1)=idx(0,0) self -> dropped;
    #   pair(pos0,pos2)=idx(0,1) kept; pair(pos1,pos2)=idx(0,1) kept -> multiplicity 2
    d, e = _induced_pair_values(dist, ec, np.array([0, 0, 1]))
    assert d.tolist() == [1.0, 1.0]
    assert e.tolist() == [4.0, 4.0]


from evaluation.stats import correlation_vertex_bca_ci


def _monotone_matrices(n=40, seed=0):
    # Build a cohort where embedding distance is monotone in EC distance -> tau_b ~ 1.
    rng = np.random.default_rng(seed)
    ec_level = rng.integers(0, 5, size=n).astype(float)  # per-protein "function score"
    dist = np.abs(ec_level[:, None] - ec_level[None, :]).astype(float)
    ec = dist.copy()  # EC distance == embedding distance by construction
    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(ec, 0.0)
    return dist, ec


def _noisy_matrices(n=30, seed=0):
    # Moderate (not perfect) correlation, so BOTH CIs are non-degenerate and the
    # vertex-vs-pair width comparison is meaningful (a perfectly monotone cohort
    # collapses both CIs to ~[1, 1] and the discriminator becomes vacuous).
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    ec = np.abs(score[:, None] - score[None, :])
    dist = ec + rng.normal(scale=1.0, size=(n, n))
    dist = (dist + dist.T) / 2.0  # symmetrise the noise
    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(ec, 0.0)
    return dist, ec


def test_point_estimate_high_on_monotone_cohort():
    dist, ec = _monotone_matrices()
    lo, hi, point, degenerate, diverged = correlation_vertex_bca_ci(
        dist, ec, statistic="tau_b", n_boot=300, alpha=0.1, seed=42
    )
    assert not degenerate
    assert point > 0.9
    assert lo <= point <= hi
    assert -1.0 <= lo and hi <= 1.0


def test_degenerate_when_too_few_proteins():
    dist = np.zeros((3, 3))
    ec = np.zeros((3, 3))
    lo, hi, point, degenerate, _ = correlation_vertex_bca_ci(
        dist, ec, statistic="tau_b", n_boot=50, seed=1
    )
    assert degenerate and np.isnan(lo) and np.isnan(hi)


def test_degenerate_on_constant_ec_margin():
    dist, ec = _monotone_matrices()
    ec[:] = 1.0
    np.fill_diagonal(ec, 0.0)
    _, _, _, degenerate, _ = correlation_vertex_bca_ci(
        dist, ec, statistic="tau_b", n_boot=50, seed=1
    )
    assert degenerate


def test_vertex_interval_wider_than_pair_interval():
    # The discriminator: resampling proteins (vertices) must give a WIDER CI than
    # resampling pairs i.i.d., because pairs sharing a protein are correlated. Uses a
    # NOISY cohort so both CIs are non-degenerate (a perfectly monotone cohort would
    # collapse both to ~[1, 1] and the comparison would be vacuous).
    from evaluation.stats import _pair_bootstrap_ci_width
    dist, ec = _noisy_matrices(n=30, seed=3)
    vlo, vhi, point, degenerate, _ = correlation_vertex_bca_ci(
        dist, ec, statistic="spearman", n_boot=800, alpha=0.1, seed=7)
    assert not degenerate
    vertex_width = vhi - vlo
    pair_width = _pair_bootstrap_ci_width(
        dist, ec, statistic="spearman", n_boot=800, alpha=0.1, seed=7)
    assert vertex_width > pair_width


def test_negative_correlation_interval_on_correct_side():
    # All fixtures above are positive; this pins the SIGN/asymmetry of the BCa shift.
    # Embedding distance ANTI-correlates with EC distance -> tau_b < 0, CI fully < 0.
    dist, ec = _monotone_matrices(n=24, seed=2)
    dist = dist.max() - dist          # invert -> strong negative association
    np.fill_diagonal(dist, 0.0)
    lo, hi, point, degenerate, _ = correlation_vertex_bca_ci(
        dist, ec, statistic="tau_b", n_boot=400, alpha=0.1, seed=11)
    assert not degenerate
    assert point < -0.5
    assert hi < 0.0                    # the whole interval is on the negative side
    assert lo <= point <= hi
