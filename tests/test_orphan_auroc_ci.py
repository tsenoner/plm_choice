"""Unit 3 — vertex-bootstrap AUROC BCa CI bound over stats.vertex_bca_ci.

The dyadic-dependence CI: each Bromberg pair has BOTH endpoints in the orphan set, so
pairs sharing an orphan are correlated. The CI resamples ORPHANS (vertices) and
recomputes AUROC over the induced sparse pair set, reusing the shipped vertex_bca_ci core.
"""
import time

import numpy as np
import pandas as pd
import pytest

from evaluation.orphan_auroc_ci import (
    orphan_auroc_vertex_bca_ci,
    weighted_concordance_auc,
)


def _pp(rows):
    return pd.DataFrame(rows, columns=["p1", "p2", "cos", "snn", "tm", "sibling"])


# ── the U-form kernel vs sklearn roc_auc_score (the equivalence the binding relies on) ──
def test_weighted_auc_equals_sklearn_unit_weights():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n = 60
    cos = rng.normal(size=n)
    sib = rng.integers(0, 2, size=n).astype(bool)
    if sib.all() or not sib.any():
        sib[0], sib[1] = True, False
    w = np.ones(n)
    u = weighted_concordance_auc(cos, sib, w)
    assert u == pytest.approx(roc_auc_score(sib, cos), abs=1e-9)


def test_weighted_auc_handles_ties_like_sklearn():
    from sklearn.metrics import roc_auc_score

    cos = np.array([0.5, 0.5, 0.5, 0.2, 0.8, 0.8])
    sib = np.array([True, False, True, False, True, False])
    w = np.ones(6)
    assert weighted_concordance_auc(cos, sib, w) == pytest.approx(
        roc_auc_score(sib, cos), abs=1e-9
    )


def test_weighted_auc_weights_replicate_duplication():
    # A weight of 2 on a row must equal duplicating that row.
    cos = np.array([0.9, 0.1, 0.5])
    sib = np.array([True, False, False])
    w = np.array([2.0, 1.0, 1.0])
    u_weighted = weighted_concordance_auc(cos, sib, w)
    cos_dup = np.array([0.9, 0.9, 0.1, 0.5])
    sib_dup = np.array([True, True, False, False])
    u_dup = weighted_concordance_auc(cos_dup, sib_dup, np.ones(4))
    assert u_weighted == pytest.approx(u_dup, abs=1e-9)


def test_weighted_auc_nan_when_one_class_empty():
    cos = np.array([0.1, 0.2, 0.3])
    sib = np.array([True, True, True])
    assert np.isnan(weighted_concordance_auc(cos, sib, np.ones(3)))


# ── the vertex-BCa binding ──────────────────────────────────────────────────────────
def _separable_fixture(n_orphans=20, seed=0):
    """Many sibling pairs (high cos) + many non-sibling pairs (low cos) over n orphans."""
    rng = np.random.default_rng(seed)
    ids = [f"O{i}" for i in range(n_orphans)]
    rows = []
    # siblings: consecutive orphans, high cos
    for i in range(n_orphans - 1):
        rows.append((ids[i], ids[i + 1], 0.9 + rng.normal(scale=0.02), 0.5, 0.5, True))
    # non-siblings: distant orphans, low cos
    for i in range(0, n_orphans - 2, 2):
        rows.append((ids[i], ids[i + 2], 0.1 + rng.normal(scale=0.02), 0.5, 0.5, False))
        if i + 5 < n_orphans:
            rows.append((ids[i], ids[i + 5], 0.05 + rng.normal(scale=0.02), 0.5, 0.5, False))
    return _pp(rows)


def test_perfect_separation_auroc_one_ci_sane():
    # sibling cos all strictly above non-sibling cos -> point AUROC 1.0
    rng = np.random.default_rng(1)
    ids = [f"O{i}" for i in range(20)]
    rows = []
    for i in range(19):
        rows.append((ids[i], ids[i + 1], 0.95, 0.5, 0.5, True))
    for i in range(0, 18, 2):
        rows.append((ids[i], ids[i + 2], 0.05, 0.5, 0.5, False))
    out = orphan_auroc_vertex_bca_ci(_pp(rows), n_boot=300, alpha=0.1, seed=42)
    assert out["point"] == pytest.approx(1.0)
    assert not out["degenerate"]
    assert out["ci_lo"] <= out["point"] <= out["ci_hi"] + 1e-9
    assert 0.0 <= out["ci_lo"] and out["ci_hi"] <= 1.0


def test_random_labels_auroc_near_half():
    rng = np.random.default_rng(3)
    n = 30
    ids = [f"O{i}" for i in range(n)]
    rows = []
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.25:
                rows.append((ids[a], ids[b], rng.normal(), 0.5, 0.5, bool(rng.integers(0, 2))))
    out = orphan_auroc_vertex_bca_ci(_pp(rows), n_boot=300, alpha=0.1, seed=7)
    assert 0.3 < out["point"] < 0.7


def test_validate_point_passes():
    # point kernel == boot kernel on the identity resample (the AUROC vs U-form tie guard).
    out = orphan_auroc_vertex_bca_ci(
        _separable_fixture(20, seed=2), n_boot=200, alpha=0.1, seed=5, validate_point=True
    )
    assert not out["degenerate"]


def test_reproducible_under_fixed_seed():
    df = _separable_fixture(20, seed=4)
    a = orphan_auroc_vertex_bca_ci(df, n_boot=300, alpha=0.1, seed=99)
    b = orphan_auroc_vertex_bca_ci(df, n_boot=300, alpha=0.1, seed=99)
    assert a["ci_lo"] == b["ci_lo"] and a["ci_hi"] == b["ci_hi"]


def test_clustered_fixture_vertex_ci_wider_than_naive_pair_ci():
    # Many pairs share ONE orphan (a hub) -> the vertex CI must be WIDER than a naive
    # i.i.d.-pair bca_bootstrap CI (the dyadic-dependence anticonservativeness point).
    rng = np.random.default_rng(11)
    n = 30
    ids = [f"O{i}" for i in range(n)]
    rows = []
    # a hub orphan O0 in MANY pairs, mixed labels with noisy cos
    for i in range(1, n):
        rows.append((ids[0], ids[i], rng.normal(scale=1.0), 0.5, 0.5, bool(i % 3 == 0)))
    # plus some non-hub pairs
    for i in range(1, n - 1):
        rows.append((ids[i], ids[i + 1], rng.normal(scale=1.0), 0.5, 0.5, bool(i % 4 == 0)))
    df = _pp(rows)
    out = orphan_auroc_vertex_bca_ci(df, n_boot=600, alpha=0.1, seed=21)
    vertex_width = out["ci_hi"] - out["ci_lo"]

    # naive i.i.d.-PAIR percentile bootstrap over the SAME pairs (rows = sampling unit),
    # mirroring stats._pair_bootstrap_ci_width's construction.
    from sklearn.metrics import roc_auc_score

    cos = df["cos"].to_numpy()
    sib = df["sibling"].to_numpy().astype(bool)
    m = cos.size
    prng = np.random.default_rng(21)
    boot = []
    for _ in range(600):
        sel = prng.integers(0, m, size=m)
        s = sib[sel]
        if s.all() or not s.any():
            continue
        boot.append(roc_auc_score(s, cos[sel]))
    boot = np.asarray(boot)
    pair_width = float(np.quantile(boot, 0.95) - np.quantile(boot, 0.05))
    assert vertex_width > pair_width


def test_zero_sibling_draws_counted_and_handled():
    # A cohort with very few sibling pairs: some vertex resamples induce zero siblings ->
    # those draws are NaN-marked, counted in n_boot_undefined, and not crashing.
    rng = np.random.default_rng(31)
    n = 16
    ids = [f"O{i}" for i in range(n)]
    rows = []
    # exactly TWO sibling pairs, many non-siblings
    rows.append((ids[0], ids[1], 0.9, 0.5, 0.5, True))
    rows.append((ids[2], ids[3], 0.85, 0.5, 0.5, True))
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.4 and not (
                (a, b) in {(0, 1), (2, 3)}
            ):
                rows.append((ids[a], ids[b], rng.normal(scale=0.3), 0.5, 0.5, False))
    out = orphan_auroc_vertex_bca_ci(_pp(rows), n_boot=400, alpha=0.1, seed=13)
    assert out["n_boot_undefined"] >= 1  # at least one draw lost a class
    # whether degenerate or not, no crash and the count is surfaced
    assert "n_boot_undefined" in out


def test_incremental_jackknife_matches_from_scratch():
    # The load-bearing correctness claim: the incremental leave-one-orphan-out AUROC
    # (inclusion-exclusion over precomputed contributions) must equal a brute-force
    # recompute that drops every pair incident to k and reruns the U-form AUROC.
    from evaluation.orphan_auroc_ci import (
        _build_jackknife,  # test seam: returns (n, jack_fn, ids, u, v)
    )

    rng = np.random.default_rng(5)
    n = 22
    ids = [f"O{i}" for i in range(n)]
    rows = []
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.35:
                rows.append((ids[a], ids[b], rng.normal(), 0.5, 0.5, bool(rng.integers(0, 2))))
    # a hub so some orphan has high degree
    for i in range(1, n):
        rows.append((ids[0], ids[i], rng.normal(), 0.5, 0.5, bool(i % 2)))
    df = _pp(rows)

    nn, jack_fn, id_list, uu, vv = _build_jackknife(df)
    cos = df["cos"].to_numpy(dtype=float)
    sib = df["sibling"].to_numpy().astype(bool)

    def _brute(k):
        keep = (uu != k) & (vv != k)
        s = sib[keep]
        if s.all() or not s.any():
            return float("nan")
        return weighted_concordance_auc(cos[keep], s, np.ones(int(keep.sum())))

    for k in range(nn):
        inc = jack_fn(k)
        bru = _brute(k)
        if np.isnan(bru):
            assert np.isnan(inc)
        else:
            assert inc == pytest.approx(bru, abs=1e-9), f"orphan {k}: {inc} != {bru}"


@pytest.mark.slow
def test_incremental_jackknife_is_fast_at_scale():
    # ~2000 orphans, sparse pairs. The jackknife loop must be incremental (not O(n x pairs)
    # rebuilt-from-scratch). A from-scratch loop at this size would take many seconds.
    rng = np.random.default_rng(7)
    n = 2200
    ids = [f"O{i}" for i in range(n)]
    rows = []
    for i in range(n - 1):
        is_sib = bool(rng.random() < 0.3)
        rows.append((ids[i], ids[i + 1], rng.normal(), 0.5, 0.5, is_sib))
    # a few hubs to make some orphans high-degree
    for h in (0, 1, 2):
        for j in range(50):
            t = int(rng.integers(0, n))
            if t != h:
                rows.append((ids[h], ids[t], rng.normal(), 0.5, 0.5, bool(rng.random() < 0.3)))
    df = _pp(rows)
    t0 = time.time()
    out = orphan_auroc_vertex_bca_ci(df, n_boot=200, alpha=0.1, seed=1)
    elapsed = time.time() - t0
    assert not out["degenerate"]
    assert elapsed < 20.0, f"vertex-AUROC CI too slow ({elapsed:.1f}s) — jackknife not incremental?"
