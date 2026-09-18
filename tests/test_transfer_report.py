"""The 1-NN transfer readout must be arithmetic, not vibes.

Every number this report publishes is checked here against a definition computed a second,
slower, obvious way on a synthetic cohort small enough to verify by hand: the nearest
neighbour (including its tie-break), the identity exclusion in both directions, the exact
chance and oracle baselines, the paired bootstrap, EC multi-label level matching, GO
propagated-set F1, and the vectorised Wang BMA against the scalar reference.

The end-to-end tests build a cohort whose embeddings ARE its labels, so the transfer score
has a known answer (1.0) and a pLM that cannot beat chance is visibly at chance — the same
shape as the ``random_1024`` validity check on the real data.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from data_preparation.go_semantic_similarity import parse_obo
from evaluation.ec_hierarchy import ec_distance_set
from evaluation.go_similarity_matrix import (
    MF_ROOT,
    PROTEIN_BINDING,
    propagate_mf,
    random_ordered_pairs,
    scalar_bma_values,
)
from evaluation.transfer_report import (
    HBI_ARMS,
    SUBSET_ALL,
    SUBSET_HBI,
    SUBSET_NO_HIT,
    ECScorer,
    EligibilityVariant,
    GOScorer,
    arm_name,
    bootstrap_means,
    bootstrap_weights,
    build_parser,
    build_variants,
    hbi_neighbours,
    load_arm_matrix,
    main,
    nearest_neighbour_block,
    run_transfer_report,
    variant_baselines,
)

# --------------------------------------------------------------------------- #
#                                  fixtures
# --------------------------------------------------------------------------- #

#: 12 proteins in 4 EC groups of 3. Groups 1.1.1.1/1.1.1.2 share three fields, 1.1.2.1
#: shares two, 2.7.7.7 shares nothing — so every EC depth is exercised, and P11 is
#: deliberately bifunctional (its second EC reaches into another group).
EC_LABELS: dict[str, str] = {
    "P00": "1.1.1.1", "P01": "1.1.1.1", "P02": "1.1.1.1",
    "P03": "1.1.1.2", "P04": "1.1.1.2", "P05": "1.1.1.2",
    "P06": "1.1.2.1", "P07": "1.1.2.1", "P08": "1.1.2.1",
    "P09": "2.7.7.7", "P10": "2.7.7.7", "P11": "2.7.7.7;1.1.1.1",
}
EC_IDS: list[str] = sorted(EC_LABELS)
EC_SETS: list[frozenset[str]] = [frozenset(EC_LABELS[p].split(";")) for p in EC_IDS]
GROUP = {p: EC_LABELS[p].split(";")[0] for p in EC_IDS}


def write_freeze(path: Path, ids: list[str]) -> Path:
    path.write_text(json.dumps({"ids": ids, "n": len(ids), "set_name": "test"}))
    return path


def write_ec_labels(path: Path, labels: dict[str, str] = EC_LABELS) -> Path:
    rows = ["Entry\tEC number"] + [f"{pid}\t{ec}" for pid, ec in sorted(labels.items())]
    path.write_text("\n".join(rows) + "\n")
    return path


def write_h5(path: Path, vectors: dict[str, np.ndarray], *, two_d: bool = False) -> Path:
    with h5py.File(path, "w") as handle:
        for pid, vec in vectors.items():
            handle.create_dataset(pid, data=vec[None, :] if two_d else vec)
    return path


def group_embeddings(ids: list[str], *, jitter: float = 0.01, seed: int = 0) -> dict:
    """One-hot on the EC group + tiny jitter: the nearest neighbour is always same-group."""
    groups = sorted({GROUP[p] for p in ids})
    rng = np.random.default_rng(seed)
    out = {}
    for pid in ids:
        vec = np.zeros(len(groups))
        vec[groups.index(GROUP[pid])] = 1.0
        out[pid] = vec + jitter * rng.standard_normal(len(groups))
    return out


def noise_embeddings(ids: list[str], dim: int = 8, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    return {pid: rng.standard_normal(dim) for pid in ids}


def write_m8(path: Path, hits: list[tuple[str, str, float, float]]) -> Path:
    """query, target, fident, evalue (alnlen/qcov/tcov are padding the reader ignores)."""
    path.write_text(
        "".join(f"{q}\t{t}\t{f}\t{e}\t100\t0.9\t0.9\n" for q, t, f, e in hits)
    )
    return path


# --------------------------------------------------------------------------- #
#                            nearest neighbour + ties
# --------------------------------------------------------------------------- #


def test_nearest_neighbour_picks_the_minimum_and_ignores_ineligible():
    dist = np.array([[0.0, 5.0, 1.0, 2.0], [5.0, 0.0, 9.0, 3.0]])
    eligible = np.array([[False, True, False, True], [True, False, True, True]])
    nn, best, tied, dropped = nearest_neighbour_block(dist, eligible)
    assert nn.tolist() == [3, 3]  # column 2 is nearer for row 0 but is not eligible
    assert best.tolist() == [2.0, 3.0]
    assert not tied.any() and not dropped.any()


def test_nearest_neighbour_breaks_ties_by_lowest_index_and_counts_them():
    dist = np.array([[9.0, 1.0, 1.0, 1.0]])
    eligible = np.array([[False, True, True, True]])
    nn, _, tied, _ = nearest_neighbour_block(dist, eligible)
    assert nn.tolist() == [1]
    assert tied.tolist() == [True]


def test_nearest_neighbour_drops_a_query_with_no_eligible_neighbour():
    dist = np.array([[0.0, 1.0], [1.0, 0.0]])
    eligible = np.array([[False, False], [True, False]])
    nn, best, tied, dropped = nearest_neighbour_block(dist, eligible)
    assert dropped.tolist() == [True, False]
    assert not np.isfinite(best[0]) and best[1] == 1.0
    assert tied.tolist() == [False, False]  # an all-ineligible row is not "tied"


def test_self_is_never_its_own_neighbour():
    variant = EligibilityVariant("all", "", None, 4)
    block = variant.block(0, 4)
    assert np.array_equal(np.diag(block), np.zeros(4, dtype=bool))


# --------------------------------------------------------------------------- #
#                            identity-control variants
# --------------------------------------------------------------------------- #


def test_identity_variants_exclude_in_both_directions(tmp_path):
    ids = ["A", "B", "C", "D"]
    # Only A->B is reported by MMseqs2; B->A must be excluded too. C->D is a weak hit:
    # below the identity threshold but well inside the E-value threshold.
    m8 = write_m8(
        tmp_path / "hits.m8",
        [("A", "B", 0.80, 1e-40), ("C", "D", 0.12, 1e-20), ("A", "A", 1.0, 0.0)],
    )
    variants, hits, stats = build_variants(m8, ids)
    by_name = {v.name: v for v in variants}
    assert hits.has_hit.tolist() == [True, True, True, True]  # every id is in a hit
    assert set(by_name) == {"all", "fident_lt_0.3", "no_hit_evalue_0.001"}
    assert stats["n_hits_total"] == 3 and stats["n_hits_in_cohort"] == 2  # self hit dropped

    ident = by_name["fident_lt_0.3"].block(0, 4)
    assert not ident[0, 1] and not ident[1, 0]  # both directions of the A/B hit
    assert ident[2, 3] and ident[3, 2]  # 0.12 identity is below the threshold

    any_hit = by_name["no_hit_evalue_0.001"].block(0, 4)
    assert not any_hit[0, 1] and not any_hit[1, 0]
    assert not any_hit[2, 3] and not any_hit[3, 2]  # E=1e-20 disqualifies it


def test_identity_hits_outside_the_cohort_are_ignored(tmp_path):
    m8 = write_m8(tmp_path / "hits.m8", [("A", "ZZZ", 0.99, 0.0), ("A", "B", 0.99, 0.0)])
    variants, hits, stats = build_variants(m8, ["A", "B"])
    assert stats["n_hits_in_cohort"] == 1
    assert hits.has_hit.tolist() == [True, True]  # the out-of-cohort hit is not a hit
    assert not variants[1].block(0, 2)[0, 1]


def test_without_an_m8_only_the_unrestricted_variant_exists():
    variants, hits, stats = build_variants(None, ["A", "B", "C"])
    assert [v.name for v in variants] == ["all"]
    assert stats["m8"] is None
    assert hits is None  # no hit table means no homology baseline


# --------------------------------------------------------------------------- #
#                          EC multi-label level matching
# --------------------------------------------------------------------------- #


def _brute_force_ec(ec_sets, depth_key):
    threshold = {"exact": 0, "share3": 1, "share2": 2, "class": 3}[depth_key]
    n = len(ec_sets)
    return np.array(
        [
            [float(ec_distance_set(ec_sets[i], ec_sets[j], agg="min") <= threshold) for j in range(n)]
            for i in range(n)
        ]
    )


@pytest.mark.parametrize("score", ["exact", "share3", "share2", "class"])
def test_ec_scorer_block_matches_the_set_valued_hierarchy(score):
    block = ECScorer(EC_SETS).block(0, len(EC_SETS))[score]
    assert np.array_equal(block, _brute_force_ec(EC_SETS, score))


def test_ec_scorer_is_multi_label_any_pair_matches():
    """P11 carries 2.7.7.7 AND 1.1.1.1, so it matches group 1.1.1.1 exactly."""
    scorer = ECScorer(EC_SETS)
    block = scorer.block(0, len(EC_SETS))
    i, j = EC_IDS.index("P11"), EC_IDS.index("P00")
    assert block["exact"][i, j] == 1.0
    assert block["exact"][EC_IDS.index("P09"), j] == 0.0  # 2.7.7.7 alone does not
    assert block["class"][EC_IDS.index("P09"), j] == 0.0  # different EC class


def test_ec_scorer_pairs_agrees_with_its_block():
    scorer = ECScorer(EC_SETS)
    block = scorer.block(0, len(EC_SETS))
    rows, cols = random_ordered_pairs(len(EC_SETS), 50, seed=1)
    pairs = scorer.pairs(rows, cols)
    for name in scorer.names:
        assert np.array_equal(pairs[name], block[name][rows, cols])


# --------------------------------------------------------------------------- #
#                        chance / oracle arithmetic
# --------------------------------------------------------------------------- #


def test_chance_and_oracle_are_exact_row_statistics(tmp_path):
    m8 = write_m8(tmp_path / "hits.m8", [("P00", "P01", 0.9, 1e-50)])
    scorer = ECScorer(EC_SETS)
    variants, _hits, _ = build_variants(m8, EC_IDS)
    baselines = variant_baselines(scorer, variants, len(EC_IDS), block_size=5)
    full = scorer.block(0, len(EC_IDS))
    for variant in variants:
        eligible = variant.block(0, len(EC_IDS))
        for score in scorer.names:
            matrix = full[score]
            expected_chance = np.array(
                [matrix[i][eligible[i]].mean() for i in range(len(EC_IDS))]
            )
            expected_oracle = np.array(
                [matrix[i][eligible[i]].max() for i in range(len(EC_IDS))]
            )
            assert baselines[variant.name][f"chance_{score}"] == pytest.approx(expected_chance)
            assert baselines[variant.name][f"oracle_{score}"] == pytest.approx(expected_oracle)


def test_oracle_of_a_binary_score_is_the_fraction_with_such_a_neighbour():
    """Design: for EC exact the oracle is "has any exact-EC eligible neighbour"."""
    scorer = ECScorer(EC_SETS)
    variants, _hits, _ = build_variants(None, EC_IDS)
    baselines = variant_baselines(scorer, variants, len(EC_IDS), block_size=4)
    oracle = baselines["all"]["oracle_exact"]
    # Every protein has >=1 same-EC partner except none: all four groups have >=3 members.
    assert oracle.mean() == 1.0
    assert set(np.unique(oracle)) <= {0.0, 1.0}


def test_chance_of_the_class_score_is_the_eligible_same_class_share():
    scorer = ECScorer(EC_SETS)
    variants, _hits, _ = build_variants(None, EC_IDS)
    chance = variant_baselines(scorer, variants, len(EC_IDS), block_size=12)["all"]["chance_class"]
    # P00 is EC class 1; 8 of the other 11 proteins are class 1, plus P11's second EC.
    same_class = sum(1 for p in EC_IDS if p != "P00" and any(e.startswith("1.") for e in EC_LABELS[p].split(";")))
    assert chance[EC_IDS.index("P00")] == pytest.approx(same_class / 11)


# --------------------------------------------------------------------------- #
#                                  bootstrap
# --------------------------------------------------------------------------- #


def test_bootstrap_weights_are_multiplicities_of_a_resample():
    weights = bootstrap_weights(20, 64, seed=42)
    assert weights.shape == (64, 20)
    assert np.all(weights.sum(axis=1) == 20)
    assert np.array_equal(weights, bootstrap_weights(20, 64, seed=42))
    assert not np.array_equal(weights, bootstrap_weights(20, 64, seed=43))


def test_bootstrap_means_equal_an_explicit_resampling_loop():
    rng = np.random.default_rng(3)
    values = rng.random(15)
    collected = {("euclidean", "all", "all_queries", "exact"): {"armA": values}}
    boot = bootstrap_means(collected, {("all", "all_queries"): 15}, n_boot=32, seed=42)[
        "euclidean", "all", "all_queries", "exact"
    ]
    reference = np.empty(32)
    ref_rng = np.random.default_rng(42)
    for b in range(32):
        reference[b] = values[ref_rng.integers(0, 15, size=15)].mean()
    assert boot["armA"] == pytest.approx(reference)


def test_paired_bootstrap_uses_the_same_resamples_for_every_arm():
    """The difference of two arms' bootstrap means must equal the bootstrap of the
    difference — that identity only holds if both arms saw the identical resamples."""
    rng = np.random.default_rng(5)
    a, b = rng.random(30), rng.random(30)
    key = ("cosine", "all", "all_queries", "f1")
    boot = bootstrap_means({key: {"armA": a, "armB": b}}, {("all", "all_queries"): 30},
                           n_boot=40, seed=42)[key]
    paired = bootstrap_means({key: {"diff": a - b}}, {("all", "all_queries"): 30},
                             n_boot=40, seed=42)[key]["diff"]
    assert boot["armA"] - boot["armB"] == pytest.approx(paired)


# --------------------------------------------------------------------------- #
#                                GO scoring
# --------------------------------------------------------------------------- #


MINI_OBO = """format-version: 1.2

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function

[Term]
id: GO:0003824
name: catalytic activity
namespace: molecular_function
is_a: GO:0003674

[Term]
id: GO:0016787
name: hydrolase activity
namespace: molecular_function
is_a: GO:0003824

[Term]
id: GO:0004175
name: endopeptidase activity
namespace: molecular_function
is_a: GO:0016787

[Term]
id: GO:0016301
name: kinase activity
namespace: molecular_function
alt_id: GO:0099999
is_a: GO:0003824
relationship: part_of GO:0016787

[Term]
id: GO:0005488
name: binding
namespace: molecular_function
is_a: GO:0003674

[Term]
id: GO:0005515
name: protein binding
namespace: molecular_function
is_a: GO:0005488

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Typedef]
id: part_of
name: part_of
"""

#: Q06 and Q07 carry protein binding ALONE, so the --drop-protein-binding sensitivity
#: must remove them from the cohort; Q04 carries it beside a real term and survives.
GO_LABELS: dict[str, list[str]] = {
    "Q00": ["GO:0004175"], "Q01": ["GO:0004175"], "Q02": ["GO:0016787"],
    "Q03": ["GO:0016301"], "Q04": ["GO:0016301", PROTEIN_BINDING], "Q05": ["GO:0005488"],
    "Q06": [PROTEIN_BINDING], "Q07": [PROTEIN_BINDING], "Q08": ["GO:0003824", "GO:0005488"],
}
GO_IDS: list[str] = sorted(GO_LABELS)


@pytest.fixture
def mini_obo(tmp_path) -> Path:
    path = tmp_path / "mini.obo"
    path.write_text(MINI_OBO)
    return path


def write_go_labels(path: Path, labels: dict[str, list[str]] = GO_LABELS) -> Path:
    rows = ["protein_id\tGO_term"]
    for pid in sorted(labels):
        rows += [f"{pid}\t{term}" for term in labels[pid]]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_go_propagated_f1_matches_the_explicit_set_definition(mini_obo):
    go_terms = parse_obo(mini_obo)
    sets = [frozenset(GO_LABELS[p]) for p in GO_IDS]
    f1 = GOScorer(sets, go_terms).f1_block(0, len(sets))
    propagated = propagate_mf(sets, go_terms)
    for i, q in enumerate(propagated):
        for j, neighbour in enumerate(propagated):
            expected = 2 * len(q & neighbour) / (len(q) + len(neighbour))
            assert f1[i, j] == pytest.approx(expected, abs=1e-12)


def test_go_propagation_follows_part_of_and_drops_the_root(mini_obo):
    go_terms = parse_obo(mini_obo)
    (kinase,) = propagate_mf([frozenset({"GO:0016301"})], go_terms)
    # is_a to catalytic activity AND part_of hydrolase activity; the root is never scored.
    assert kinase == {"GO:0016301", "GO:0003824", "GO:0016787"}
    assert MF_ROOT not in kinase


def test_vectorised_wang_bma_equals_the_scalar_implementation(mini_obo):
    """Design item 11: the matrix BMA is only admissible if it is the same number."""
    go_terms = parse_obo(mini_obo)
    rng = np.random.default_rng(0)
    pool = sorted(t for t, v in go_terms.items() if v.namespace == "molecular_function" and t != MF_ROOT)
    sets = [
        frozenset(rng.choice(pool, size=int(rng.integers(1, 4)), replace=False).tolist())
        for _ in range(40)
    ]
    scorer = GOScorer(sets, go_terms)
    rows, cols = random_ordered_pairs(len(sets), 250, seed=42)
    assert rows.size >= 200
    ours = scorer.pairs(rows, cols)["wang_bma"]
    assert np.max(np.abs(ours - scalar_bma_values(sets, rows, cols, go_terms))) < 1e-9
    block = scorer.bma_block(0, len(sets))
    assert np.max(np.abs(block[rows, cols] - ours)) < 1e-12


def test_go_scorer_block_and_pairs_agree(mini_obo):
    go_terms = parse_obo(mini_obo)
    sets = [frozenset(GO_LABELS[p]) for p in GO_IDS]
    scorer = GOScorer(sets, go_terms)
    block = scorer.block(0, len(sets))
    rows, cols = random_ordered_pairs(len(sets), 40, seed=2)
    pairs = scorer.pairs(rows, cols, chunk=7)  # chunked path, not one shot
    for name in scorer.names:
        assert pairs[name] == pytest.approx(block[name][rows, cols], abs=1e-12)


# --------------------------------------------------------------------------- #
#                            end-to-end: EC report
# --------------------------------------------------------------------------- #


def _ec_report(tmp_path, **overrides):
    emb = tmp_path / "emb"
    emb.mkdir(parents=True, exist_ok=True)
    write_h5(emb / "perfect.h5", group_embeddings(EC_IDS))
    write_h5(emb / "noise.h5", noise_embeddings(EC_IDS), two_d=True)  # ProtT5-style (1, D)
    kwargs = dict(
        labels_kind="ec",
        freeze=write_freeze(tmp_path / "freeze.json", EC_IDS),
        labels=write_ec_labels(tmp_path / "labels.tsv"),
        emb_dir=emb,
        out_dir=tmp_path / "out",
        identity_m8=write_m8(tmp_path / "hits.m8", [("P00", "P01", 0.9, 1e-50)]),
        n_boot=64,
        block_size=5,
    )
    kwargs.update(overrides)
    return run_transfer_report(**kwargs), Path(kwargs["out_dir"])


def test_ec_report_end_to_end(tmp_path):
    manifest, out_dir = _ec_report(tmp_path)
    summary = pd.read_csv(out_dir / "summary.csv")

    assert set(summary["arm"]) == {"perfect", "noise", "hbi_evalue", "hbi_fident"}
    assert set(summary["distance"]) == {"euclidean", "cosine"}
    assert set(summary["variant"]) == {"all", "fident_lt_0.3", "no_hit_evalue_0.001"}
    assert set(summary["score"]) == {"exact", "share3", "share2", "class"}
    # One row per (arm, distance, variant, subset, score). The embedding arms cover every
    # subset the manifest reports; HBI only its own, and only where it can answer at all.
    expected = 0
    for variant in manifest["variants"]:
        for subset in variant["subsets"]:
            n_arms = 2 + (2 if subset == "hbi_answerable" else 0)
            expected += n_arms * 2 * 4  # distances x scores
    assert len(summary) == expected

    perfect = summary[(summary["arm"] == "perfect") & (summary["score"] == "exact")]
    assert (perfect["mean"] == 1.0).all()  # the embedding IS the label
    assert (perfect["mean"] >= perfect["chance"]).all()
    assert (perfect["mean"] <= perfect["oracle"] + 1e-12).all()
    assert (summary["ci_lo"] <= summary["mean"] + 1e-12).all()
    assert (summary["ci_hi"] >= summary["mean"] - 1e-12).all()

    # ProtT5-style (1, D) datasets were flattened, not left 2-D.
    assert (summary[summary["arm"] == "noise"]["n_queries"] > 0).all()
    # No tau_b for the EC arm: ec_report owns that number.
    assert not (out_dir / "tau_b.csv").exists()
    assert manifest["n_cohort"] == len(EC_IDS)
    assert set(manifest["sha256"]) >= {str(manifest["inputs"]["freeze"])}


def test_ec_report_per_query_parquet_reproduces_the_summary_mean(tmp_path):
    """Every published mean must be re-derivable from the per-query rows and the subset
    flags — the flags are the only record of which queries a subset row covers."""
    _, out_dir = _ec_report(tmp_path)
    summary = pd.read_csv(out_dir / "summary.csv")
    per_query = pd.read_parquet(out_dir / "per_query.parquet")
    flags = pd.read_parquet(out_dir / "per_query_baseline.parquet")
    for _, row in summary.iterrows():
        cell = per_query[
            (per_query["arm"] == row["arm"])
            & (per_query["distance"] == row["distance"])
            & (per_query["variant"] == row["variant"])
        ]
        in_subset = flags[(flags["variant"] == row["variant"]) & flags[f"in_{row['subset']}"]]
        cell = cell[cell["query_id"].isin(set(in_subset["query_id"]))]
        assert len(cell) == row["n_queries"]
        assert cell[f"score_{row['score']}"].mean() == pytest.approx(row["mean"])
        assert int(cell["tied"].sum()) == row["n_ties"]


def test_paired_differences_cover_every_arm_pair_and_match_the_means(tmp_path):
    _, out_dir = _ec_report(tmp_path)
    summary = pd.read_csv(out_dir / "summary.csv")
    indexed = summary.set_index(["arm", "distance", "variant", "subset", "score"])
    paired = pd.read_csv(out_dir / "paired_differences.csv")
    # C(k, 2) pairs in a cell of k arms — 2 embedding arms everywhere, 4 where HBI answers.
    cells = summary.groupby(["distance", "variant", "subset", "score"], observed=True)["arm"].nunique()
    assert len(paired) == int((cells * (cells - 1) // 2).sum())
    for _, row in paired.iterrows():
        key = (row["distance"], row["variant"], row["subset"], row["score"])
        mean_a = indexed.loc[(row["arm_a"], *key), "mean"]
        mean_b = indexed.loc[(row["arm_b"], *key), "mean"]
        assert row["mean_diff"] == pytest.approx(mean_a - mean_b)
        assert row["ci_lo"] <= row["mean_diff"] + 1e-9
        assert row["ci_hi"] >= row["mean_diff"] - 1e-9


def test_identity_variant_shrinks_the_eligible_neighbourhood(tmp_path):
    """Excluding homologous neighbours must be visible in the manifest, not silent."""
    hits = [(a, b, 0.9, 1e-50) for a in EC_IDS for b in EC_IDS if a < b and GROUP[a] == GROUP[b]]
    manifest, out_dir = _ec_report(tmp_path, identity_m8=write_m8(tmp_path / "many.m8", hits))
    by_name = {v["name"]: v for v in manifest["variants"]}
    assert by_name["all"]["median_eligible_neighbours"] == len(EC_IDS) - 1
    assert by_name["fident_lt_0.3"]["median_eligible_neighbours"] < len(EC_IDS) - 1
    summary = pd.read_csv(out_dir / "summary.csv")
    restricted = summary[(summary["variant"] == "fident_lt_0.3") & (summary["score"] == "exact")]
    # With every same-group neighbour excluded, only P11's second EC can still score.
    assert restricted["oracle"].max() < 1.0


def test_report_is_deterministic_under_the_seed(tmp_path):
    first, dir_a = _ec_report(tmp_path / "a")
    second, dir_b = _ec_report(tmp_path / "b")
    for name in ("summary.csv", "paired_differences.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(dir_a / name), pd.read_csv(dir_b / name))
    assert first["sha256"][first["inputs"]["labels"]] != ""
    assert second["parameters"]["seed"] == 42


# --------------------------------------------------------------------------- #
#                    homology baseline (HBI) — the headline comparison
# --------------------------------------------------------------------------- #

#: A hit table with a known answer for every branch of the baseline: P00's best hit differs
#: between the two criteria, P05 is only ever a TARGET (so it must still find P04), P06/P07
#: are a weak-identity hit that survives the fident variant, P10/P11 are a weak hit that
#: survives even the no-hit variant, and P02/P08 have no hit at all.
HBI_HITS: list[tuple[str, str, float, float]] = [
    ("P00", "P01", 0.40, 1e-50),
    ("P00", "P03", 0.95, 1e-10),
    ("P04", "P05", 0.70, 1e-30),
    ("P09", "P00", 0.55, 1e-20),
    ("P06", "P07", 0.10, 1e-5),
    ("P10", "P11", 0.05, 0.5),
]
HBI_NO_HIT = {"P02", "P08"}


def _hbi_variants(tmp_path, hits=None):
    m8 = write_m8(tmp_path / "hbi.m8", HBI_HITS if hits is None else hits)
    variants, table, _ = build_variants(m8, EC_IDS)
    return {v.name: v for v in variants}, table


def _hbi_report(tmp_path, **overrides):
    return _ec_report(
        tmp_path, identity_m8=write_m8(tmp_path / "hbi.m8", HBI_HITS), **overrides
    )


def test_hbi_picks_the_best_hit_by_each_criterion(tmp_path):
    by_name, table = _hbi_variants(tmp_path)
    nn_e, value_e, tied_e = hbi_neighbours(table, by_name["all"], "evalue")
    nn_f, value_f, _ = hbi_neighbours(table, by_name["all"], "fident")
    p00 = EC_IDS.index("P00")
    assert nn_e[p00] == EC_IDS.index("P01") and value_e[p00] == 1e-50
    assert nn_f[p00] == EC_IDS.index("P03")  # a weaker E-value but 0.95 identity
    assert value_f[p00] == pytest.approx(1.0 - 0.95)
    # "in either direction": P05 is only ever a target, and must still find P04.
    assert nn_e[EC_IDS.index("P05")] == EC_IDS.index("P04")
    # A query without a hit is -1/inf, never a silent neighbour 0.
    for pid in HBI_NO_HIT:
        assert nn_e[EC_IDS.index(pid)] == -1
        assert np.isinf(value_e[EC_IDS.index(pid)])
    assert not tied_e.any()


def test_hbi_transfers_only_from_neighbours_the_variant_allows(tmp_path):
    """The identity control must bind the sequence search exactly as it binds the pLMs."""
    by_name, table = _hbi_variants(tmp_path)
    answerable = {
        name: set(np.array(EC_IDS)[hbi_neighbours(table, variant, "evalue")[0] >= 0])
        for name, variant in by_name.items()
    }
    assert answerable["all"] == set(EC_IDS) - HBI_NO_HIT
    # Only the two sub-0.30 hits survive the identity gate...
    assert answerable["fident_lt_0.3"] == {"P06", "P07", "P10", "P11"}
    # ...and only the E=0.5 pair survives "no hit at all".
    assert answerable["no_hit_evalue_0.001"] == {"P10", "P11"}


def test_hbi_ties_break_by_the_secondary_criterion_and_are_counted(tmp_path):
    by_name, table = _hbi_variants(
        tmp_path, hits=[("P00", "P01", 0.50, 1e-20), ("P00", "P02", 0.90, 1e-20)]
    )
    nn, _, tied = hbi_neighbours(table, by_name["all"], "evalue")
    p00 = EC_IDS.index("P00")
    assert nn[p00] == EC_IDS.index("P02")  # equal E-value, higher identity wins
    assert tied[p00]  # and the tie is reported, because the answer depends on the order


def test_hbi_is_scored_on_the_queries_it_can_answer(tmp_path):
    manifest, out_dir = _hbi_report(tmp_path)
    by_variant = {v["name"]: v for v in manifest["variants"]}
    assert by_variant["all"]["subsets"] == {
        "all_queries": len(EC_IDS),
        "hbi_answerable": len(EC_IDS) - len(HBI_NO_HIT),
        "no_hit": len(HBI_NO_HIT),
    }
    assert manifest["hbi"]["n_queries_without_any_cohort_hit"] == len(HBI_NO_HIT)
    assert manifest["hbi_arms"] == list(HBI_ARMS)

    summary = pd.read_csv(out_dir / "summary.csv")
    hbi = summary[summary["arm"].isin(HBI_ARMS)]
    assert set(hbi["subset"]) == {SUBSET_HBI}  # never averaged over queries it cannot answer
    under_all = hbi[hbi["variant"] == "all"]
    assert set(under_all["n_queries"]) == {len(EC_IDS) - len(HBI_NO_HIT)}
    assert set(under_all["n_dropped"]) == {len(HBI_NO_HIT)}
    # hbi_evalue transfers P00 -> P01 (same EC) and P09 -> P00 (different class), so its
    # exact score is neither 0 nor 1: the baseline is a real competitor, not a straw man.
    exact = under_all[(under_all["arm"] == "hbi_evalue") & (under_all["score"] == "exact")]
    assert 0.0 < exact["mean"].iloc[0] < 1.0


def test_hbi_does_not_depend_on_the_embedding_distance(tmp_path):
    """Its rows are repeated under each distance only so the paired difference against an
    embedding arm sits in one cell; the numbers themselves must be identical."""
    _, out_dir = _hbi_report(tmp_path)
    summary = pd.read_csv(out_dir / "summary.csv")
    hbi = summary[summary["arm"] == "hbi_evalue"]
    wide = hbi.pivot_table(index=["variant", "subset", "score"], columns="distance", values="mean")
    assert wide["euclidean"].to_numpy() == pytest.approx(wide["cosine"].to_numpy())


def test_the_pLM_minus_hbi_difference_is_the_gap_on_the_same_queries(tmp_path):
    """The headline number: a paired difference is only paired if both arms were scored on
    the identical query list, in the identical order."""
    _, out_dir = _hbi_report(tmp_path)
    per_query = pd.read_parquet(out_dir / "per_query.parquet")
    flags = pd.read_parquet(out_dir / "per_query_baseline.parquet")
    answerable = set(
        flags[(flags["variant"] == "all") & flags[f"in_{SUBSET_HBI}"]]["query_id"]
    )

    def scores(arm):
        rows = per_query[
            (per_query["arm"] == arm)
            & (per_query["distance"] == "euclidean")
            & (per_query["variant"] == "all")
        ]
        rows = rows[rows["query_id"].isin(answerable)]
        return rows.sort_values("query_id")["score_exact"].to_numpy()

    gap = scores("perfect").mean() - scores("hbi_evalue").mean()
    paired = pd.read_csv(out_dir / "paired_differences.csv")
    row = paired[
        (paired["arm_a"] == "perfect")
        & (paired["arm_b"] == "hbi_evalue")
        & (paired["distance"] == "euclidean")
        & (paired["variant"] == "all")
        & (paired["subset"] == SUBSET_HBI)
        & (paired["score"] == "exact")
    ]
    assert len(row) == 1
    assert row["mean_diff"].iloc[0] == pytest.approx(gap)
    assert row["ci_lo"].iloc[0] <= gap <= row["ci_hi"].iloc[0]


def test_the_no_hit_subset_is_where_only_an_embedding_can_answer(tmp_path):
    _, out_dir = _hbi_report(tmp_path)
    flags = pd.read_parquet(out_dir / "per_query_baseline.parquet")
    under_all = flags[flags["variant"] == "all"]
    assert set(under_all[under_all[f"in_{SUBSET_NO_HIT}"]]["query_id"]) == HBI_NO_HIT
    assert not (under_all[f"in_{SUBSET_NO_HIT}"] & under_all[f"in_{SUBSET_HBI}"]).any()

    summary = pd.read_csv(out_dir / "summary.csv")
    no_hit = summary[summary["subset"] == SUBSET_NO_HIT]
    assert set(no_hit["arm"]) == {"perfect", "noise"}  # HBI has no answer here by construction
    assert set(no_hit[no_hit["variant"] == "all"]["n_queries"]) == {len(HBI_NO_HIT)}


def test_without_an_m8_there_is_no_homology_baseline(tmp_path):
    manifest, out_dir = _ec_report(tmp_path, identity_m8=None)
    summary = pd.read_csv(out_dir / "summary.csv")
    assert set(summary["subset"]) == {SUBSET_ALL}
    assert set(summary["arm"]) == {"perfect", "noise"}
    assert manifest["hbi_arms"] == []
    assert manifest["hbi"]["n_queries_without_any_cohort_hit"] == 0


# --------------------------------------------------------------------------- #
#                            end-to-end: GO report
# --------------------------------------------------------------------------- #


def _go_report(tmp_path, mini_obo, **overrides):
    emb = tmp_path / "emb"
    emb.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    terms = sorted({t for v in GO_LABELS.values() for t in v})
    write_h5(
        emb / "perfect.h5",
        {p: np.array([1.0 if t in GO_LABELS[p] else 0.0 for t in terms]) for p in GO_IDS},
    )
    write_h5(emb / "random_init_toy_seed0.h5", {p: rng.standard_normal(6) for p in GO_IDS})
    kwargs = dict(
        labels_kind="go",
        freeze=write_freeze(tmp_path / "freeze.json", GO_IDS),
        labels=write_go_labels(tmp_path / "go_labels.tsv"),
        go_obo=mini_obo,
        emb_dir=emb,
        out_dir=tmp_path / "out",
        identity_m8=None,
        n_boot=32,
        block_size=4,
    )
    kwargs.update(overrides)
    return run_transfer_report(**kwargs), Path(kwargs["out_dir"])


def test_go_report_end_to_end(tmp_path, mini_obo):
    manifest, out_dir = _go_report(tmp_path, mini_obo)
    summary = pd.read_csv(out_dir / "summary.csv")
    assert set(summary["score"]) == {"f1", "wang_bma"}
    assert set(summary["arm"]) == {"perfect", "randinit_toy"}
    assert manifest["wang_bma_check"]["max_abs_diff_vs_scalar"] < 1e-9

    tau = pd.read_csv(out_dir / "tau_b.csv")
    assert len(tau) == 2 * 2  # arms x distances
    assert tau["n_pairs"].eq(len(GO_IDS) * (len(GO_IDS) - 1) // 2).all()
    assert not tau["subsampled"].any()

    baseline = pd.read_parquet(out_dir / "per_query_baseline.parquet")
    assert set(baseline.columns) >= {"variant", "query_id", "n_eligible", "chance_f1", "oracle_f1"}
    assert (baseline["n_eligible"] == len(GO_IDS) - 1).all()


def test_go_scores_lie_between_chance_and_oracle(tmp_path, mini_obo):
    _, out_dir = _go_report(tmp_path, mini_obo)
    summary = pd.read_csv(out_dir / "summary.csv")
    assert (summary["mean"] <= summary["oracle"] + 1e-12).all()
    assert (summary["mean"] >= 0.0).all() and (summary["mean"] <= 1.0 + 1e-12).all()


def test_drop_protein_binding_removes_proteins_left_termless(tmp_path, mini_obo):
    manifest, out_dir = _go_report(tmp_path, mini_obo, drop_protein_binding=True)
    # Q06 and Q07 are annotated with GO:0005515 only, so they leave the cohort.
    assert manifest["n_frozen"] == len(GO_IDS)
    assert manifest["n_cohort"] == len(GO_IDS) - 2
    assert manifest["label_cleaning"]["proteins_dropped"] == 2
    assert manifest["label_cleaning"]["dropped_term"] == 3  # Q04, Q06, Q07
    per_query = pd.read_parquet(out_dir / "per_query.parquet")
    assert not {"Q06", "Q07"} & set(per_query["query_id"].astype(str))
    assert not {"Q06", "Q07"} & set(per_query["neighbour_id"].astype(str))


def test_tau_b_subsampling_is_recorded(tmp_path, mini_obo):
    _, out_dir = _go_report(tmp_path, mini_obo, tau_max_pairs=8)
    tau = pd.read_csv(out_dir / "tau_b.csv")
    assert tau["subsampled"].all()
    assert (tau["n_pairs"] < len(GO_IDS) * (len(GO_IDS) - 1) // 2).all()


def test_alt_ids_are_mapped_not_dropped(tmp_path, mini_obo):
    labels = {**GO_LABELS, "Q03": ["GO:0099999"]}  # secondary id of GO:0016301
    manifest, _ = _go_report(
        tmp_path, mini_obo, labels=write_go_labels(tmp_path / "alt.tsv", labels)
    )
    assert manifest["label_cleaning"]["alt_id_mapped"] == 1
    assert manifest["n_cohort"] == len(GO_IDS)


# --------------------------------------------------------------------------- #
#                              loud failures
# --------------------------------------------------------------------------- #


def test_missing_frozen_id_in_an_arm_is_fatal(tmp_path):
    emb = tmp_path / "emb"
    emb.mkdir()
    vectors = group_embeddings(EC_IDS)
    vectors.pop("P05")
    write_h5(emb / "short.h5", vectors)
    with pytest.raises(Exception, match="missing from the slice"):
        run_transfer_report(
            labels_kind="ec",
            freeze=write_freeze(tmp_path / "freeze.json", EC_IDS),
            labels=write_ec_labels(tmp_path / "labels.tsv"),
            emb_dir=emb,
            out_dir=tmp_path / "out",
            n_boot=8,
        )


def test_a_label_free_frozen_id_is_fatal(tmp_path):
    emb = tmp_path / "emb"
    emb.mkdir()
    write_h5(emb / "a.h5", group_embeddings(EC_IDS))
    partial = {pid: ec for pid, ec in EC_LABELS.items() if pid != "P07"}
    with pytest.raises(Exception, match="no fully-specified EC"):
        run_transfer_report(
            labels_kind="ec",
            freeze=write_freeze(tmp_path / "freeze.json", EC_IDS),
            labels=write_ec_labels(tmp_path / "labels.tsv", partial),
            emb_dir=emb,
            out_dir=tmp_path / "out",
            n_boot=8,
        )


def test_non_finite_embeddings_are_fatal(tmp_path):
    emb = tmp_path / "emb"
    emb.mkdir()
    vectors = group_embeddings(EC_IDS)
    vectors["P02"] = vectors["P02"] * np.nan
    path = write_h5(emb / "nan.h5", vectors)
    with pytest.raises(Exception, match="non-finite"):
        load_arm_matrix(path, EC_IDS, EC_IDS)


def test_a_go_run_without_an_obo_is_fatal(tmp_path):
    emb = tmp_path / "emb"
    emb.mkdir()
    write_h5(emb / "a.h5", {p: np.ones(3) for p in GO_IDS})
    with pytest.raises(Exception, match="--go-obo"):
        run_transfer_report(
            labels_kind="go",
            freeze=write_freeze(tmp_path / "freeze.json", GO_IDS),
            labels=write_go_labels(tmp_path / "go.tsv"),
            emb_dir=emb,
            out_dir=tmp_path / "out",
            n_boot=8,
        )


def test_a_termless_protein_without_the_sensitivity_flag_is_fatal(tmp_path, mini_obo):
    emb = tmp_path / "emb"
    emb.mkdir()
    write_h5(emb / "a.h5", {p: np.ones(3) for p in GO_IDS})
    labels = {**GO_LABELS, "Q05": ["GO:0008150"]}  # a BP term: nothing scoreable in MF
    with pytest.raises(Exception, match="no scoreable MF term"):
        run_transfer_report(
            labels_kind="go",
            freeze=write_freeze(tmp_path / "freeze.json", GO_IDS),
            labels=write_go_labels(tmp_path / "bp.tsv", labels),
            go_obo=mini_obo,
            emb_dir=emb,
            out_dir=tmp_path / "out",
            n_boot=8,
        )


# --------------------------------------------------------------------------- #
#                                    CLI
# --------------------------------------------------------------------------- #


def test_arm_name_labels_the_random_init_controls():
    assert arm_name("/x/random_init_ankh_base_seed0.h5") == "randinit_ankh_base"
    assert arm_name("/x/random_init_prot_t5_seed0.h5") == "randinit_prot_t5"
    assert arm_name("/x/esm2_3b.h5") == "esm2_3b"
    assert arm_name("/x/random_1024.h5") == "random_1024"


def test_cli_defaults_match_the_design():
    args = build_parser().parse_args(
        ["--labels-kind", "ec", "--freeze", "f", "--labels", "l", "--emb-dir", "e", "--out-dir", "o"]
    )
    assert args.distances == ["euclidean", "cosine"]
    assert args.n_boot == 2000 and args.seed == 42
    assert args.fident_max == 0.30 and args.evalue_max == 1e-3
    assert args.arms is None and args.drop_protein_binding is False


def test_cli_runs_and_reports_input_faults_as_exit_2(tmp_path, capsys):
    emb = tmp_path / "emb"
    emb.mkdir()
    write_h5(emb / "perfect.h5", group_embeddings(EC_IDS))
    argv = [
        "--labels-kind", "ec",
        "--freeze", str(write_freeze(tmp_path / "freeze.json", EC_IDS)),
        "--labels", str(write_ec_labels(tmp_path / "labels.tsv")),
        "--emb-dir", str(emb),
        "--out-dir", str(tmp_path / "out"),
        "--n-boot", "16",
        "--distances", "cosine",
    ]
    assert main(argv) == 0
    assert (tmp_path / "out" / "manifest.json").exists()
    capsys.readouterr()
    assert main(argv + ["--arms", "nope"]) == 2
    assert "INPUT ERROR" in capsys.readouterr().err
