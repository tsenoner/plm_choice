#!/usr/bin/env python3
"""The empirical noise floor and the hubness behind it — the side measurement SUMMARY.md
quotes for the ``random_1024`` validity check.

Why it exists. ``random_1024`` is the pipeline's validity check: i.i.d. noise must land on
the exact chance expectation. Its published CI is a bootstrap over QUERIES with the
embedding held fixed, so it carries none of the randomness of the noise draw itself, and on
the euclidean axis it excludes chance in 13 of 48 EC cells and all 12 GO cells. This script
measures the floor the honest way — ``--draws`` fresh i.i.d. Gaussian embeddings, scored
end to end by the same code — and measures why a single draw wanders: under euclidean,
nearest neighbour in i.i.d. Gaussian space is dominated by the NEIGHBOUR's norm, so the
low-norm proteins become hubs that most queries transfer from.

Outputs ``noise_floor_<kind>.json`` next to this file. Nothing here is typed by hand into
SUMMARY.md; the generator reads this JSON.

    PYTHONPATH=src python side_measurements/noise_floor.py --kind ec --draws 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# These paths were absolute to one machine. They are environment variables now, so an
# unset one fails here by name rather than as a FileNotFoundError further down.
import os


def _need(var: str) -> str:
    """The value of `var`, or a message naming what to set."""
    try:
        return os.environ[var]
    except KeyError:
        raise SystemExit(f"set {var} before running this script") from None


ARTEFACTS = _need("PAPER_ARTEFACTS")
REPO = _need("REPO")

sys.path.insert(0, f"{REPO}/src")

from data_preparation.go_semantic_similarity import parse_obo  # noqa: E402
from evaluation.analysis_io import load_frozen_ids  # noqa: E402
from evaluation.go_similarity_matrix import parse_alt_ids  # noqa: E402
from evaluation.transfer_report import (  # noqa: E402
    VARIANT_ALL,
    BlockDistances,
    ECScorer,
    EligibilityVariant,
    GOScorer,
    load_ec_labels,
    load_go_labels,
    nearest_neighbour_block,
)

RESULTS = Path(f"{ARTEFACTS}/functional_2026-09-17")
BACKUP = Path(ARTEFACTS)
OBO = Path(f"{REPO}/data/reference/go/go-basic.obo")
INPUTS = {
    "ec": {
        "freeze": BACKUP / "c1_strat_2026-09-15/ec_strat_v2_freeze.json",
        "labels": BACKUP / "c1_strat_2026-09-15/ec_labels_v2.tsv",
        "run": RESULTS / "ec_v2_transfer",
    },
    "go": {
        "freeze": RESULTS / "go_mf_cohort_freeze.json",
        "labels": RESULTS / "go_mf_labels.tsv",
        "run": RESULTS / "go_mf_transfer",
    },
}


def build_scorer(kind: str):
    ids = list(load_frozen_ids(INPUTS[kind]["freeze"]))
    if kind == "ec":
        return ids, ECScorer(load_ec_labels(INPUTS[kind]["labels"], ids))
    go_terms = parse_obo(OBO)
    ids, term_sets, _ = load_go_labels(
        INPUTS[kind]["labels"], ids, go_terms, parse_alt_ids(OBO), drop_protein_binding=False
    )
    return ids, GOScorer(term_sets, go_terms)


def one_draw(scorer, n: int, dim: int, seed: int, distance: str, variant) -> dict[str, float]:
    """Mean transfer score of ONE fresh i.i.d. Gaussian embedding, same code path as a run."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim))
    blocks = BlockDistances(x, distance)
    nn = np.zeros(n, dtype=np.int64)
    for lo in range(0, n, 1024):
        hi = min(lo + 1024, n)
        nn[lo:hi] = nearest_neighbour_block(blocks.block(lo, hi), variant.block(lo, hi))[0]
    rows = np.arange(n)
    scores = scorer.pairs(rows, nn)
    out = {name: float(values.mean()) for name, values in scores.items()}
    deg = np.bincount(nn, minlength=n)
    order = np.argsort(deg)[::-1]
    norms = np.linalg.norm(x, axis=1)
    out["_hub_distinct_neighbours"] = float(np.count_nonzero(deg))
    out["_hub_top_in_degree"] = float(deg.max())
    out["_hub_top10_share"] = float(deg[order[:10]].sum() / n)
    out["_hub_corr_indegree_norm"] = float(np.corrcoef(deg, norms)[0, 1])
    out["_hub_top20_mean_norm"] = float(norms[order[:20]].mean())
    out["_cohort_mean_norm"] = float(norms.mean())
    return out


def random_neighbour_floor(scorer, n: int, draws: int, seed: int) -> dict[str, dict[str, float]]:
    """The other floor: one uniformly random ELIGIBLE neighbour per query, ``draws`` times.

    This is the sampling spread the chance expectation really has; comparing it with the
    fresh-embedding spread separates "a random draw wanders" from "a single embedding has
    hubs".
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(n)
    means: dict[str, list[float]] = {}
    for _ in range(draws):
        pick = rng.integers(0, n - 1, size=n)
        pick += (pick >= rows).astype(pick.dtype)  # never itself
        for name, values in scorer.pairs(rows, pick).items():
            means.setdefault(name, []).append(float(values.mean()))
    return {
        name: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "draws": len(v)}
        for name, v in means.items()
    }


def published_hubness(run_dir: Path, arms: list[str]) -> dict:
    """In-degree of the published nearest-neighbour choices (variant `all`), per arm."""
    frame = pd.read_parquet(
        run_dir / "per_query.parquet", columns=["arm", "distance", "variant", "neighbour_id"]
    )
    frame = frame[frame["variant"] == VARIANT_ALL]
    out: dict[str, dict] = {}
    for distance in ("euclidean", "cosine"):
        sub = frame[frame["distance"] == distance]
        for arm in arms:
            rows = sub[sub["arm"] == arm]
            if rows.empty:
                continue
            deg = rows["neighbour_id"].value_counts()
            deg = deg[deg > 0]
            n = len(rows)
            out[f"{arm}/{distance}"] = {
                "n_queries": int(n),
                "distinct_neighbours": int(len(deg)),
                "top_in_degree": [int(v) for v in deg.to_numpy()[:3]],
                "top10_share": float(deg.to_numpy()[:10].sum() / n),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("ec", "go"), required=True)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--dim", type=int, default=1024, help="random_1024's dimension")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    ids, scorer = build_scorer(args.kind)
    n = len(ids)
    variant = EligibilityVariant(VARIANT_ALL, "every other cohort protein", None, n)
    print(f"{args.kind}: n={n}, scores={list(scorer.names)}", flush=True)

    fresh: dict[str, dict[str, list[float]]] = {}
    for distance in ("euclidean", "cosine"):
        per_key: dict[str, list[float]] = {}
        for k in range(args.draws):
            got = one_draw(scorer, n, args.dim, args.seed + k, distance, variant)
            for key, value in got.items():
                per_key.setdefault(key, []).append(value)
            print(f"  {distance} draw {k}: "
                  + ", ".join(f"{s}={got[s]:.5f}" for s in scorer.names), flush=True)
        fresh[distance] = per_key

    summary = {
        "kind": args.kind,
        "n_cohort": n,
        "dim": args.dim,
        "draws": args.draws,
        "seed0": args.seed,
        "fresh_embeddings": {
            distance: {
                key: {
                    "mean": float(np.mean(v)),
                    "sd": float(np.std(v, ddof=1)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                }
                for key, v in per_key.items()
            }
            for distance, per_key in fresh.items()
        },
        "random_neighbour_draw": random_neighbour_floor(scorer, n, 200, args.seed),
        "published_hubness": published_hubness(
            INPUTS[args.kind]["run"],
            ["random_1024", "clean", "prottucker", "randinit_esm1b", "esmc_600m"],
        ),
    }
    out = Path(__file__).with_name(f"noise_floor_{args.kind}.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
