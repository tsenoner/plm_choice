#!/usr/bin/env python3
"""Re-score the saved Swiss-Prot test predictions of one probe grid on subsets of the test pairs.

    python scripts/paper/rescore_test_predictions.py \
        --preds ~/plm_choice/models/full100_p10 \
        --pairs $DSS/plm_choice_data/data/processed/sprot_pre2024_e1/sets/test.parquet \
        --emb-dir $DSS/plm_choice_data/data/processed/full100_p10/embeddings \
        --caps-dir <out>/pair_caps --caps-tag test_full --caps 10 50 100 \
        --out <out>/rescore

Two checks of task D14, both from predictions that already exist (nothing is re-run):

* **R1.6 / M-40 -- TM-score by sequence identity.** The TM-score probe and the raw distance
  scored within PIDE bins [0.3,0.5), [0.5,0.7), [0.7,0.9), [0.9,1.0] (the bins of
  pair_redundancy.py), plus the pairs that carry no PIDE (no MMseqs2 hit that passed the
  reliability filters -- NOT low identity, see pair_redundancy.py --realign).
* **R1.3 / M-34 -- per-protein pair caps.** Every target scored on the greedy random caps
  written by pair_redundancy.py (a pair kept while both its proteins are below k), so the
  ranking can be compared with the uncapped one.

**Attaching pair ids.** The prediction archives hold only `predictions` and `targets`. They
are written in the order of shared.datasets._load_and_filter_data: the test parquet's rows,
minus rows with a null target, minus pairs with a protein outside (the arm's HDF5 keys - the
committed exclusion freeze). This script replays exactly that, per arm, and refuses any
archive whose `targets` are not that column in that order -- only then are the parquet's
query/target/fident borrowed. As a second check the replayed full-set Spearman must equal the
one in the cell's own metrics file.

Score = Spearman rho oriented so that higher is better: rho for the FNN probe, -rho for the
Euclidean read-out (a distance) -- the paper's magnitude. Ranks are among the 14 pLMs (the
Gaussian control is scored but never ranked).

Outputs (--out): rescore_arms.csv (subset x read-out x target x arm), rescore_rank_agreement.csv
(each subset's ranking against the all-pairs one), rescore_summary.json (inputs, hashes,
replay checks, headline facts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from shared.protein_cohort import load_excluded_proteins, restrict_to_cohort  # noqa: E402

TARGETS = ("fident", "alntmscore", "hfsp")
READ_OUTS = ("fnn", "euclidean")
CONTROL = "random_1024"
SPECIALISTS = {"clean", "prottucker"}
# pair_redundancy.py's populations: the capped lists of "pide_hfsp" serve both PIDE and HFSP.
CAP_POPULATION = {"fident": "pide_hfsp", "hfsp": "pide_hfsp", "alntmscore": "tmscore"}
PIDE_EDGES = (0.3, 0.5, 0.7, 0.9)
PIDE_BIN_NAMES = ("[0.3,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pide_bins(fident: np.ndarray) -> np.ndarray:
    """Bin label per pair, exactly as pair_redundancy.pide_bin (last bin closed at 1.0)."""
    out = np.full(fident.size, "outside_bins", dtype=object)
    out[np.isnan(fident)] = "no_pide"
    for lo, hi, name in zip(PIDE_EDGES, (*PIDE_EDGES[1:], None), PIDE_BIN_NAMES, strict=True):
        sel = (fident >= lo) & ((fident < hi) if hi is not None else (fident <= 1.0))
        out[sel] = name
    return out


def ranked(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def rho_from_ranks(rank_y: np.ndarray, x: np.ndarray) -> float:
    """Spearman rho = Pearson of average ranks (what scipy.stats.spearmanr computes)."""
    return float(np.corrcoef(rank_y, ranked(x))[0, 1])


def metrics_spearman(npz: Path) -> float:
    txt = npz.with_name(npz.name.replace("_predictions_targets.npz", "_metrics.txt"))
    for line in txt.read_text().splitlines():
        if line.startswith("Spearman:"):
            return float(line.split(":", 1)[1])
    raise SystemExit(f"{txt}: no Spearman line")


def ranks_of(score: dict[str, float]) -> dict[str, int]:
    order = sorted((a for a in score if a != CONTROL), key=score.get, reverse=True)
    return {a: i + 1 for i, a in enumerate(order)}


def agreement(base: dict[str, float], sub: dict[str, float]) -> dict:
    arms = sorted(a for a in base if a != CONTROL and a in sub)
    rb, rs = ranks_of({a: base[a] for a in arms}), ranks_of({a: sub[a] for a in arms})
    found = [a for a in arms if a not in SPECIALISTS]
    shift = [abs(rb[a] - rs[a]) for a in arms]
    top = lambda r: " > ".join(sorted(r, key=r.get)[:3])  # noqa: E731
    return {"n_models": len(arms),
            "rank_rho_all": float(stats.spearmanr([base[a] for a in arms], [sub[a] for a in arms])[0]),
            "kendall_tau_all": float(stats.kendalltau([base[a] for a in arms], [sub[a] for a in arms])[0]),
            "rank_rho_foundation": float(stats.spearmanr([base[a] for a in found], [sub[a] for a in found])[0]),
            "same_best": sorted(rb, key=rb.get)[0] == sorted(rs, key=rs.get)[0],
            "top3_all_pairs": top(rb), "top3_subset": top(rs),
            "max_rank_shift": int(max(shift)), "n_models_rank_changed": int(sum(s > 0 for s in shift)),
            "median_abs_delta_score": float(np.median([abs(sub[a] - base[a]) for a in arms])),
            "max_abs_delta_score": float(max(abs(sub[a] - base[a]) for a in arms))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True, help="models/<grid> with <read-out>/<target>/<arm>/")
    ap.add_argument("--pairs", type=Path, required=True, help="the test parquet the grid was scored on")
    ap.add_argument("--emb-dir", type=Path, required=True, help="the embeddings the grid was trained and tested on")
    ap.add_argument("--caps-dir", type=Path, help="pair_redundancy.py --out-caps")
    ap.add_argument("--caps-tag", default="test_full")
    ap.add_argument("--caps", type=int, nargs="*", default=[10, 50, 100])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pairs = pl.read_parquet(args.pairs, columns=["query", "target", *TARGETS])
    n_unique = pairs.select("query", "target").unique().height
    if n_unique != pairs.height:
        raise SystemExit(f"{args.pairs}: (query, target) is not unique ({n_unique} of {pairs.height})")
    excluded = load_excluded_proteins()
    if not excluded:
        raise SystemExit("no exclusion freeze found: the replay would not match the probe loader")
    report: dict = {"inputs": {"preds": str(args.preds), "pairs": str(args.pairs), "pairs_sha256": sha256(args.pairs),
                               "pairs_rows": pairs.height, "emb_dir": str(args.emb_dir),
                               "n_excluded_freeze": len(excluded), "caps_dir": str(args.caps_dir),
                               "caps": args.caps, "seed": args.seed},
                    "replay": {}}

    arms = sorted({p.name for p in args.preds.glob("*/*/*") if p.is_dir()})
    # Arms with the same cohort see the same rows, so the replay is done once per distinct key set.
    cohort_of: dict[str, int] = {}
    cohorts: list[frozenset[str]] = []
    for arm in arms:
        with h5py.File(args.emb_dir / f"{arm}.h5", "r") as h:
            kept, summary = restrict_to_cohort(set(h.keys()), excluded)
        kept = frozenset(kept)
        if kept not in cohorts:
            cohorts.append(kept)
        cohort_of[arm] = cohorts.index(kept)
        report["replay"][f"cohort/{arm}"] = summary.kept
    report["replay"]["n_distinct_cohorts"] = len(cohorts)

    caps: dict[tuple[str, int], pl.DataFrame] = {}
    if args.caps_dir:
        for k in args.caps:
            for pop in set(CAP_POPULATION.values()):
                path = args.caps_dir / f"{args.caps_tag}_{pop}_cap{k}_seed{args.seed}.parquet"
                caps[(pop, k)] = pl.read_parquet(path, columns=["query", "target"])
                report["inputs"][f"cap_file/{pop}/cap{k}"] = {"path": str(path), "rows": caps[(pop, k)].height}

    def replay(target: str, cohort: frozenset[str]):
        """The rows one arm's archive holds, in loader order, plus every subset as a mask."""
        ids = pl.Series(sorted(cohort))
        sub = (pairs.filter(pl.col(target).is_not_null())
               .filter(pl.col("query").is_in(ids.implode()) & pl.col("target").is_in(ids.implode())).with_row_index("row"))
        y = sub[target].to_numpy().astype("float32")  # the loader's dtype, as the archive stores it
        masks = {"all": np.ones(sub.height, bool)}
        if target == "alntmscore":
            b = pide_bins(sub["fident"].fill_null(np.nan).to_numpy().astype("float64"))
            for name in ("no_pide", *PIDE_BIN_NAMES, "outside_bins"):
                if name != "outside_bins" or (b == name).any():
                    masks[f"pide_bin:{name}"] = b == name
        for k in args.caps if args.caps_dir else []:
            cap = caps[(CAP_POPULATION[target], k)]
            m = np.zeros(sub.height, bool)
            m[sub.join(cap, on=["query", "target"], how="semi")["row"].to_numpy()] = True
            if m.sum() != cap.height:
                raise SystemExit(f"{target}: {cap.height - m.sum()} capped pairs are not among the scored rows "
                                 "-- the cap was drawn on a different population")
            masks[f"cap:{k}"] = m
        return y, masks, {name: ranked(y[m]) for name, m in masks.items() if m.sum() > 2}

    rows = []
    for target in TARGETS:
        cache: dict[int, tuple] = {}
        for arm in arms:
            if cohort_of[arm] not in cache:
                cache[cohort_of[arm]] = replay(target, cohorts[cohort_of[arm]])
            y, masks, ry = cache[cohort_of[arm]]
            for ro in READ_OUTS:
                found = sorted((args.preds / ro / target / arm / "evaluation_results").glob(
                    "test_test_*_predictions_targets.npz"))
                if not found:
                    continue
                if len(found) != 1:
                    raise SystemExit(f"{ro}/{target}/{arm}: {len(found)} test prediction archives")
                z = np.load(found[0])
                if not np.array_equal(z["targets"], y):
                    raise SystemExit(f"{found[0]}: targets are not the replayed {target} column in loader order")
                pred = z["predictions"].astype("float64")
                sign = -1.0 if ro == "euclidean" else 1.0
                full = rho_from_ranks(ry["all"], pred)
                stored = metrics_spearman(found[0])
                report["replay"][f"{ro}/{target}/{arm}"] = {"n": int(y.size), "rho_replayed": full,
                                                            "rho_metrics_file": stored, "abs_diff": abs(full - stored)}
                if abs(full - stored) > 1e-6:
                    raise SystemExit(f"{ro}/{target}/{arm}: replayed rho {full} != stored {stored}")
                for name, m in masks.items():
                    r = rho_from_ranks(ry[name], pred[m]) if name in ry else np.nan
                    rows.append({"subset": name, "read_out": ro, "target": target, "arm": arm,
                                 "n_pairs": int(m.sum()), "rho": r, "score": sign * r,
                                 "y_mean": float(y[m].mean()) if m.any() else np.nan,
                                 "y_sd": float(y[m].std(ddof=1)) if m.sum() > 1 else np.nan})
            print(f"{target}/{arm}: {y.size:,} pairs, {len(masks)} subsets", flush=True)

    df = pl.DataFrame(rows)
    ranks = []
    agree = []
    for (subset, ro, target), g in df.group_by(["subset", "read_out", "target"], maintain_order=True):
        score = dict(zip(g["arm"], g["score"], strict=True))
        rk = ranks_of({a: s for a, s in score.items() if np.isfinite(s)})
        ranks += [{"subset": subset, "read_out": ro, "target": target, "arm": a, "rank_of_14": rk.get(a)}
                  for a in score]
        if subset != "all":
            base = df.filter((pl.col("subset") == "all") & (pl.col("read_out") == ro) & (pl.col("target") == target))
            agree.append({"subset": subset, "read_out": ro, "target": target,
                          "n_pairs": int(g["n_pairs"].max()), "n_pairs_all": int(base["n_pairs"].max()),
                          **agreement(dict(zip(base["arm"], base["score"], strict=True)),
                                      {a: s for a, s in score.items() if np.isfinite(s)})})
    df = df.join(pl.DataFrame(ranks, schema_overrides={"rank_of_14": pl.Int64}),
                 on=["subset", "read_out", "target", "arm"], how="left")
    ag = pl.DataFrame(agree)

    args.out.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.out / "rescore_arms.csv", float_precision=6)
    ag.write_csv(args.out / "rescore_rank_agreement.csv", float_precision=6)
    report["max_abs_diff_replayed_vs_stored_rho"] = max(
        v["abs_diff"] for k, v in report["replay"].items() if isinstance(v, dict))
    report["rank_agreement"] = ag.to_dicts()
    (args.out / "rescore_summary.json").write_text(json.dumps(report, indent=1, default=str))
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=250):
        print(ag.select("subset", "read_out", "target", "n_pairs", "rank_rho_all", "rank_rho_foundation",
                        "same_best", "max_rank_shift", "top3_all_pairs", "top3_subset"))
    print(f"wrote rescore_arms.csv, rescore_rank_agreement.csv, rescore_summary.json -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
