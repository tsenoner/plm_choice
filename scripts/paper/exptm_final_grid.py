#!/usr/bin/env python3
"""Does Figure 1B's TM-score ranking hold against EXPERIMENTAL structures? On the paper's own probes.

    uv run python scripts/paper/exptm_final_grid.py --metrics <probe_metrics.csv> --out <dir>

Reviewer R2.2 (task D14, ledger M-47). exptm_and_training_summary.py answered this from the
84 exptm_* cells, which are the 10%-of-pairs / patience-5 probes, so the answer could not go in
the paper (it never mixes grids, M-26). scripts/lrz/probe_eval_external.sbatch with
SRC_DATASET=full100_p10 scored Figure 1B's probes (ESM-2 3B included) on the same three
experimental-TM pair sets, without retraining, into exptm_{afdbtarget,matched,wholechain}_full100_p10:

    afdbtarget   the training target itself (Foldseek alntmscore on AlphaFold models), on the
                 experimental pairs -- isolates the change of pairs from the change of target
    matched      US-align TM-score normalised by the SHORTER chain (closest to alntmscore's
                 alignment-span normalisation)
    wholechain   US-align TM-score normalised by the LONGER chain

This checks that every exptm FNN checkpoint is the full100_p10 checkpoint of the same arm, then
compares per-arm Spearman rho (higher = better: -rho for the Euclidean read-out) between Figure
1B (full test split) and each exptm set, for the FNN probe and the raw Euclidean distance.

Outputs (--out): exptm_final_ranking.csv, exptm_final_rank_agreement.csv, exptm_final.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exptm_and_training_summary import (  # noqa: E402
    CKPT,
    DISPLAY,
    FOUNDATION,
    PLMS,
    n_from_p,
    oriented,
    rank_agreement,
    ranks,
)

FIG1 = "full100_p10"
SETS = ("afdbtarget", "matched", "wholechain")
EXPTM = {s: f"exptm_{s}_{FIG1}" for s in SETS}
OLD = {s: f"exptm_{s}" for s in SETS}  # the 10%/patience-5 probes on the same pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", type=Path, required=True, help="probe_metrics.csv with the *_full100_p10 exptm cells")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    m = pd.read_csv(args.metrics)
    tm = m[m["target"] == "alntmscore"].copy()
    tm["score"] = oriented(tm)
    missing = [d for d in EXPTM.values() if d not in set(tm["dataset"])]
    if missing:
        raise SystemExit(f"{args.metrics}: no cells for {missing}")
    facts: dict = {"metrics": str(args.metrics)}

    # --- the exptm probes must be Figure 1B's: same checkpoint name, arm by arm -----------------
    fnn = tm[tm["model_type"] == "fnn"].copy()
    fnn["ckpt"] = fnn["file"].map(lambda s: "-".join(g.groups()) if (g := CKPT.search(s)) else np.nan)
    ck = fnn.pivot_table(index="arm", columns="dataset", values="ckpt", aggfunc="first")
    for ds in EXPTM.values():
        bad = [a for a in ck[ds].dropna().index if ck.loc[a, ds] != ck.loc[a, FIG1]]
        if bad:
            raise SystemExit(f"{ds}: checkpoints differ from {FIG1} for {bad}")
    facts["exptm_arms"] = {s: sorted(ck[ds].dropna().index) for s, ds in EXPTM.items()}
    facts["exptm_arms_missing"] = {s: [a for a in DISPLAY if a not in ck[ds].dropna().index] for s, ds in EXPTM.items()}

    # --- pair counts, recovered from the random arm's p-values (the metrics carry no n) ---------
    facts["n_pairs_inferred"] = {}
    for ds in (*EXPTM.values(), *OLD.values()):
        r = tm[(tm["dataset"] == ds) & (tm["arm"] == "random_1024")]
        ns = [n_from_p(x.Spearman, x.Spearman_p_value) for x in r.itertuples() if 0 < x.Spearman_p_value < 1]
        ns += [n_from_p(x.Pearson, x.Pearson_p_value) for x in r.itertuples() if 0 < x.Pearson_p_value < 1]
        if ns:
            facts["n_pairs_inferred"][ds] = int(round(np.median(ns)))

    rows, agree = [], []
    for ro in ("fnn", "euclidean"):
        d = tm[tm["model_type"] == ro]
        score = {ds: dict(zip(g["arm"], g["score"], strict=True)) for ds, g in d.groupby("dataset")}
        se = {ds: dict(zip(g["arm"], g["Spearman_SE"], strict=True)) for ds, g in d.groupby("dataset")}
        rk = {ds: ranks(score[ds], PLMS) for ds in (FIG1, *EXPTM.values())}
        for a in DISPLAY:
            row = {"read_out": ro, "arm": a, "model": DISPLAY[a], "foundation": a in FOUNDATION,
                   "rho_fig1b": score[FIG1].get(a, np.nan), "se_fig1b": se[FIG1].get(a, np.nan),
                   "rank_fig1b_of14": rk[FIG1].get(a)}
            for s, ds in EXPTM.items():
                row[f"rho_{s}"] = score[ds].get(a, np.nan)
                row[f"se_{s}"] = se[ds].get(a, np.nan)
                row[f"rank_{s}_of14"] = rk[ds].get(a)
                row[f"rho_{s}_10pct_p5_probes"] = score.get(OLD[s], {}).get(a, np.nan)
            row["delta_matched_minus_afdbtarget"] = row["rho_matched"] - row["rho_afdbtarget"]
            row["delta_wholechain_minus_afdbtarget"] = row["rho_wholechain"] - row["rho_afdbtarget"]
            rows.append(row)
        comparisons = [(FIG1, EXPTM["afdbtarget"], "Figure 1B -> experimental pairs, same AFDB target"),
                       (FIG1, EXPTM["matched"], "Figure 1B -> experimental pairs and target (shorter chain)"),
                       (FIG1, EXPTM["wholechain"], "Figure 1B -> experimental pairs and target (longer chain)"),
                       (EXPTM["afdbtarget"], EXPTM["matched"], "same pairs; AFDB -> experimental target (shorter)"),
                       (EXPTM["afdbtarget"], EXPTM["wholechain"], "same pairs; AFDB -> experimental target (longer)"),
                       (EXPTM["matched"], EXPTM["wholechain"], "same pairs; normalisation only")]
        if ro == "fnn":
            comparisons += [(OLD[s], EXPTM[s], f"grid change on {s}: 10%/p5 -> 100%/p10 probes")
                            for s in SETS if OLD[s] in score]
        for a_ds, b_ds, what in comparisons:
            for set_name, arms in (("foundation_12", FOUNDATION), ("all_14", PLMS)):
                agree.append({"read_out": ro, "a": a_ds, "b": b_ds, "model_set": set_name, "what": what,
                              **rank_agreement(score[a_ds], score[b_ds], arms, se[a_ds], se[b_ds])})

    ranking, agreement = pd.DataFrame(rows), pd.DataFrame(agree)
    plm = ranking[ranking["arm"].isin(PLMS)]
    for ro, g in plm.groupby("read_out"):
        facts[f"{ro}/drop_afdbtarget_to_matched"] = [float(g.delta_matched_minus_afdbtarget.min()),
                                                     float(g.delta_matched_minus_afdbtarget.max())]
        facts[f"{ro}/drop_afdbtarget_to_wholechain"] = [float(g.delta_wholechain_minus_afdbtarget.min()),
                                                        float(g.delta_wholechain_minus_afdbtarget.max())]
        facts[f"{ro}/drop_fig1b_to_matched"] = [float((g.rho_matched - g.rho_fig1b).min()),
                                                float((g.rho_matched - g.rho_fig1b).max())]
        for s in SETS:
            facts[f"{ro}/rho_{s}_range"] = [float(g[f"rho_{s}"].min()), float(g[f"rho_{s}"].max())]
    w = plm.pivot_table(index="arm", columns="read_out", values=[f"rho_{s}" for s in SETS])
    facts["fnn_minus_euclidean"] = {s: [float((w[(f"rho_{s}", "fnn")] - w[(f"rho_{s}", "euclidean")]).min()),
                                        float((w[(f"rho_{s}", "fnn")] - w[(f"rho_{s}", "euclidean")]).max())]
                                    for s in SETS}
    facts["rank_agreement"] = agreement.to_dict(orient="records")

    args.out.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out / "exptm_final_ranking.csv", index=False, float_format="%.6g")
    agreement.to_csv(args.out / "exptm_final_rank_agreement.csv", index=False, float_format="%.6g")
    (args.out / "exptm_final.json").write_text(json.dumps(facts, indent=1, default=str))
    show = agreement[agreement["model_set"] == "all_14"]
    print(show[["read_out", "a", "b", "n_models", "spearman_rank_rho", "kendall_tau", "top3_a", "top3_b"]]
          .to_string(index=False, float_format="%.3f"))
    print(json.dumps({k: v for k, v in facts.items() if k != "rank_agreement"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
