#!/usr/bin/env python3
"""Two checks (task D14) measured from probe_metrics.csv alone.

    uv run python scripts/paper/exptm_and_training_summary.py \
        --metrics manuscript/paper_inputs/probe_2026-10-08/probe_metrics.csv \
        --out manuscript/paper_inputs/d14_2026-10-16

**Reviewer R2.2 -- does the TM-score ranking of Figure 1B hold against EXPERIMENTAL
structures?** probe_metrics.csv holds 84 cells (14 arms x {fnn, euclidean} x 3 targets) in which
already-trained probes were scored, without retraining, on the test pairs that have an
experimental PDB chain at both ends (scripts/lrz/probe_eval_external.sbatch):

    exptm_afdbtarget   the training target itself (Foldseek alntmscore on AlphaFold models)
    exptm_matched      US-align TM-score normalised by the SHORTER chain, max(TM1, TM2): the
                       normalisation closest to alntmscore's alignment-span one
    exptm_wholechain   US-align TM-score normalised by the LONGER chain, min(TM1, TM2)

Those probes are NOT Figure 1B's. Every exptm checkpoint name equals the
sprot_pre2024_e1_sub10 checkpoint of the same arm (the reduced protocol: 10% of pairs,
patience 5); this script checks that rather than assuming it. So afdbtarget-vs-experimental is a same-probe comparison
(only the target changes), while experimental-vs-Figure-1B (full100_p10: 100% of pairs,
patience 10) also changes the probe. The Euclidean read-out trains nothing, so for it only the
pairs and the target change.

The metrics files carry no pair count. It is recovered exactly from the one arm whose p-value
does not underflow to 0 (random_1024): scipy's spearmanr/pearsonr p-value is a t-test with
n - 2 degrees of freedom, so (rho, p) fixes n.

**Reviewer R1.10 -- the final train/validation loss per model.** train.py logs
train_loss and val_loss once per epoch to an offline wandb run only; the one copy that reaches
probe_metrics.csv is the best checkpoint's file name, best-{epoch}-{step}-{val_loss:.3f}, i.e.
the validation MSE at 3 decimals (collect_probe_metrics.py parses it from there). The test MSE
of the same checkpoint is in the metrics file at full precision and is reported beside it.
Steps per epoch (= step / (epoch + 1), validation once per epoch) translate the best epoch into
passes over the training pairs; with early stopping at patience P the run ended P epochs after
the best one unless it hit max_epochs=100, which gives the epochs actually trained.

Outputs (all in --out): exptm_ranking.csv, exptm_rank_agreement.csv, training_summary.csv,
d14_exptm_training.json and README_exptm_training.md (what each file is, and the caveats).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

DISPLAY = {"prott5": "ProtT5", "prottucker": "ProtTucker", "esm1b": "ESM1b", "clean": "CLEAN",
           "esm2_8m": "ESM2 8M", "esm2_35m": "ESM2 35M", "esm2_150m": "ESM2 150M",
           "esm2_650m": "ESM2 650M", "esm2_3b": "ESM2 3B", "esm3_open": "ESM3",
           "esmc_300m": "ESMC 300M", "esmc_600m": "ESMC 600M", "ankh_base": "Ankh base",
           "ankh_large": "Ankh large", "random_1024": "Gaussian control"}
SPECIALISTS = {"clean", "prottucker"}
FOUNDATION = [a for a in DISPLAY if a not in SPECIALISTS and a != "random_1024"]  # Figure 1's twelve
PLMS = [a for a in DISPLAY if a != "random_1024"]                                 # all fourteen

EXPTM = ("exptm_afdbtarget", "exptm_matched", "exptm_wholechain")
FIG1 = "full100_p10"            # Figure 1B / Table S2 grid
SUB10 = "sprot_pre2024_e1_sub10"  # the grid the exptm probes were trained in
# From the launch records: the sprot_pre2024_e1_sub10 grids ran at patience 5; full100_p10
# and randinit_full100 at PATIENCE=10; `patience10` holds 5 cells of the 10% splits trained
# at patience 10.
PATIENCE = {SUB10: 5, FIG1: 10, "randinit_full100": 10, "patience10": 10}
MAX_EPOCHS = 100
CKPT = re.compile(r"epoch=(\d+)-step=(\d+)-val_loss=([\d.]+)")


def oriented(df: pd.DataFrame) -> pd.Series:
    """Spearman rho with higher = better: rho for the probe, -rho for a distance (the paper's magnitude)."""
    return np.where(df["model_type"] == "euclidean", -df["Spearman"], df["Spearman"])


def n_from_p(r: float, p: float) -> float:
    """The n at which a t-test on correlation r (n - 2 df) gives two-sided p."""
    def f(n):
        t = r * np.sqrt((n - 2) / (1 - r * r))
        return np.log(2) + stats.t.logsf(abs(t), n - 2) - np.log(p)
    return optimize.brentq(f, 10, 1e9, xtol=1e-6)


def ranks(score: dict, arms: list[str]) -> dict:
    """1 = best, among the listed arms that have a score."""
    order = sorted((a for a in arms if a in score), key=score.get, reverse=True)
    return {a: i + 1 for i, a in enumerate(order)}


def rank_agreement(a: dict, b: dict, arms: list[str], se_a: dict, se_b: dict) -> dict:
    """Spearman/Kendall between two per-arm score vectors over the arms both have, plus top-3s.

    n_tied_with_best counts arms whose score is within 1.96 * sqrt(se_best^2 + se_arm^2) of the
    best -- pair-bootstrap SEs treated as independent. Pairs share proteins, which makes those
    SEs too narrow, and the arms share pairs, which makes independence too wide: indicative only.
    """
    common = [x for x in arms if x in a and x in b and np.isfinite(a[x]) and np.isfinite(b[x])]
    va, vb = [a[x] for x in common], [b[x] for x in common]

    def top(d, se):
        order = sorted(common, key=d.get, reverse=True)
        best = order[0]
        tied = sum(1 for x in order[1:] if d[best] - d[x] < 1.96 * np.hypot(se.get(best, 0), se.get(x, 0)))
        return order, tied

    oa, ta = top(a, se_a)
    ob, tb = top(b, se_b)
    return {"n_models": len(common),
            "spearman_rank_rho": stats.spearmanr(va, vb)[0],
            "kendall_tau": stats.kendalltau(va, vb)[0],
            "best_a": oa[0], "best_b": ob[0],
            "top3_a": " > ".join(oa[:3]), "top3_b": " > ".join(ob[:3]),
            "n_tied_with_best_a": ta, "n_tied_with_best_b": tb,
            "missing": " ".join(x for x in arms if x not in common)}


def exptm_tables(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tm = m[m["target"] == "alntmscore"].copy()
    tm["score"] = oriented(tm)
    facts: dict = {}

    # --- which grid trained the exptm probes: compare checkpoint names arm by arm ---------------
    fnn = tm[tm["model_type"] == "fnn"].copy()
    fnn["ckpt"] = fnn["file"].map(lambda s: "-".join(CKPT.search(s).groups()) if CKPT.search(s) else np.nan)
    ck = fnn.pivot_table(index="arm", columns="dataset", values="ckpt", aggfunc="first")
    exp_arms = ck[EXPTM[0]].dropna().index
    same = {ds: int(sum(ck.loc[a, ds] == ck.loc[a, EXPTM[0]] for a in exp_arms if pd.notna(ck.loc[a].get(ds))))
            for ds in ck.columns if ds not in EXPTM}
    for ds in EXPTM[1:]:
        assert all(ck.loc[a, ds] == ck.loc[a, EXPTM[0]] for a in exp_arms), "exptm targets used different probes"
    steps = tm[(tm["model_type"] == "fnn") & tm["dataset"].isin(EXPTM + (FIG1, SUB10))].copy()
    steps["spe"] = steps["step"] / (steps["epoch"] + 1)
    facts["grid_identification"] = {
        "exptm_arms": sorted(exp_arms),
        "exptm_arms_missing": [a for a in PLMS if a not in exp_arms],
        "exptm_checkpoint_name_matches_per_dataset": {k: f"{v}/{len(exp_arms)}" for k, v in same.items()},
        "steps_per_epoch_alntmscore": {ds: sorted(g["spe"].unique().tolist()) for ds, g in steps.groupby("dataset")},
        "patience10_rerun_cells_matching_exptm": [a for a in exp_arms if pd.notna(ck.loc[a].get("patience10"))
                                                  and ck.loc[a, "patience10"] == ck.loc[a, EXPTM[0]]],
        "patience10_rerun_cells_differing_from_exptm": [a for a in exp_arms if pd.notna(ck.loc[a].get("patience10"))
                                                        and ck.loc[a, "patience10"] != ck.loc[a, EXPTM[0]]],
    }

    # --- pair counts, recovered from the random arm's p-values ----------------------------------
    rnd = tm[tm["arm"] == "random_1024"]
    facts["n_test_pairs_inferred"] = {}
    for ds, g in rnd.groupby("dataset"):
        ns = [n_from_p(r.Spearman, r.Spearman_p_value) for r in g.itertuples() if 0 < r.Spearman_p_value < 1]
        ns += [n_from_p(r.Pearson, r.Pearson_p_value) for r in g.itertuples() if 0 < r.Pearson_p_value < 1]
        if ns:
            facts["n_test_pairs_inferred"][ds] = {"n": int(round(np.median(ns))),
                                                  "max_abs_dev": float(np.max(np.abs(np.array(ns) - np.median(ns))))}

    # --- per-arm table ---------------------------------------------------------------------------
    rows, agree = [], []
    for ro in ("fnn", "euclidean"):
        d = tm[tm["model_type"] == ro]
        score = {ds: dict(zip(g["arm"], g["score"], strict=True)) for ds, g in d.groupby("dataset")}
        se = {ds: dict(zip(g["arm"], g["Spearman_SE"], strict=True)) for ds, g in d.groupby("dataset")}
        r2 = {ds: dict(zip(g["arm"], g["Pearson_r2"], strict=True)) for ds, g in d.groupby("dataset")}
        common13 = [a for a in PLMS if all(a in score[ds] for ds in EXPTM)]
        common11 = [a for a in FOUNDATION if a in common13]

        rk = {(ds, name): ranks(score[ds], arms) for ds in (FIG1, SUB10) + EXPTM
              for name, arms in (("13", common13), ("11", common11))}
        rk[(FIG1, "fig1b_12")] = ranks(score[FIG1], FOUNDATION)
        rk[(FIG1, "all_14")] = ranks(score[FIG1], PLMS)
        for a in DISPLAY:
            row = {"read_out": ro, "arm": a, "model": DISPLAY[a], "foundation": a in FOUNDATION,
                   "in_exptm": a in common13}
            for ds, tag in ((FIG1, "fig1b_full100_p10"), (SUB10, "sub10_fulltest"), *((e, e[6:]) for e in EXPTM)):
                row[f"rho_{tag}"] = score.get(ds, {}).get(a, np.nan)
                row[f"se_{tag}"] = se.get(ds, {}).get(a, np.nan)
            for e in EXPTM:
                row[f"pearson_r2_{e[6:]}"] = r2.get(e, {}).get(a, np.nan)
            row["rank_fig1b_of12"] = rk[(FIG1, "fig1b_12")].get(a)
            row["rank_fig1b_of14"] = rk[(FIG1, "all_14")].get(a)
            for ds, tag in ((FIG1, "fig1b"), (SUB10, "sub10"), *((e, e[6:]) for e in EXPTM)):
                row[f"rank_{tag}_of13"] = rk[(ds, "13")].get(a)
                row[f"rank_{tag}_of11"] = rk[(ds, "11")].get(a)
            row["delta_matched_minus_afdbtarget"] = row["rho_matched"] - row["rho_afdbtarget"]
            row["delta_wholechain_minus_afdbtarget"] = row["rho_wholechain"] - row["rho_afdbtarget"]
            if ro == "fnn" and a in ck.index and pd.notna(ck.loc[a].get(EXPTM[0])):
                ep, st, vl = ck.loc[a, EXPTM[0]].split("-")
                row.update(exptm_ckpt_epoch=int(ep), exptm_ckpt_step=int(st), exptm_ckpt_val_loss=float(vl),
                           exptm_ckpt_is_sub10=ck.loc[a, EXPTM[0]] == ck.loc[a].get(SUB10))
            rows.append(row)

        comparisons = [
            ("exptm_afdbtarget", "exptm_matched", "same probes; target AFDB -> experimental (shorter-chain norm.)"),
            ("exptm_afdbtarget", "exptm_wholechain", "same probes; target AFDB -> experimental (longer-chain norm.)"),
            ("exptm_matched", "exptm_wholechain", "same probes; normalisation only"),
            (SUB10, "exptm_afdbtarget", "same probes and target; full 10% test -> experimental-pair subset"),
            (SUB10, "exptm_matched", "same probes; full 10% test, AFDB target -> experimental subset and target"),
            (FIG1, "exptm_matched", "CROSS-GRID: Figure 1B (100%/p10) vs 10%/p5 probes on experimental (shorter)"),
            (FIG1, "exptm_wholechain", "CROSS-GRID: Figure 1B (100%/p10) vs 10%/p5 probes on experimental (longer)"),
            (FIG1, "exptm_afdbtarget", "CROSS-GRID: Figure 1B vs 10%/p5 probes, AFDB target on experimental subset"),
            (FIG1, SUB10, "grid effect alone: 100%/p10 vs 10%/p5 probes, AFDB target, full test"),
        ]
        for a_ds, b_ds, what in comparisons:
            for set_name, arms in (("foundation_12", FOUNDATION), ("all_14", PLMS)):
                r = rank_agreement(score[a_ds], score[b_ds], arms, se[a_ds], se[b_ds])
                # Only a trained read-out can cross grids; the Euclidean one is the same embeddings.
                agree.append({"read_out": ro, "a": a_ds, "b": b_ds, "model_set": set_name,
                              "cross_grid": ro == "fnn" and FIG1 in (a_ds, b_ds),
                              "what": what if ro == "fnn" else
                              what.replace("CROSS-GRID: ", "").replace("probes", "raw distances"), **r})
    return pd.DataFrame(rows), pd.DataFrame(agree), facts


def training_table(m: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    f = m[(m["model_type"] == "fnn") & m["dataset"].isin(datasets)].copy()
    f["steps_per_epoch"] = f["step"] / (f["epoch"] + 1)
    assert (f["steps_per_epoch"] % 1 == 0).all(), "a checkpoint step is not a whole number of epochs"
    ref = (m[(m["model_type"] == "fnn") & (m["dataset"] == FIG1)]
           .assign(spe=lambda x: x["step"] / (x["epoch"] + 1)).groupby("target")["spe"].agg(lambda s: s.mode()[0]))
    # The same arm under the reduced protocol (10% of pairs, patience 5), to show how far the
    # protocol moves it.
    reduced = {FIG1: SUB10, "randinit_full100": "sprot_pre2024_e1_sub10_randinit"}
    sub = m[m["model_type"] == "fnn"].set_index(["dataset", "target", "arm"])["Spearman"]
    rows = []
    for r in f.sort_values(["dataset", "target", "arm"]).itertuples():
        pat = PATIENCE[r.dataset]
        ep, spe = int(r.epoch), int(r.steps_per_epoch)
        ran = min(ep + 1 + pat, MAX_EPOCHS)
        twin = re.sub(r"^random_init_(.*)_seed\d+$", r"\1", r.arm).replace("prot_t5", "prott5")
        s10 = sub.get((reduced.get(r.dataset), r.target, r.arm), np.nan)
        rows.append({
            "dataset": r.dataset, "target": r.target, "arm": r.arm,
            "model": DISPLAY.get(twin, twin) + (" (untrained)" if r.arm.startswith("random_init") else ""),
            "patience": pat, "best_epoch": ep, "passes_to_best": ep + 1, "best_step": int(r.step),
            "steps_per_epoch": spe, "train_pairs_min": (spe - 1) * 1024 + 1, "train_pairs_max": spe * 1024,
            "steps_per_epoch_pretrained_grid": int(ref[r.target]),
            "extra_batches_vs_pretrained_grid": spe - int(ref[r.target]),
            "epochs_trained_inferred": ran, "stopped_by": "max_epochs" if ran == MAX_EPOCHS else "early_stopping",
            "best_epoch_le1": ep <= 1,
            "val_loss_mse_3dp": float(r.val_loss), "test_mse": float(r.MSE), "test_rmse": float(r.RMSE),
            "test_spearman": float(r.Spearman),
            "test_spearman_10pct_p5_grid": s10, "delta_vs_10pct_p5_grid": float(r.Spearman) - s10,
            "train_loss": np.nan,  # logged to the offline wandb run on LRZ only; see the JSON
        })
    return pd.DataFrame(rows)


README = """# D14 -- experimental-TM ranking (R2.2) and per-probe training summary (R1.10)

Written by `scripts/paper/exptm_and_training_summary.py` (public repo) from `probe_metrics.csv` alone:

    uv run python scripts/paper/exptm_and_training_summary.py \\
        --metrics manuscript/paper_inputs/probe_2026-10-08/probe_metrics.csv --out <this dir>

Score = Spearman rho, higher is better (-rho for the Euclidean read-out: the paper's magnitude).

| file | what |
|---|---|
| `exptm_ranking.csv` | read-out x arm: rho on Figure 1B (`full100_p10`), on the 10%/p5 grid's full test, and on the experimental pairs under the three targets; SEs, Pearson R2, ranks among the 13 / 11 models in common, AFDB-to-experimental drop, the exptm checkpoint |
| `exptm_rank_agreement.csv` | rank agreement (Spearman, Kendall) between two conditions over the 12 foundation models or all 14 (ESM-2 3B has no exptm cell, so 11 / 13 in common), top-3 of each; `cross_grid` marks FNN comparisons that change the probe |
| `training_summary.csv` | FNN cell of `full100_p10` / `randinit_full100`: best epoch, step, steps/epoch, train-pair bounds, val MSE (3 dp, checkpoint name), test MSE (full precision), inferred epochs trained, the same arm's rho under the reduced protocol (10%/p5) |
| `d14_exptm_training.json` | grid identification, inferred pair counts, patience check, 100%/p10-vs-10%/p5 deltas, FNN vs Euclidean on exptm, flagged cells, where the train loss lives |

Read before quoting:

- The exptm cells are the **10%-of-pairs / patience-5 probes** (`sprot_pre2024_e1_sub10`), not Figure 1's:
  checkpoint names match {sub10_match} arms of that grid and {fig_match} of `full100_p10`; {spe_exp} steps/epoch
  against {spe_fig}. Missing from exptm: {missing}.
- Pairs: **{n_afdb:,}** (`exptm_afdbtarget`) and **{n_exp:,}** (`exptm_matched`, `exptm_wholechain`), recovered
  exactly from the random arm's p-values -- not 10,400 / 10,740, and not one identical pair set. Whether the
  first is a subset of the second cannot be checked locally (the parquets are on LRZ).
- `matched` = US-align TM normalised by the SHORTER chain, max(TM1, TM2); `wholechain` = by the LONGER chain,
  min(TM1, TM2) (`scripts/lrz/probe_eval_external.sbatch`, `src/evaluation/analyze_experimental_tm.py`). The
  script that wrote the exptm parquets is not in the repo, so this rests on those two descriptions.
- SEs are pair-bootstrap, too narrow because pairs share proteins; `n_tied_with_best` is indicative only.
- `val_loss_mse_3dp` is the checkpoint name's 3-decimal value; HFSP losses are in HFSP^2 units.
  `epochs_trained_inferred` = best epoch + 1 + patience (no cell reaches max_epochs 100): inferred from the
  early-stopping rule, not observed.
- **Train loss is not recoverable locally.** It was logged only to the offline wandb runs on LRZ
  (`models/<dataset>/fnn/<target>/<arm>/wandb/`, excluded from the Zenodo tars); the SLURM logs
  `logs/probe-grid-<job>_<idx>.out` carry it in the progress bar and print `Best validation loss` at 6 dp.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", type=Path, required=True, help="probe_metrics.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--training-datasets", nargs="+", default=[FIG1, "randinit_full100"])
    args = ap.parse_args()

    m = pd.read_csv(args.metrics)
    ranking, agree, facts = exptm_tables(m)
    train = training_table(m, args.training_datasets)

    # Side checks that the text leans on.
    pp = m[(m["model_type"] == "fnn") & m["dataset"].isin([SUB10, "patience10"])]
    pp = pp.pivot_table(index=["target", "arm"], columns="dataset", values="Spearman").dropna()
    facts["patience5_vs_patience10_same_10pct_pairs"] = {
        f"{t}/{a}": {"p5": round(r[SUB10], 4), "p10": round(r["patience10"], 4),
                     "delta": round(r["patience10"] - r[SUB10], 4)} for (t, a), r in pp.iterrows()}
    g = m[(m["dataset"].isin([FIG1, SUB10])) & (m["arm"].isin(PLMS))]
    g = g.pivot_table(index=["model_type", "target", "arm"], columns="dataset", values="Spearman")
    d = (g[FIG1] - g[SUB10]).abs().groupby(level=["model_type", "target"])
    facts["protocol_abs_delta_rho_100p10_vs_10p5"] = {
        f"{mt}/{t}": {"median": round(float(s.median()), 4), "max": round(float(s.max()), 4),
                      "argmax": s.droplevel([0, 1]).idxmax(), "n_arms": int(s.size)} for (mt, t), s in d}
    # Against what the protocol moves a cell by: the gaps between neighbouring models in Figure 1.
    fig = m[(m["dataset"] == FIG1) & (m["model_type"] == "fnn") & m["arm"].isin(FOUNDATION)]
    facts["fig1_fnn_adjacent_rank_gap_12_foundation"] = {
        t: {"median": round(float(np.median(-np.diff(np.sort(g["Spearman"].to_numpy())[::-1]))), 4),
            "min": round(float(np.min(-np.diff(np.sort(g["Spearman"].to_numpy())[::-1]))), 4),
            "max": round(float(np.max(-np.diff(np.sort(g["Spearman"].to_numpy())[::-1]))), 4)}
        for t, g in fig.groupby("target")}
    fe = ranking.pivot_table(index="arm", columns="read_out", values=["rho_matched", "rho_wholechain", "rho_afdbtarget"])
    fe = fe[fe.index.isin(PLMS)].dropna()
    facts["fnn_minus_euclidean_on_exptm"] = {
        t: {"min": round(float((fe[(f"rho_{t}", "fnn")] - fe[(f"rho_{t}", "euclidean")]).min()), 4),
            "max": round(float((fe[(f"rho_{t}", "fnn")] - fe[(f"rho_{t}", "euclidean")]).max()), 4),
            "arms_where_euclidean_ge_fnn": sorted(fe.index[fe[(f"rho_{t}", "euclidean")] >= fe[(f"rho_{t}", "fnn")]])}
        for t in ("afdbtarget", "matched", "wholechain")}
    facts["euclidean_rho_up_from_afdbtarget_to_matched"] = sorted(
        ranking[(ranking.read_out == "euclidean") & (ranking.delta_matched_minus_afdbtarget > 0)
                & ranking.arm.isin(PLMS)]["arm"])
    tr = train[~train["arm"].str.contains("random_1024")]
    facts["training"] = {
        "cells": int(len(train)),
        "best_epoch_le1": [f"{r.dataset}/{r.target}/{r.arm} (epoch {r.best_epoch})"
                           for r in train[train.best_epoch_le1].itertuples()],
        "best_epoch_range_excl_random": {f"{ds}/{t}": [int(g.best_epoch.min()), int(g.best_epoch.max())]
                                         for (ds, t), g in tr.groupby(["dataset", "target"])},
        "stopped_by_max_epochs": int((train.stopped_by == "max_epochs").sum()),
        "steps_per_epoch": {f"{ds}/{t}": sorted(set(g.steps_per_epoch)) for (ds, t), g in train.groupby(["dataset", "target"])},
        "off_pretrained_cohort_arms": sorted(set(train.loc[train.extra_batches_vs_pretrained_grid != 0, "arm"])),
        # Per target: HFSP's MSE is in HFSP units (tens), PIDE's and TM's in [0, 1] units.
        "val_vs_test_mse_spearman_per_target": {
            t: round(float(stats.spearmanr(g.val_loss_mse_3dp, g.test_mse)[0]), 4) for t, g in tr.groupby("target")},
        "train_loss_source": "train.py logs train_loss/val_loss per epoch to WandbLogger only, offline on LRZ "
                             "(models/<dataset>/fnn/<target>/<arm>/wandb/); the SLURM logs "
                             "logs/probe-grid-<job>_<idx>.out carry the progress bar and 'Best validation "
                             "loss: x.xxxxxx'. Neither is local; wandb/ was excluded from the Zenodo tars.",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out / "exptm_ranking.csv", index=False, float_format="%.6g")
    agree.to_csv(args.out / "exptm_rank_agreement.csv", index=False, float_format="%.6g")
    train.to_csv(args.out / "training_summary.csv", index=False, float_format="%.6g")
    facts["rank_agreement"] = agree.to_dict(orient="records")
    (args.out / "d14_exptm_training.json").write_text(json.dumps(facts, indent=1, default=str))
    gi, nt = facts["grid_identification"], facts["n_test_pairs_inferred"]
    (args.out / "README_exptm_training.md").write_text(README.format(
        sub10_match=gi["exptm_checkpoint_name_matches_per_dataset"][SUB10],
        fig_match=gi["exptm_checkpoint_name_matches_per_dataset"][FIG1],
        spe_exp=int(gi["steps_per_epoch_alntmscore"][EXPTM[0]][0]),
        spe_fig=int(gi["steps_per_epoch_alntmscore"][FIG1][0]),
        missing=", ".join(gi["exptm_arms_missing"]) or "none",
        n_afdb=nt[EXPTM[0]]["n"], n_exp=nt[EXPTM[1]]["n"]))

    show = agree[agree.a.isin(EXPTM[:1] + (FIG1, SUB10)) & (agree.model_set == "all_14")]
    print(show[["read_out", "a", "b", "n_models", "spearman_rank_rho", "kendall_tau", "top3_a", "top3_b"]]
          .to_string(index=False, float_format="%.3f"))
    print(f"wrote exptm_ranking.csv, exptm_rank_agreement.csv, training_summary.csv, "
          f"d14_exptm_training.json, README_exptm_training.md -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
