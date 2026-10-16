#!/usr/bin/env python3
"""Re-derive, in Spearman rho on the final probe grid, a set of text claims (task D14).

    PAPER_ARTEFACTS=<manuscript>/paper_inputs MANUSCRIPT_DIR=<manuscript>/bib_2026 \
        uv run python scripts/paper/spearman_facts.py --out <paper_inputs>/d14_2026-10-16

Why: a set of proposed text edits (task D14) was written against a version that still
reported Pearson R2 on the 10%-of-pairs / patience-5 grid. They claim that the two
task-specific models (CLEAN = contrastive head on ESM-1b, ProtTucker = contrastive head
on ProtT5) lead the raw-distance ranking, that their probes add little over their raw
distances, and that the PIDE and HFSP probes saturate. The paper now reports one score,
Spearman rho (for the Euclidean read-out its magnitude), from the 100%-of-pairs /
patience-10 grid. Each claim has to be measured again on that grid before any of it can
go into the manuscript, and the Results sentences that quote rho ranges are re-checked
against the same numbers.

What it measures, from probe_metrics.csv (dataset full100_p10, random_1024 excluded from
every ranking and range; it is a control, not a candidate):
  1. the raw-distance (|Euclidean rho|) ranking of the 14 pLMs per target, the ranks of
     CLEAN and ProtTucker, and ties at two decimals (the precision the paper prints);
  2. the probe-minus-raw gain, FNN rho - |Euclidean rho|, per model and target;
  3. saturation: max/min/spread of FNN rho, models within 0.02 of the best, spread of
     the top five, headroom 1 - max;
  4. the min-max ranges quoted in Results, over all 14 pLMs and over the 12 foundation
     models, and which population each printed range matches;
  5. the best non-specialist raw distance per target;
plus a cell-by-cell check of Table S2 (as generated, and as printed in the supplement)
against the CSV, and a few further numbers quoted in the same Results/Discussion
paragraphs.

Outputs (in --out): spearman_facts.json, spearman_facts_per_model.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


def _need(var: str) -> str:
    """The value of `var`, or a message naming what to set."""
    try:
        return os.environ[var]
    except KeyError:
        raise SystemExit(f"set {var} before running this script") from None


DATASET = "full100_p10"
RANDINIT = "randinit_full100"
CONTROL = "random_1024"
TARGETS = {"fident": "PIDE", "alntmscore": "TM-score", "hfsp": "HFSP"}
SPECIALISTS = {"clean": "esm1b", "prottucker": "prott5"}  # specialist -> its parent
# Display label as Table S2 prints it -> arm key in probe_metrics.csv.
LABELS = {
    "Ankh Base": "ankh_base", "Ankh Large": "ankh_large", "CLEAN": "clean",
    "ESM-1b": "esm1b", "ESM2-8M": "esm2_8m", "ESM2-35M": "esm2_35m",
    "ESM2-150M": "esm2_150m", "ESM2-650M": "esm2_650m", "ESM2-3B": "esm2_3b",
    "ESM3": "esm3_open", "ESM C-300M": "esmc_300m", "ESM C-600M": "esmc_600m",
    "ProtT5": "prott5", "ProtTucker": "prottucker", "Random": "random_1024",
}
NAME = {v: k for k, v in LABELS.items()}
BLOCK = {"PIDE": "fident", "TM-score": "alntmscore", "HFSP": "hfsp"}

# Ranges printed in 04.results.md, "Supervised extraction" paragraph (probe vs raw).
PRINTED_RANGES = {
    "fident": {"fnn": "0.76-0.95", "euclidean": "0.59-0.77"},
    "alntmscore": {"fnn": "0.53-0.80", "euclidean": "0.45-0.63"},
    "hfsp": {"fnn": "0.69-0.94", "euclidean": "0.62-0.78"},
}
EPS = 1e-12


def r4(x: float) -> float:
    return round(float(x), 4)


def f2(x: float) -> str:
    return f"{x:.2f}"


def load(metrics: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(metrics)
    d = df[df.dataset == DATASET].copy()
    d["score"] = d["Spearman"].abs()  # |rho|; FNN rho is positive, Euclidean negative
    wide = d.pivot_table(index="arm", columns=["model_type", "target"], values="score")
    if wide.isna().any().any() or len(wide) != 15:
        raise SystemExit(f"{DATASET}: expected 15 arms x 6 cells, got\n{wide}")
    return df, wide


def ranking(s: pd.Series) -> list[dict]:
    """Descending ranking with the rank by exact value and the tie group at 2 decimals."""
    s = s.sort_values(ascending=False)
    rounded = s.map(f2)
    out = []
    for i, (arm, v) in enumerate(s.items(), 1):
        same = [NAME[a] for a in s.index if rounded[a] == rounded[arm]]
        first = 1 + sum(rounded[a] > rounded[arm] for a in s.index)  # competition rank at 2 dp
        out.append({
            "rank": i, "model": NAME[arm], "arm": arm, "rho": r4(v), "rho_2dp": rounded[arm],
            "rank_2dp": first if len(same) == 1 else f"{first}-{first + len(same) - 1} (tie)",
            "tied_at_2dp_with": [m for m in same if m != NAME[arm]],
        })
    return out


def rng_str(s: pd.Series) -> str:
    return f"{f2(s.min())}-{f2(s.max())}"


def parse_table(text: str) -> dict[tuple[str, str, str], tuple[str, str]]:
    """{(target, read-out, arm): (value, mark)} from a pandoc grid table; mark is best/second/''."""
    cells, block = {}, None
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != 4 or parts[1] == "Embedding":
            continue
        param, emb, a, b = parts
        if param:
            block = BLOCK.get(param)
        if block is None or emb not in LABELS:
            continue
        for mt, raw in (("euclidean", a), ("fnn", b)):
            m = re.search(r"(\d\.\d\d)", raw)
            mark = "best" if raw.startswith("**") else "second" if raw.startswith("*") else ""
            cells[(block, mt, LABELS[emb])] = (m.group(1) if m else raw, mark)
    return cells


def check_table(path: Path, wide: pd.DataFrame, section: str | None = None) -> dict:
    text = path.read_text()
    if section:  # cut the tbls:performance grid table out of the supplement
        lines, seen = [], False
        for line in text[text.index(section):].splitlines():
            if line.startswith(("+", "|")):
                seen = True
                lines.append(line)
            elif seen:
                break
        text = "\n".join(lines)
    cells = parse_table(text)
    mismatches = []
    for (tgt, mt, arm), (val, mark) in cells.items():
        want = f2(wide[(mt, tgt)][arm])
        if val != want:
            mismatches.append({"cell": f"{TARGETS[tgt]}/{mt}/{NAME[arm]}", "printed": val, "csv": want})
        ranked = wide[(mt, tgt)].drop(CONTROL).sort_values(ascending=False).index
        want_mark = "best" if arm == ranked[0] else "second" if arm == ranked[1] else ""
        if mark != want_mark:
            mismatches.append({"cell": f"{TARGETS[tgt]}/{mt}/{NAME[arm]}", "printed_mark": mark,
                               "csv_mark": want_mark})
    return {"file": str(path), "cells_parsed": len(cells), "mismatches": mismatches}


def readme(f: dict) -> list[str]:
    """The headline answers, generated from the facts so they cannot drift from them."""
    q1, q2, q3, q4 = (f[k] for k in ("q1_raw_distance_ranking", "q2_probe_minus_raw_gain",
                                     "q3_saturation_fnn", "q4_results_ranges"))
    labs = list(TARGETS.values())
    sat = q3["14_pLMs"]
    return [
        "Made by scripts/paper/spearman_facts.py (public repo); outputs spearman_facts.json and "
        "spearman_facts_per_model.csv. Score: Spearman rho, |rho| for Euclidean; 14 pLMs, random_1024 excluded.",
        "Raw-distance ranks (PIDE/TM/HFSP): CLEAN " + "/".join(str(q1[x]["CLEAN_rank"]) for x in labs)
        + ", ProtTucker " + "/".join(str(q1[x]["ProtTucker_rank"]) for x in labs)
        + f"; top four on every target: {q1['claim_top4_on_every_target']}.",
        "FNN - raw gain, CLEAN vs ESM-1b: " + "; ".join(
            f"{x} {q2[x]['CLEAN']:.3f} vs {q2[x]['ESM-1b']:.3f}" for x in labs) + ".",
        "FNN top-5 spread: " + "; ".join(f"{x} {sat[x]['top5_spread']:.3f} (max {sat[x]['max']:.3f})"
                                         for x in labs) + ".",
        "Results ranges matching neither 14 nor 12 models: "
        + (", ".join(k for k, v in q4.items() if v["matches"] == ["NEITHER"]) or "none")
        + "; matching only the 12 foundation models: "
        + (", ".join(k for k, v in q4.items() if v["matches"] == ["foundation_12"]) or "none") + ".",
        "Table S2 mismatches: " + ", ".join(f"{Path(c['file']).name} {len(c['mismatches'])}"
                                            for c in f["tableS2_check"]) + ".",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    artefacts, manuscript = Path(_need("PAPER_ARTEFACTS")), Path(_need("MANUSCRIPT_DIR"))
    metrics = a.metrics or artefacts / "probe_2026-10-08/probe_metrics.csv"
    df, wide = load(metrics)
    plms = wide.drop(CONTROL)
    found = plms.drop(list(SPECIALISTS))
    facts: dict = {
        "source": str(metrics), "dataset": DATASET,
        "score": "Spearman rho; |rho| for the Euclidean read-out",
        "population": {
            "models": "14 pLMs (12 foundation + CLEAN + ProtTucker); random_1024 excluded",
            "pairs": "Swiss-Prot pre-2024 test split; per Supplementary Methods 4,867,244 PIDE/HFSP "
                     "and 7,973,695 TM-score pairs before the <=1022-residue cohort filter (1.2% of all "
                     "pairs removed at load). The evaluated n per arm is not recorded in probe_metrics.csv.",
            "max_bootstrap_SE": round(float(df[(df.dataset == DATASET) & (df.arm != CONTROL)]
                                            ["Spearman_SE"].max()), 6),
        },
    }

    # 1. raw-distance ranking -------------------------------------------------------
    q1 = {}
    for tgt, lab in TARGETS.items():
        rk = ranking(plms[("euclidean", tgt)])
        pos = {r["arm"]: r for r in rk}
        q1[lab] = {"ranking": rk, **{f"{NAME[s]}_rank": pos[s]["rank"] for s in SPECIALISTS},
                   **{f"{NAME[s]}_rank_2dp": pos[s]["rank_2dp"] for s in SPECIALISTS},
                   "leader": rk[0]["model"]}
    worst = {NAME[s]: max(q1[lab][f"{NAME[s]}_rank"] for lab in q1) for s in SPECIALISTS}
    q1["claim_top4_on_every_target"] = all(v <= 4 for v in worst.values())
    q1["worst_rank_per_specialist"] = worst
    q1["specialists_top_k_on_every_target"] = max(worst.values())
    facts["q1_raw_distance_ranking"] = q1

    # 2. probe-minus-raw gain -------------------------------------------------------
    gain = plms["fnn"] - plms["euclidean"]
    q2 = {lab: {NAME[m]: r4(gain.loc[m, tgt]) for m in gain.index} for tgt, lab in TARGETS.items()}
    q2["specialist_vs_parent"] = {
        TARGETS[tgt]: {
            f"{NAME[s]}_vs_{NAME[p]}": {
                "fnn": [r4(plms.loc[s, ("fnn", tgt)]), r4(plms.loc[p, ("fnn", tgt)])],
                "euclidean": [r4(plms.loc[s, ("euclidean", tgt)]), r4(plms.loc[p, ("euclidean", tgt)])],
                "gain": [r4(gain.loc[s, tgt]), r4(gain.loc[p, tgt])],
            } for s, p in SPECIALISTS.items()
        } for tgt in TARGETS
    }
    q2["clean_fnn_below_esm1b_on_all_targets"] = bool(
        (plms.loc["clean", "fnn"] < plms.loc["esm1b", "fnn"]).all())
    q2["prottucker_fnn_below_prott5_on_all_targets"] = bool(
        (plms.loc["prottucker", "fnn"] < plms.loc["prott5", "fnn"]).all())
    q2["smallest_gain"] = {lab: {"model": NAME[gain[tgt].idxmin()], "gain": r4(gain[tgt].min())}
                           for tgt, lab in TARGETS.items()}
    q2["every_pLM_gain_positive"] = bool((gain > 0).all().all())
    # 1 = smallest gain of the 14; "headroom lost" should show as a low rank.
    q2["gain_rank_ascending_of_14"] = {
        TARGETS[t]: {NAME[sp]: int(gain[t].rank()[sp]) for sp in SPECIALISTS} for t in TARGETS}
    q2["specialist_minus_parent_raw"] = {
        TARGETS[t]: {f"{NAME[sp]}-{NAME[par]}": r4(plms.loc[sp, ("euclidean", t)] - plms.loc[par, ("euclidean", t)])
                     for sp, par in SPECIALISTS.items()} for t in TARGETS}
    # The control's signed rho: both read-outs sit at zero, so its "gain" is noise.
    ctrl = df[(df.dataset == DATASET) & (df.arm == CONTROL)]
    signed: dict = {}
    for r in ctrl.itertuples():
        signed.setdefault(TARGETS[r.target], {})[r.model_type] = round(float(r.Spearman), 6)
    q2["control_random_1024_signed_rho"] = signed
    facts["q2_probe_minus_raw_gain"] = q2

    # 3. saturation -----------------------------------------------------------------
    q3 = {}
    for pop_name, pop in (("14_pLMs", plms), ("12_foundation", found)):
        for tgt, lab in TARGETS.items():
            s = pop[("fnn", tgt)].sort_values(ascending=False)
            near = s[s >= s.iloc[0] - 0.02 - EPS]
            q3.setdefault(pop_name, {})[lab] = {
                "max": r4(s.iloc[0]), "max_model": NAME[s.index[0]],
                "min": r4(s.iloc[-1]), "min_model": NAME[s.index[-1]],
                "spread_max_minus_min": r4(s.iloc[0] - s.iloc[-1]),
                "sd_over_models": r4(s.std(ddof=1)),
                "headroom_1_minus_max": r4(1 - s.iloc[0]),
                "n_within_0.02_of_best_incl_best": len(near),
                "within_0.02": {NAME[m]: r4(v) for m, v in near.items()},
                "gap_1st_to_2nd": r4(s.iloc[0] - s.iloc[1]),
                "top5": {NAME[m]: r4(v) for m, v in s.iloc[:5].items()},
                "top5_spread": r4(s.iloc[0] - s.iloc[4]),
            }
    facts["q3_saturation_fnn"] = q3

    # 4. Results ranges -------------------------------------------------------------
    q4 = {}
    for tgt, lab in TARGETS.items():
        for mt in ("fnn", "euclidean"):
            r14, r12 = rng_str(plms[(mt, tgt)]), rng_str(found[(mt, tgt)])
            printed = PRINTED_RANGES[tgt][mt]
            q4[f"{lab}/{mt}"] = {
                "printed": printed, "all_14": r14, "foundation_12": r12,
                "all_14_exact": [r4(plms[(mt, tgt)].min()), r4(plms[(mt, tgt)].max())],
                "foundation_12_exact": [r4(found[(mt, tgt)].min()), r4(found[(mt, tgt)].max())],
                "matches": [p for p, r in (("all_14", r14), ("foundation_12", r12)) if r == printed]
                or ["NEITHER"],
            }
    facts["q4_results_ranges"] = q4

    # 5. best non-specialist raw distance -------------------------------------------
    q5 = {}
    for tgt, lab in TARGETS.items():
        s = found[("euclidean", tgt)].sort_values(ascending=False)
        q5[lab] = {"best": NAME[s.index[0]], "rho": r4(s.iloc[0]),
                   "second": NAME[s.index[1]], "second_rho": r4(s.iloc[1])}
    facts["q5_best_foundation_raw"] = q5

    # Further numbers quoted in the same paragraphs ---------------------------------
    rnd = df[df.dataset == RANDINIT].copy()
    rnd["arm"] = (rnd.arm.str.removeprefix("random_init_").str.removesuffix("_seed0")
                  .replace({"prot_t5": "prott5"}))
    rnd["score"] = rnd["Spearman"].abs()
    rw = rnd.pivot_table(index="arm", columns=["model_type", "target"], values="score")
    twins = sorted(set(rw.index) & set(found.index))
    extra = {
        "randinit_twins": [NAME[t] for t in twins],
        "randinit_esm2_3b_euclid_PIDE": r4(rw.loc["esm2_3b", ("euclidean", "fident")]),
        "esm2_35m_euclid_PIDE": r4(plms.loc["esm2_35m", ("euclidean", "fident")]),
        "esm3_euclid_PIDE": r4(plms.loc["esm3_open", ("euclidean", "fident")]),
        "pretraining_gain_euclid_mean": {
            TARGETS[t]: r4(found.loc[twins, ("euclidean", t)].mean() - rw.loc[twins, ("euclidean", t)].mean())
            for t in TARGETS},
        "fnn_rank_TM_foundation": [NAME[m] for m in found[("fnn", "alntmscore")]
                                   .sort_values(ascending=False).index],
        "fnn_rank_PIDE_14": [NAME[m] for m in plms[("fnn", "fident")].sort_values(ascending=False).index],
        "fnn_rank_HFSP_14": [NAME[m] for m in plms[("fnn", "hfsp")].sort_values(ascending=False).index],
        "esm2_ladder_fnn": {TARGETS[t]: {NAME[m]: r4(plms.loc[m, ("fnn", t)]) for m in
                                         ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b"]}
                            for t in TARGETS},
        "ankh_fnn": {TARGETS[t]: [r4(plms.loc["ankh_base", ("fnn", t)]), r4(plms.loc["ankh_large", ("fnn", t)])]
                     for t in TARGETS},
    }
    facts["extra_same_paragraphs"] = extra

    # Table S2 cross-check -----------------------------------------------------------
    facts["tableS2_check"] = [
        check_table(artefacts / "figures_spearman_2026-10-09/tableS2.md", wide),
        check_table(manuscript / "sections/90.supplementary.md", wide, section="[]{#tbls:performance}"),
    ]

    facts["readme"] = readme(facts)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "spearman_facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    rows = []
    for tgt, lab in TARGETS.items():
        er = {r["arm"]: r for r in ranking(plms[("euclidean", tgt)])}
        fr = {r["arm"]: r for r in ranking(plms[("fnn", tgt)])}
        for m in plms.index:
            rows.append({"target": lab, "model": NAME[m], "arm": m,
                         "specialist": m in SPECIALISTS,
                         "euclid_abs_rho": r4(plms.loc[m, ("euclidean", tgt)]),
                         "euclid_rank_14": er[m]["rank"], "euclid_rank_2dp": er[m]["rank_2dp"],
                         "fnn_rho": r4(plms.loc[m, ("fnn", tgt)]),
                         "fnn_rank_14": fr[m]["rank"], "fnn_rank_2dp": fr[m]["rank_2dp"],
                         "gain_fnn_minus_euclid": r4(gain.loc[m, tgt])})
    pd.DataFrame(rows).to_csv(a.out / "spearman_facts_per_model.csv", index=False)
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
