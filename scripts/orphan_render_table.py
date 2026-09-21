#!/usr/bin/env python
"""Render the orphan arm CSV as the markdown tables that go into RESULTS/SUMMARY.md."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def f(v, n=3):
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return "n/a"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--strata", default=None)
    args = ap.parse_args(argv)
    df = pd.read_csv(args.csv)
    # pandas reads the literal string "null" (the random arm's kind) as NaN.
    df["kind"] = df["kind"].fillna("null")
    order = {"embedding": 0, "floor": 1, "null": 2, "label_ingredient": 3}
    df["_k"] = df["kind"].map(order).fillna(9)
    df = df.sort_values(["_k", "auroc"], ascending=[True, False])

    print("| arm | kind | AUROC | vertex 95% CI | component 95% CI | naive pair 95% CI |"
          " rho(S,SNN) | dS | Recall@PPf<0.5 | n pairs | n sib | n vert |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        print(
            f"| {r['arm']} | {r['kind']} | {f(r['auroc'])} | "
            f"[{f(r['vertex_ci_lo'])}, {f(r['vertex_ci_hi'])}] | "
            f"[{f(r['component_ci_lo'])}, {f(r['component_ci_hi'])}] | "
            f"[{f(r['naive_pair_ci_lo'])}, {f(r['naive_pair_ci_hi'])}] | "
            f"{f(r['spearman_vs_SNN'])} | {float(r['pu_delta_s']):+.3f} | "
            f"{f(r['pu_recall_max_ppf50'])} | {int(r['n_pairs'])} | "
            f"{int(r['n_siblings'])} | {int(r['n_vertices_in_pairs'])} |"
        )

    print("\n\n### CI width comparison (95% interval width)\n")
    print("| arm | vertex | component | naive pair | vertex / naive |")
    print("|---|---|---|---|---|")
    for _, r in df.iterrows():
        vw = r["vertex_ci_hi"] - r["vertex_ci_lo"]
        cw = r["component_ci_hi"] - r["component_ci_lo"]
        nw = r["naive_pair_ci_hi"] - r["naive_pair_ci_lo"]
        ratio = f(vw / nw, 1) if nw == nw and nw > 0 else "n/a"
        print(f"| {r['arm']} | {f(vw, 4)} | {f(cw, 4)} | {f(nw, 4)} | {ratio}x |")

    print("\n\n### PU-aware detail\n")
    print("| arm | mean S sibling | mean S unlabelled | dS_raw | Recall@F1max | dS = "
          "Recall x dS_raw | Recall@PPf<0.5 | enrichment | balanced ROC-AUC | balanced PR-AUC | F1max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        print(
            f"| {r['arm']} | {f(r['pu_mean_S_sibling'], 4)} | {f(r['pu_mean_S_unlabelled'], 4)} | "
            f"{float(r['pu_delta_s_raw']):+.4f} | {f(r['pu_recall_at_f1max'])} | "
            f"{float(r['pu_delta_s']):+.4f} | {f(r['pu_recall_max_ppf50'])} | "
            f"{f(r['pu_enrichment_vs_ppf'], 2)} | {f(r['pu_roc_auc_balanced'])} | "
            f"{f(r['pu_pr_auc_balanced'])} | {f(r['pu_f1max'])} |"
        )

    if args.strata and Path(args.strata).exists():
        s = pd.read_csv(args.strata)
        print("\n\n### Confound-stratified AUROC (point estimates)\n")
        print("| arm | rho(S, len-ratio) | rho(S, pident) | AUROC all | AUROC len-ratio>0.9 | "
              "AUROC len-ratio<=0.9 | AUROC pident<30 | AUROC pident<20 |")
        print("|---|---|---|---|---|---|---|---|")
        for _, r in s.iterrows():
            print(
                f"| {r['arm']} | {f(r['spearman_vs_lenratio'])} | {f(r['spearman_vs_pident'])} | "
                f"{f(r['auroc_all'])} | {f(r['auroc_lenratio_gt_0.9'])} | "
                f"{f(r['auroc_lenratio_le_0.9'])} | {f(r['auroc_pident_lt_30'])} | "
                f"{f(r['auroc_pident_lt_20'])} |"
            )
        print(f"\nStrata sizes: n(all)={int(s.iloc[0]['n_all'])} "
              f"sib={int(s.iloc[0]['nsib_all'])}; "
              f"n(len-ratio>0.9)={int(s.iloc[0]['n_lenratio_gt_0.9'])} "
              f"sib={int(s.iloc[0]['nsib_lenratio_gt_0.9'])}; "
              f"n(len-ratio<=0.9)={int(s.iloc[0]['n_lenratio_le_0.9'])} "
              f"sib={int(s.iloc[0]['nsib_lenratio_le_0.9'])}; "
              f"n(pident<30)={int(s.iloc[0]['n_pident_lt_30'])} "
              f"sib={int(s.iloc[0]['nsib_pident_lt_30'])}; "
              f"n(pident<20)={int(s.iloc[0]['n_pident_lt_20'])} "
              f"sib={int(s.iloc[0]['nsib_pident_lt_20'])}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
