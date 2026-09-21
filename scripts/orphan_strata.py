#!/usr/bin/env python
"""Confound-stratified AUROC for the orphan arm.

Two confounds are measurable on this pair table and both are large:

* **protein length** -- the sibling rule requires a TM >= 0.7 structural alignment, which
  in practice requires similar lengths. A pure length ranker,
  ``S = min(L1,L2)/max(L1,L2)``, is a strong classifier on this label.
* **within-set sequence identity** -- the pair table's own ``pident`` column (dropped by
  ``orphan_io``) is itself a strong ranker, although these proteins are "orphans" only
  with respect to UniRef100, not to each other.

This script re-scores every arm inside confound-controlled strata, using the per-pair
parquets the main run wrote, so the headline AUROC can be read against what survives
when the confound is held down. Point estimates only (no bootstrap) -- the strata are
here to show direction and magnitude, not to carry an interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _auroc(y, s):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    return float(roc_auc_score(y, s))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--fasta", required=True)
    args = ap.parse_args(argv)

    res = Path(args.results_dir)
    raw = pd.read_csv(args.pairs, sep="\t")
    key = raw["p1"] + "\t" + raw["p2"]
    pident = pd.Series(raw["pident"].to_numpy(), index=key)

    seqs, pid, n = {}, None, 0
    for line in open(args.fasta):
        line = line.rstrip()
        if line.startswith(">"):
            if pid:
                seqs[pid] = n
            pid = line[1:].split()[0]
            n = 0
        elif line:
            n += len(line)
    seqs[pid] = n
    lens = pd.Series(seqs)

    rows = []
    for pq in sorted(res.glob("cells/orphan_*_raw_cosine.parquet")):
        arm = pq.name[len("orphan_"):-len("_raw_cosine.parquet")]
        df = pd.read_parquet(pq)
        k = df["p1"] + "\t" + df["p2"]
        pid_v = pident.reindex(k).to_numpy()
        l1 = lens.reindex(df["p1"]).to_numpy()
        l2 = lens.reindex(df["p2"]).to_numpy()
        ratio = np.minimum(l1, l2) / np.maximum(l1, l2)
        sib = df["sibling"].to_numpy().astype(bool)
        extra = {
            "seqlen": ratio,
            "pident": pid_v,
            "tm": df["tm"].to_numpy(),
            "snn": df["snn"].to_numpy(),
        }
        for name, score in [(arm, df["cos"].to_numpy())] + (
            list(extra.items()) if arm == "aac20" else []
        ):
            strata = {
                "all": np.ones(sib.size, bool),
                "lenratio_gt_0.9": ratio > 0.9,
                "lenratio_le_0.9": ratio <= 0.9,
                "pident_lt_20": pid_v < 20,
                "pident_lt_30": pid_v < 30,
            }
            from scipy.stats import spearmanr

            row = {
                "arm": name,
                "spearman_vs_lenratio": float(spearmanr(score, ratio).correlation),
                "spearman_vs_pident": float(spearmanr(score, pid_v).correlation),
            }
            for sname, mask in strata.items():
                row[f"auroc_{sname}"] = _auroc(sib[mask], score[mask])
                row[f"n_{sname}"] = int(mask.sum())
                row[f"nsib_{sname}"] = int(sib[mask].sum())
            rows.append(row)

    out = pd.DataFrame(rows).drop_duplicates(subset=["arm"]).sort_values(
        "auroc_all", ascending=False
    )
    out.to_csv(res / "orphan_strata_table.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {res / 'orphan_strata_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
