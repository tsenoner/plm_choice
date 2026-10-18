#!/usr/bin/env python3
"""Tabulate the D14 open-decision runs against the final grid (R1.8 linear read-outs, R2.5 head width).

    # pull the small outputs from LRZ first (metrics files, ridge JSONs; no checkpoints)
    R=<manuscript>/paper_inputs/d14_open_2026-10-17/raw
    rsync -am --include='*/' --include='*.json' --include='*.npz' --exclude='*' \
        ai:/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2/d14_open/ridge/ $R/ridge/
    for d in full100_p10_h256 full100_p10_h16 full100_p10_lindist; do
        rsync -am --include='*/' --include='*_metrics.txt' --include='training_complete.txt' \
            --exclude='*' ai:plm_choice/models/$d/ $R/models/$d/; done
    python scripts/paper/d14_open_collect.py --root <manuscript>/paper_inputs

Every score is test-split Spearman rho, oriented so higher is better (the Euclidean read-out is a
distance, so its rho is negated -- Table S2's convention). The hidden-64 FNN and Euclidean values
are the paper's own (``probe_2026-10-08/probe_metrics.csv``, dataset full100_p10). Ranks are among
the 14 pLMs; the Gaussian control is listed but never ranked. Cells still training are left
empty, so the script can be re-run as they finish.

Writes, under d14_open_2026-10-17/:
  linear_probe/linear_probe_rho.csv     target x arm: euclidean, ridge |a-b|, ridge (a-b)^2,
                                        linear_distance (SGD, final protocol), fnn; ranks; gap closed
  linear_probe/rank_agreement.csv       per target, Spearman/Kendall between read-out rankings
  probe_capacity/capacity_rho.csv       arm x hidden size (16/64/256) rho, delta vs 64, best epoch
  probe_capacity/capacity_summary.json  rank agreement, delta vs embedding size
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from scipy import stats

TARGETS = ("fident", "alntmscore", "hfsp")
CONTROL = "random_1024"
DIMS = {
    "ankh_base": 768, "ankh_large": 1536, "clean": 128, "esm1b": 1280, "esm2_8m": 320,
    "esm2_35m": 480, "esm2_150m": 640, "esm2_650m": 1280, "esm2_3b": 2560, "esm3_open": 1536,
    "esmc_300m": 960, "esmc_600m": 1152, "prott5": 1024, "prottucker": 128, "random_1024": 1024,
}


def metrics_file(run_dir: Path) -> tuple[float | None, int | None]:
    """(Spearman, best epoch) of one cell, or (None, None) while it is not finished."""
    rho = None
    for f in sorted((run_dir / "evaluation_results").glob("*_metrics.txt")):
        for line in f.read_text().splitlines():
            if line.startswith("Spearman:"):
                rho = float(line.split(":", 1)[1])
    epoch = None
    done = run_dir / "training_complete.txt"
    if done.is_file():
        m = re.search(r"epoch=(\d+)", done.read_text())
        epoch = int(m.group(1)) if m else None
    return rho, epoch


def ranks(values: dict[str, float]) -> dict[str, int]:
    arms = [a for a in values if a != CONTROL and values[a] is not None and not math.isnan(values[a])]
    order = sorted(arms, key=lambda a: -values[a])
    return {a: i + 1 for i, a in enumerate(order)}


def agreement(a: dict[str, float], b: dict[str, float]) -> dict:
    arms = sorted(k for k in a if k != CONTROL and a.get(k) is not None and b.get(k) is not None)
    if len(arms) < 3:
        return {"n": len(arms)}
    x, y = [a[k] for k in arms], [b[k] for k in arms]
    return {
        "n": len(arms),
        "spearman": float(stats.spearmanr(x, y).statistic),
        "kendall": float(stats.kendalltau(x, y).statistic),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="manuscript/paper_inputs")
    args = ap.parse_args()
    out = args.root / "d14_open_2026-10-17"
    raw = out / "raw"

    paper: dict[tuple[str, str, str], float] = {}
    paper_epoch: dict[tuple[str, str], int] = {}
    with open(args.root / "probe_2026-10-08/probe_metrics.csv") as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] != "full100_p10":
                continue
            sign = -1.0 if r["model_type"] == "euclidean" else 1.0
            paper[(r["model_type"], r["target"], r["arm"])] = sign * float(r["Spearman"])
            if r["model_type"] == "fnn" and r["epoch"]:
                paper_epoch[(r["target"], r["arm"])] = int(r["epoch"])
    arms = sorted({k[2] for k in paper})

    # --- R1.8: linear read-outs -------------------------------------------------------------
    rows, agree = [], []
    for t in TARGETS:
        col: dict[str, dict[str, float | None]] = {
            "euclidean": {a: paper.get(("euclidean", t, a)) for a in arms},
            "fnn": {a: paper.get(("fnn", t, a)) for a in arms},
            "ridge_abs": {}, "ridge_sq": {}, "linear_distance": {},
        }
        alpha = {}
        for a in arms:
            for feat in ("abs", "sq"):
                f = raw / "ridge" / a / f"{t}_{feat}.json"
                if f.is_file():
                    j = json.loads(f.read_text())
                    col[f"ridge_{feat}"][a] = j["selected"]["test_spearman"]
                    alpha[(a, feat)] = j["selected"]["alpha"]
                    eu = paper.get(("euclidean", t, a))
                    if eu is not None and abs(-j["euclidean_spearman_replayed"] - eu) > 1e-4:
                        raise SystemExit(f"{a}/{t}: ridge pairs differ from the paper's ({j['euclidean_spearman_replayed']})")
                else:
                    col[f"ridge_{feat}"][a] = None
            rho, _ = metrics_file(raw / "models/full100_p10_lindist/linear_distance" / t / a)
            col["linear_distance"][a] = rho
        rk = {k: ranks(v) for k, v in col.items()}
        for a in arms:
            r = {"target": t, "arm": a, "dims": DIMS.get(a)}
            for k, v in col.items():
                r[k] = None if v.get(a) is None else round(v[a], 4)
                r[f"rank_{k}"] = rk[k].get(a)
            r["ridge_abs_alpha"] = alpha.get((a, "abs"))
            eu, fn, ra = col["euclidean"].get(a), col["fnn"].get(a), col["ridge_abs"].get(a)
            r["gap_closed_ridge_abs"] = (
                None if None in (eu, fn, ra) or abs(fn - eu) < 1e-6 else round((ra - eu) / (fn - eu), 3)
            )
            rows.append(r)
        names = list(col)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                agree.append({"target": t, "a": x, "b": y, **agreement(col[x], col[y])})

    (out / "linear_probe").mkdir(parents=True, exist_ok=True)
    with open(out / "linear_probe/linear_probe_rho.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "linear_probe/rank_agreement.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "a", "b", "n", "spearman", "kendall"])
        w.writeheader()
        w.writerows(agree)

    # --- R2.5: head width ----------------------------------------------------------------------
    cap_rows = []
    for t in TARGETS:
        cells = {}
        for h, ds in ((16, "full100_p10_h16"), (256, "full100_p10_h256")):
            for d in sorted((raw / "models" / ds / "fnn" / t).glob("*")) if (raw / "models" / ds).is_dir() else []:
                cells[(h, d.name)] = metrics_file(d)
        for a in sorted({k[1] for k in cells}):
            base = paper.get(("fnn", t, a))
            r = {"target": t, "arm": a, "dims": DIMS.get(a), "rho_h64": base,
                 "epoch_h64": paper_epoch.get((t, a))}
            for h in (16, 256):
                rho, ep = cells.get((h, a), (None, None))
                r[f"rho_h{h}"] = rho
                r[f"epoch_h{h}"] = ep
                r[f"delta_h{h}"] = None if rho is None or base is None else round(rho - base, 4)
            r["euclidean"] = paper.get(("euclidean", t, a))
            cap_rows.append(r)
    summary = {}
    if cap_rows:
        (out / "probe_capacity").mkdir(parents=True, exist_ok=True)
        with open(out / "probe_capacity/capacity_rho.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cap_rows[0]))
            w.writeheader()
            w.writerows(cap_rows)
        for t in TARGETS:
            sub = [r for r in cap_rows if r["target"] == t]
            if not sub:
                continue
            s = {}
            for h in (16, 256):
                # The Gaussian control has no signal to gain; keep it out of the size trend.
                done = [r for r in sub if r[f"rho_h{h}"] is not None and r["arm"] != CONTROL]
                s[f"h{h}"] = {
                    "n_done": len(done),
                    "rank_vs_h64": agreement({r["arm"]: r["rho_h64"] for r in done},
                                             {r["arm"]: r[f"rho_h{h}"] for r in done}),
                    "delta_vs_log2_dims_spearman": (
                        float(stats.spearmanr([math.log2(r["dims"]) for r in done],
                                              [r[f"delta_h{h}"] for r in done]).statistic)
                        if len(done) >= 3 else None
                    ),
                    "delta": {r["arm"]: r[f"delta_h{h}"] for r in done},
                }
            summary[t] = s
        (out / "probe_capacity/capacity_summary.json").write_text(json.dumps(summary, indent=1))

    # console summary
    for t in TARGETS:
        print(f"\n== {t} ==")
        for r in [x for x in rows if x["target"] == t]:
            print(f"  {r['arm']:<12} eu {r['euclidean']!s:<7} ridge|d| {r['ridge_abs']!s:<7} "
                  f"ridge d2 {r['ridge_sq']!s:<7} linSGD {r['linear_distance']!s:<7} fnn {r['fnn']!s:<7} "
                  f"gap {r['gap_closed_ridge_abs']}")
        for g in [x for x in agree if x["target"] == t and "spearman" in x]:
            print(f"  rank {g['a']} vs {g['b']}: rho {g['spearman']:.2f} tau {g['kendall']:.2f} (n={g['n']})")
    if summary:
        print("\n== head width ==")
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
