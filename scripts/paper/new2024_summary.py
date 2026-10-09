#!/usr/bin/env python3
"""New2024 per-model generalisation with PROTEIN-level confidence intervals (plan task C9).

    uv run python scripts/paper/new2024_summary.py \
        --preds <models/new2024_full100_p10> \
        --pairs data/processed/2024_new_e1/sets/test.parquet \
        --sprot-metrics <probe_metrics.csv> --sprot-dataset full100_p10 \
        --out <dir>

Why protein-level: New2024 has 5,466 pairs over only 673 proteins, so pairs that share a
protein are not independent and a pair bootstrap (what evaluate.py reports) is too narrow.
Each replicate resamples proteins with replacement; a pair enters mult(query) * mult(target)
times (the scheme of src/evaluation/analyze_experimental_tm.py:cluster_bootstrap_r).

The prediction archives carry no protein ids. They are written in test-parquet row order with
that target's null rows dropped, which this script checks per file (targets must equal the
parquet column exactly) before borrowing the parquet's query/target ids.

Every arm of one target is scored on the SAME replicate, so differences between arms (the
size ladders) get paired CIs. "score" is Spearman rho oriented so that higher is better:
rho for the FNN probe, -rho for the Euclidean read-out (a distance).

Outputs: new2024_arms.csv (one row per read-out x target x arm x subset) and
new2024_ladders.csv (paired differences along each family's size ladder).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TARGETS = ("fident", "alntmscore", "hfsp")
LADDERS = {
    "ESM-2": ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b"],
    "Ankh": ["ankh_base", "ankh_large"],
    "ESM-C": ["esmc_300m", "esmc_600m"],
}
# "all" = every pair of the target; "lt95" = drops near-identical pairs (PIDE >= 0.95), which
# are ~half of New2024's PIDE pairs and compress every model towards the same score. A pair
# with no PIDE (no MMseqs2 hit, i.e. below the 0.3 identity threshold) stays in.
SUBSETS = ("all", "lt95")


def load_arms(preds: Path, pairs: pd.DataFrame, read_out: str, target: str, subset: str):
    """{arm: oriented predictions} plus the shared (query, target, y) of that target's pairs."""
    keep = pairs[target].notna().to_numpy()
    sub = pairs[keep].reset_index(drop=True)
    y_all = sub[target].to_numpy(dtype="float32")
    mask = np.ones(len(sub), bool) if subset == "all" else ~(sub["fident"].to_numpy() >= 0.95)
    arms = {}
    for npz in sorted(preds.glob(f"{read_out}/{target}/*/evaluation_results/test_new2024_*_predictions_targets.npz")):
        arm = npz.parts[-3]
        z = np.load(npz)
        if not np.array_equal(z["targets"], y_all):
            raise SystemExit(f"{npz}: targets are not the parquet's {target} column in order; "
                             "cannot attach protein ids")
        sign = -1.0 if read_out == "euclidean" else 1.0
        arms[arm] = sign * z["predictions"][mask].astype("float64")
    return arms, sub["query"].to_numpy()[mask], sub["target"].to_numpy()[mask], y_all[mask].astype("float64")


def bootstrap(arms: dict, q: np.ndarray, t: np.ndarray, y: np.ndarray, n_boot: int, seed: int):
    """Point rho and n_boot protein-resampled rhos per arm, all arms on the same replicates."""
    prots, inv = np.unique(np.concatenate([q, t]), return_inverse=True)
    qi, ti = inv[: len(q)], inv[len(q):]
    point = {a: stats.spearmanr(p, y)[0] for a, p in arms.items()}
    rng = np.random.default_rng(seed)
    boots = {a: [] for a in arms}
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(prots), len(prots)), minlength=len(prots))
        sel = np.repeat(np.arange(len(y)), mult[qi] * mult[ti])
        ys = y[sel]
        for a, p in arms.items():
            boots[a].append(stats.spearmanr(p[sel], ys)[0])
    return point, {a: np.asarray(v) for a, v in boots.items()}, len(prots)


def ci(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", type=Path, required=True, help="models/<new2024 dataset> root")
    ap.add_argument("--pairs", type=Path, required=True, help="New2024 test.parquet")
    ap.add_argument("--sprot-metrics", type=Path, required=True, help="probe_metrics.csv")
    ap.add_argument("--sprot-dataset", default="full100_p10")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.pairs)
    sp = pd.read_csv(args.sprot_metrics)
    sp = sp[sp["dataset"] == args.sprot_dataset]

    arm_rows, ladder_rows, agree_rows = [], [], []
    for read_out in ("fnn", "euclidean"):
        for target in TARGETS:
            for subset in SUBSETS:
                arms, q, t, y = load_arms(args.preds, pairs, read_out, target, subset)
                if not arms:
                    continue
                point, boots, n_prot = bootstrap(arms, q, t, y, args.n_boot, args.seed)
                sign = -1.0 if read_out == "euclidean" else 1.0
                ref = {r.arm: sign * r.Spearman for r in sp[(sp.model_type == read_out) & (sp.target == target)].itertuples()}
                for a in arms:
                    lo, hi = ci(boots[a])
                    arm_rows.append({"read_out": read_out, "target": target, "subset": subset, "arm": a,
                                     "n_pairs": len(y), "n_proteins": n_prot, "new2024": point[a],
                                     "ci_lo": lo, "ci_hi": hi, "sprot": ref.get(a, np.nan)})
                for fam, ladder in LADDERS.items():
                    for small, big in zip(ladder, ladder[1:], strict=False):
                        if small in arms and big in arms:
                            lo, hi = ci(boots[big] - boots[small])
                            ladder_rows.append({"read_out": read_out, "target": target, "subset": subset,
                                                "family": fam, "smaller": small, "larger": big,
                                                "delta_new2024": point[big] - point[small], "ci_lo": lo, "ci_hi": hi,
                                                "excludes_0": lo > 0 or hi < 0,
                                                "delta_sprot": ref.get(big, np.nan) - ref.get(small, np.nan)})
                # Is "the ranking does not carry over" itself resolved? Rank agreement with Swiss-Prot per
                # replicate gives it a CI; and how many arms the best one cannot be told apart from.
                common = [a for a in arms if a in ref]
                per_rep = [stats.spearmanr([boots[a][i] for a in common], [ref[a] for a in common])[0]
                           for i in range(len(boots[common[0]]))]
                best = max(arms, key=point.get)
                tied = sum(1 for a in arms if a != best and ci(boots[best] - boots[a])[0] <= 0)
                lo, hi = ci(np.asarray(per_rep))
                agree_rows.append({"read_out": read_out, "target": target, "subset": subset, "n_arms": len(common),
                                   "rank_agreement": stats.spearmanr([point[a] for a in common], [ref[a] for a in common])[0],
                                   "ci_lo": lo, "ci_hi": hi, "best": best, "n_tied_with_best": tied})
                print(f"{read_out:9} {target:10} {subset:4} {len(arms):2} arms  {len(y):5} pairs / {n_prot} proteins",
                      file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    arms_df, lad_df = pd.DataFrame(arm_rows), pd.DataFrame(ladder_rows)
    arms_df.to_csv(args.out / "new2024_arms.csv", index=False)
    lad_df.to_csv(args.out / "new2024_ladders.csv", index=False)
    pd.DataFrame(agree_rows).to_csv(args.out / "new2024_rank_agreement.csv", index=False)

    # Does the cross-model ranking carry over? Spearman between arms' New2024 and Swiss-Prot scores.
    for (ro, tg, ss), g in arms_df.groupby(["read_out", "target", "subset"]):
        g = g.dropna(subset=["sprot"])
        r = stats.spearmanr(g["new2024"], g["sprot"])[0] if len(g) > 2 else np.nan
        print(f"{ro:9} {tg:10} {ss:4} range {g['new2024'].min():.3f}-{g['new2024'].max():.3f}  "
              f"rank agreement with Swiss-Prot rho={r:.2f} (n={len(g)})")
    (args.out / "new2024_table.md").write_text(table_md(arms_df))
    print(f"wrote {args.out}/new2024_arms.csv, new2024_ladders.csv, new2024_table.md")
    return 0


# Table S1 order and names, so the supplement's two per-model tables read the same way.
DISPLAY = [("prott5", "ProtT5"), ("prottucker", "ProtTucker"), ("esm1b", "ESM-1b"), ("clean", "CLEAN"),
           ("esm2_8m", "ESM-2 8M"), ("esm2_35m", "ESM-2 35M"), ("esm2_150m", "ESM-2 150M"),
           ("esm2_650m", "ESM-2 650M"), ("esm2_3b", "ESM-2 3B"), ("esm3_open", "ESM-3"),
           ("esmc_300m", "ESM-C 300M"), ("esmc_600m", "ESM-C 600M"), ("ankh_base", "Ankh base"),
           ("ankh_large", "Ankh large")]


def table_md(arms_df: pd.DataFrame) -> str:
    """Pandoc grid table, two-row header: FNN Spearman rho, Swiss-Prot vs New2024 (protein-level 95% CI).

    Best and second best per column are marked as in Table S2 (bold underlined, italic underlined),
    on point estimates; the caption says how many arms the New2024 best cannot be told apart from.
    """
    d = arms_df[(arms_df.read_out == "fnn") & (arms_df.subset == "all")].set_index(["target", "arm"])
    marks = {}
    for tg in TARGETS:
        for col in ("sprot", "new2024"):
            first, second = d.loc[tg][col].sort_values(ascending=False).index[:2]
            marks[(tg, col, first)], marks[(tg, col, second)] = "best", "second"

    def mark(txt: str, kind: str | None) -> str:
        return {"best": f"**[{txt}]{{.underline}}**", "second": f"*[{txt}]{{.underline}}*"}.get(kind, txt)

    w = [12] + [22, 35] * len(TARGETS)          # model, then (Swiss-Prot, New2024) per target
    span = [a + b + 3 for a, b in zip(w[1::2], w[2::2], strict=True)]  # one target header over two columns

    def row(cells: list[str], widths: list[int]) -> str:
        # A cell wider than its column breaks the grid, so this refuses one.
        assert all(len(c) <= n for c, n in zip(cells, widths, strict=True)), cells
        return "| " + " | ".join(c.ljust(n) for c, n in zip(cells, widths, strict=True)) + " |"

    def rule(widths: list[int], ch: str = "-", blank_first: bool = False) -> str:
        return "+" + "+".join((" " if blank_first and i == 0 else ch) * (n + 2) for i, n in enumerate(widths)) + "+"

    out = [rule([w[0]] + span),
           row(["", "PIDE", "TM-score", "HFSP"], [w[0]] + span),
           rule(w, blank_first=True),            # the model cell spans both header rows
           row(["Model"] + ["Swiss-Prot", "New2024"] * len(TARGETS), w),
           rule(w, "=")]
    for arm, name in DISPLAY:
        cells = [name]
        for tg in TARGETS:
            r = d.loc[(tg, arm)]
            cells += [mark(f"{r.sprot:.2f}", marks.get((tg, "sprot", arm))),
                      mark(f"{r.new2024:.2f}", marks.get((tg, "new2024", arm))) + f" ({r.ci_lo:.2f}--{r.ci_hi:.2f})"]
        out += [row(cells, w), rule(w)]
    return "\n".join(out) + "\n"

if __name__ == "__main__":
    raise SystemExit(main())
