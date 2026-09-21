#!/usr/bin/env python3
"""The four extra manuscript numbers, recomputed from the REFRESHED probe grid.

Each of these was computed while the ESM-2 3B alntmscore cell was still missing, so
each averaged over one arm too few. Nothing here is hard-coded from the old run.

1. Euclidean pre-training gain: mean |rho| pretrained - mean |rho| untrained twin.
2. FNN pre-training gain, same construction.
3. Rank agreement (FNN, |rho|) against the PUBLISHED ranking.
4. Per-family FNN Pearson R2 on alntmscore: the ESM-2 ladder and Ankh base vs large.
"""

from __future__ import annotations

# These paths were absolute to one machine. They are environment variables now, so an
# unset one fails here by name rather than as a FileNotFoundError further down.
import os
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def _need(var: str) -> str:
    """The value of `var`, or a message naming what to set."""
    try:
        return os.environ[var]
    except KeyError:
        raise SystemExit(f"set {var} before running this script") from None


ARTEFACTS = _need("PAPER_ARTEFACTS")
MANUSCRIPT = _need("MANUSCRIPT_DIR")

METRICS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    ARTEFACTS,
    "probe_e1_2026-09-18/probe_metrics.csv"
)
PUBLISHED = Path(
    MANUSCRIPT,
    "archive/original_submission_2025/xother_data/plm_ranking_by_spearman.csv"
)

PRE = "sprot_pre2024_e1_sub10"
RAND = "sprot_pre2024_e1_sub10_randinit"
CONTROL = "random_1024"  # i.i.d. Gaussian control, never a ranking candidate

# The randinit arms spell ProtT5 "prot_t5"; the pretrained arms spell it "prott5".
ALIAS = {"prot_t5": "prott5"}


def twin_to_arm(twin: str) -> str:
    core = twin.removeprefix("random_init_").removesuffix("_seed0")
    return ALIAS.get(core, core)


def main() -> None:
    df = pd.read_csv(METRICS)
    pre = df[df.dataset == PRE]
    rnd = df[df.dataset == RAND]

    print(f"metrics file : {METRICS}")
    print("=" * 78)

    # --- which arms actually have an untrained twin ---------------------------
    twins = sorted(rnd.arm.unique())
    mapped = {t: twin_to_arm(t) for t in twins}
    print(f"\n[0] untrained twins present: {len(twins)}")
    for t in twins:
        print(f"      {t:<34} -> {mapped[t]}")
    unmatched = [m for m in mapped.values() if m not in set(pre.arm.unique())]
    print(f"    twins with no pretrained counterpart: {unmatched or 'none'}")
    no_twin = sorted(set(pre.arm.unique()) - set(mapped.values()))
    print(f"    pretrained arms with NO twin (excluded): {no_twin}")

    # --- 1 & 2: pre-training gain in |Spearman| -------------------------------
    for mt in ("euclidean", "fnn"):
        print(f"\n[{1 if mt == 'euclidean' else 2}] pre-training gain, {mt} read-out"
              f"  (mean |rho| pretrained - mean |rho| untrained)")
        for target in ("fident", "alntmscore"):
            p = pre[(pre.target == target) & (pre.model_type == mt)]
            r = rnd[(rnd.target == target) & (rnd.model_type == mt)]
            p_s = p.set_index("arm")["Spearman"].abs()
            r_s = {twin_to_arm(a): v for a, v in
                   r.set_index("arm")["Spearman"].abs().items()}
            pairs = sorted(set(p_s.index) & set(r_s))
            gains = [(a, p_s[a], r_s[a], p_s[a] - r_s[a]) for a in pairs]
            mean_pre = sum(g[1] for g in gains) / len(gains)
            mean_unt = sum(g[2] for g in gains) / len(gains)
            print(f"    {target:<11} n={len(pairs):>2}  "
                  f"pretrained {mean_pre:.4f}  untrained {mean_unt:.4f}  "
                  f"GAIN {mean_pre - mean_unt:+.4f}")
            for a, pv, rv, d in sorted(gains, key=lambda g: -g[3]):
                print(f"         {a:<14} {pv:.4f} - {rv:.4f} = {d:+.4f}")

    # --- 3: rank agreement with the published ranking -------------------------
    print("\n[3] rank agreement vs PUBLISHED ranking (FNN, |rho|, control excluded)")
    pub = pd.read_csv(PUBLISHED).set_index("Embedding")
    for target in ("fident", "alntmscore", "hfsp"):
        s = pre[(pre.target == target) & (pre.model_type == "fnn")]
        s = s.set_index("arm")["Spearman"].abs()
        p = pub["Abs_Spearman_" + target]
        common = sorted((set(s.index) & set(p.index)) - {CONTROL})
        rho = spearmanr(s[common], p[common])
        print(f"    {target:<11} n={len(common):>2}  rho={rho.statistic:.4f}  "
              f"p={rho.pvalue:.3g}")
        if target == "alntmscore":
            print(f"      arms: {common}")

    # --- 4: per-family FNN Pearson R2 on alntmscore ---------------------------
    print("\n[4] FNN Pearson R2 on alntmscore, by family")
    fnn = pre[(pre.target == "alntmscore") & (pre.model_type == "fnn")]
    fnn = fnn.set_index("arm")["Pearson_r2"]
    ladder = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b"]
    sizes = {"esm2_8m": "8M", "esm2_35m": "35M", "esm2_150m": "150M",
             "esm2_650m": "650M", "esm2_3b": "3B"}
    print("    ESM-2 ladder (ascending size):")
    for a in ladder:
        v = fnn.get(a)
        print(f"      {sizes[a]:>5} {a:<12} "
              f"{'MISSING' if v is None else f'{v:.4f}'}")
    vals = [fnn.get(a) for a in ladder]
    if all(v is not None for v in vals):
        mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        print(f"      monotone increasing in size? {mono}")
        print(f"      argmax = {ladder[vals.index(max(vals))]} ({max(vals):.4f})")
    print("    Ankh:")
    for a in ("ankh_base", "ankh_large"):
        v = fnn.get(a)
        print(f"      {a:<12} {'MISSING' if v is None else f'{v:.4f}'}")
    if fnn.get("ankh_base") is not None and fnn.get("ankh_large") is not None:
        print(f"      large > base? {fnn['ankh_large'] > fnn['ankh_base']}")


if __name__ == "__main__":
    main()
