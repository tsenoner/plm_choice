#!/usr/bin/env python3
"""Does Figure 3's Wasserstein ordering depend on min-max scaling? (D14; R1.7, R2.3)

What it measures. Figure 3 (upper triangle) compares the 14 pLMs' distributions of
pairwise Euclidean embedding distance by the 1-Wasserstein distance W after min-max
scaling each model's distances to [0, 1]. W is not invariant to a per-model rescaling,
so its *values* change with the scale; the paper's claim rests on its *ordering* (which
model pairs are close in shape, and that families sit together). Both reviewers asked
for z-score and IQR scaling. This script recomputes W on exactly the published pair
population under

  * minmax  (x - min) / (max - min)          -- the published Figure 3 scale
  * z-score (x - mean) / sd                  -- sd with ddof=0
  * IQR     (x - median) / (q75 - q25)
  * p99     x / p99                          -- already in the reducer's JSON; re-derived

and reports how far the 91-element W vector reorders between scales (Spearman rho,
Kendall tau-b), whether each model keeps its nearest model, where the within-family
pairs rank, which pairs move most, and whether W separates within-family from
cross-family pairs under each scale (AUC, with a family-label permutation p-value
because the 91 cells share models and are not independent). Two diagnostics make the
answer interpretable: a split-half check (rows [0, n/2) vs [n/2, n), two independent
uniform samples) showing each scale's ordering is reproducible on its own, so a low
agreement between scales is a difference in what they measure and not sampling noise;
and rank correlations of W with |difference| in scale-free per-arm descriptors
(min-max position of the median, tail length, skewness, kurtosis, QCD) showing what
each scale's W actually reads.

Why it is computed this way. Every scale is a positive affine map of the raw distances,
so sorting commutes with it: the sorted scaled sample is the scaled sorted sample, and
each arm is sorted once. All 14 arms are reduced on the same 5,000,000 rows, so the two
empirical distributions in every cell have equal n and W1 = mean(|sort(a) - sort(b)|)
is exact. That is the estimator ``scripts/fingerprint_full_reduce.py`` used for the
published matrix (including its float32 storage of the sorted min-max values), and the
first thing this script does is reproduce that matrix from the raw pairs and report the
deviation; it refuses to continue past a deviation larger than --max-dev. A few cells
are additionally checked against ``scipy.stats.wasserstein_distance`` on the unsorted
scaled samples. Quantiles (median, q25, q75, p99) use the reducer's order-statistic
convention ``xs[round(q * (n - 1))]`` so the per-arm numbers match its JSON.

Population. The 5,000,000 uniformly random pairs (seed 42) of the 526,871-protein
cohort in ``random_pairs_distances.parquet``; the sampler already dropped self-pairs and
identical-sequence pairs (by SHA-1 of the sequence). This script re-checks both: no row
with query == target, and, with --fasta, no row whose two sequences hash equal. The
i.i.d. Gaussian control ``dist_random_1024`` is excluded, as in Figure 3.

    python scripts/paper/wasserstein_scaling.py \
        --pairs  $PAPER_ARTEFACTS/final_figures_2026-09-20/random_pairs/random_pairs_distances.parquet \
        --fingerprint $PAPER_ARTEFACTS/final_figures_2026-09-20/fp/out/fingerprint_full.json \
        --fasta data/raw/sprot_2024/sprot.fasta \
        --out $PAPER_ARTEFACTS/d14_2026-10-16/wasserstein_scaling
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import kendalltau, mannwhitneyu, spearmanr, wasserstein_distance

#: Figure 3's arms, in the reducer's order. ``random_1024`` is excluded by design.
ARMS = (
    "ankh_base", "ankh_large", "clean", "esm1b", "esm2_8m", "esm2_35m", "esm2_150m",
    "esm2_650m", "esm2_3b", "esm3_open", "esmc_300m", "esmc_600m", "prott5", "prottucker",
)

#: Copy of ``src/visualization/plm_constants.EMBEDDING_FAMILY_MAP`` for these arms
#: (CLEAN is an ESM-1b head, ProtTucker a ProtT5 head).
FAMILY = {
    "ankh_base": "Ankh", "ankh_large": "Ankh",
    "clean": "ESM1", "esm1b": "ESM1",
    "esm2_8m": "ESM2", "esm2_35m": "ESM2", "esm2_150m": "ESM2",
    "esm2_650m": "ESM2", "esm2_3b": "ESM2",
    "esm3_open": "ESM3",
    "esmc_300m": "ESMC", "esmc_600m": "ESMC",
    "prott5": "ProtT5", "prottucker": "ProtT5",
}

SCALES = ("minmax", "zscore", "iqr", "p99")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def q_index(q: float, n: int) -> int:
    """The reducer's order-statistic quantile index."""
    return min(int(round(q * (n - 1))), n - 1)


def sequence_hashes(fasta: Path, wanted: set[str]) -> dict[str, str]:
    """SHA-1 of the sequence for every id in `wanted`, streaming a plain FASTA."""
    out: dict[str, str] = {}
    current, chunks = None, []
    with fasta.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if current in wanted:
                    out[current] = hashlib.sha1("".join(chunks).encode()).hexdigest()
                header = line[1:].split()[0]
                current = header.split("|")[1] if "|" in header else header
                chunks = []
            else:
                chunks.append(line.strip())
    if current in wanted:
        out[current] = hashlib.sha1("".join(chunks).encode()).hexdigest()
    return out


def check_population(pairs: Path, fasta: Path | None) -> dict:
    """Self-pair and identical-sequence checks on the query/target columns."""
    t = pq.read_table(pairs, columns=["query", "target"])
    q, s = t.column("query"), t.column("target")
    n_self = int(pc.sum(pc.equal(q, s)).as_py() or 0)
    res: dict = {"n_rows": t.num_rows, "n_self_pairs": n_self}
    if fasta is not None:
        qs = q.to_pandas()
        ts = s.to_pandas()
        ids = set(pd.unique(qs)) | set(pd.unique(ts))
        h = sequence_hashes(fasta, ids)
        res["n_ids"] = len(ids)
        res["n_ids_without_sequence"] = len(ids - h.keys())
        code = {pid: i for i, pid in enumerate(sorted(ids))}
        # Map each hash to one integer so the 5M-row comparison is vectorised.
        hcode = {}
        id2h = np.full(len(code), -1, dtype=np.int64)
        for pid, i in code.items():
            if pid in h:
                id2h[i] = hcode.setdefault(h[pid], len(hcode))
        qi = qs.map(code).to_numpy()
        ti = ts.map(code).to_numpy()
        hq, ht = id2h[qi], id2h[ti]
        both = (hq >= 0) & (ht >= 0)
        res["n_rows_with_both_sequences"] = int(both.sum())
        res["n_identical_sequence_pairs"] = int(((hq == ht) & both).sum())
        res["fasta"] = str(fasta)
    return res


def reduce_arm(x: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Sorted sample under every scale, plus the per-arm statistics used."""
    xs = np.sort(x)
    n = xs.size
    lo, hi = float(xs[0]), float(xs[-1])
    mean, sd = float(xs.mean()), float(xs.std(ddof=0))
    q25, med, q75 = (float(xs[q_index(q, n)]) for q in (0.25, 0.50, 0.75))
    p99 = float(xs[q_index(0.99, n)])
    if not (hi > lo and sd > 0 and q75 > q25 and p99 > 0):
        raise ValueError(f"degenerate arm: min={lo} max={hi} sd={sd} iqr={q75 - q25}")
    scaled = {
        # float32, as the reducer stored them -- for the bit-for-bit reproduction.
        "minmax_f32": ((xs - lo) / (hi - lo)).astype(np.float32),
        "p99_f32": (xs / p99).astype(np.float32),
        "minmax": (xs - lo) / (hi - lo),
        "zscore": (xs - mean) / sd,
        "iqr": (xs - med) / (q75 - q25),
        "p99": xs / p99,
    }
    z = scaled["zscore"]
    stats = {
        "n": n, "min": lo, "max": hi, "mean": mean, "sd": sd,
        "p25": q25, "median": med, "p75": q75, "iqr": q75 - q25, "p99": p99,
        # Scale-free descriptors used only to explain what each W reads.
        "minmax_median": (med - lo) / (hi - lo),
        "tail_max_minus_p99_over_iqr": (hi - p99) / (q75 - q25),
        "skewness": float(np.mean(z**3)),
        "excess_kurtosis": float(np.mean(z**4) - 3.0),
        "qcd": (q75 - q25) / (q75 + q25),
    }
    return scaled, stats


def w_matrix(sorted_scaled: dict[str, dict[str, np.ndarray]], scale: str) -> np.ndarray:
    """W1 between every pair of arms from equal-n sorted samples (exact)."""
    k = len(ARMS)
    w = np.zeros((k, k))
    for i, j in combinations(range(k), 2):
        a, b = sorted_scaled[ARMS[i]][scale], sorted_scaled[ARMS[j]][scale]
        w[i, j] = w[j, i] = float(np.mean(np.abs(a - b), dtype=np.float64))
    return w


def family_auc_permutation(w: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    """AUC = P(W_within < W_cross), and its p-value under family-label permutation.

    The 91 cells are not independent (each arm sits in 13 of them), so a Mann-Whitney
    p-value over cells overstates the evidence. Permuting which arm carries which
    family label -- family sizes fixed -- keeps that dependence and asks whether the
    real family assignment separates W better than an arbitrary one of the same shape.
    """
    k = len(ARMS)
    iu = np.triu_indices(k, 1)
    r = pd.Series(w[iu]).rank(method="average").to_numpy()
    fams = sorted(set(FAMILY.values()))
    labels = np.array([fams.index(FAMILY[a]) for a in ARMS])

    def auc(lab: np.ndarray) -> np.ndarray:
        # lab: (P, k) family labels -> (P,) AUC
        within = lab[:, iu[0]] == lab[:, iu[1]]
        nw = within.sum(axis=1)
        nc = within.shape[1] - nw
        u_w = (r[None, :] * within).sum(axis=1) - nw * (nw + 1) / 2.0
        return 1.0 - u_w / (nw * nc)

    obs = float(auc(labels[None, :])[0])
    rng = np.random.default_rng(seed)
    perm = np.stack([labels[rng.permutation(k)] for _ in range(n_perm)])
    null = auc(perm)
    p = float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1))
    return obs, p


def rank_of(v: np.ndarray) -> np.ndarray:
    """1 = smallest W (most similar shape) among the 91 pairs."""
    return pd.Series(v).rank(method="min").to_numpy().astype(int)


def render_readme(summary: dict, df: pd.DataFrame, agree_df: pd.DataFrame,
                  fam_df: pd.DataFrame, nn_df: pd.DataFrame, split_df: pd.DataFrame,
                  drive_df: pd.DataFrame, args: argparse.Namespace) -> str:
    """A short README for the output directory, every number taken from this run."""
    pop, rep = summary["population"], summary["reproduction"]
    L = [
        "# D14 (R1.7, R2.3): Figure 3's Wasserstein triangle under other scalings",
        "",
        f"Generated by `scripts/paper/wasserstein_scaling.py` on {time.strftime('%Y-%m-%d')}.",
        f"Inputs: `{args.pairs}`, `{args.fingerprint}`"
        + (f", identical-sequence re-check against `{args.fasta}`." if args.fasta else "."),
        "",
        "## Population",
        "",
        f"{pop['n_pairs_used']:,} uniformly random pairs, {pop['n_models']} pLMs, "
        f"{pop['n_model_pairs']} model pairs; excluded: {pop['excluded_arm']}. Re-checked: "
        f"{pop['n_self_pairs']} self-pairs"
        + (f", {pop['n_identical_sequence_pairs']} identical-sequence pairs (SHA-1; "
           f"{pop['n_ids_without_sequence']} of {pop['n_ids']:,} ids without a sequence)"
           if "n_identical_sequence_pairs" in pop else "")
        + ". Every cell uses the same rows.",
        "",
        "## Reproduction of the published min-max matrix",
        "",
        f"Max abs deviation from `fingerprint_full.json[\"distances\"]`: "
        f"{rep['minmax_f32_vs_published_max_abs_dev']:.3g} (float32 replica of the reducer), "
        f"{rep['minmax_f64_vs_published_max_abs_dev']:.3g} (float64). p99 matrix: "
        f"{rep['p99_f32_vs_published_max_abs_dev']:.3g}. Per-arm quantiles vs JSON: "
        f"{rep['per_arm_quantiles_vs_json_max_abs_dev']:.3g}. scipy "
        f"`wasserstein_distance` cross-check: {summary['scipy_crosscheck_max_abs_dev']:.3g}.",
        "",
        "## Rank agreement of the 91 W values between scales",
        "",
        "| scales | Spearman rho | Kendall tau-b | overlap of 10 closest pairs | max abs rank change |",
        "|---|---|---|---|---|",
    ]
    for r in agree_df.itertuples():
        L.append(f"| {r.scale_a} vs {r.scale_b} | {r.spearman_rho:.4f} | {r.kendall_tau_b:.4f} "
                 f"| {r.overlap_of_10_closest_pairs}/10 | {r.max_abs_rank_change} |")
    sh = split_df[split_df["scale"].isin(SCALES)]
    L += ["", "Split-half (two independent halves of "
          f"{int(sh['n_pairs_per_half'].iloc[0]):,} pairs) Spearman rho of each scale's W "
          "ordering, and max abs W change between halves:", ""]
    for r in sh.itertuples():
        L.append(f"* {r.scale}: rho {r.spearman_rho_half1_vs_half2:.4f}; max |dW| "
                 f"{r.max_abs_w_diff_half1_vs_half2:.4f} ({r.max_diff_cell}), median "
                 f"{r.median_abs_w_diff_half1_vs_half2:.4f}")
    xs = split_df[~split_df["scale"].isin(SCALES)]
    L.append("* cross-scale rho within each half: " + "; ".join(
        f"{r.scale} {r.spearman_rho_cross_scale:.3f}" for r in xs.itertuples()))
    L += ["", "What each W reads -- Spearman of W with |difference| in a scale-free per-arm "
          "descriptor:", "", "| descriptor | " + " | ".join(SCALES) + " |",
          "|---|" + "---|" * len(SCALES)]
    for r in drive_df.itertuples(index=False):
        L.append(f"| {r[0]} | " + " | ".join(f"{v:.4f}" for v in r[1:]) + " |")
    L += ["", "## Family separation", "",
          "AUC = P(W_within < W_cross); p from family-label permutations (cells share "
          "models, so a per-cell test is not valid).", "",
          "| scale | AUC | permutation p | median rank within / cross |", "|---|---|---|---|"]
    for r in fam_df.itertuples():
        L.append(f"| {r.scale} | {r.auc_within_lower:.4f} | {r.p_family_label_permutation:.4f} "
                 f"| {r.median_rank_within:g} / {r.median_rank_cross:g} |")
    L += ["", "## Within-family pairs: rank among 91 (1 = smallest W)", "",
          "| pair | " + " | ".join(SCALES) + " |", "|---|" + "---|" * len(SCALES)]
    for r in df[df["within_family"]].itertuples():
        L.append(f"| {r.model_a} / {r.model_b} | "
                 + " | ".join(str(getattr(r, f"rank_{s}")) for s in SCALES) + " |")
    same = nn_df[nn_df["nearest_same_minmax_z_iqr"]]["arm"].tolist()
    diff = nn_df[~nn_df["nearest_same_minmax_z_iqr"]]
    L += ["", "## Nearest model (lowest W)", "",
          f"Same under min-max, z-score and IQR for {len(same)}/{len(nn_df)} arms: "
          + ", ".join(same) + ". Changes for: "
          + "; ".join(f"{r.arm} ({r.nearest_minmax} -> {r.nearest_zscore} / {r.nearest_iqr})"
                      for r in diff.itertuples()) + ".",
          "", "## Largest movers (max |rank change| min-max -> z or IQR)", ""]
    for r in df.sort_values("max_abs_drank_z_iqr", ascending=False).head(10).itertuples():
        L.append(f"* {r.model_a} / {r.model_b}: rank {r.rank_minmax} -> {r.rank_zscore} (z), "
                 f"{r.rank_iqr} (IQR)")
    L += ["", "## Files", "",
          "`w_pairs.csv` (91 pairs: W and rank per scale), `w_rank_agreement.csv`, "
          "`w_nearest_model.csv`, `w_family_separation.csv`, `w_split_half.csv`, "
          "`w_drivers.csv`, `w_per_arm_scale.csv`, `w_scipy_crosscheck.csv`, "
          "`wasserstein_scaling.json` (all of it plus the four 14x14 matrices).", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, type=Path,
                    help="random_pairs_distances.parquet (query, target, dist_<arm>)")
    ap.add_argument("--fingerprint", required=True, type=Path,
                    help="fingerprint_full.json the published Figure 3 was drawn from")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--fasta", type=Path, default=None,
                    help="optional FASTA to re-check the identical-sequence exclusion")
    ap.add_argument("--max-dev", type=float, default=1e-6,
                    help="refuse to continue if the min-max reproduction deviates more")
    ap.add_argument("--scipy-cells", type=int, default=3,
                    help="number of cells cross-checked with scipy per scale")
    ap.add_argument("--n-perm", type=int, default=100_000,
                    help="family-label permutations for the family-separation p-value")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    fp = json.loads(args.fingerprint.read_text())
    if list(fp["columns"]) != list(ARMS):
        raise SystemExit(f"fingerprint arms {fp['columns']} != {list(ARMS)}")
    w_pub = np.asarray(fp["distances"], dtype=float)
    w_pub_p99 = np.asarray(fp["distances_p99"], dtype=float)
    n_pub = int(fp["metadata"]["n_pairs_used"])

    # --- Population ---------------------------------------------------------
    pop = check_population(args.pairs, args.fasta)
    log(f"population: {pop}")
    if pop["n_self_pairs"]:
        raise SystemExit(f"{pop['n_self_pairs']} self-pairs in {args.pairs}")
    if pop.get("n_identical_sequence_pairs"):
        raise SystemExit(f"{pop['n_identical_sequence_pairs']} identical-sequence pairs")

    # --- Per-arm reduction --------------------------------------------------
    cols = [f"dist_{a}" for a in ARMS]
    table = pq.read_table(args.pairs, columns=cols)
    raw = {a: table.column(f"dist_{a}").to_numpy().astype(np.float64) for a in ARMS}
    del table
    n = raw[ARMS[0]].size
    finite = np.ones(n, dtype=bool)
    for a in ARMS:
        finite &= np.isfinite(raw[a])
    n_used = int(finite.sum())
    if n_used != n_pub:
        raise SystemExit(f"{n_used:,} usable rows here vs {n_pub:,} in the fingerprint JSON")
    log(f"{n:,} rows, {n_used:,} finite in all {len(ARMS)} arms")

    # Unsorted scaled samples for the scipy cross-check, kept for a few arms only.
    check_cells = [("clean", "esm1b"), ("ankh_base", "ankh_large"), ("esm1b", "esmc_600m"),
                   ("esm2_8m", "prottucker"), ("esm3_open", "prott5")][: args.scipy_cells]
    check_arms = {a for c in check_cells for a in c}
    unsorted: dict[str, dict[str, np.ndarray]] = {}

    sorted_scaled: dict[str, dict[str, np.ndarray]] = {}
    per_arm: dict[str, dict] = {}
    # Split halves: the sampler drew pairs i.i.d., so rows [0, n/2) and [n/2, n) are two
    # independent uniform samples. If a scale's ordering reproduces across halves, a low
    # agreement BETWEEN scales is a real difference in what they measure, not noise.
    half = n_used // 2
    halves: list[dict[str, dict[str, np.ndarray]]] = [{}, {}]
    for a in ARMS:
        x = raw.pop(a)[finite]
        sorted_scaled[a], per_arm[a] = reduce_arm(x)
        for h, part in enumerate((x[:half], x[half : 2 * half])):
            sc, _ = reduce_arm(part)
            halves[h][a] = {s: sc[s] for s in SCALES}
        if a in check_arms:
            st = per_arm[a]
            unsorted[a] = {
                "minmax": (x - st["min"]) / (st["max"] - st["min"]),
                "zscore": (x - st["mean"]) / st["sd"],
                "iqr": (x - st["median"]) / st["iqr"],
                "p99": x / st["p99"],
            }
        del x
        log(f"reduced {a}")

    # Per-arm statistics against the reducer's JSON.
    pub_arm = fp["metadata"]["per_arm"]
    arm_rows = []
    for a in ARMS:
        st = per_arm[a]
        row = {"arm": a, "family": FAMILY[a], **st}
        for k in ("min", "max", "p25", "median", "p75", "p99"):
            row[f"absdev_{k}_vs_json"] = abs(st[k] - pub_arm[a][k])
        arm_rows.append(row)
    arm_df = pd.DataFrame(arm_rows)
    max_stat_dev = float(arm_df.filter(like="absdev_").to_numpy().max())

    # --- W matrices ---------------------------------------------------------
    k = len(ARMS)
    W = {s: w_matrix(sorted_scaled, s) for s in ("minmax_f32", "p99_f32", *SCALES)}
    iu = np.triu_indices(k, 1)
    repro = {
        "minmax_f32_vs_published_max_abs_dev": float(np.abs(W["minmax_f32"] - w_pub)[iu].max()),
        "minmax_f64_vs_published_max_abs_dev": float(np.abs(W["minmax"] - w_pub)[iu].max()),
        "p99_f32_vs_published_max_abs_dev": float(np.abs(W["p99_f32"] - w_pub_p99)[iu].max()),
        "p99_f64_vs_published_max_abs_dev": float(np.abs(W["p99"] - w_pub_p99)[iu].max()),
        "per_arm_quantiles_vs_json_max_abs_dev": max_stat_dev,
        "published_minmax_W_range": [float(w_pub[iu].min()), float(w_pub[iu].max())],
    }
    log(f"reproduction: {repro}")
    if repro["minmax_f32_vs_published_max_abs_dev"] > args.max_dev:
        raise SystemExit(
            f"cannot reproduce the published min-max W matrix: max abs deviation "
            f"{repro['minmax_f32_vs_published_max_abs_dev']:.3g} > {args.max_dev:g}"
        )

    # scipy cross-check on unsorted samples (independent implementation of W1).
    scipy_rows = []
    for a, b in check_cells:
        i, j = ARMS.index(a), ARMS.index(b)
        for s in SCALES:
            ws = float(wasserstein_distance(unsorted[a][s], unsorted[b][s]))
            scipy_rows.append({"model_a": a, "model_b": b, "scale": s, "w_sorted_quantile": W[s][i, j],
                               "w_scipy": ws, "abs_dev": abs(ws - W[s][i, j])})
    del unsorted
    scipy_df = pd.DataFrame(scipy_rows)
    log(f"scipy cross-check max abs dev {scipy_df['abs_dev'].max():.3g}")

    # --- Pair table, ranks, movers ------------------------------------------
    pairs = [(ARMS[i], ARMS[j]) for i, j in zip(*iu, strict=True)]
    df = pd.DataFrame({"model_a": [p[0] for p in pairs], "model_b": [p[1] for p in pairs]})
    df["family_a"] = df["model_a"].map(FAMILY)
    df["family_b"] = df["model_b"].map(FAMILY)
    df["within_family"] = df["family_a"] == df["family_b"]
    df["w_minmax_published"] = w_pub[iu]
    for s in SCALES:
        df[f"w_{s}"] = W[s][iu]
    for s in SCALES:
        df[f"rank_{s}"] = rank_of(df[f"w_{s}"].to_numpy())
    for s in ("zscore", "iqr", "p99"):
        df[f"drank_{s}_vs_minmax"] = df[f"rank_{s}"] - df["rank_minmax"]
    df["max_abs_drank_z_iqr"] = df[["drank_zscore_vs_minmax", "drank_iqr_vs_minmax"]].abs().max(axis=1)

    agree = []
    for s1, s2 in (("minmax", "zscore"), ("minmax", "iqr"), ("zscore", "iqr"),
                   ("minmax", "p99"), ("zscore", "p99"), ("iqr", "p99")):
        v1, v2 = df[f"w_{s1}"].to_numpy(), df[f"w_{s2}"].to_numpy()
        rho, p_rho = spearmanr(v1, v2)
        tau, p_tau = kendalltau(v1, v2)
        same_10 = len(set(df.nsmallest(10, f"w_{s1}").index) & set(df.nsmallest(10, f"w_{s2}").index))
        agree.append({"scale_a": s1, "scale_b": s2, "n_pairs_compared": len(df),
                      "spearman_rho": float(rho), "spearman_p": float(p_rho),
                      "kendall_tau_b": float(tau), "kendall_p": float(p_tau),
                      "max_abs_rank_change": int(np.abs(df[f"rank_{s1}"] - df[f"rank_{s2}"]).max()),
                      "mean_abs_rank_change": float(np.abs(df[f"rank_{s1}"] - df[f"rank_{s2}"]).mean()),
                      "overlap_of_10_closest_pairs": same_10})
    agree_df = pd.DataFrame(agree)

    # Nearest model (lowest W) per arm under each scale.
    nn_rows = []
    for i, a in enumerate(ARMS):
        row = {"arm": a, "family": FAMILY[a]}
        for s in SCALES:
            w = W[s][i].copy()
            w[i] = np.inf
            o = np.argsort(w)
            row[f"nearest_{s}"] = ARMS[o[0]]
            row[f"w_nearest_{s}"] = float(w[o[0]])
            row[f"second_{s}"] = ARMS[o[1]]
        row["nearest_same_minmax_z_iqr"] = (
            row["nearest_minmax"] == row["nearest_zscore"] == row["nearest_iqr"]
        )
        row["nearest_same_all_four"] = row["nearest_same_minmax_z_iqr"] and (
            row["nearest_minmax"] == row["nearest_p99"]
        )
        row["nearest_is_same_family_minmax"] = FAMILY[row["nearest_minmax"]] == FAMILY[a]
        row["nearest_is_same_family_zscore"] = FAMILY[row["nearest_zscore"]] == FAMILY[a]
        row["nearest_is_same_family_iqr"] = FAMILY[row["nearest_iqr"]] == FAMILY[a]
        nn_rows.append(row)
    nn_df = pd.DataFrame(nn_rows)

    # Family separation: P(W_within < W_cross) (Mann-Whitney AUC), per scale.
    fam_rows = []
    wf = df["within_family"].to_numpy()
    for s in SCALES:
        v = df[f"w_{s}"].to_numpy()
        u, p = mannwhitneyu(v[wf], v[~wf], alternative="less")
        auc_perm, p_perm = family_auc_permutation(W[s], args.n_perm, seed=0)
        fam_rows.append({
            "scale": s, "n_within": int(wf.sum()), "n_cross": int((~wf).sum()),
            "auc_within_lower": float(1 - u / (wf.sum() * (~wf).sum())),
            "auc_check": auc_perm,
            "p_family_label_permutation": p_perm,
            "n_perm": args.n_perm,
            "mannwhitney_p_one_sided_cells_not_independent": float(p),
            "median_rank_within": float(np.median(df.loc[wf, f"rank_{s}"])),
            "median_rank_cross": float(np.median(df.loc[~wf, f"rank_{s}"])),
            "n_within_in_10_closest": int(df.nsmallest(10, f"w_{s}")["within_family"].sum()),
            "n_within_in_14_closest": int(df.nsmallest(14, f"w_{s}")["within_family"].sum()),
        })
    fam_df = pd.DataFrame(fam_rows)

    # Split-half reproducibility of each scale's ordering.
    split_rows = []
    W_half = [{s: w_matrix(halves[h], s) for s in SCALES} for h in (0, 1)]
    del halves
    for s in SCALES:
        v0, v1 = W_half[0][s][iu], W_half[1][s][iu]
        split_rows.append({
            "scale": s, "n_pairs_per_half": half, "n_model_pairs": len(df),
            "spearman_rho_half1_vs_half2": float(spearmanr(v0, v1)[0]),
            "kendall_tau_b_half1_vs_half2": float(kendalltau(v0, v1)[0]),
            "max_abs_w_diff_half1_vs_half2": float(np.abs(v0 - v1).max()),
            "max_diff_cell": "/".join(pairs[int(np.argmax(np.abs(v0 - v1)))]),
            "median_abs_w_diff_half1_vs_half2": float(np.median(np.abs(v0 - v1))),
            "spearman_rho_half1_vs_full": float(spearmanr(v0, W[s][iu])[0]),
        })
    # Cross-scale agreement within each half, to show it matches the full-sample value.
    for s1, s2 in (("minmax", "zscore"), ("minmax", "iqr"), ("zscore", "iqr")):
        for h in (0, 1):
            split_rows.append({
                "scale": f"{s1}_vs_{s2}_half{h + 1}", "n_pairs_per_half": half,
                "n_model_pairs": len(df),
                "spearman_rho_cross_scale": float(spearmanr(W_half[h][s1][iu], W_half[h][s2][iu])[0]),
            })
    split_df = pd.DataFrame(split_rows)
    log("split-half:\n" + split_df.to_string(index=False))

    # What each W reads: rank-correlate the 91 W values with |difference| in a
    # scale-free per-arm descriptor.
    desc = ("minmax_median", "tail_max_minus_p99_over_iqr", "skewness", "excess_kurtosis", "qcd")
    drive_rows = []
    for d in desc:
        vals = np.array([per_arm[a][d] for a in ARMS])
        absdiff = np.abs(vals[iu[0]] - vals[iu[1]])
        row = {"descriptor": f"abs diff {d}"}
        for s in SCALES:
            row[f"spearman_with_w_{s}"] = float(spearmanr(absdiff, W[s][iu])[0])
        drive_rows.append(row)
    drive_df = pd.DataFrame(drive_rows)
    log("drivers:\n" + drive_df.to_string(index=False))

    # --- Write --------------------------------------------------------------
    split_df.to_csv(args.out / "w_split_half.csv", index=False, float_format="%.6g")
    drive_df.to_csv(args.out / "w_drivers.csv", index=False, float_format="%.4f")
    df.to_csv(args.out / "w_pairs.csv", index=False, float_format="%.6f")
    agree_df.to_csv(args.out / "w_rank_agreement.csv", index=False, float_format="%.6g")
    nn_df.to_csv(args.out / "w_nearest_model.csv", index=False, float_format="%.6f")
    fam_df.to_csv(args.out / "w_family_separation.csv", index=False, float_format="%.6g")
    arm_df.to_csv(args.out / "w_per_arm_scale.csv", index=False, float_format="%.8g")
    scipy_df.to_csv(args.out / "w_scipy_crosscheck.csv", index=False, float_format="%.10g")
    summary = {
        "what": "1-Wasserstein between per-model pairwise-distance distributions under four scalings",
        "pairs_file": str(args.pairs),
        "fingerprint_file": str(args.fingerprint),
        "population": {
            **pop,
            "n_pairs_used": n_used,
            "n_models": len(ARMS),
            "n_model_pairs": len(df),
            "excluded_arm": "random_1024 (i.i.d. Gaussian control), as in Figure 3",
        },
        "estimator": "W1 = mean(|sort(a) - sort(b)|) on equal-n samples (exact); "
                     "quantiles xs[round(q*(n-1))]; sd ddof=0",
        "reproduction": repro,
        "scipy_crosscheck_max_abs_dev": float(scipy_df["abs_dev"].max()),
        "rank_agreement": agree,
        "family_separation": fam_rows,
        "split_half": split_rows,
        "drivers": drive_rows,
        "nearest_model_same_minmax_z_iqr": int(nn_df["nearest_same_minmax_z_iqr"].sum()),
        "matrices": {s: W[s].tolist() for s in SCALES},
        "columns": list(ARMS),
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "wasserstein_scaling.json").write_text(json.dumps(summary, indent=1))
    (args.out / "README.md").write_text(
        render_readme(summary, df, agree_df, fam_df, nn_df, split_df, drive_df, args)
    )
    log(f"wrote outputs to {args.out} in {time.time() - t0:.1f}s")
    print(agree_df.to_string(index=False))
    print(fam_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
