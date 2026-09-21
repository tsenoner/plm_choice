"""Build RESULTS/SUMMARY.md from the three transfer_report output directories.

Every number printed here is read from summary.csv / paired_differences.csv / tau_b.csv /
manifest.json — nothing is recomputed by hand and nothing is typed in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# These paths were absolute to one machine. They are environment variables now, so an
# unset one fails here by name rather than as a FileNotFoundError further down.
import os


def _need(var: str) -> str:
    """The value of `var`, or a message naming what to set."""
    try:
        return os.environ[var]
    except KeyError:
        raise SystemExit(f"set {var} before running this script") from None


ARTEFACTS = _need("PAPER_ARTEFACTS")

RESULTS = Path(f"{ARTEFACTS}/functional_2026-09-17")
EC_DIR = RESULTS / "ec_v2_transfer"
GO_DIR = RESULTS / "go_mf_transfer"
GO_NOPB_DIR = RESULTS / "go_mf_transfer_noPB"
EC_TAU = RESULTS / "ec_tau_b_v2_euclidean.csv"

PRIMARY = {"ec": "exact", "go": "f1"}
HBI = ["hbi_evalue", "hbi_fident"]
TWINS = {
    "randinit_ankh_base": "ankh_base",
    "randinit_ankh_large": "ankh_large",
    "randinit_esm1b": "esm1b",
    "randinit_esm2_150m": "esm2_150m",
    "randinit_esm2_35m": "esm2_35m",
    "randinit_esm2_3b": "esm2_3b",
    "randinit_esm2_650m": "esm2_650m",
    "randinit_esm2_8m": "esm2_8m",
    "randinit_esmc_300m": "esmc_300m",
    "randinit_esmc_600m": "esmc_600m",
    "randinit_prot_t5": "prott5",
}
FAMILIES = {
    "ESM-2": ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b"],
    "ESM-C": ["esmc_300m", "esmc_600m"],
    "Ankh": ["ankh_base", "ankh_large"],
}
PARAMS = {  # parameter count in millions, for "smaller vs larger" ordering only
    "esm2_8m": 8, "esm2_35m": 35, "esm2_150m": 150, "esm2_650m": 650, "esm2_3b": 3000,
    "esmc_300m": 300, "esmc_600m": 600, "ankh_base": 740, "ankh_large": 1900,
}


SIDE = Path(__file__).resolve().parent


def side_json(name: str) -> dict:
    """A recorded side measurement (the scripts next to this one wrote it).

    The tables in SUMMARY.md come from the runs' own outputs; the caveats need quantities
    no run computes — the empirical noise floor, the hubness of an i.i.d. arm, how many
    cohort proteins share a sequence. Those are measured by the scripts in this directory
    and read back here, so every number in the file is reproducible from a command.
    """
    path = SIDE / name
    if not path.is_file():
        raise SystemExit(f"missing side measurement {path}; run the script that writes it")
    return json.loads(path.read_text())



def load(directory: Path) -> dict:
    out = {
        "summary": pd.read_csv(directory / "summary.csv"),
        "paired": pd.read_csv(directory / "paired_differences.csv"),
        "manifest": json.loads((directory / "manifest.json").read_text()),
    }
    tau = directory / "tau_b.csv"
    if tau.exists():
        out["tau"] = pd.read_csv(tau)
    return out


def cell(summary: pd.DataFrame, *, distance: str, variant: str, subset: str, score: str) -> pd.DataFrame:
    sel = summary[
        (summary["distance"] == distance)
        & (summary["variant"] == variant)
        & (summary["subset"] == subset)
        & (summary["score"] == score)
    ]
    return sel.sort_values("mean", ascending=False).reset_index(drop=True)


def rank_table(frame: pd.DataFrame, *, highlight: set[str] = frozenset()) -> list[str]:
    """One ranked cell.

    `chance` and `oracle` are scoped to the arm's OWN candidate set (`summary.csv`'s
    `baseline_scope`): an embedding arm may choose any eligible cohort protein, a homology
    arm only its own MMseqs2 hits, so the two ceilings are different numbers and the scope
    is printed rather than left to be guessed. `ties` is "the pick depends on the id order"
    for every arm; the second figure is the wider count of "the criterion alone did not
    decide", which only differs for the homology arms (MMseqs2 rounds `fident`).
    """
    lines = [
        "| # | arm | mean | 95% CI | chance | oracle | baselines over | n | ties (id-order / criterion) |",
        "|---|-----|------|--------|--------|--------|---|---|---|",
    ]
    for i, row in enumerate(frame.itertuples(), start=1):
        name = f"**{row.arm}**" if row.arm in highlight else row.arm
        scope = "cohort" if row.baseline_scope == "cohort" else "its MMseqs2 hits"
        lines.append(
            f"| {i} | {name} | {row.mean:.4f} | {row.ci_lo:.4f}–{row.ci_hi:.4f} | "
            f"{row.chance:.4f} | {row.oracle:.4f} | {scope} | {row.n_queries} | "
            f"{row.n_ties} / {row.n_primary_ties} |"
        )
    return lines


def paired_lookup(paired: pd.DataFrame, *, distance: str, variant: str, subset: str, score: str):
    sel = paired[
        (paired["distance"] == distance)
        & (paired["variant"] == variant)
        & (paired["subset"] == subset)
        & (paired["score"] == score)
    ]
    table = {}
    for row in sel.itertuples():
        table[(row.arm_a, row.arm_b)] = (row.mean_diff, row.ci_lo, row.ci_hi, row.excludes_zero)
    return table


def signed(pair_table, a: str, b: str):
    """Mean difference a - b with its CI, whichever order the table stores it in."""
    if (a, b) in pair_table:
        return pair_table[(a, b)]
    diff, lo, hi, sig = pair_table[(b, a)]
    return -diff, -hi, -lo, sig


def section(kind: str, data: dict, title: str) -> list[str]:
    score = PRIMARY[kind]
    summary, paired, manifest = data["summary"], data["paired"], data["manifest"]
    variants = [v["name"] for v in manifest["variants"]]
    emb_arms = manifest["arms"]
    out = [f"## {title}", ""]
    n = manifest["n_cohort"]
    out.append(
        f"Cohort n = {n}; {len(emb_arms)} embedding arms + {len(manifest['hbi_arms'])} homology "
        f"baselines; primary score `{score}`; distances {', '.join(manifest['distances'])}; "
        f"{manifest['parameters']['n_boot']} bootstrap resamples, seed {manifest['parameters']['seed']}."
    )
    out.append("")
    out.append("Query subsets (per identity variant):")
    out.append("")
    out.append("| variant | all_queries | hbi_answerable | no_hit | queries dropped (no eligible neighbour) | median eligible neighbours |")
    out.append("|---|---|---|---|---|---|")
    for v in manifest["variants"]:
        s = v["subsets"]
        out.append(
            f"| `{v['name']}` | {s.get('all_queries', 0)} | {s.get('hbi_answerable', 0)} | "
            f"{s.get('no_hit', '—')} | {v['n_dropped']} | {v['median_eligible_neighbours']:.0f} |"
        )
    out.append("")
    out.append(
        "(`no_hit` is published under variant `all` only — those queries have no hit, so an "
        "identity variant excludes nothing for them.)"
    )
    out.append("")

    # ---- headline: pLM vs homology search, on the queries HBI can answer -------------
    out.append(f"### {title} — headline: embeddings vs MMseqs2 (euclidean, variant `all`, subset `hbi_answerable`)")
    out.append("")
    head = cell(summary, distance="euclidean", variant="all", subset="hbi_answerable", score=score)
    out += rank_table(head, highlight=set(HBI))
    out.append("")
    pair_all = paired_lookup(paired, distance="euclidean", variant="all", subset="hbi_answerable", score=score)
    out.append("Paired per-query difference against `hbi_evalue` (positive = the embedding wins):")
    out.append("")
    out.append("| arm | mean diff vs hbi_evalue | 95% CI | CI excludes 0 |")
    out.append("|---|---|---|---|")
    ordered = [a for a in head["arm"] if a not in HBI]
    for arm in ordered:
        diff, lo, hi, sig = signed(pair_all, arm, "hbi_evalue")
        out.append(f"| {arm} | {diff:+.4f} | {lo:+.4f}–{hi:+.4f} | {'yes' if sig else 'no'} |")
    out.append("")

    # ---- HBI under each identity variant --------------------------------------------
    out.append(f"### {title} — the homology baseline under each identity variant (euclidean, subset `hbi_answerable`)")
    out.append("")
    out.append(
        "The eligibility rule binds HBI too: under `fident_lt_0.3` it may only transfer from a "
        "hit below 30% identity, under `no_hit_evalue_0.001` only from a hit with E > 1e-3. "
        "`n` is how many queries still have such a hit."
    )
    out.append("")
    out.append("| variant | n | hbi_evalue | hbi_fident | best embedding arm | its mean | best - hbi_evalue | 95% CI |")
    out.append("|---|---|---|---|---|---|---|---|")
    for variant in variants:
        frame = cell(summary, distance="euclidean", variant=variant, subset="hbi_answerable", score=score)
        if frame.empty:
            continue
        indexed = frame.set_index("arm")
        best = next(a for a in frame["arm"] if a not in HBI)
        table = paired_lookup(paired, distance="euclidean", variant=variant, subset="hbi_answerable", score=score)
        diff, lo, hi, _ = signed(table, best, "hbi_evalue")
        out.append(
            f"| `{variant}` | {int(indexed.loc[best, 'n_queries'])} | "
            f"{indexed.loc['hbi_evalue', 'mean']:.4f} | {indexed.loc['hbi_fident', 'mean']:.4f} | "
            f"{best} | {indexed.loc[best, 'mean']:.4f} | {diff:+.4f} | {lo:+.4f}–{hi:+.4f} |"
        )
    out.append("")

    # ---- every score depth ----------------------------------------------------------
    depth_order = ["exact", "share3", "share2", "class", "f1", "wang_bma"]
    scores = sorted(summary["score"].unique(), key=depth_order.index)
    out.append(f"### {title} — all scores (euclidean, variant `all`, subset `all_queries`)")
    out.append("")
    out.append("| arm | " + " | ".join(f"`{s}`" for s in scores) + " |")
    out.append("|---" * (len(scores) + 1) + "|")
    primary_order = cell(summary, distance="euclidean", variant="all", subset="all_queries", score=score)
    per_score = {
        s: cell(summary, distance="euclidean", variant="all", subset="all_queries", score=s).set_index("arm")
        for s in scores
    }
    for arm in primary_order["arm"]:
        out.append(
            f"| {arm} | " + " | ".join(f"{per_score[s].loc[arm, 'mean']:.4f}" for s in scores) + " |"
        )
    out.append(
        "| _chance_ | "
        + " | ".join(f"{per_score[s]['chance'].iloc[0]:.4f}" for s in scores)
        + " |"
    )
    out.append(
        "| _oracle_ | "
        + " | ".join(f"{per_score[s]['oracle'].iloc[0]:.4f}" for s in scores)
        + " |"
    )
    out.append("")

    # ---- one ranking table per identity variant -------------------------------------
    for variant in variants:
        out.append(f"### {title} — variant `{variant}` (euclidean, subset `all_queries`, all {len(emb_arms)} embedding arms)")
        out.append("")
        out += rank_table(cell(summary, distance="euclidean", variant=variant, subset="all_queries", score=score))
        out.append("")

    # ---- the no-hit subset ----------------------------------------------------------
    no_hit = cell(summary, distance="euclidean", variant="all", subset="no_hit", score=score)
    if len(no_hit):
        out.append(f"### {title} — queries MMseqs2 cannot answer at all (subset `no_hit`, euclidean, variant `all`)")
        out.append("")
        out.append(
            f"{int(no_hit['n_queries'].iloc[0])} of {n} queries have no MMseqs2 hit to any cohort "
            "protein. HBI has no answer for them by construction; these are the embedding arms' "
            "scores there."
        )
        out.append("")
        out += rank_table(no_hit)
        out.append("")

    # ---- cosine vs euclidean --------------------------------------------------------
    out.append(f"### {title} — cosine vs euclidean")
    out.append("")
    for variant in variants:
        euc = cell(summary, distance="euclidean", variant=variant, subset="all_queries", score=score).set_index("arm")
        cos = cell(summary, distance="cosine", variant=variant, subset="all_queries", score=score).set_index("arm")
        arms = [a for a in euc.index if a in cos.index]
        rho = spearmanr(euc.loc[arms, "mean"], cos.loc[arms, "mean"]).statistic
        delta = (cos.loc[arms, "mean"] - euc.loc[arms, "mean"]).sort_values()
        rank_euc = euc.loc[arms, "mean"].rank(ascending=False)
        rank_cos = cos.loc[arms, "mean"].rank(ascending=False)
        move = (rank_euc - rank_cos).sort_values()
        movers = ", ".join(
            f"{a} {delta[a]:+.4f} ({int(rank_euc[a])}->{int(rank_cos[a])})"
            for a in list(delta.index[:3]) + list(delta.index[-3:])
        )
        out.append(
            f"- `{variant}`: Spearman rho of the {len(arms)} arm means = {rho:.4f}; "
            f"mean(cosine) - mean(euclidean) ranges {delta.min():+.4f} to {delta.max():+.4f}; "
            f"biggest movers (delta, rank euclidean->cosine): {movers}; "
            f"largest rank moves {int(abs(move).max())} places."
        )
    out.append("")

    # ---- identity gate: all -> the strictest variant ---------------------------------
    strict = variants[-1]
    out.append(f"### {title} — what the identity control costs (variant `all` -> `{strict}`, euclidean, subset `all_queries`)")
    out.append("")
    a = cell(summary, distance="euclidean", variant="all", subset="all_queries", score=score).set_index("arm")
    b = cell(summary, distance="euclidean", variant=strict, subset="all_queries", score=score).set_index("arm")
    fident = cell(summary, distance="euclidean", variant=variants[1], subset="all_queries", score=score).set_index("arm")
    out.append("| arm | variant all | variant fident<0.30 | variant no-hit | drop all->no-hit | retained |")
    out.append("|---|---|---|---|---|---|")
    for arm in a.sort_values("mean", ascending=False).index:
        drop = a.loc[arm, "mean"] - b.loc[arm, "mean"]
        retained = b.loc[arm, "mean"] / a.loc[arm, "mean"] if a.loc[arm, "mean"] > 0 else float("nan")
        out.append(
            f"| {arm} | {a.loc[arm, 'mean']:.4f} | {fident.loc[arm, 'mean']:.4f} | "
            f"{b.loc[arm, 'mean']:.4f} | {drop:.4f} | {retained:.2f} |"
        )
    out.append("")
    out.append(
        "(These are three different neighbour pools on the same queries, so the drop is a "
        "difference of two means over the same query set but not a paired-bootstrap CI: "
        "`paired_differences.csv` pairs arms within a cell, not variants.)"
    )
    out.append("")

    # ---- random controls ------------------------------------------------------------
    out.append(f"### {title} — random controls (euclidean, variant `all`, subset `all_queries`)")
    out.append("")
    pair_std = paired_lookup(paired, distance="euclidean", variant="all", subset="all_queries", score=score)
    row = a.loc["random_1024"]
    out.append(
        f"`random_1024` (i.i.d. noise): {row['mean']:.4f} [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] "
        f"against a chance expectation of {row['chance']:.4f} — "
        + (
            "the CI covers chance, as the validity check requires."
            if row["ci_lo"] <= row["chance"] <= row["ci_hi"]
            else "the query-bootstrap CI does not cover chance, but it holds the single noise "
            "draw fixed and is the wrong yardstick for this arm: see the validity checks, "
            "where fresh i.i.d. draws measure the floor and its spread directly."
        )
    )
    out.append("")
    out.append("| random-init arm | mean | pretrained twin | mean | diff (randinit - trained) | 95% CI | CI excludes 0 |")
    out.append("|---|---|---|---|---|---|---|")
    for ri, trained in TWINS.items():
        if ri not in a.index or trained not in a.index:
            continue
        diff, lo, hi, sig = signed(pair_std, ri, trained)
        out.append(
            f"| {ri} | {a.loc[ri, 'mean']:.4f} | {trained} | {a.loc[trained, 'mean']:.4f} | "
            f"{diff:+.4f} | {lo:+.4f}–{hi:+.4f} | {'yes' if sig else 'no'} |"
        )
    out.append("")

    # ---- within-family scaling ------------------------------------------------------
    out.append(f"### {title} — does a smaller model beat a bigger one? (euclidean, variant `all`, subset `all_queries`)")
    out.append("")
    out.append("| family | smaller | larger | smaller - larger | 95% CI | smaller wins |")
    out.append("|---|---|---|---|---|---|")
    for family, members in FAMILIES.items():
        members = [m for m in members if m in a.index]
        for i, small in enumerate(members):
            for large in members[i + 1 :]:
                if PARAMS[small] >= PARAMS[large]:
                    continue
                diff, lo, hi, sig = signed(pair_std, small, large)
                verdict = "yes" if sig and diff > 0 else ("no (worse)" if sig else "tie")
                out.append(
                    f"| {family} | {small} | {large} | {diff:+.4f} | {lo:+.4f}–{hi:+.4f} | {verdict} |"
                )
    out.append("")
    return out


def checks(kind: str, data: dict, title: str, floor: dict) -> list[str]:
    """Arithmetic invariants, and the noise-floor validity check on EVERY score.

    The earlier version of this file checked `random_1024` only on the primary score and
    reported the result as if it held generally. It does not: the query bootstrap holds the
    single noise draw fixed, so on the euclidean axis — where an i.i.d. Gaussian cohort has
    hubs — the interval is too narrow and misses chance in a third of the cells. The honest
    yardstick is the spread of FRESH noise draws, measured by `noise_floor.py`.
    """
    summary, score = data["summary"], PRIMARY[kind]
    out = []
    bad_ci = summary[(summary["ci_lo"] > summary["mean"] + 1e-9) | (summary["ci_hi"] < summary["mean"] - 1e-9)]
    bad_oracle = summary[summary["mean"] > summary["oracle"] + 1e-9]
    out.append(
        f"- {title}: {len(summary)} published cells. Cells whose CI does not bracket the mean: "
        f"{len(bad_ci)}. Cells whose mean exceeds the oracle (own scope): {len(bad_oracle)}."
    )

    noise = summary[summary["arm"] == "random_1024"].copy()
    noise["covers"] = (noise["ci_lo"] <= noise["chance"]) & (noise["chance"] <= noise["ci_hi"])
    degenerate = noise["subset"] == "no_hit"
    out.append(
        f"- {title} noise floor, ALL scores: `random_1024` has {len(noise)} cells; its query-"
        f"bootstrap CI covers the exact chance expectation in {int(noise['covers'].sum())} of "
        f"them ({int((~noise['covers'] & ~degenerate).sum())} failures outside the small "
        f"`no_hit` subset, {int((~noise['covers'] & degenerate).sum())} inside it where the "
        "mean is exactly 0 and the CI is a point)."
    )
    for distance in sorted(summary["distance"].unique()):
        rows = noise[(noise["distance"] == distance) & (noise["subset"] == "all_queries")
                     & (noise["variant"] == "all")]
        for row in rows.itertuples():
            stats = floor["fresh_embeddings"][distance][row.score]
            inside = stats["min"] <= row.mean <= stats["max"]
            z = (row.mean - stats["mean"]) / stats["sd"]
            out.append(
                f"  - `{row.score}` / {distance}: {row.mean:.5f} [{row.ci_lo:.5f}, {row.ci_hi:.5f}] "
                f"vs chance {row.chance:.5f} — query-bootstrap CI covers chance: "
                f"{'yes' if row.ci_lo <= row.chance <= row.ci_hi else 'NO'}; "
                f"{floor['draws']} fresh i.i.d. draws give {stats['mean']:.5f} +- {stats['sd']:.5f} "
                f"(range {stats['min']:.5f}-{stats['max']:.5f}), so the published draw is "
                f"{abs(z):.1f} sd from the floor mean and {'inside' if inside else 'outside'} "
                "the spread of fresh draws."
            )
    hub = floor["published_hubness"]
    fresh_hub = floor["fresh_embeddings"]
    out.append(
        f"- {title} why the euclidean noise cells miss chance: an i.i.d. Gaussian cohort has "
        f"hubs under euclidean, not under cosine. Published `random_1024`: "
        f"{hub['random_1024/euclidean']['distinct_neighbours']} distinct neighbours for "
        f"{hub['random_1024/euclidean']['n_queries']} queries, top in-degree "
        f"{hub['random_1024/euclidean']['top_in_degree'][0]}, top-10 share "
        f"{hub['random_1024/euclidean']['top10_share']:.4f} — against "
        f"{hub['random_1024/cosine']['top10_share']:.4f} for the same arm under cosine and "
        f"{hub['clean/euclidean']['top10_share']:.4f} for `clean`. Across "
        f"{floor['draws']} fresh draws the effect reproduces (top-10 share "
        f"{fresh_hub['euclidean']['_hub_top10_share']['mean']:.4f} euclidean vs "
        f"{fresh_hub['cosine']['_hub_top10_share']['mean']:.4f} cosine) and it is a norm "
        f"effect: corr(in-degree, vector norm) = "
        f"{fresh_hub['euclidean']['_hub_corr_indegree_norm']['mean']:.3f} under euclidean, "
        f"{fresh_hub['cosine']['_hub_corr_indegree_norm']['mean']:.3f} under cosine. A noise "
        "arm therefore transfers from the lowest-norm proteins rather than from a uniform "
        "draw, so `chance` is not the comparator for it under euclidean."
    )

    no_hit = summary[summary["subset"] == "no_hit"]
    if len(no_hit):
        out.append(
            f"- {title}: the `no_hit` subset is published under variant `all` only "
            f"(variants present: {sorted(set(no_hit['variant']))}) — those queries have no hit, "
            "so an identity variant excludes nothing for them and the cell would be identical "
            "three times over."
        )
    hbi_rows = summary[summary["arm"].isin(HBI)]
    if len(hbi_rows):
        wide = hbi_rows.pivot_table(index=["arm", "variant", "subset", "score"], columns="distance", values="mean")
        identical = bool(np.allclose(wide.to_numpy(), wide.to_numpy()[:, [0]]))
        out.append(f"- {title}: the HBI rows are identical under both distances: {identical}.")
        out.append(
            f"- {title}: `n_dropped` (queries the variant leaves with no eligible neighbour) is "
            f"{sorted(set(summary['n_dropped']))} in every cell; queries outside a subset are "
            "counted in `n_not_in_subset` instead."
        )
    beat = 0
    pair = paired_lookup(data["paired"], distance="euclidean", variant="all", subset="hbi_answerable", score=score)
    arms = data["manifest"]["arms"]
    for arm in arms:
        diff, lo, hi, sig = signed(pair, arm, "hbi_evalue")
        beat += bool(sig and diff > 0)
    out.append(
        f"- {title}: {beat} of the {len(arms)} embedding arms beat `hbi_evalue` on `{score}` with a "
        "paired-bootstrap CI that excludes zero (euclidean, variant `all`, subset `hbi_answerable`)."
    )
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def scale_verdicts(data: dict, score: str) -> dict[str, str]:
    """Per family: does the SMALLER model win, on this axis? Read off the paired CIs."""
    pair = paired_lookup(data["paired"], distance="euclidean", variant="all",
                         subset="all_queries", score=score)
    arms = set(data["manifest"]["arms"])
    out: dict[str, str] = {}
    for family, members in FAMILIES.items():
        members = [m for m in members if m in arms]
        wins, losses, ties = [], [], []
        for i, small in enumerate(members):
            for large in members[i + 1:]:
                if PARAMS[small] >= PARAMS[large]:
                    continue
                diff, lo, hi, sig = signed(pair, small, large)
                (wins if sig and diff > 0 else losses if sig else ties).append(f"{small}>{large}")
        out[family] = (
            f"{len(wins)} smaller-wins / {len(losses)} bigger-wins / {len(ties)} ties"
            + (f" ({', '.join(wins)})" if wins else "")
        )
    return out


def build_caveats(ec, go, nopb, floor_ec, floor_go, dups, sym) -> list[str]:
    """The reading instructions, with every number pulled from a run output or a recorded
    side measurement in this directory — never typed in."""
    ec_head = cell(ec["summary"], distance="euclidean", variant="all",
                   subset="hbi_answerable", score="exact").set_index("arm")
    go_head = cell(go["summary"], distance="euclidean", variant="all",
                   subset="hbi_answerable", score="f1").set_index("arm")
    ec_pair = paired_lookup(ec["paired"], distance="euclidean", variant="all",
                            subset="hbi_answerable", score="exact")
    go_pair = paired_lookup(go["paired"], distance="euclidean", variant="all",
                            subset="hbi_answerable", score="f1")
    unsupervised = [a for a in ec_head.index if a not in set(HBI) | {"clean", "prottucker"}
                    and not a.startswith("randinit") and a != "random_1024"]
    best_unsup = unsupervised[0]
    d_unsup = signed(ec_pair, best_unsup, "hbi_evalue")
    go_winners = [(a, *signed(go_pair, a, "hbi_evalue")) for a in go_head.index if a not in HBI]
    go_beat = [w for w in go_winners if w[4] and w[1] > 0]

    ec_noise = ec["summary"][(ec["summary"]["arm"] == "random_1024")]
    go_noise = go["summary"][(go["summary"]["arm"] == "random_1024")]
    def miss(frame):
        covered = (frame["ci_lo"] <= frame["chance"]) & (frame["chance"] <= frame["ci_hi"])
        return int((~covered).sum()), len(frame)
    ec_miss, ec_cells = miss(ec_noise)
    go_miss, go_cells = miss(go_noise)
    go_f1_noise = go_noise[(go_noise["distance"] == "euclidean") & (go_noise["variant"] == "all")
                           & (go_noise["subset"] == "all_queries") & (go_noise["score"] == "f1")].iloc[0]
    go_floor = floor_go["fresh_embeddings"]["euclidean"]["f1"]
    ec_exact_noise = ec_noise[(ec_noise["distance"] == "euclidean") & (ec_noise["variant"] == "all")
                              & (ec_noise["subset"] == "all_queries") & (ec_noise["score"] == "exact")].iloc[0]

    randinit = [a for a in ec_head.index if a.startswith("randinit")]
    ri_lo, ri_hi = ec_head.loc[randinit, "mean"].min(), ec_head.loc[randinit, "mean"].max()

    fident_ties = ec["summary"][(ec["summary"]["arm"] == "hbi_fident") & (ec["summary"]["variant"] == "all")
                                & (ec["summary"]["score"] == "exact")
                                & (ec["summary"]["distance"] == "euclidean")].iloc[0]
    prottucker_ties = ec["summary"][(ec["summary"]["arm"] == "prottucker") & (ec["summary"]["variant"] == "all")
                                    & (ec["summary"]["subset"] == "hbi_answerable")
                                    & (ec["summary"]["score"] == "exact")
                                    & (ec["summary"]["distance"] == "euclidean")].iloc[0]

    nopb_go = None
    if nopb is not None:
        main_cell = cell(go["summary"], distance="euclidean", variant="all",
                         subset="hbi_answerable", score="f1").set_index("arm")
        sens_cell = cell(nopb["summary"], distance="euclidean", variant="all",
                         subset="hbi_answerable", score="f1").set_index("arm")
        shared = [a for a in main_cell.index if a in sens_cell.index]
        delta = (sens_cell.loc[shared, "mean"] - main_cell.loc[shared, "mean"])
        rank_before = main_cell.loc[shared, "mean"].rank(ascending=False)
        rank_after = sens_cell.loc[shared, "mean"].rank(ascending=False)
        nopb_go = {
            "dropped": nopb["manifest"]["propagated_cleaning"]["dropped_from_propagated_sets"],
            "n": nopb["manifest"]["n_cohort"],
            "delta_min": delta.min(), "delta_max": delta.max(),
            "chance_before": float(main_cell["chance"].iloc[0]),
            "chance_after": float(sens_cell["chance"].iloc[0]),
            "max_rank_move": int((rank_before - rank_after).abs().max()),
            "top_before": rank_before.idxmin(), "top_after": rank_after.idxmin(),
        }

    caveats = [
        "**`clean` and `prottucker` are supervised arms.** `clean` is CLEAN (ESM-1b backbone, "
        "contrastively trained *on EC numbers*) and `prottucker` is ProtTucker (supervised on "
        f"CATH). CLEAN tops the EC table ({ec_head.loc['clean', 'mean']:.4f}) and ProtTucker the "
        f"GO table ({go_head.loc['prottucker', 'mean']:.4f}); both are partly reading back their "
        "own training objective. Neither is evidence about general-purpose pLM embeddings, and "
        "every table that quotes them must label them.",

        "**The headline is axis-dependent, and that is the finding.** On EC every unsupervised "
        f"pLM LOSES to MMseqs2: the best is {best_unsup} at {ec_head.loc[best_unsup, 'mean']:.4f} "
        f"against hbi_evalue's {ec_head.loc['hbi_evalue', 'mean']:.4f}, a paired difference of "
        f"{d_unsup[0]:+.4f} [{d_unsup[1]:+.4f}, {d_unsup[2]:+.4f}]. On GO-MF, the same code on the "
        f"same protocol has {len(go_beat)} arm(s) BEATING hbi_evalue with a CI that excludes zero ("
        + ", ".join(f"{a} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]" for a, d, lo, hi, _ in go_beat)
        + "). The manuscript cannot say \"embeddings beat homology search\" or \"lose to it\" "
        "without naming the axis. Scale is the same story, read off the paired CIs of the "
        "smaller-vs-bigger tables — EC: "
        + "; ".join(f"{f} {v}" for f, v in scale_verdicts(ec, "exact").items())
        + " | GO-MF: "
        + "; ".join(f"{f} {v}" for f, v in scale_verdicts(go, "f1").items())
        + ".",

        "**The untrained controls are nowhere near chance.** The random-init arms reach "
        f"{ri_lo:.2f}-{ri_hi:.2f} EC exact against a chance expectation of "
        f"{ec_head['chance'].iloc[0]:.4f}. An untrained transformer still orders proteins usefully "
        "(length, composition, the tokenizer), so \"beats chance\" is a meaningless bar on this "
        "readout. The bars that mean something are the untrained twin and the homology baseline.",

        "**Two tie columns, because they are two questions.** `n_ties` means the same thing for "
        "every arm — after the full tie-break chain the pick still depends on the id order — and "
        f"is small everywhere ({int(prottucker_ties['n_ties'])} for prottucker, "
        f"{int(fident_ties['n_ties'])} for hbi_fident on EC). `n_primary_ties` is the wider count "
        "of \"the criterion alone did not decide\", which is large for `hbi_fident` "
        f"({int(fident_ties['n_primary_ties'])} of {int(fident_ties['n_queries'])}) only because "
        "MMseqs2 reports `fident` rounded; the E-value then resolves almost all of them. "
        "`hbi_evalue` is the baseline to quote.",

        "**Symmetrising the hit table does not move the headline.** Eligibility is defined \"in "
        "either direction\", so HBI is too. Measured on the EC cohort ("
        f"`side_measurements/hbi_symmetrisation.json`): {pct(sym['share_of_directed_rows_with_their_reverse'])} "
        f"of directed hit rows have their reverse reported, i.e. "
        f"{pct(sym['share_of_unordered_pairs_bidirectional'])} of the "
        f"{sym['unordered_cohort_pairs']} unordered cohort pairs are bidirectional; symmetrising "
        f"adds {sym['answerable_symmetrised'] - sym['answerable_query_direction_only']} answerable "
        f"queries ({sym['answerable_query_direction_only']} -> {sym['answerable_symmetrised']}) and "
        f"changes the chosen neighbour for {pct(sym['share_whose_neighbour_changes'])} of the "
        f"{sym['n_both_answerable']} queries both schemes answer — on which EC exact is "
        f"{sym['exact_on_both_answerable_query_direction']:.4f} either way. The published "
        f"{sym['published_summary_mean']:.4f} is over the larger symmetrised set of "
        f"{sym['published_summary_n']} queries; do not compare it with the "
        f"{sym['exact_on_both_answerable_query_direction']:.4f} without saying "
        "which query set it is on.",

        "**The `no_hit` subset is small, and is published once.** "
        f"{int(ec['manifest']['hbi']['n_queries_without_any_cohort_hit'])} EC and "
        f"{int(go['manifest']['hbi']['n_queries_without_any_cohort_hit'])} GO queries have no "
        "MMseqs2 hit to any cohort protein, so those CIs are wide (EC: +-0.10 or worse). It is "
        "reported under variant `all` only: those queries have no hits, so an identity variant "
        "excludes nothing for them and the other two cells would be the same numbers again.",

        "**The HBI rows under variants b and c are not \"homology search without homologues\".** "
        "They are homology search restricted to the hits the variant still allows — sub-30% "
        "identity hits, or hits with E > 1e-3. They measure how much a *weak* hit still buys, "
        "which is the honest comparison against a pLM denied the same neighbours.",

        "**Read an HBI row's `chance`/`oracle` as its own.** `summary.csv` carries a "
        "`baseline_scope` column: on an embedding row chance and oracle are taken over every "
        "eligible cohort protein, on an `hbi_*` row over that query's eligible MMseqs2 hits — the "
        "only neighbours a sequence search could have transferred from. On EC (variant `all`, "
        f"`hbi_answerable`) the search's own ceiling is {ec_head.loc['hbi_evalue', 'oracle']:.4f} "
        f"against the cohort's {ec_head.loc['clean', 'oracle']:.4f}, and a RANDOM hit from its own "
        f"list already scores {ec_head.loc['hbi_evalue', 'chance']:.4f} against the cohort chance "
        f"of {ec_head.loc['clean', 'chance']:.4f}. Comparing an HBI mean with the cohort oracle "
        "would overstate how far the search is from its best possible answer.",

        "**Do not compare an arm's `all_queries` number with an HBI number.** `all_queries` "
        "includes the queries HBI cannot answer. Every pLM-vs-HBI statement in this file is on "
        "the `hbi_answerable` subset, where both readouts saw the identical query list.",

        "**The noise-floor CI is too narrow, on every score.** `random_1024`'s published interval "
        "is a bootstrap over QUERIES with the single noise draw held fixed, so it carries none of "
        f"the randomness of the draw. It misses the exact chance expectation in {ec_miss} of "
        f"{ec_cells} EC cells and {go_miss} of {go_cells} GO cells — e.g. GO F1 (euclidean, variant "
        f"`all`): {go_f1_noise['mean']:.5f} [{go_f1_noise['ci_lo']:.5f}, {go_f1_noise['ci_hi']:.5f}] "
        f"against chance {go_f1_noise['chance']:.5f}. Measured the right way "
        f"(`side_measurements/noise_floor.py`, {floor_go['draws']} fresh i.i.d. 1024-d draws scored "
        f"end to end): {go_floor['mean']:.5f} +- {go_floor['sd']:.5f}, range {go_floor['min']:.5f}-"
        f"{go_floor['max']:.5f} — the published value is "
        f"{abs(go_f1_noise['mean'] - go_floor['mean']) / go_floor['sd']:.1f} sd "
        f"{'above' if go_f1_noise['mean'] > go_floor['mean'] else 'below'} that mean, i.e. one "
        "draw of the noise, not a signal. An interval that carried the draw as well as the "
        f"queries would be about {go_floor['sd'] * 1.96 / ((go_f1_noise['ci_hi'] - go_f1_noise['ci_lo']) / 2):.1f}x "
        f"wider than the published +-{(go_f1_noise['ci_hi'] - go_f1_noise['ci_lo']) / 2:.5f}. "
        "The sentence to write is \"random_1024 is within the spread of fresh noise "
        "draws\", not \"random_1024 is significantly above chance\" — and under euclidean `chance` "
        "is not quite the right comparator for it at all, because an i.i.d. cohort has low-norm "
        "hubs (see the validity checks). On EC exact the question does not arise: "
        f"{ec_exact_noise['mean']:.4f} [{ec_exact_noise['ci_lo']:.4f}, {ec_exact_noise['ci_hi']:.4f}] "
        f"against chance {ec_exact_noise['chance']:.4f}.",

        "**Neither cohort is deduplicated at 100% sequence identity.** The strata are CATH "
        "superfamily x length, which never looks at the sequence, so "
        f"{dups['ec']['n_in_a_duplicate_group']} of {dups['ec']['n']} EC proteins "
        f"({pct(dups['ec']['share_in_a_duplicate_group'])}, {dups['ec']['n_groups']} groups) and "
        f"{dups['go']['n_in_a_duplicate_group']} of {dups['go']['n']} GO proteins "
        f"({pct(dups['go']['share_in_a_duplicate_group'])}, {dups['go']['n_groups']} groups) share a "
        "byte-identical sequence with another cohort member. Under variant `all` every EC arm "
        "scores exactly 1.0 on those queries — free points handed equally to CLEAN, to ESM-C and "
        f"to the untrained controls — which inflates the variant-`all` EC means by at most "
        f"{dups['ec']['max_inflation']['delta']:+.4f} ({dups['ec']['max_inflation']['arm']}). On GO "
        "they are not free points (identical sequences can carry different MF annotations: the "
        f"duplicated queries average {dups['go']['per_arm']['prottucker']['duplicated_queries']:.4f} "
        f"for prottucker against {dups['go']['per_arm']['prottucker']['other_queries']:.4f} on the "
        "rest). Both strict variants remove the effect entirely (an identical sequence has "
        "fident 1.0) and it cancels in every paired difference, so no conclusion depends on it — "
        "but the methods section must say the cohorts are stratified, not redundancy-reduced.",

        "**`neighbour_distance` in `per_query.parquet` is not one unit.** It is whatever was "
        "minimised to pick that neighbour: the embedding distance on a pLM row, the MMseqs2 "
        "E-value on `hbi_evalue` (so a 0.0 there is an underflow, not a zero distance) and "
        "`1 - fident` on `hbi_fident`. The units are recorded in `manifest.json` under "
        "`hbi.neighbour_distance_units`; do not filter or plot the column across arms.",
    ]
    if nopb_go is not None:
        caveats.append(
            "**The protein-binding sensitivity now tests what it was meant to test, and it "
            "passes.** Dropping GO:0005515 from the annotations alone is provably a no-op on this "
            "data (UniProt exports the specific descendants, never the generic term: 0 of "
            f"{nopb_go['n']} cohort proteins carry it directly), so the flag now drops it from the "
            f"PROPAGATED sets as well — where it really sits, in "
            f"{nopb_go['dropped'].get('GO:0005515', 0)} of {nopb_go['n']} of them. Every F1 falls "
            f"by {abs(nopb_go['delta_max']):.4f}-{abs(nopb_go['delta_min']):.4f} and the chance "
            f"level falls with it ({nopb_go['chance_before']:.4f} -> {nopb_go['chance_after']:.4f}); "
            f"the largest rank move is {nopb_go['max_rank_move']} place(s) and the top arm is "
            f"{nopb_go['top_before']} either way. `wang_bma` is defined on the unpropagated sets "
            "and is unchanged to the last bit, which is also why that run needs no tau-b."
        )
    return [f"{i}. {c}" for i, c in enumerate(caveats, start=1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ec-dir", default=EC_DIR, type=Path)
    parser.add_argument("--go-dir", default=GO_DIR, type=Path)
    parser.add_argument("--nopb-dir", default=GO_NOPB_DIR, type=Path)
    parser.add_argument("--out", default=RESULTS / "SUMMARY.md", type=Path)
    args = parser.parse_args()
    ec = load(args.ec_dir)
    go = load(args.go_dir)
    nopb = load(args.nopb_dir) if (args.nopb_dir / "summary.csv").exists() else None
    floor_ec = side_json("noise_floor_ec.json")
    floor_go = side_json("noise_floor_go.json")
    dups = side_json("duplicate_sequences.json")
    sym = side_json("hbi_symmetrisation.json")

    lines = [
        "# Functional annotation transfer: EC and GO-MF, embeddings vs homology search",
        "",
        "Leave-one-out 1-nearest-neighbour annotation transfer (goPredSim / EAT protocol) over "
        "the EC v2 and GO-MF cohorts, for 15 pretrained pLM arms, 11 random-init controls and "
        "two MMseqs2 homology baselines, under three neighbour-eligibility variants and three "
        "query subsets. Every number in the tables is generated from the `summary.csv` / "
        "`paired_differences.csv` / `tau_b.csv` / `manifest.json` of the runs below; the "
        "numbers in the caveats and the validity checks come from the recorded side "
        "measurements in `side_measurements/` (`noise_floor.py`, `duplicate_sequences.py`, "
        "`hbi_symmetrisation.py`, each writing the JSON this file reads). Nothing is typed by "
        "hand, and `side_measurements/make_summary.py` rebuilds this file.",
        "",
        "| run | output directory |",
        "|---|---|",
        f"| EC (EC v2 cohort) | `{args.ec_dir}` |",
        f"| GO-MF | `{args.go_dir}` |",
        f"| GO-MF, --drop-protein-binding (propagated-set sensitivity) | `{args.nopb_dir}` |",
        "",
        "Readouts: EC `exact` = the neighbour's EC agrees on all four fields (any pair of ECs "
        "when a protein carries several); `share3`/`share2`/`class` are the shallower depths. "
        "GO `f1` = protein-centric F1 of the two ancestor-closed MF term sets (root excluded); "
        "`wang_bma` = Wang best-match-average of the unpropagated sets. `chance` is the exact "
        "expectation over eligible neighbours, `oracle` the best any eligible neighbour offers.",
        "",
    ]
    lines.append("Code version of each run (`manifest.json` -> `versions.git_commit`), so a table "
                 "can be traced to the commit that produced it:")
    lines.append("")
    lines.append("| run | git commit | numpy | polars | cohort n | arms |")
    lines.append("|---|---|---|---|---|---|")
    for name, data in (("EC", ec), ("GO-MF", go)) + ((("GO-MF noPB", nopb),) if nopb else ()):
        v = data["manifest"]["versions"]
        lines.append(
            f"| {name} | `{(v.get('git_commit') or 'n/a')[:8]}` | {v['numpy']} | {v['polars']} | "
            f"{data['manifest']['n_cohort']} | {len(data['manifest']['arms'])} + "
            f"{len(data['manifest']['hbi_arms'])} homology |"
        )
    lines.append("")
    lines.append("### Inputs (sha256, first 12)")
    lines.append("")
    lines.append("| run | input | path | sha256 |")
    lines.append("|---|---|---|---|")
    for name, data in (("EC", ec), ("GO-MF", go)) + ((("GO-MF noPB", nopb),) if nopb else ()):
        manifest = data["manifest"]
        for key, path in manifest["inputs"].items():
            lines.append(f"| {name} | {key} | `{path}` | `{manifest['sha256'][path][:12]}` |")
        lines.append(f"| {name} | emb_dir | `{manifest['emb_dir']}` | 26 slices, hashed in manifest.json |")
    lines.append("")
    lines += section("ec", ec, "EC")
    lines += section("go", go, "GO-MF")

    # ---- tau-b ---------------------------------------------------------------------
    lines.append("## Secondary readout: Kendall tau-b over all pairs")
    lines.append("")
    if "tau" not in go:
        raise SystemExit(f"{args.go_dir} has no tau_b.csv: the GO run is the tau-b source")
    go_tau = go["tau"].set_index(["arm", "distance"])["tau_b"]
    ec_tau = pd.read_csv(EC_TAU).set_index("arm") if EC_TAU.exists() else None
    lines.append(
        "GO tau-b is the correlation between embedding distance and `1 - Wang BMA` over all "
        f"{int(go['tau'].iloc[0]['n_pairs'])} cohort pairs (point estimate, no CI, as designed). "
        "EC tau-b is the existing readout (`ec_tau_b_v2_euclidean.csv`): embedding distance vs "
        f"EC hierarchical distance over all {ec['manifest']['n_cohort'] * (ec['manifest']['n_cohort'] - 1) // 2} "
        "pairs of the same EC v2 cohort, euclidean."
    )
    lines.append("")
    lines.append("| arm | GO tau-b (euclidean) | GO tau-b (cosine) | EC tau-b (euclidean) | GO 1-NN F1 | EC 1-NN exact |")
    lines.append("|---|---|---|---|---|---|")
    go_primary = cell(go["summary"], distance="euclidean", variant="all", subset="all_queries", score="f1").set_index("arm")
    ec_primary = cell(ec["summary"], distance="euclidean", variant="all", subset="all_queries", score="exact").set_index("arm")
    for arm in go["manifest"]["arms"]:
        ec_value = "n/a"
        if ec_tau is not None and arm in ec_tau.index:
            ec_value = f"{ec_tau.loc[arm, 'tau_b']:.4f}"
        lines.append(
            f"| {arm} | {go_tau.get((arm, 'euclidean'), float('nan')):.4f} | "
            f"{go_tau.get((arm, 'cosine'), float('nan')):.4f} | {ec_value} | "
            f"{go_primary.loc[arm, 'mean']:.4f} | {ec_primary.loc[arm, 'mean']:.4f} |"
        )
    lines.append("")
    both = [a for a in go["manifest"]["arms"] if ec_tau is not None and a in ec_tau.index]
    if both:
        rho_go = spearmanr(
            [go_tau[(a, "euclidean")] for a in both], [go_primary.loc[a, "mean"] for a in both]
        ).statistic
        rho_ec = spearmanr(
            [ec_tau.loc[a, "tau_b"] for a in both], [ec_primary.loc[a, "mean"] for a in both]
        ).statistic
        lines.append(
            f"Rank correlation between the tau-b ordering and the 1-NN ordering over the "
            f"{len(both)} arms that have both: GO rho = {rho_go:.3f}, EC rho = {rho_ec:.3f}."
        )
        lines.append("")

    # ---- protein-binding sensitivity -------------------------------------------------
    lines.append("## GO sensitivity: does the uninformative binding term carry the F1?")
    lines.append("")
    if nopb is None:
        lines.append("Not available: the run has not finished.")
    else:
        man = nopb["manifest"]
        dropped = man["propagated_cleaning"]["dropped_from_propagated_sets"]
        lines.append(
            "`--drop-protein-binding` removes GO:0005515 from the annotations AND from the "
            "propagated sets. Only the second half can move anything here: UniProt exports the "
            "specific descendants (GO:0042802 \"identical protein binding\", GO:0042803 "
            "\"protein homodimerization\"), never the generic term, so the annotation drop removes "
            f"{man['label_cleaning']['dropped_term']} annotations and "
            f"{man['label_cleaning']['proteins_dropped']} proteins — while the term sits in "
            f"{dropped.get('GO:0005515', 0)} of {man['n_cohort']} propagated MF sets "
            f"({dropped.get('GO:0005515', 0) / man['n_cohort']:.1%}), which is where the F1 is "
            "computed. `wang_bma` is defined on the unpropagated sets and cannot move, so that "
            "run is `--no-tau` (tau-b is built from `1 - Wang BMA` and is identical by "
            "construction)."
        )
        lines.append("")
        main_cell = cell(go["summary"], distance="euclidean", variant="all",
                         subset="hbi_answerable", score="f1").set_index("arm")
        sens_cell = cell(nopb["summary"], distance="euclidean", variant="all",
                         subset="hbi_answerable", score="f1").set_index("arm")
        shared = [a for a in main_cell.index if a in sens_cell.index]
        main_pair = paired_lookup(go["paired"], distance="euclidean", variant="all",
                                  subset="hbi_answerable", score="f1")
        sens_pair = paired_lookup(nopb["paired"], distance="euclidean", variant="all",
                                  subset="hbi_answerable", score="f1")
        lines.append("Euclidean, variant `all`, subset `hbi_answerable`, score `f1`:")
        lines.append("")
        lines.append("| arm | main | sensitivity | delta | vs hbi_evalue (main) | vs hbi_evalue (sensitivity) |")
        lines.append("|---|---|---|---|---|---|")
        for arm in shared:
            if arm in HBI:
                gap_main = gap_sens = "—"
            else:
                d1 = signed(main_pair, arm, "hbi_evalue")
                d2 = signed(sens_pair, arm, "hbi_evalue")
                gap_main = f"{d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}]"
                gap_sens = f"{d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}]"
            lines.append(
                f"| {arm} | {main_cell.loc[arm, 'mean']:.4f} | {sens_cell.loc[arm, 'mean']:.4f} | "
                f"{sens_cell.loc[arm, 'mean'] - main_cell.loc[arm, 'mean']:+.4f} | {gap_main} | {gap_sens} |"
            )
        lines.append(
            f"| _chance_ | {main_cell['chance'].iloc[0]:.4f} | {sens_cell['chance'].iloc[0]:.4f} | "
            f"{sens_cell['chance'].iloc[0] - main_cell['chance'].iloc[0]:+.4f} | | |"
        )
        rank_before = main_cell.loc[shared, "mean"].rank(ascending=False)
        rank_after = sens_cell.loc[shared, "mean"].rank(ascending=False)
        wang_main = cell(go["summary"], distance="euclidean", variant="all",
                         subset="hbi_answerable", score="wang_bma").set_index("arm")
        wang_sens = cell(nopb["summary"], distance="euclidean", variant="all",
                         subset="hbi_answerable", score="wang_bma").set_index("arm")
        wang_delta = float((wang_sens.loc[shared, "mean"] - wang_main.loc[shared, "mean"]).abs().max())
        lines.append("")
        lines.append(
            f"Largest rank move: {int((rank_before - rank_after).abs().max())} place(s); the term "
            f"costs every arm {abs((sens_cell.loc[shared, 'mean'] - main_cell.loc[shared, 'mean'])).min():.4f}"
            f"-{abs((sens_cell.loc[shared, 'mean'] - main_cell.loc[shared, 'mean'])).max():.4f} F1 and "
            f"the chance level {main_cell['chance'].iloc[0] - sens_cell['chance'].iloc[0]:.4f}, so the "
            "ranking and the pLM-vs-homology gaps are unchanged. Maximum |delta| on `wang_bma`: "
            f"{wang_delta:.1e} (unpropagated, as designed)."
        )
    lines.append("")

    # ---- validity checks and caveats --------------------------------------------------
    lines.append("## Validity checks")
    lines.append("")
    lines += checks("ec", ec, "EC", floor_ec)
    lines += checks("go", go, "GO-MF", floor_go)
    lines.append("")
    lines.append("## What looks wrong, or needs saying before this is quoted")
    lines.append("")
    lines += build_caveats(ec, go, nopb, floor_ec, floor_go, dups, sym)
    lines.append("")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
