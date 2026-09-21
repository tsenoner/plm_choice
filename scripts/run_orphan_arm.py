#!/usr/bin/env python
"""Run the C8 orphan / metagenomic out-of-distribution arm over every embedding arm.

Drives the SHIPPED orphan arm (``evaluation.orphan_{io,score,report,auroc_ci,freeze}``)
across the 12 pLM embedding files, plus four non-pLM reference arms, and adds:

* a **component (cluster) bootstrap BCa CI** beside the shipped vertex-bootstrap CI
  (:mod:`evaluation.orphan_component_ci`);
* the two **positive-unlabelled-aware** read-outs the source paper defines,
  DeltaS (Eq. 9/10) and Recall_maxPPf50 (Eqs. 11/12) (:mod:`evaluation.orphan_pu`);
* the paper's **balanced under-sampling protocol** (100 iterations) so F1max /
  Precision / Recall / PR-AUC / ROC-AUC are on the same footing as its Table 1;
* the paper's **random-classifier null**, measured on this exact pair table;
* per-arm Spearman rho against the pair table's own ``pident`` (the within-set pairwise
  sequence identity that ``orphan_io`` drops), because a sequence-identity ranker is
  itself a strong baseline on this set.

Reference (non-pLM) arms:
  ``aac20``     -- 20-d amino-acid composition cosine: the sequence-composition floor.
  ``seqlen``    -- min(L1,L2)/max(L1,L2): a pure protein-length ranker.
  ``pident``    -- the pair table's own sequence identity (a confound baseline).
  ``tm``/``snn``-- the two ingredients of the sibling label itself (circular by
                   construction; reported as the ceiling/definition reference).
  ``random``    -- Uniform(0,1) per pair, the paper's random classifier.

Usage::

    PYTHONPATH=src python scripts/run_orphan_arm.py \
        --pairs  data/raw/orphan_stockpile/orphan_sibling_score.tsv.gz \
        --fasta  data/raw/orphan_stockpile/orphan_sequences_11444.fasta \
        --emb-dir <dir with the 12 *.h5> --out-dir <results dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.analysis_io import json_safe, load_embeddings_h5
from evaluation.orphan_auroc_ci import orphan_auroc_vertex_bca_ci
from evaluation.orphan_component_ci import orphan_auroc_component_bca_ci, pair_components
from evaluation.orphan_freeze import derive_orphan_freeze, verify_orphan_freeze
from evaluation.orphan_io import load_orphan_pairs
from evaluation.orphan_pu import (
    balanced_threshold_metrics,
    cosine_to_similarity,
    delta_s_components,
    random_classifier_null,
    recall_max_ppf50,
)
from evaluation.orphan_report import orphan_correlation_report

PLMS = [
    "ankh_base", "ankh_large", "esm1b", "esm2_8m", "esm2_35m", "esm2_150m",
    "esm2_650m", "esm2_3b", "esm3_open", "esmc_300m", "esmc_600m", "prott5",
]


def _parse_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    pid = None
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if pid is not None:
                    seqs[pid] = "".join(buf)
                pid = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line)
    if pid is not None:
        seqs[pid] = "".join(buf)
    return seqs


AA = "ACDEFGHIKLMNPQRSTVWY"


def _aac_embeddings(seqs: dict[str, str]) -> dict[str, np.ndarray]:
    index = {a: i for i, a in enumerate(AA)}
    out = {}
    for pid, s in seqs.items():
        v = np.zeros(20, dtype=np.float32)
        for ch in s:
            j = index.get(ch)
            if j is not None:
                v[j] += 1.0
        tot = v.sum()
        out[pid] = v / tot if tot > 0 else v
    return out


def _pu_block(score: np.ndarray, sibling: np.ndarray, *, seed: int) -> dict:
    """DeltaS (Eq. 9/10) + Recall_maxPPf50 (Eq. 12) + the balanced protocol."""
    ds = delta_s_components(score, sibling)
    rc = recall_max_ppf50(score, sibling)
    bal = balanced_threshold_metrics(score, sibling, n_iter=100, seed=seed)
    delta_s = ds["delta_s_raw"] * bal["recall_at_f1max"]
    return {
        **{f"pu_{k}": v for k, v in ds.items()},
        **{f"pu_{k}": v for k, v in rc.items()},
        **{f"pu_{k}": v for k, v in bal.items()},
        "pu_delta_s": float(delta_s),
    }


def run_arm(
    *,
    name: str,
    kind: str,
    per_pair: pd.DataFrame,
    pident: np.ndarray,
    out_dir: Path,
    n_boot: int,
    seed: int,
    manifest: dict | None = None,
) -> dict:
    """Common downstream for one arm: CIs + PU stats + identity correlation."""
    from scipy.stats import spearmanr

    t0 = time.time()
    score_raw = per_pair["cos"].to_numpy(dtype=np.float64)
    sibling = per_pair["sibling"].to_numpy().astype(bool)

    if manifest is None:  # score-only arm: compute the vertex CI ourselves
        vci = orphan_auroc_vertex_bca_ci(per_pair, n_boot=n_boot, alpha=0.05, seed=seed)
        manifest = {
            "plm": name,
            "siblings_AUROC": vci["point"],
            "ci_lo": vci["ci_lo"], "ci_hi": vci["ci_hi"],
            "ci_degenerate": vci["degenerate"],
            "percentile_diverged": vci["diverged"],
            "n_boot_undefined": vci["n_boot_undefined"],
            "naive_ci_lo": float("nan"), "naive_ci_hi": float("nan"),
            "spearman_cos_vs_SNN": float(spearmanr(score_raw, per_pair["snn"]).correlation),
            "spearman_cos_vs_TM": float(spearmanr(score_raw, per_pair["tm"]).correlation),
            "n_pairs": int(len(per_pair)),
            "n_pairs_dropped": 0,
            "n_siblings": int(sibling.sum()),
            "n_proteins": int(len(set(per_pair["p1"]) | set(per_pair["p2"]))),
        }

    cci = orphan_auroc_component_bca_ci(per_pair, n_boot=n_boot, alpha=0.05, seed=seed)

    # The paper scores embeddings with S = (cos + 1)/2 (Eq. 5). For the non-cosine
    # reference arms the score is already in [0, 1] (pident is rescaled to [0, 1]).
    if kind == "embedding":
        score = cosine_to_similarity(score_raw)
    elif name == "pident":
        score = score_raw / 100.0
    else:
        score = score_raw
    pu = _pu_block(score, sibling, seed=seed)

    rho_pid = float(spearmanr(score_raw, pident).correlation)
    n_vert = len(set(per_pair["p1"].astype(str)) | set(per_pair["p2"].astype(str)))

    row = {
        "arm": name,
        "kind": kind,
        "auroc": manifest["siblings_AUROC"],
        "vertex_ci_lo": manifest["ci_lo"],
        "vertex_ci_hi": manifest["ci_hi"],
        "vertex_ci_degenerate": manifest["ci_degenerate"],
        "vertex_ci_diverged": manifest["percentile_diverged"],
        "component_ci_lo": cci["ci_lo"],
        "component_ci_hi": cci["ci_hi"],
        "component_ci_degenerate": cci["degenerate"],
        "component_ci_diverged": cci["diverged"],
        "n_components": cci["n_components"],
        "naive_pair_ci_lo": manifest["naive_ci_lo"],
        "naive_pair_ci_hi": manifest["naive_ci_hi"],
        "spearman_vs_SNN": manifest["spearman_cos_vs_SNN"],
        "spearman_vs_TM": manifest["spearman_cos_vs_TM"],
        "spearman_vs_pident": rho_pid,
        "n_pairs": manifest["n_pairs"],
        "n_pairs_dropped": manifest["n_pairs_dropped"],
        "n_siblings": manifest["n_siblings"],
        "n_proteins_embedded": manifest["n_proteins"],
        "n_vertices_in_pairs": int(n_vert),
        "seconds": round(time.time() - t0, 1),
        **pu,
    }
    (out_dir / f"arm_{name}.json").write_text(
        json.dumps(json_safe({"row": row, "report_manifest": manifest, "component_ci": cci}),
                   indent=2) + "\n"
    )
    print(f"  {name:12s} AUROC={row['auroc']:.4f} "
          f"vertex[{row['vertex_ci_lo']:.4f},{row['vertex_ci_hi']:.4f}] "
          f"comp[{row['component_ci_lo']:.4f},{row['component_ci_hi']:.4f}] "
          f"dS={row['pu_delta_s']:+.4f} R@PPf50={row['pu_recall_max_ppf50']:.4f} "
          f"({row['seconds']}s)", flush=True)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plms", nargs="*", default=PLMS)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)

    # ── inputs + sanity checks ────────────────────────────────────────────────────────
    pairs_raw, n_mal, n_self, n_reord, n_dup = load_orphan_pairs(args.pairs, strict=False)
    pairs = load_orphan_pairs(args.pairs)  # strict path; must not raise
    raw = pd.read_csv(args.pairs, sep="\t")
    assert (raw["p1"].to_numpy() == pairs["p1"].to_numpy()).all()
    assert (raw["p2"].to_numpy() == pairs["p2"].to_numpy()).all()
    pident = raw["pident"].to_numpy(dtype=np.float64)

    seqs = _parse_fasta(Path(args.fasta))
    freeze = derive_orphan_freeze(
        pairs, derived_from=str(Path(args.pairs).name), source_tsv=str(args.pairs)
    )
    verify_orphan_freeze(freeze, pairs)
    (out_dir / "orphan_bromberg_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    expected_ids = freeze["ids"]

    sibling_all = pairs["sibling"].to_numpy().astype(bool)
    sib_sub = pairs[pairs["sibling"]]
    _, n_comp_full, n_vert_full = pair_components(
        pairs.assign(cos=0.0)[["p1", "p2", "cos", "sibling"]]
    )
    _, n_comp_sib, n_vert_sib = pair_components(
        sib_sub.assign(cos=0.0)[["p1", "p2", "cos", "sibling"]]
    )
    checks = {
        "n_fasta_sequences": len(seqs),
        "n_pairs_rows_kept": int(len(pairs)),
        "n_malformed_rows": n_mal,
        "n_self_pairs": n_self,
        "n_reordered_rows": n_reord,
        "n_duplicate_rows": n_dup,
        "n_siblings": int(sibling_all.sum()),
        "sibling_pair_fraction": float(sibling_all.mean()),
        "n_proteins_in_any_pair": int(len(expected_ids)),
        "n_proteins_in_a_sibling_pair": int(n_vert_sib),
        "n_proteins_in_no_pair": int(len(seqs) - len(expected_ids)),
        "full_pair_graph_n_components": int(n_comp_full),
        "sibling_graph_n_components": int(n_comp_sib),
        "freeze_content_sha256": freeze["content_sha256"],
    }
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    for label, sub in (("full", pairs), ("sibling", sib_sub)):
        ids = sorted(set(sub["p1"]) | set(sub["p2"]))
        vid = {p: i for i, p in enumerate(ids)}
        u = np.array([vid[a] for a in sub["p1"]])
        v = np.array([vid[b] for b in sub["p2"]])
        _, lab = connected_components(
            coo_matrix((np.ones(u.size), (u, v)), shape=(len(ids), len(ids))), directed=False
        )
        sizes = np.bincount(lab)
        checks[f"{label}_graph_largest_component"] = int(sizes.max())
        checks[f"{label}_graph_component_size_p50"] = float(np.median(sizes))
    checks["random_classifier_null"] = random_classifier_null(sibling_all, n_iter=100, seed=0)
    print(json.dumps(checks, indent=2), flush=True)
    (out_dir / "sanity_checks.json").write_text(json.dumps(json_safe(checks), indent=2) + "\n")

    rows: list[dict] = []

    # ── 12 pLM arms, through the shipped report ───────────────────────────────────────
    for plm in args.plms:
        h5 = Path(args.emb_dir) / f"{plm}.h5"
        if not h5.exists():
            print(f"  SKIP {plm}: {h5} missing", file=sys.stderr, flush=True)
            continue
        emb = load_embeddings_h5(h5)
        manifest = orphan_correlation_report(
            emb, pairs, out_dir / "cells", plm=plm, expected_ids=expected_ids,
            representation="raw", distance="cosine", seed=args.seed,
            n_boot=args.n_boot, ci_alpha=0.05, overwrite=True,
        )
        per_pair = pd.read_parquet(manifest["path"])
        manifest["embedding_dim"] = int(next(iter(emb.values())).shape[0])
        rows.append(run_arm(name=plm, kind="embedding", per_pair=per_pair, pident=pident,
                            out_dir=out_dir / "cells", n_boot=args.n_boot, seed=args.seed,
                            manifest=manifest))
        pd.DataFrame(rows).to_csv(out_dir / "orphan_arm_table.csv", index=False)

    # ── reference / floor arms ────────────────────────────────────────────────────────
    aac = _aac_embeddings(seqs)
    manifest = orphan_correlation_report(
        aac, pairs, out_dir / "cells", plm="aac20", expected_ids=expected_ids,
        representation="raw", distance="cosine", seed=args.seed,
        n_boot=args.n_boot, ci_alpha=0.05, overwrite=True,
    )
    rows.append(run_arm(name="aac20", kind="floor",
                        per_pair=pd.read_parquet(manifest["path"]), pident=pident,
                        out_dir=out_dir / "cells", n_boot=args.n_boot, seed=args.seed,
                        manifest=manifest))

    lengths = {pid: float(len(s)) for pid, s in seqs.items()}
    l1 = np.array([lengths[a] for a in pairs["p1"]])
    l2 = np.array([lengths[b] for b in pairs["p2"]])
    extra = {
        "seqlen": ("floor", np.minimum(l1, l2) / np.maximum(l1, l2)),
        "pident": ("floor", pident.copy()),
        "random": ("null", np.random.default_rng(0).random(len(pairs))),
        "tm": ("label_ingredient", pairs["tm"].to_numpy(dtype=np.float64)),
        "snn": ("label_ingredient", pairs["snn"].to_numpy(dtype=np.float64)),
    }
    for name, (kind, score) in extra.items():
        pp = pd.DataFrame({
            "p1": pairs["p1"].to_numpy(), "p2": pairs["p2"].to_numpy(),
            "cos": score, "snn": pairs["snn"].to_numpy(), "tm": pairs["tm"].to_numpy(),
            "sibling": sibling_all,
        })
        rows.append(run_arm(name=name, kind=kind, per_pair=pp, pident=pident,
                            out_dir=out_dir / "cells", n_boot=args.n_boot, seed=args.seed))
        pd.DataFrame(rows).to_csv(out_dir / "orphan_arm_table.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "orphan_arm_table.csv", index=False)
    print(f"\nwrote {out_dir / 'orphan_arm_table.csv'} ({len(df)} arms)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
