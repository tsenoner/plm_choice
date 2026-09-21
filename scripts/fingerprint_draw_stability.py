#!/usr/bin/env python3
"""Is the random-pair fingerprint stable across independent draws, and what did the published one measure?

Two questions, one script.

1. STABILITY. The convergence table shipped with the 5M sample used *nested* subsamples of one
   draw, which bounds the stability of that estimate but says nothing about draw-to-draw
   variation. This re-draws independently, with different seeds, and reports the spread.

2. WHAT THE PUBLISHED FIGURE MEASURED. The published fingerprint puts CLEAN against its parent
   ESM-1b at rho = 0.38. On aligner-found pairs we measure 0.83, on uniformly random pairs
   -0.13 -- the published value sits between the two, and its own input is gone. The surviving
   clue is that its medians match a small all-vs-all table. An all-vs-all over a SMALL protein
   set is not the same population as uniformly random pairs over a large one: every protein
   appears in every pair, so the sample is dominated by whatever that small set happens to
   contain. This measures that directly, at several set sizes, to see whether a small all-vs-all
   reproduces the published value.

    python scripts/fingerprint_draw_stability.py --emb-dir <cohort2k> --out <dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

ARMS = ["clean", "esm1b", "prott5", "prottucker", "ankh_base", "ankh_large", "esmc_300m", "esm2_8m"]


def load_matrix(path: Path, ids: list[str]) -> np.ndarray:
    # mean(axis=0) over a 2-D dataset: the protein-level pooling rule
    # ridge_pair_distances.py and distance_computation.py share. Identical to a
    # flatten on the (1, D) cohort2k arms, and not a flatten on an (L, D) one.
    out = None
    with h5py.File(path, "r") as h:
        for i, pid in enumerate(ids):
            emb = np.asarray(h[pid])
            if emb.ndim > 1:
                emb = emb.mean(axis=0)
            if out is None:
                out = np.empty((len(ids), emb.size), dtype=np.float32)
            out[i] = emb
    return out


def distances(mat: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    out = np.empty(a.size, dtype=np.float64)
    for s in range(0, a.size, chunk):
        e = min(s + chunk, a.size)
        d = mat[a[s:e]].astype(np.float64) - mat[b[s:e]].astype(np.float64)
        out[s:e] = np.sqrt(np.einsum("ij,ij->i", d, d))
    return out


def pair_rhos(mats: dict[str, np.ndarray], a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = {arm: distances(m, a, b) for arm, m in mats.items()}
    return {
        f"{x}|{y}": float(spearmanr(d[x], d[y]).statistic)
        for x, y in itertools.combinations(ARMS, 2)
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-pairs", type=int, default=1_000_000)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--allvsall-sizes", type=int, nargs="*", default=[500, 1225, 3000])
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.emb_dir / "clean.h5") as h:
        ids = sorted(h.keys())
    mats = {arm: load_matrix(args.emb_dir / f"{arm}.h5", ids) for arm in ARMS}
    print(f"{len(ids):,} proteins, {len(ARMS)} arms loaded", flush=True)

    report: dict[str, object] = {"n_proteins": len(ids), "arms": ARMS}

    # 1) independent draws of uniformly random pairs
    draws = []
    for seed in range(args.draws):
        rng = np.random.default_rng(1000 + seed)
        a = rng.integers(0, len(ids), size=args.n_pairs)
        b = rng.integers(0, len(ids), size=args.n_pairs)
        keep = a != b
        rhos = pair_rhos(mats, a[keep], b[keep])
        draws.append({"seed": 1000 + seed, "n_pairs": int(keep.sum()), **rhos})
        print(f"  draw {seed}: clean|esm1b = {rhos['clean|esm1b']:+.4f}", flush=True)
    keys = [k for k in draws[0] if "|" in k]
    report["random_draws"] = draws
    report["random_spread"] = {
        k: {
            "mean": float(np.mean([d[k] for d in draws])),
            "sd": float(np.std([d[k] for d in draws], ddof=1)),
            "min": float(np.min([d[k] for d in draws])),
            "max": float(np.max([d[k] for d in draws])),
        }
        for k in keys
    }

    # 2) all-vs-all over a small protein set, the shape the published figure appears to have had
    allvsall = []
    for n_prot in args.allvsall_sizes:
        for seed in range(3):
            rng = np.random.default_rng(9000 + seed)
            sel = rng.choice(len(ids), n_prot, replace=False)
            ii, jj = np.triu_indices(n_prot, k=1)
            a, b = sel[ii], sel[jj]
            rhos = pair_rhos(mats, a, b)
            allvsall.append({"n_proteins": n_prot, "seed": 9000 + seed, "n_pairs": int(a.size), **rhos})
            print(f"  all-vs-all n={n_prot} seed={9000+seed}: clean|esm1b = {rhos['clean|esm1b']:+.4f}", flush=True)
    report["small_all_vs_all"] = allvsall

    (args.out / "draw_stability.json").write_text(json.dumps(report, indent=2))
    ce = report["random_spread"]["clean|esm1b"]
    print("\nclean|esm1b over independent random draws: "
          f"mean {ce['mean']:+.4f}, sd {ce['sd']:.4f}, range [{ce['min']:+.4f}, {ce['max']:+.4f}]")
    for n_prot in args.allvsall_sizes:
        vals = [r["clean|esm1b"] for r in allvsall if r["n_proteins"] == n_prot]
        print(f"clean|esm1b over all-vs-all of {n_prot} proteins: {['%+.4f' % v for v in vals]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
