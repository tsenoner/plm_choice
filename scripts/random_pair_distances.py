#!/usr/bin/env python3
"""Embedding distances over a uniform random sample of ALL protein pairs.

Why this exists. Figures 2 and 3 currently describe the pairs MMseqs2/Foldseek could align —
related proteins with a real similarity gradient. That is a biased slice of protein space, and
measurably so: between-model rank correlation is 0.77 on those pairs against 0.34 on random
ones, and CLEAN anti-correlates with every other model on random pairs while agreeing at
0.67-0.83 on aligned ones. If the claim is about protein space rather than about alignable
pairs, the figure has to be drawn on a uniform sample of it.

A true all-vs-all over the 526,871-protein cohort is 1.39e11 pairs, which is neither storable
nor necessary: a uniform random sample is unbiased by construction, and its precision is set by
the sample size alone. At n = 5,000,000 the standard error of a rank correlation is about
1/sqrt(n) = 4.5e-4 and quantile error is smaller still, so the sample is three orders of
magnitude finer than any difference the figures report. --convergence recomputes the headline
statistics at 10k / 100k / 1M / 5M so that claim is shown rather than asserted.

Identical sequences deposited under two accessions are dropped (they are the same protein
twice), by SHA-1 of the sequence rather than by a zero distance — the float16 arms store those
pairs at 0.0004-0.0056, so a distance test would miss them in exactly three arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import polars as pl


def sequence_hashes(fasta: Path, ids: list[str]) -> np.ndarray:
    """SHA-1 per id, in the order of `ids`, read streaming from a plain FASTA."""
    wanted = {pid: i for i, pid in enumerate(ids)}
    out = np.zeros(len(ids), dtype=object)
    current, chunks = None, []
    with fasta.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if current is not None and current in wanted:
                    out[wanted[current]] = hashlib.sha1("".join(chunks).encode()).hexdigest()
                header = line[1:].split()[0]
                current = header.split("|")[1] if "|" in header else header
                chunks = []
            else:
                chunks.append(line.strip())
    if current is not None and current in wanted:
        out[wanted[current]] = hashlib.sha1("".join(chunks).encode()).hexdigest()
    missing = int((out == 0).sum())
    if missing:
        raise SystemExit(f"{missing} cohort ids had no sequence in {fasta}")
    return out


def pair_distances(mat: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    out = np.empty(a.size, dtype=np.float64)
    for s in range(0, a.size, chunk):
        e = min(s + chunk, a.size)
        d = mat[a[s:e]].astype(np.float64) - mat[b[s:e]].astype(np.float64)
        out[s:e] = np.sqrt(np.einsum("ij,ij->i", d, d))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dir", required=True, type=Path)
    ap.add_argument("--fasta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-pairs", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--convergence", action="store_true")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    arms = sorted(p.stem for p in args.emb_dir.glob("*.h5"))
    # The cohort is the id set every arm shares; clean/esm1b are the smallest (<=1022 aa).
    with h5py.File(args.emb_dir / "clean.h5") as h:
        ids = sorted(h.keys())
    print(f"{len(ids):,} cohort proteins, {len(arms)} arms: {arms}", flush=True)

    h = sequence_hashes(args.fasta, ids)
    print(f"{len(set(h)):,} distinct sequences among them", flush=True)

    # Oversample, then drop self-pairs and identical-sequence pairs, then trim to n.
    need = int(args.n_pairs * 1.05)
    a = rng.integers(0, len(ids), size=need)
    b = rng.integers(0, len(ids), size=need)
    keep = (a != b) & (h[a] != h[b])
    a, b = a[keep][: args.n_pairs], b[keep][: args.n_pairs]
    # The 5% headroom is a guess about the collision and redundancy rate; nothing
    # guarantees it. Without this, a draw that fell short would write the parquet
    # anyway and record the short count in summary["n_pairs"], and the docstring's
    # precision argument ("at n = 5,000,000 the standard error is about
    # 1/sqrt(n) = 4.5e-4") would be resting on a number nothing checked.
    if a.size != args.n_pairs:
        raise SystemExit(
            f"asked for {args.n_pairs:,} pairs but only {a.size:,} of the {need:,} "
            f"drawn survived the self-pair and identical-sequence filters; raise the "
            f"oversampling headroom above 1.05x"
        )
    print(f"{a.size:,} random pairs kept ({need - keep.sum():,} dropped as self or identical)", flush=True)

    frame = {"query": [ids[i] for i in a], "target": [ids[i] for i in b]}
    for arm in arms:
        with h5py.File(args.emb_dir / f"{arm}.h5") as fh:
            mat = None
            for i, pid in enumerate(ids):
                # Protein-level pooling, the rule scripts/ridge_pair_distances.py and
                # src/data_preparation/distance_computation.py both use: mean over
                # axis 0 for a 2-D dataset. Identical to a flatten for the (1, D)
                # cohort2k files, and a silent flatten of an (L, D) one without it.
                emb = np.asarray(fh[pid])
                if emb.ndim > 1:
                    emb = emb.mean(axis=0)
                if mat is None:
                    mat = np.empty((len(ids), emb.size), dtype=np.float32)
                mat[i] = emb
        frame[f"dist_{arm}"] = pair_distances(mat, a, b)
        del mat
        print(f"  {arm} done", flush=True)

    out = pl.DataFrame(frame)
    out.write_parquet(args.out_dir / "random_pairs_distances.parquet")
    print(f"wrote {args.out_dir / 'random_pairs_distances.parquet'} ({out.height:,} rows)")

    summary: dict[str, object] = {
        "n_pairs": int(a.size),
        "n_proteins_in_cohort": len(ids),
        "n_proteins_sampled": int(len(set(a.tolist()) | set(b.tolist()))),
        "seed": args.seed,
        "arms": arms,
    }
    if args.convergence:
        from scipy.stats import spearmanr

        probe = [("clean", "esm1b"), ("ankh_base", "ankh_large"), ("esmc_300m", "esmc_600m")]
        conv = []
        for n in (10_000, 100_000, 1_000_000, a.size):
            if n > a.size:
                continue
            idx = rng.choice(a.size, n, replace=False) if n < a.size else np.arange(a.size)
            row: dict[str, object] = {"n": int(n)}
            for x, y in probe:
                row[f"rho_{x}_{y}"] = float(
                    spearmanr(out[f"dist_{x}"].to_numpy()[idx], out[f"dist_{y}"].to_numpy()[idx]).statistic
                )
            for arm in ("clean", "esm2_650m"):
                q = np.percentile(out[f"dist_{arm}"].to_numpy()[idx], [25, 50, 75])
                row[f"q25_{arm}"], row[f"median_{arm}"], row[f"q75_{arm}"] = (float(v) for v in q)
            conv.append(row)
        pl.DataFrame(conv).write_csv(args.out_dir / "convergence.csv")
        summary["convergence"] = conv
        print(pl.DataFrame(conv))

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "convergence"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
