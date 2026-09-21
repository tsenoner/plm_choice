#!/usr/bin/env python3
"""Pre-flight / post-flight degeneracy check for a random-init embedding arm.

Answers ONE question: does this H5 hold a usable embedding space, or has every
protein collapsed onto (nearly) the same vector?

Layout assumed (verified against src/data_preparation/embeddings/
embedding_generation.py:939 and :1046): one top-level HDF5 *dataset per protein*,
named by the FASTA accession, shape (D,), dtype float32 (legacy pretrained files
are float16 -- both are handled).

Usage
-----
    # a 1-2k-protein smoke run, BEFORE the 39-task array
    python preflight_degeneracy_check.py smoke/random_init_esm2_650m_seed0.h5

    # calibrate against a known-good PRETRAINED arm (do this first!)
    python preflight_degeneracy_check.py smoke/random_init_prost_t5_seed0.h5 \
        --reference data/processed/sprot_pre2024/embeddings/prostt5.h5

    # optional: is the space just sequence length in disguise?
    python preflight_degeneracy_check.py FILE --fasta data/raw/sprot_2024/sprot.fasta

    # machine-readable, for a barrier / afterok gate
    python preflight_degeneracy_check.py FILE --json report.json

Exit codes: 0 = PASS/WARN, 1 = FAIL (degenerate), 2 = could not evaluate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

# -- pass criteria (see write-up; RAW cosine is REPORTED, never gated) --------
CENTERED_COS_FAIL = 0.90   # centred mean pairwise cosine at/above this = dead
CENTERED_COS_WARN = 0.50
PR_FAIL = 5.0              # participation ratio of the centred covariance
PR_WARN = 10.0
RAW_COS_SUSPECT = 0.9999   # raw cosine only matters when it is this extreme
LENGTH_VAR_WARN = 0.90     # share of centred variance explained by log(length)
DISTINCT_FRAC_FAIL = 0.50  # SwissProt really does contain identical sequences:
DISTINCT_FRAC_WARN = 0.98  # 2/1000 duplicates is normal, 500/1000 is collapse
CROSS_SEED_COS_FAIL = 0.999  # two seeds must not produce the same vectors


def sample_matrix(path: Path, n: int, seed: int):
    """Return (X, ids, n_total, d, dtype, bad_dim). X is (n, d) float64."""
    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
        if not keys:
            raise SystemExit(f"{path}: no top-level datasets (empty artifact)")
        n_total = len(keys)
        take = min(n, n_total)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=take, replace=False)
        idx.sort()
        ids = [keys[i] for i in idx]

        first = np.asarray(f[ids[0]][()])
        if first.ndim != 1:
            raise SystemExit(
                f"{path}: '{ids[0]}' has shape {first.shape}; this script expects "
                "per_protein (1-D) vectors, not per_residue matrices."
            )
        d = int(first.shape[0])
        raw_dtype = str(f[ids[0]].dtype)
        X = np.empty((take, d), dtype=np.float64)
        bad_dim = []
        for row, k in enumerate(ids):
            v = np.asarray(f[k][()])
            if v.ndim != 1 or v.shape[0] != d:
                bad_dim.append((k, tuple(v.shape)))
                X[row] = np.nan
            else:
                X[row] = v
    return X, ids, n_total, d, raw_dtype, bad_dim


def cosine_stats(X: np.ndarray) -> dict:
    """Mean/median/sd/p5/p95 over all n*(n-1)/2 pairwise cosines. Vectorised."""
    norms = np.linalg.norm(X, axis=1)
    ok = norms > 0
    Xn = X[ok] / norms[ok, None]
    # errstate: some BLAS builds (macOS Accelerate) emit spurious divide/overflow
    # warnings from matmul. The inputs are already checked finite by the caller.
    with np.errstate(all="ignore"):
        G = Xn @ Xn.T
    iu = np.triu_indices(Xn.shape[0], k=1)
    c = np.clip(G[iu], -1.0, 1.0)
    return {
        "n_pairs": int(c.size),
        "mean": float(c.mean()),
        "median": float(np.median(c)),
        "sd": float(c.std()),
        "p5": float(np.percentile(c, 5)),
        "p95": float(np.percentile(c, 95)),
        "min": float(c.min()),
        "max": float(c.max()),
    }


def spectrum_stats(Xc: np.ndarray) -> dict:
    """Effective dimensionality of the CENTRED matrix."""
    n, d = Xc.shape
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s**2
    tot = float(lam.sum())
    if tot <= 0:
        return {
            "participation_ratio": 0.0, "evr_top1": float("nan"),
            "evr_top10": float("nan"), "effective_rank": 0.0,
            "numerical_rank": 0, "max_rank": int(min(n - 1, d)),
        }
    p = lam / tot
    pr = float(tot**2 / float((lam**2).sum()))
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    tol = float(s.max()) * max(n, d) * np.finfo(np.float64).eps
    return {
        "participation_ratio": pr,
        "evr_top1": float(p[0]),
        "evr_top10": float(p[:10].sum()),
        "effective_rank": float(np.exp(ent)),
        "numerical_rank": int((s > tol).sum()),
        "max_rank": int(min(n - 1, d)),
    }


def fasta_lengths(fasta: Path, wanted: set) -> dict:
    lengths, name, n = {}, None, 0
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None and name in wanted:
                    lengths[name] = n
                name, n = line[1:].split()[0], 0
            else:
                n += len(line.strip())
    if name is not None and name in wanted:
        lengths[name] = n
    return lengths


def length_variance_share(Xc: np.ndarray, loglen: np.ndarray) -> float:
    """Share of centred variance a linear function of log(length) explains."""
    t = loglen - loglen.mean()
    with np.errstate(all="ignore"):
        denom = float(t @ t)
        if denom <= 0:
            return float("nan")
        beta = (t @ Xc) / denom
    fitted = np.outer(t, beta)
    return float((fitted**2).sum() / (Xc**2).sum())


def cross_seed_check(paths, n: int, seed: int) -> dict | None:
    """Same protein, different seed files -> the vectors must actually differ.

    This is the direct guard against the 'sd = 0.000' failure: if --random_seed
    silently does not reach the weights, all three seeds give identical output
    and the published spread is a fiction.
    """
    if len(paths) < 2:
        return None
    mats, dims = [], set()
    ref_ids = None
    for p in paths:
        X, ids, _, d, _, _ = sample_matrix(Path(p), n, seed)
        if ref_ids is None:
            ref_ids = ids
        elif ids != ref_ids:
            return {"error": "files hold different protein sets; cannot pair them"}
        dims.add(d)
        mats.append(X)
    if len(dims) != 1:
        return {"error": f"differing dimensions {sorted(dims)}; not the same model"}
    out = {"pairs": []}
    A = mats[0]
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    for j in range(1, len(mats)):
        B = mats[j]
        Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
        with np.errstate(all="ignore"):
            per_protein = np.clip((An * Bn).sum(axis=1), -1.0, 1.0)
        out["pairs"].append({
            "a": str(paths[0]), "b": str(paths[j]),
            "mean_same_protein_cosine": float(per_protein.mean()),
            "min": float(per_protein.min()), "max": float(per_protein.max()),
            "identical": bool(np.array_equal(A, B)),
        })
    return out


def analyse(path: Path, n: int, seed: int, fasta: Path | None, label: str) -> dict:
    X, ids, n_total, d, dtype, bad_dim = sample_matrix(path, n, seed)
    rep: dict = {
        "label": label, "path": str(path), "n_datasets_in_file": n_total,
        "n_sampled": int(X.shape[0]), "dim": d, "stored_dtype": dtype,
        "fatal": [], "warn": [],
    }
    if bad_dim:
        rep["fatal"].append(
            f"{len(bad_dim)} sampled vector(s) have a wrong shape, e.g. {bad_dim[:3]}"
        )

    finite_rows = np.isfinite(X).all(axis=1)
    rep["n_nonfinite"] = int((~finite_rows).sum())
    if rep["n_nonfinite"]:
        rep["fatal"].append(
            f"{rep['n_nonfinite']}/{X.shape[0]} sampled vectors contain NaN/Inf"
        )
    X = X[finite_rows]
    ids = [i for i, keep in zip(ids, finite_rows) if keep]
    if X.shape[0] < 10:
        rep["fatal"].append("fewer than 10 usable vectors; nothing to measure")
        rep["verdict"] = "FAIL"
        return rep

    norms = np.linalg.norm(X, axis=1)
    rep["norm_mean"] = float(norms.mean())
    rep["norm_cv"] = float(norms.std() / norms.mean()) if norms.mean() else float("nan")
    rep["n_zero_norm"] = int((norms == 0).sum())
    if rep["n_zero_norm"]:
        rep["fatal"].append(f"{rep['n_zero_norm']} all-zero vectors")

    # SwissProt contains genuinely identical sequences under different accessions
    # (verified: C3PAG3/Q633M2 and Q1CKF9/Q667T6 in a 1000-protein sample), so a
    # handful of bit-identical vectors is CORRECT, not a defect. Only a large
    # duplicate fraction means collapse.
    rep["n_distinct_vectors"] = int(np.unique(X, axis=0).shape[0])
    frac = rep["n_distinct_vectors"] / X.shape[0]
    if frac < DISTINCT_FRAC_FAIL:
        rep["fatal"].append(
            f"only {rep['n_distinct_vectors']}/{X.shape[0]} distinct vectors "
            f"({frac:.1%}): the model is emitting one vector for many proteins"
        )
    elif frac < DISTINCT_FRAC_WARN:
        rep["warn"].append(
            f"{X.shape[0] - rep['n_distinct_vectors']}/{X.shape[0]} bit-identical "
            "duplicate vectors (a few are expected: SwissProt has duplicate sequences)"
        )

    rep["n_dead_dims"] = int((X.std(axis=0) == 0).sum())

    rep["raw_cosine"] = cosine_stats(X)

    mu = X.mean(axis=0)
    Xc = X - mu
    rep["centered_cosine"] = cosine_stats(Xc)
    rep["spectrum"] = spectrum_stats(Xc)

    with np.errstate(all="ignore"):
        mu2 = float(mu @ mu)
    trS = float((Xc**2).sum() / (X.shape[0] - 1))
    r = trS / mu2 if mu2 > 0 else float("inf")
    rep["anisotropy_ratio_trSigma_over_mu2"] = r
    rep["raw_cosine_predicted_from_anisotropy"] = 1.0 / (1.0 + r) if np.isfinite(r) else 0.0

    rng = np.random.default_rng(seed + 12345)
    rep["gaussian_control_cosine"] = cosine_stats(rng.standard_normal(X.shape))

    if fasta is not None:
        try:
            lens = fasta_lengths(fasta, set(ids))
            pos = {k: i for i, k in enumerate(ids)}
            have = [k for k in ids if k in lens]
            if len(have) >= 10:
                sel = np.array([pos[k] for k in have])
                loglen = np.log(np.array([lens[k] for k in have], dtype=np.float64))
                share = length_variance_share(Xc[sel], loglen)
                rep["length_variance_share"] = share
                if share > LENGTH_VAR_WARN:
                    rep["warn"].append(
                        f"{100 * share:.1f}% of the centred variance is a linear function of "
                        "log(length): the space may encode little but length"
                    )
        except Exception as e:  # noqa: BLE001 - diagnostic only
            rep["warn"].append(f"length diagnostic skipped: {type(e).__name__}: {e}")

    cc = rep["centered_cosine"]["mean"]
    pr = rep["spectrum"]["participation_ratio"]
    if cc >= CENTERED_COS_FAIL:
        rep["fatal"].append(
            f"centred mean pairwise cosine {cc:+.4f} >= {CENTERED_COS_FAIL}: after removing "
            "the common mean the proteins are STILL the same vector -> degenerate"
        )
    elif abs(cc) >= CENTERED_COS_WARN:
        rep["warn"].append(
            f"centred mean pairwise cosine {cc:+.4f} is high (>{CENTERED_COS_WARN})"
        )
    if pr < PR_FAIL:
        rep["fatal"].append(
            f"participation ratio {pr:.2f} < {PR_FAIL}: the centred space has fewer than "
            f"{PR_FAIL:.0f} effective directions out of {rep['spectrum']['max_rank']}"
        )
    elif pr < PR_WARN:
        rep["warn"].append(f"participation ratio {pr:.2f} < {PR_WARN}")
    if rep["raw_cosine"]["mean"] > RAW_COS_SUSPECT:
        rep["warn"].append(
            f"raw mean cosine {rep['raw_cosine']['mean']:.6f} > {RAW_COS_SUSPECT}: extreme; "
            "confirm the residual survives float32 storage"
        )
    if rep["centered_cosine"]["sd"] < 1e-6:
        rep["fatal"].append("centred pairwise cosines have no spread at all")

    rep["verdict"] = "FAIL" if rep["fatal"] else ("WARN" if rep["warn"] else "PASS")
    return rep


def show(rep: dict) -> None:
    print(f"\n=== {rep['label']} ===")
    print(f"  file                  {rep['path']}")
    print(f"  datasets in file      {rep['n_datasets_in_file']}")
    print(f"  sampled x dim (dtype) {rep['n_sampled']} x {rep['dim']} ({rep['stored_dtype']})")
    if "norm_mean" not in rep:
        for m in rep["fatal"]:
            print(f"  FATAL {m}")
        print(f"  VERDICT: {rep['verdict']}")
        return
    c, cc = rep["raw_cosine"], rep["centered_cosine"]
    g, sp = rep["gaussian_control_cosine"], rep["spectrum"]
    print(f"  ||x|| mean / CV       {rep['norm_mean']:.4g} / {rep['norm_cv']:.4g}")
    print("  distinct / nonfinite / zero-norm / dead dims  "
          f"{rep['n_distinct_vectors']} / {rep['n_nonfinite']} / "
          f"{rep['n_zero_norm']} / {rep['n_dead_dims']}")
    print(f"  {'':<22}{'mean':>10}{'median':>10}{'sd':>10}{'p5':>10}{'p95':>10}")
    for name, s in (("RAW cosine", c), ("CENTRED cosine", cc), ("Gaussian control", g)):
        print(f"  {name:<22}{s['mean']:>10.4f}{s['median']:>10.4f}"
              f"{s['sd']:>10.4f}{s['p5']:>10.4f}{s['p95']:>10.4f}")
    print(f"  pairs per matrix      {c['n_pairs']}")
    print(f"  anisotropy r=tr(S)/||mu||^2   {rep['anisotropy_ratio_trSigma_over_mu2']:.4g}"
          f"  (predicts raw cos ~ {rep['raw_cosine_predicted_from_anisotropy']:.4f})")
    print(f"  participation ratio   {sp['participation_ratio']:.2f}  of max rank {sp['max_rank']}")
    print(f"  effective rank(exp H) {sp['effective_rank']:.2f}   numerical rank {sp['numerical_rank']}")
    print(f"  top-1 / top-10 EVR    {sp['evr_top1']:.4f} / {sp['evr_top10']:.4f}")
    if "length_variance_share" in rep:
        print(f"  var expl. by log(len) {rep['length_variance_share']:.4f}")
    for m in rep["warn"]:
        print(f"  WARN  {m}")
    for m in rep["fatal"]:
        print(f"  FATAL {m}")
    print(f"  VERDICT: {rep['verdict']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("h5", type=Path, nargs="+", help="random_init_<model>_seed<N>.h5")
    ap.add_argument("--n", type=int, default=1000, help="proteins to sample (default 1000)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (fixed => reproducible)")
    ap.add_argument("--reference", type=Path, default=None,
                    help="a known-good PRETRAINED .h5 to calibrate against")
    ap.add_argument("--fasta", type=Path, default=None,
                    help="optional FASTA for the log(length) diagnostic")
    ap.add_argument("--json", type=Path, default=None, help="write the full report as JSON")
    ap.add_argument("--no-cross-seed", action="store_true",
                    help="skip the seed-0/1/2 identity check when >1 file is given")
    args = ap.parse_args()

    reports = []
    cross = None
    try:
        if args.reference is not None:
            reports.append(analyse(args.reference, args.n, args.seed, args.fasta,
                                   f"REFERENCE (pretrained) {args.reference.name}"))
        for p in args.h5:
            reports.append(analyse(p, args.n, args.seed, args.fasta, p.name))
        if not args.no_cross_seed:
            cross = cross_seed_check(args.h5, args.n, args.seed)
    except (OSError, SystemExit) as e:
        print(f"[FATAL] could not evaluate: {e}", file=sys.stderr)
        return 2

    for r in reports:
        show(r)

    cross_failed = False
    if cross:
        print("\n=== cross-seed identity check ===")
        if "error" in cross:
            print(f"  skipped: {cross['error']}")
        for pr in cross.get("pairs", []):
            bad = pr["identical"] or pr["mean_same_protein_cosine"] > CROSS_SEED_COS_FAIL
            cross_failed |= bad
            print(f"  {'FATAL' if bad else 'ok   '} same-protein cosine "
                  f"{pr['mean_same_protein_cosine']:.6f} "
                  f"(min {pr['min']:.4f}) {Path(pr['a']).name} vs {Path(pr['b']).name}"
                  + ("  [BIT-IDENTICAL FILES: --random_seed had no effect]"
                     if pr["identical"] else ""))

    if args.json:
        args.json.write_text(json.dumps({"arms": reports, "cross_seed": cross}, indent=2))
        print(f"\nwrote {args.json}")

    print("\n--- summary ---")
    for r in reports:
        print(f"  {r['verdict']:<5} {r['label']}")
    failed = any(r["verdict"] == "FAIL" for r in reports) or cross_failed
    print(f"  overall: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
