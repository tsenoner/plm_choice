#!/usr/bin/env python3
"""B4 degeneracy gate — is an arm's embedding space usable, or has it collapsed?

Samples N proteins per arm and reports, per arm:

  raw cosine      mean pairwise cos(x_i, x_j).  REPORTED, NEVER GATED ON.
                  Transformers are anisotropic: all vectors sit in a narrow cone,
                  so this runs high even for healthy models. The real pretrained
                  prostt5 arm measures 0.647, and a synthetic *healthy* control
                  measures 0.97 -- so the plan's ">0.95 = worthless" rule would
                  discard a good arm.
  centred cosine  same, after subtracting the mean vector. Removes the common
                  offset and asks the real question: do proteins differ at all?
  PR              participation ratio of the centred covariance,
                  (sum lambda)^2 / sum(lambda^2) -- how many directions the
                  variance actually occupies. Needed because a rank-1 collapse
                  gives centred cosine ~0 (which LOOKS healthy) at PR ~1.5.

Fails on centred cosine or PR. Both, because they catch different collapses.

Sampling note: the arms hold ~542k top-level datasets. Enumerating f.keys() with
an isinstance() check opens every object and takes longer than the whole rest of
the job on GPFS, so names are fetched by INDEX -- 1000 lookups, not 542,238.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

CENTRED_FAIL, CENTRED_WARN = 0.90, 0.50
PR_FAIL, PR_WARN = 5.0, 10.0


def sample(path: str, n: int, seed: int, window: int = 20000):
    """Sample n vectors without enumerating all ~542k links.

    h5py's get_objname_by_idx is O(number_of_links) PER CALL, so 1000 of them on
    a 542k-object file costs 86 s locally and ~27 min over GPFS -- longer than
    everything else combined. Lazy iteration of f.keys() is ~430x faster because
    it walks the link table once and can stop early.

    So: walk a bounded prefix of `window` links and stride through it for n
    samples. That is a PREFIX sample (datasets were created in FASTA/accession
    order), not a uniform one over the whole file. Acceptable here -- the question
    is whether protein vectors differ from each other at all, and a collapsed
    space is collapsed everywhere. If the prefix happened to be unusually
    homogeneous the bias would raise centred cosine and lower PR, i.e. toward a
    false ALARM rather than a false pass, which is the safe direction for a gate.
    """
    import itertools
    with h5py.File(path, "r") as f:
        total = f.id.get_num_objs()
        win = min(window, total)
        step = max(1, win // n)
        names = list(itertools.islice(f.keys(), 0, win, step))[:n]
        # prott5.h5 / prottucker.h5 ship (1, D) with a leading singleton axis
        # where every other arm ships (D,). Squeeze it rather than crash, and let
        # the caller see that the layout differs.
        X = np.stack([np.asarray(f[nm][()], dtype=np.float64).reshape(-1) for nm in names])
    return X, total


def stats(X: np.ndarray):
    n, d = X.shape
    finite = np.isfinite(X).all(axis=1)
    nonfinite = int((~finite).sum())
    X = X[finite]
    norms = np.linalg.norm(X, axis=1)
    zero = int((norms == 0).sum())
    X = X[norms > 0]
    norms = norms[norms > 0]

    def pairwise(M):
        nrm = np.linalg.norm(M, axis=1, keepdims=True)
        M = M[(nrm > 0).ravel()]
        nrm = nrm[(nrm > 0).ravel()]
        U = M / nrm
        G = U @ U.T
        iu = np.triu_indices(len(U), k=1)
        return G[iu]

    raw = pairwise(X)
    Xc = X - X.mean(axis=0, keepdims=True)
    keep = np.linalg.norm(Xc, axis=1) > 0
    cen = pairwise(Xc[keep]) if keep.sum() > 1 else np.array([0.0])

    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s**2 / max(len(Xc) - 1, 1)
    pr = float(lam.sum() ** 2 / np.sum(lam**2)) if lam.sum() > 0 else 0.0

    g = np.random.default_rng(0).standard_normal(X.shape)
    ctrl = pairwise(g)

    distinct = len({r.tobytes() for r in np.ascontiguousarray(X)})
    return dict(
        n=int(len(X)), d=int(d), nonfinite=nonfinite, zero_norm=zero,
        distinct=distinct, norm_cv=float(norms.std() / norms.mean()),
        raw_mean=float(raw.mean()), raw_med=float(np.median(raw)),
        raw_p5=float(np.percentile(raw, 5)), raw_p95=float(np.percentile(raw, 95)),
        cen_mean=float(cen.mean()), cen_med=float(np.median(cen)),
        pr=pr, max_rank=int(min(len(Xc) - 1, d)),
        top1_evr=float(lam[0] / lam.sum()) if lam.sum() > 0 else 0.0,
        ctrl_mean=float(ctrl.mean()),
    )


def verdict(s):
    why = []
    if s["cen_mean"] >= CENTRED_FAIL:
        why.append(f"centred cosine {s['cen_mean']:.3f} >= {CENTRED_FAIL}")
    if s["pr"] <= PR_FAIL:
        why.append(f"participation ratio {s['pr']:.2f} <= {PR_FAIL}")
    if s["nonfinite"]:
        why.append(f"{s['nonfinite']} non-finite vectors")
    if s["zero_norm"]:
        why.append(f"{s['zero_norm']} zero-norm vectors")
    if s["distinct"] < 0.5 * s["n"]:
        why.append(f"only {s['distinct']}/{s['n']} distinct")
    if why:
        return "FAIL", "; ".join(why)
    warn = []
    if s["cen_mean"] >= CENTRED_WARN:
        warn.append(f"centred cosine {s['cen_mean']:.3f}")
    if s["pr"] <= PR_WARN:
        warn.append(f"PR {s['pr']:.2f}")
    return ("WARN", "; ".join(warn)) if warn else ("PASS", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    files = sorted({p for pat in a.paths for p in glob.glob(pat)})
    print(f"{'arm':<32}{'keys':>9}{'dim':>6}{'raw':>8}{'centred':>9}{'PR':>8}{'ctrl':>8}  verdict")
    print("-" * 96)
    out, worst = {}, "PASS"
    for p in files:
        name = os.path.basename(p).replace(".h5", "")
        try:
            X, total = sample(p, a.n, a.seed)
            s = stats(X)
            s["keys"] = total
            v, why = verdict(s)
            s["verdict"], s["why"] = v, why
            flags = []
            if s['nonfinite']:
                flags.append(f"nonfinite={s['nonfinite']}")
            if s['zero_norm']:
                flags.append(f"zeronorm={s['zero_norm']}")
            if s['distinct'] < s['n']:
                flags.append(f"distinct={s['distinct']}/{s['n']}")
            print(f"{name:<32}{total:>9,}{s['d']:>6}{s['raw_mean']:>8.4f}"
                  f"{s['cen_mean']:>9.4f}{s['pr']:>8.2f}{s['ctrl_mean']:>8.4f}  {v}"
                  + (f"  ({why})" if why else "")
                  + (f"  [{', '.join(flags)}]" if flags else ""), flush=True)
            out[name] = s
            if v == "FAIL" or (v == "WARN" and worst == "PASS"):
                worst = v
        except Exception as e:
            print(f"{name:<32}  ERROR {type(e).__name__}: {e}", flush=True)
            out[name] = {"verdict": "ERROR", "why": str(e)}
            worst = "FAIL"
    print("-" * 96)
    print(f"OVERALL: {worst}   (gate: centred cosine >= {CENTRED_FAIL} or PR <= {PR_FAIL} => FAIL)")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
