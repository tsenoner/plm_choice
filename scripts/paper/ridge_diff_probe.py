#!/usr/bin/env python3
"""Closed-form ridge regression on |a-b| or (a-b)^2: the transparent estimator of R1.8.

    python scripts/paper/ridge_diff_probe.py \
        --embedding $DSS/plm_choice_data/data/processed/full100_p10/embeddings/prott5.h5 \
        --sets-dir $DSS/plm_choice_data/data/processed/sprot_pre2024_e1/sets \
        --out $DSS/d14_open/ridge

The reviewer asked for "a simpler, transparent estimator (e.g., linear/ridge on |a-b|)". This
is that estimator, fitted exactly rather than by SGD, on the same pairs as the paper's probes:

* **Pairs.** Train, validation and test splits are read through
  ``shared.datasets._load_and_filter_data`` -- the loader the FNN and Euclidean cells use -- so
  null targets and proteins outside the committed cohort freeze are dropped exactly as there.
  All training pairs are used (the final protocol), not a subsample.
* **Features.** ``abs``: |a-b| (the reviewer's wording). ``sq``: (a-b)^2, which is the model
  ``LinearDistancePredictor`` (``src/training/models.py``) trains by SGD: a learned per-dimension
  weighting of the squared Euclidean distance. With equal negative weights it ranks pairs as
  the Euclidean read-out does, so under Spearman ``sq`` nests that baseline.
* **Fit.** Sufficient statistics (n, sum x, X'X, X'y) accumulate over the training pairs on the
  GPU in float64. Features are standardised with the training moments; the ridge solution for
  every alpha in the grid comes from one eigendecomposition of the correlation matrix (alpha is
  relative to n, i.e. in units of the standardised X'X/n). Alpha is chosen by validation MSE;
  the test split is touched once, for every alpha, and the selected alpha is reported.
* **Check.** The test pass also scores -||a-b|| -- the Euclidean read-out -- whose Spearman
  must equal the one in the paper's Euclidean metrics file for the arm; ``--check-metrics``
  asserts it (to 1e-4) so the pairs are proven identical, not assumed.

Output, per (target, feature): one JSON with every alpha's validation/test scores and the
selected one, plus ``<target>_<feature>_weights.npz`` (weights in the original units, bias, alpha).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from shared.datasets import _load_and_filter_data  # noqa: E402

TARGETS = ("fident", "alntmscore", "hfsp")
FEATURES = ("abs", "sq")
ALPHAS = (1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def load_embeddings(h5_path: Path, ids: list[str]) -> np.ndarray:
    """One row per id, in the order given (float32, as the probe loader returns them)."""
    with h5py.File(h5_path, "r") as f:
        d = int(np.prod(f[ids[0]].shape))
        emb = np.empty((len(ids), d), np.float32)
        for i, pid in enumerate(ids):
            emb[i] = f[pid][:].reshape(-1)
    return emb


def pair_index(df: pl.DataFrame, index: pl.DataFrame, param: str):
    """query/target ids -> row numbers in the embedding matrix; target column as float64."""
    out = (
        df.with_row_index("_row")
        .join(index.rename({"pid": "query", "idx": "qi"}), on="query", how="left")
        .join(index.rename({"pid": "target", "idx": "ti"}), on="target", how="left")
        .sort("_row")
    )
    if out["qi"].null_count() or out["ti"].null_count():
        raise RuntimeError("a pair protein is missing from the embedding index")
    return (
        out["qi"].to_numpy().astype(np.int64),
        out["ti"].to_numpy().astype(np.int64),
        out[param].to_numpy().astype(np.float64),
    )


def chunks(n: int, size: int):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def features(emb: torch.Tensor, qi: torch.Tensor, ti: torch.Tensor, kind: str):
    diff = emb[qi] - emb[ti]
    feat = diff.abs() if kind == "abs" else diff * diff
    return feat.double(), (diff * diff).sum(1)


def fit(emb, qi, ti, y, kind, chunk, dev):
    d = emb.shape[1]
    sx = torch.zeros(d, dtype=torch.float64, device=dev)
    sxx = torch.zeros(d, d, dtype=torch.float64, device=dev)
    sxy = torch.zeros(d, dtype=torch.float64, device=dev)
    sy = syy = 0.0
    n = len(y)
    for s in chunks(n, chunk):
        x, _ = features(emb, qi[s], ti[s], kind)
        yy = torch.as_tensor(y[s], device=dev)
        sx += x.sum(0)
        sxx += x.T @ x
        sxy += x.T @ yy
        sy += float(yy.sum())
        syy += float((yy * yy).sum())
    mu, ybar = sx / n, sy / n
    cov = sxx / n - torch.outer(mu, mu)
    cxy = sxy / n - mu * ybar
    sd = torch.sqrt(torch.clamp(torch.diagonal(cov), min=0.0))
    live = sd > 1e-12 * max(float(sd.max()), 1e-300)
    sd = torch.where(live, sd, torch.ones_like(sd))
    corr = cov / torch.outer(sd, sd)
    r = torch.where(live, cxy / sd, torch.zeros_like(cxy))
    corr[~live, :] = 0.0
    corr[:, ~live] = 0.0
    evals, evecs = torch.linalg.eigh(corr)
    proj = evecs.T @ r
    weights, biases = [], []
    for a in ALPHAS:
        w_std = evecs @ (proj / (torch.clamp(evals, min=0.0) + a))
        w_std = torch.where(live, w_std, torch.zeros_like(w_std))
        w = w_std / sd
        weights.append(w)
        biases.append(ybar - float(mu @ w))
    return torch.stack(weights, 1), torch.tensor(biases, dtype=torch.float64, device=dev), {
        "n": n,
        "dims": d,
        "dims_constant": int((~live).sum()),
        "y_var": syy / n - ybar**2,
        "corr_eig_min": float(evals.min()),
        "corr_eig_max": float(evals.max()),
    }


def predict(emb, qi, ti, W, b, kind, chunk, dev):
    preds, d2 = [], []
    for s in chunks(len(qi), chunk):
        x, dist2 = features(emb, qi[s], ti[s], kind)
        preds.append((x @ W + b).cpu().numpy())
        d2.append(dist2.double().cpu().numpy())
    return np.concatenate(preds), np.concatenate(d2)


def euclid_spearman_from_metrics(run_dir: Path) -> float | None:
    for f in sorted((run_dir / "evaluation_results").glob("*_metrics.txt")):
        for line in f.read_text().splitlines():
            if line.startswith("Spearman:"):
                return float(line.split(":", 1)[1])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", type=Path, required=True)
    ap.add_argument("--sets-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=TARGETS)
    ap.add_argument("--features", nargs="+", default=list(FEATURES), choices=FEATURES)
    ap.add_argument(
        "--check-metrics",
        type=Path,
        help="models/<grid>/euclidean dir: assert the replayed Euclidean Spearman equals the paper's",
    )
    args = ap.parse_args()

    arm = args.embedding.stem
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.out / arm
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    splits = {
        (t, s): _load_and_filter_data(str(args.sets_dir / f"{s}.parquet"), str(args.embedding), t)
        for t in args.targets
        for s in ("train", "val", "test")
    }
    ids = sorted(
        set().union(*(set(df["query"].to_list()) | set(df["target"].to_list()) for df in splits.values()))
    )
    index = pl.DataFrame({"pid": ids, "idx": np.arange(len(ids), dtype=np.int64)})
    emb_np = load_embeddings(args.embedding, ids)
    emb = torch.as_tensor(emb_np, device=dev)
    d = emb.shape[1]
    chunk = max(1 << 14, (1 << 28) // d)
    print(f"{arm}: {len(ids):,} proteins x {d} dims loaded in {time.time() - t0:.0f}s on {dev}", flush=True)

    for target in args.targets:
        idx = {s: pair_index(splits[(target, s)], index, target) for s in ("train", "val", "test")}
        gpu = {s: (torch.as_tensor(q, device=dev), torch.as_tensor(t, device=dev)) for s, (q, t, _) in idx.items()}
        for kind in args.features:
            t1 = time.time()
            W, b, info = fit(emb, *gpu["train"], idx["train"][2], kind, chunk, dev)
            pv, _ = predict(emb, *gpu["val"], W, b, kind, chunk, dev)
            yv = idx["val"][2]
            val_mse = ((pv - yv[:, None]) ** 2).mean(0)
            best = int(np.argmin(val_mse))
            pt, d2 = predict(emb, *gpu["test"], W, b, kind, chunk, dev)
            yt = idx["test"][2]
            rows = []
            for k, a in enumerate(ALPHAS):
                rows.append(
                    {
                        "alpha": a,
                        "val_mse": float(val_mse[k]),
                        "val_spearman": float(stats.spearmanr(pv[:, k], yv).statistic),
                        "test_mse": float(((pt[:, k] - yt) ** 2).mean()),
                        "test_spearman": float(stats.spearmanr(pt[:, k], yt).statistic),
                        "test_pearson": float(stats.pearsonr(pt[:, k], yt).statistic),
                    }
                )
            euclid = float(stats.spearmanr(-np.sqrt(d2), yt).statistic)
            check = None
            if args.check_metrics is not None:
                ref = euclid_spearman_from_metrics(args.check_metrics / target / arm)
                check = {"paper_euclidean_spearman": ref, "replayed": -euclid}
                if ref is None or abs(ref - (-euclid)) > 1e-4:
                    raise RuntimeError(f"{arm}/{target}: replayed Euclidean {-euclid:.6f} != paper {ref}")
            res = {
                "arm": arm,
                "target": target,
                "feature": kind,
                "n_train": len(idx["train"][2]),
                "n_val": len(yv),
                "n_test": len(yt),
                "fit": info,
                "alphas": rows,
                "selected": rows[best],
                "euclidean_spearman_replayed": -euclid,
                "euclidean_check": check,
                "embedding": str(args.embedding),
                "seconds": round(time.time() - t1, 1),
            }
            (out / f"{target}_{kind}.json").write_text(json.dumps(res, indent=1))
            np.savez(
                out / f"{target}_{kind}_weights.npz",
                weights=W[:, best].cpu().numpy(),
                bias=float(b[best]),
                alpha=ALPHAS[best],
            )
            s = rows[best]
            print(
                f"{arm} {target} {kind}: alpha={s['alpha']:g} test rho={s['test_spearman']:.4f} "
                f"(euclid {-euclid:.4f}; n_train={len(idx['train'][2]):,}) {res['seconds']}s",
                flush=True,
            )
        del gpu
    print(f"{arm}: done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
