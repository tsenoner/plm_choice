#!/usr/bin/env python3
"""UMAP of a superkingdom-stratified protein subsample, and what a 2-D map keeps (R1.3).

    python scripts/paper/umap_subsample.py \
        --sprot-json $DSS/plm_choice_data/data/raw/sprot_2024/sprot.json \
        --emb-dir $DSS/plm_choice_data/data/processed/full100_p10/embeddings \
        --arms prott5 esmc_600m random_1024 --out $DSS/d14_open/umap

The reviewer asked for "a UMAP or tSNE of the embedding space". The paper declines it on the
grounds that a 2-D projection distorts the distances the paper measures. This script makes that
a measured statement instead of an assertion, and draws the map so the user can judge it:

* **Labels.** Superkingdom = the first lineage element of each Swiss-Prot 2024_01 entry
  (Bacteria / Eukaryota / Archaea / Viruses). CATH class = the first digit of the entry's Gene3D
  cross-references (1 mainly alpha, 2 mainly beta, 3 alpha-beta, 4 few secondary structures),
  kept only when every Gene3D domain of the protein falls in one class.
* **Sample.** Proteins of the shared cohort (the arm's HDF5 keys minus the committed exclusion
  freeze, as the probes use), up to ``--per-stratum`` per superkingdom, seed 42; the same
  proteins for every arm.
* **Map.** umap-learn, Euclidean metric (the paper's distance), n_neighbors 15, min_dist 0.1,
  random_state 42.
* **What the map keeps,** per arm, in the full embedding and in the 2-D map: Spearman rho
  between Euclidean distances over 1,000,000 random protein pairs of the sample (does the map
  preserve the paper's quantity?); 10-NN superkingdom and CATH-class accuracy (5-fold CV; does
  the label structure exist before projection?); sklearn trustworthiness at k=15 on 5,000
  proteins (are 2-D neighbours real neighbours?).

Writes umap_coords.parquet, umap_metrics.json, labels.parquet (cache) and umap_<colouring>.png.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.manifold import trustworthiness  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from shared.protein_cohort import load_excluded_proteins, restrict_to_cohort  # noqa: E402

SUPERKINGDOMS = ("Bacteria", "Eukaryota", "Archaea", "Viruses")
CATH = {"1": "1 mainly α", "2": "2 mainly β", "3": "3 α/β", "4": "4 few SS"}
SK_COLOURS = {"Bacteria": "#1f77b4", "Eukaryota": "#d62728", "Archaea": "#2ca02c", "Viruses": "#9467bd"}
CATH_COLOURS = {"1 mainly α": "#e6550d", "2 mainly β": "#3182bd", "3 α/β": "#31a354", "4 few SS": "#756bb1"}
DISPLAY = {"prott5": "ProtT5", "esmc_600m": "ESM C-600M", "esm2_3b": "ESM2-3B", "random_1024": "Random (control)",
           "clean": "CLEAN", "ankh_large": "Ankh Large", "esm2_8m": "ESM2-8M", "prottucker": "ProtTucker"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_labels(sprot_json: Path) -> pl.DataFrame:
    """Accession -> superkingdom, CATH class; sprot.json holds one entry per line."""
    rows = []
    with open(sprot_json, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith('{"entryType"'):
                continue
            e = json.loads(line.rstrip(","))
            lineage = e.get("organism", {}).get("lineage") or [None]
            classes = {
                x["id"].split(".", 1)[0]
                for x in e.get("uniProtKBCrossReferences", [])
                if x.get("database") == "Gene3D"
            }
            cath = CATH.get(next(iter(classes))) if len(classes) == 1 else ("multiple" if classes else None)
            rows.append((e["primaryAccession"], lineage[0], cath, e.get("sequence", {}).get("length")))
    return pl.DataFrame(rows, schema=["acc", "superkingdom", "cath_class", "length"], orient="row")


def load(h5: Path, ids: list[str]) -> np.ndarray:
    with h5py.File(h5, "r") as f:
        return np.stack([f[i][:].reshape(-1).astype(np.float32) for i in ids])


def pair_rho(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_pairs: int) -> float:
    a = rng.integers(0, len(x), n_pairs)
    b = rng.integers(0, len(x), n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    dx = np.concatenate([np.linalg.norm(x[a[s:s + 100_000]] - x[b[s:s + 100_000]], axis=1)
                         for s in range(0, len(a), 100_000)])
    dy = np.linalg.norm(y[a] - y[b], axis=1)
    return float(stats.spearmanr(dx, dy).statistic)


def knn_acc(x: np.ndarray, labels: np.ndarray) -> float:
    clf = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
    return float(cross_val_score(clf, x, labels, cv=5).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sprot-json", type=Path, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["prott5", "esmc_600m", "random_1024"])
    ap.add_argument("--per-stratum", type=int, default=12_500)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import umap  # noqa: PLC0415  (only this script needs umap-learn)

    cache = args.out / "labels.parquet"
    if cache.is_file():
        labels = pl.read_parquet(cache)
    else:
        log("parsing sprot.json")
        labels = parse_labels(args.sprot_json)
        labels.write_parquet(cache)
    log(f"{labels.height:,} entries; superkingdoms {labels['superkingdom'].value_counts().sort('count').rows()}")

    with h5py.File(args.emb_dir / f"{args.arms[0]}.h5", "r") as f:
        keys = set(f.keys())
    cohort, summary = restrict_to_cohort(keys, load_excluded_proteins())
    print(summary.describe(args.arms[0]))
    pool = labels.filter(pl.col("acc").is_in(cohort) & pl.col("superkingdom").is_in(SUPERKINGDOMS))
    sample = pl.concat(
        [g.sample(min(g.height, args.per_stratum), seed=42) for _, g in pool.group_by("superkingdom", maintain_order=True)]
    ).sort("acc")
    ids = sample["acc"].to_list()
    sk = sample["superkingdom"].to_numpy()
    cath = sample["cath_class"].to_numpy()
    single = np.array([c in CATH.values() for c in cath])
    log(f"cohort {len(cohort):,}; pool {pool.height:,}; sample {len(ids):,}: "
        f"{sample['superkingdom'].value_counts().sort('superkingdom').rows()}; single CATH class {single.sum():,}")

    rng = np.random.default_rng(42)
    tw_idx = rng.choice(len(ids), min(5000, len(ids)), replace=False)
    coords, metrics = [], {"sample": {"n": len(ids), "per_superkingdom": dict(sample["superkingdom"].value_counts().rows()),
                                      "n_single_cath_class": int(single.sum()),
                                      "cath_counts": dict(sample.filter(pl.col("cath_class").is_in(list(CATH.values())))
                                                          ["cath_class"].value_counts().rows())},
                           "params": {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "random_state": 42},
                           "arms": {}}
    for arm in args.arms:
        t0 = time.time()
        x = load(args.emb_dir / f"{arm}.h5", ids)
        log(f"{arm}: {x.shape} loaded in {time.time() - t0:.0f}s")
        y = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42).fit_transform(x)
        log(f"{arm}: UMAP in {time.time() - t0:.0f}s")
        m = {
            "dims": int(x.shape[1]),
            "distance_rho_full_vs_2d": pair_rho(x, y, np.random.default_rng(7), 1_000_000),
            "knn10_superkingdom_full": knn_acc(x, sk),
            "knn10_superkingdom_2d": knn_acc(y, sk),
            "knn10_cath_full": knn_acc(x[single], cath[single]),
            "knn10_cath_2d": knn_acc(y[single], cath[single]),
            "trustworthiness_k15": float(trustworthiness(x[tw_idx], y[tw_idx], n_neighbors=15)),
            "seconds": round(time.time() - t0, 1),
        }
        metrics["arms"][arm] = m
        log(f"{arm}: {m}")
        coords.append(pl.DataFrame({"acc": ids, "arm": arm, "x": y[:, 0], "y": y[:, 1]}))
        (args.out / "umap_metrics.json").write_text(json.dumps(metrics, indent=1))

    sk_major = max(metrics["sample"]["per_superkingdom"].values())
    metrics["chance"] = {
        "superkingdom_majority": sk_major / len(ids),
        "cath_majority": max(metrics["sample"]["cath_counts"].values()) / int(single.sum()),
    }
    (args.out / "umap_metrics.json").write_text(json.dumps(metrics, indent=1))
    allc = pl.concat(coords).join(sample, on="acc")
    allc.write_parquet(args.out / "umap_coords.parquet")

    for colouring, col, palette in (("superkingdom", "superkingdom", SK_COLOURS), ("cath", "cath_class", CATH_COLOURS)):
        fig, axes = plt.subplots(1, len(args.arms), figsize=(6.2 * len(args.arms), 6.4))
        for ax, arm in zip(np.atleast_1d(axes), args.arms, strict=True):
            d = allc.filter(pl.col("arm") == arm)
            other = d.filter(~pl.col(col).is_in(list(palette)) | pl.col(col).is_null())
            ax.scatter(other["x"], other["y"], s=0.4, c="#d9d9d9", alpha=0.4, rasterized=True, linewidths=0)
            dd = d.filter(pl.col(col).is_in(list(palette))).sample(fraction=1.0, shuffle=True, seed=1)
            ax.scatter(dd["x"], dd["y"], s=0.6, c=[palette[v] for v in dd[col]], alpha=0.6, rasterized=True,
                       linewidths=0)
            m = metrics["arms"][arm]
            key = "superkingdom" if colouring == "superkingdom" else "cath"
            ax.set_title(f"{DISPLAY.get(arm, arm)}\n10-NN {colouring} acc. {m[f'knn10_{key}_full']:.2f} full / "
                         f"{m[f'knn10_{key}_2d']:.2f} 2-D;  distance ρ(full, 2-D) {m['distance_rho_full_vs_2d']:.2f}",
                         fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k, markersize=7) for k, c in palette.items()]
        if colouring == "cath":
            handles.append(plt.Line2D([], [], marker="o", ls="", color="#d9d9d9", label="none / multiple",
                                      markersize=7))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10)
        fig.suptitle(f"UMAP of {len(ids):,} cohort proteins (≤{args.per_stratum:,} per superkingdom), "
                     f"coloured by {'superkingdom' if colouring == 'superkingdom' else 'CATH class (Gene3D)'}",
                     fontsize=12, weight="bold")
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.savefig(args.out / f"umap_{colouring}.png", dpi=200)
        plt.close(fig)
    log("done")


if __name__ == "__main__":
    main()
