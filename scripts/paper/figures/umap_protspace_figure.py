#!/usr/bin/env python3
"""Supplementary UMAP figure (R1.3) from a ProtSpace bundle.

    # the bundle: ProtSpace 4.17.0 on the 37,500-protein sample of umap_subsample.py --export-sample
    uvx --from protspace==4.17.0 protspace prepare -i prott5.h5:prott5 -i esmc_600m.h5:esmc_600m \
        -i random_1024.h5:random_1024 -m umap2 -a annotations.csv --stats -o out
    python scripts/paper/figures/umap_protspace_figure.py \
        --bundle <paper_inputs>/d14_open_2026-10-17/umap/protspace/data.parquetbundle \
        --metrics <paper_inputs>/d14_open_2026-10-17/umap/umap_metrics.json \
        --out <paper_inputs>/d14_open_2026-10-17/umap/protspace

The projection, the faithfulness statistics and the category colours are ProtSpace's (UMAP,
Euclidean, 25 neighbours, min_dist 0.1, seed 42; ``--stats``), read from the bundle, so the figure
and protspace.app show the same map. This script only draws it and adds one number ProtSpace does
not compute: 10-nearest-neighbour superkingdom accuracy in the map (5-fold CV), next to the same
accuracy in the full embedding from ``umap_subsample.py`` (same proteins).

Writes umap_protspace.png and umap_protspace_summary.json.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

DELIM = b"---PARQUET_DELIMITER---"
PANELS = (("prott5", "ProtT5"), ("esmc_600m", "ESM C-600M"), ("random_1024", "Random (Gaussian control)"))
ORDER = ("Bacteria", "Eukaryota", "Archaea")


def decode(parts, column: str, codes: np.ndarray) -> np.ndarray:
    """Annotation cells are stored as integer codes; their strings sit in a (name, data) part as
    ``dict:<column>`` (concatenated UTF-8) and ``dict:<column>:len`` (int32 byte lengths)."""
    table = next(p for p in parts if list(p.columns) == ["name", "data"])
    blob = dict(zip(table["name"], table["data"], strict=True))
    lengths = np.frombuffer(blob[f"dict:{column}:len"], dtype="<i4")
    raw, values, at = blob[f"dict:{column}"], [], 0
    for n in lengths:
        values.append(raw[at:at + n].decode("utf-8"))
        at += n
    return np.array(values, dtype=object)[codes]


def read_bundle(path: Path):
    parts = [pq.read_table(io.BytesIO(p)).to_pandas() for p in path.read_bytes().split(DELIM)]
    annotations, meta, coords, settings = parts[0], parts[1], parts[2], parts[3]
    sk = decode(parts, "superkingdom", annotations["superkingdom"].to_numpy())
    return sk, meta, coords, json.loads(settings["settings_json"][0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True, help="umap_metrics.json of umap_subsample.py")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sk, meta, coords, settings = read_bundle(args.bundle)
    full = json.loads(args.metrics.read_text())["arms"]
    palette = {k: v["color"] for k, v in settings["superkingdom"]["categories"].items()}

    summary = {"bundle": str(args.bundle), "n": int(len(sk)), "projections": {}}
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 6.2))
    rng = np.random.default_rng(1)
    order = rng.permutation(len(sk))  # draw order shuffled so no class paints over another
    for ax, (src, label) in zip(axes, PANELS, strict=True):
        row = meta[meta["source"] == src].iloc[0]
        name = row["projection_name"]
        q = {k: v["value"] for k, v in json.loads(row["info_json"])["quality"].items()}
        xy = coords[[f"{name}__x", f"{name}__y"]].to_numpy()
        knn2d = float(cross_val_score(KNeighborsClassifier(n_neighbors=10, n_jobs=-1), xy, sk, cv=5).mean())
        summary["projections"][src] = {
            "projection": name, **{f"protspace_{k}": v for k, v in q.items()},
            "knn10_superkingdom_2d": knn2d, "knn10_superkingdom_full": full[src]["knn10_superkingdom_full"],
        }
        ax.scatter(xy[order, 0], xy[order, 1], s=0.5, c=[palette[v] for v in sk[order]], alpha=0.7,
                   linewidths=0, rasterized=True)
        ax.set_title(f"{label}\ndistance ρ(embedding, map) {q['spearman_distance']:.2f} · "
                     f"trustworthiness {q['trustworthiness']:.2f}\n"
                     f"10-NN superkingdom accuracy {full[src]['knn10_superkingdom_full']:.2f} embedding, "
                     f"{knn2d:.2f} map", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=palette[k], label=k, markersize=8)
               for k in ORDER if k in palette]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=12)
    fig.suptitle(f"UMAP (ProtSpace) of {len(sk):,} Swiss-Prot proteins, 12,500 per superkingdom",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(args.out / "umap_protspace.png", dpi=300)
    plt.close(fig)
    (args.out / "umap_protspace_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
