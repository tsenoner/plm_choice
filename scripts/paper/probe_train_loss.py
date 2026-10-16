#!/usr/bin/env python3
"""Per-epoch train and validation loss of every FNN probe, read from its offline wandb runs.

    python scripts/paper/probe_train_loss.py --models ~/plm_choice/models \
        --datasets full100_p10 randinit_full100 --out <dir>

Reviewer R1.10 (task D14, ledger M-39) asked for the final train/validation loss per model.
train.py logs `train_loss` and `val_loss` once per epoch (on_epoch=True) to a WandbLogger in
offline mode only, so the numbers exist nowhere but in models/<dataset>/fnn/<target>/<arm>/
wandb/offline-run-*/run-*.wandb on LRZ. Those files are mostly captured console output (the
progress bar); the history records are few and are what this reads.

A cell can hold several runs: an aborted first launch (no history), the training run, a resumed
training run (same wandb id, epochs continue), and the evaluation run (history without
train_loss). Runs are read in start-time order and per-epoch values of a later run replace an
earlier run's, so a restart from scratch and a resume both come out right.

`train_loss` is the mean MSE over that epoch's training batches while the weights move (the FNN
has no dropout); `val_loss` is the MSE on the validation split at the epoch's end. The selected
probe is the best-val_loss checkpoint; its epoch comes from the checkpoint name and is checked
against the logged minimum. HFSP losses are in HFSP^2 units, PIDE and TM-score in [0, 1]^2.

Outputs (--out): train_loss_per_cell.csv (one row per cell), train_loss_per_epoch.csv (the
curves), train_loss_summary.json (ranges per dataset x target, checks).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

CKPT = re.compile(r"best-epoch=(\d+)-step=(\d+)-val_loss=([\d.]+)\.ckpt$")


def history(path: Path) -> list[dict]:
    """The history rows of one offline .wandb file (each row a flat {key: value})."""
    ds = datastore.DataStore()
    ds.open_for_scan(str(path))
    rows = []
    while (data := ds.scan_data()) is not None:
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            rows.append({(i.key or "/".join(i.nested_key)): json.loads(i.value_json) for i in rec.history.item})
    return rows


def cell_curve(cell: Path) -> tuple[pd.DataFrame, list[str]]:
    """Per-epoch train/val loss of one cell, later runs overriding earlier ones epoch by epoch."""
    per_epoch: dict[int, dict] = {}
    used = []
    for run in sorted(cell.glob("wandb/offline-run-*/run-*.wandb"), key=lambda p: p.parent.name):
        rows = [r for r in history(run) if "epoch" in r and ("train_loss" in r or "val_loss" in r)]
        if not rows:
            continue
        used.append(run.parent.name)
        fresh: dict[int, dict] = {}
        for r in rows:
            e = int(r["epoch"])
            d = fresh.setdefault(e, {"epoch": e, "run": run.parent.name})
            for key in ("train_loss", "val_loss"):
                if key in r:
                    d[key] = float(r[key])
            if "trainer/global_step" in r:
                d["global_step"] = int(r["trainer/global_step"])
        per_epoch.update(fresh)
    return pd.DataFrame(sorted(per_epoch.values(), key=lambda d: d["epoch"])), used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["full100_p10", "randinit_full100"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cells, curves, problems = [], [], []
    for dataset in args.datasets:
        for cell in sorted((args.models / dataset / "fnn").glob("*/*")):
            target, arm = cell.parent.name, cell.name
            ck = [m for p in cell.glob("checkpoints/best-*.ckpt") if (m := CKPT.search(p.name))]
            if len(ck) != 1:
                problems.append(f"{dataset}/{target}/{arm}: {len(ck)} best checkpoints")
                continue
            best_epoch, best_step, name_val = int(ck[0][1]), int(ck[0][2]), float(ck[0][3])
            curve, used = cell_curve(cell)
            if curve.empty or "train_loss" not in curve:
                problems.append(f"{dataset}/{target}/{arm}: no train_loss in any wandb run")
                continue
            curves.append(curve.assign(dataset=dataset, target=target, arm=arm))
            c = curve.set_index("epoch")
            last = int(c.index.max())
            vmin_epoch = int(c["val_loss"].idxmin())
            row = {"dataset": dataset, "target": target, "arm": arm,
                   "best_epoch": best_epoch, "best_step": best_step, "val_loss_ckpt_name_3dp": name_val,
                   "epochs_logged": int(c.index.size), "last_epoch": last,
                   "epochs_contiguous": bool((np.diff(c.index.to_numpy()) == 1).all() and c.index.min() == 0),
                   "train_loss_best_epoch": c.at[best_epoch, "train_loss"] if best_epoch in c.index else np.nan,
                   "val_loss_best_epoch": c.at[best_epoch, "val_loss"] if best_epoch in c.index else np.nan,
                   "train_loss_last_epoch": c.at[last, "train_loss"], "val_loss_last_epoch": c.at[last, "val_loss"],
                   "train_loss_min": c["train_loss"].min(), "val_loss_min": c["val_loss"].min(),
                   "epoch_of_val_min": vmin_epoch, "wandb_runs_used": " ".join(used)}
            row["val_over_train_best_epoch"] = row["val_loss_best_epoch"] / row["train_loss_best_epoch"]
            # The checkpoint name's val_loss is the same number at 3 dp, and early stopping kept the minimum.
            row["ckpt_name_matches_log"] = bool(abs(round(row["val_loss_best_epoch"], 3) - name_val) < 1e-9)
            row["best_is_logged_min"] = vmin_epoch == best_epoch
            for flag in ("ckpt_name_matches_log", "best_is_logged_min", "epochs_contiguous"):
                if not row[flag]:
                    problems.append(f"{dataset}/{target}/{arm}: {flag} is False")
            cells.append(row)
            print(f"{dataset}/{target}/{arm}: epochs 0-{last}, best {best_epoch}, "
                  f"train {row['train_loss_best_epoch']:.6g} val {row['val_loss_best_epoch']:.6g}", flush=True)

    df = pd.DataFrame(cells)
    summary: dict = {"datasets": args.datasets, "cells": int(len(df)), "problems": problems}
    for (ds, t), g in df.groupby(["dataset", "target"]):
        g = g[~g["arm"].str.contains("random_1024")]
        summary[f"{ds}/{t}"] = {
            "n_cells": int(len(g)),
            "train_loss_best_epoch_range": [float(g.train_loss_best_epoch.min()), float(g.train_loss_best_epoch.max())],
            "val_loss_best_epoch_range": [float(g.val_loss_best_epoch.min()), float(g.val_loss_best_epoch.max())],
            "val_over_train_range": [float(g.val_over_train_best_epoch.min()), float(g.val_over_train_best_epoch.max())],
            "val_over_train_median": float(g.val_over_train_best_epoch.median()),
            "train_loss_last_epoch_range": [float(g.train_loss_last_epoch.min()), float(g.train_loss_last_epoch.max())],
            "epochs_trained_range": [int(g.last_epoch.min()) + 1, int(g.last_epoch.max()) + 1],
            "epochs_after_best_range": [int((g.last_epoch - g.best_epoch).min()), int((g.last_epoch - g.best_epoch).max())],
        }

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "train_loss_per_cell.csv", index=False, float_format="%.6g")
    if curves:
        pd.concat(curves).reindex(columns=["dataset", "target", "arm", "epoch", "train_loss", "val_loss",
                                           "global_step", "run"]).to_csv(
            args.out / "train_loss_per_epoch.csv", index=False, float_format="%.6g")
    (args.out / "train_loss_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
