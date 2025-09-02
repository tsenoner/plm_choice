#!/usr/bin/env python3
"""
infer_pairs.py
==============

Predict a numeric property for protein–protein (or generic) pairs.

INPUT
-----
  • checkpoint (.ckpt)
  • pair table   (.parquet)   – must contain:
        <query_col>   sequence identifier 1
        <target_col>  sequence identifier 2
        <truth_col>   ground-truth numeric value
  • HDF5 embeddings for each sequence id

OUTPUT
------
  <output_dir>/<prefix>_predictions.npz  with keys
      - predictions  (float32, N)
      - targets      (float32, N)
      - query_id     (object,  N)
      - target_id    (object,  N)

Example
-------
python infer_pairs.py \
       --checkpoint models/run_42/checkpoints/best.ckpt \
       --model-type fnn \
       --pairs      new_pairs.parquet \
       --hdf5       embeddings.hdf5 \
       --truth-col  true_pide
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

# --------------------------------------------------------------------------- #
# Project-specific imports – adjust package path if needed
from src.shared.datasets import create_single_loader
from src.training.models import (
    FNNPredictor,
    LinearRegressionPredictor,
    LinearDistancePredictor,
)
from src.shared.helpers import get_device
# --------------------------------------------------------------------------- #

MODEL_CLASSES = {
    "fnn": FNNPredictor,
    "linear": LinearRegressionPredictor,
    "linear_distance": LinearDistancePredictor,
}


# ─────────────────────────── helpers ────────────────────────────────────────
def load_model(
    checkpoint: Path, model_type: str, device: torch.device
) -> pl.LightningModule:
    if model_type not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model-type '{model_type}'. Choose from {', '.join(MODEL_CLASSES)}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"• loading {model_type} model  ←  {checkpoint.name}")
    model = MODEL_CLASSES[model_type].load_from_checkpoint(str(checkpoint))
    model.eval().to(device)
    return model


def make_loader(
    pairs_pq: Path, hdf5: Path, truth_col: str, batch_size: int
) -> torch.utils.data.DataLoader:
    if not pairs_pq.is_file():
        raise FileNotFoundError(f"Pairs file not found: {pairs_pq}")
    if not hdf5.is_file():
        raise FileNotFoundError(f"HDF5 embeddings not found: {hdf5}")

    return create_single_loader(
        parquet_file=str(pairs_pq),
        hdf_file=str(hdf5),
        param_name=truth_col,
        batch_size=batch_size,
        shuffle=False,
    )


@torch.no_grad()
def predict(
    model: pl.LightningModule, loader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    preds, tgts = [], []
    for q_emb, t_emb, val in loader:
        q_emb = q_emb.to(device, non_blocking=True)
        t_emb = t_emb.to(device, non_blocking=True)
        pred = model(q_emb, t_emb)
        preds.append(pred.cpu())
        tgts.append(val.cpu())
    return (
        torch.cat(preds).numpy().astype(np.float32).flatten(),
        torch.cat(tgts).numpy().astype(np.float32).flatten(),
    )


def save_npz(
    out_path: Path,
    predictions: np.ndarray,
    targets: np.ndarray,
    q_ids: np.ndarray,
    t_ids: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        predictions=predictions,
        targets=targets,
        query_id=q_ids,
        target_id=t_ids,
    )
    print(f"• saved predictions →  {out_path}")


# ─────────────────────────── CLI & main ─────────────────────────────────────
def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run model inference on pair embeddings.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-type", choices=list(MODEL_CLASSES), required=True)
    p.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="Parquet file with query/target/truth columns",
    )
    p.add_argument("--hdf5", type=Path, required=True, help="Sequence-level embeddings")
    p.add_argument(
        "--truth-col",
        required=True,
        help="Column in <pairs> that holds the ground-truth value",
    )
    # Defaults align with `src.shared.datasets` which expects 'query' and 'target'
    p.add_argument("--query-col", default="query")
    p.add_argument("--target-col", default="target")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the .npz (default = <ckpt>/inference_results)",
    )
    p.add_argument(
        "--device", default=None, help="'cpu' | 'cuda[:idx]' (auto if omitted)"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_cli(argv)
    device = torch.device(args.device) if args.device else get_device()
    print(f"• inference device: {device}")

    # 1) read pairs once (ids & truth)
    df_pairs = pd.read_parquet(args.pairs)
    required = {args.query_col, args.target_col, args.truth_col}
    missing = required - set(df_pairs.columns)
    if missing:
        raise ValueError(f"{args.pairs} is missing columns: {', '.join(missing)}")

    loader = make_loader(args.pairs, args.hdf5, args.truth_col, args.batch_size)
    model = load_model(args.checkpoint, args.model_type, device)

    print("• running inference …")
    predictions, targets = predict(model, loader, device)
    print(f"  done. N = {len(predictions)}")

    # 2) save
    out_dir = args.output_dir or (args.checkpoint.parent / "inference_results")
    prefix = Path(args.pairs).stem
    # Extract PLM name from HDF5 filename (e.g., "ankh_base.h5" -> "ankh_base")
    plm_name = Path(args.hdf5).stem
    npz_path = out_dir / f"{prefix}_{plm_name}_pred.npz"
    save_npz(
        npz_path,
        predictions,
        targets,
        df_pairs[args.query_col].to_numpy(),
        df_pairs[args.target_col].to_numpy(),
    )

    print("✓ inference complete.")


if __name__ == "__main__":
    main()
