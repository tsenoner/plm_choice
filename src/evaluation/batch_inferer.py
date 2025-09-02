#!/usr/bin/env python3
"""
batch_infer.py
==============

Batch-run inference for *multiple* target parameters and *all* embeddings
found in a directory.  For every (param, embedding) combination the script:

  1. finds the most recent model checkpoint under <models_root>
     that fits the directory pattern:
         **/<model_type>/<param>/<embedding_name>/**/checkpoints/best-*.ckpt
     (model_type defaults to fnn → linear_distance → linear, but can be set)
   2. calls the single-run inference helper (infer_pairs.py) in-process
   3. stores the resulting NPZ under:
         <output_dir>/<param>/<pairs_stem>_<embedding_name>_pred.npz

Ground-truth values are required in the pair Parquet file.

Typical call
------------

python batch_infer.py \
       --emb-dir    data/processed/sprot_embs \
       --pairs      new_pairs.parquet \
       --truth-params fident alntmscore \
       --models-root models \
       --output     out/batch_predictions
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
from src.evaluation import infer_pairs as infer_mod


# --------------------------------------------------------------------------- #
# 1.  CLI                                                                     #
# --------------------------------------------------------------------------- #
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch inference for many embeddings / params"
    )
    p.add_argument(
        "--emb-dir",
        type=Path,
        required=True,
        help="Directory with *.h5 embedding files",
    )
    p.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="Parquet file holding query/target columns + ground-truth params",
    )
    p.add_argument(
        "--truth-params",
        nargs="+",
        required=True,
        help="One or more column names in --pairs to be predicted",
    )
    p.add_argument(
        "--models-root",
        type=Path,
        required=True,
        help="High-level directory where models were saved (see project spec)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write the *.npz prediction archives",
    )
    p.add_argument(
        "--model-types",
        nargs="+",
        default=["fnn", "linear_distance", "linear"],
        help="Model types to try (first match wins)",
    )
    p.add_argument("--query-col", default="query")
    p.add_argument("--target-col", default="target")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument(
        "--device", default=None, help="'cuda[:idx]' | 'cpu'  (default: auto)"
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 2.  Locating a checkpoint                                                   #
# --------------------------------------------------------------------------- #
def find_best_ckpt(
    models_root: Path, model_types: list[str], param: str, embedding_name: str
) -> Path | None:
    """
    Return the *newest* best-*.ckpt for the given (param, embedding), trying
    the model_types in the given order.  `None` if nothing is found.
    """
    candidates: list[Path] = []
    for m_type in model_types:
        pattern = f"**/{m_type}/{param}/{embedding_name}/**/checkpoints/best-*.ckpt"
        candidates.extend(models_root.glob(pattern))
    if not candidates:
        return None
    # newest = largest mtime
    return max(candidates, key=lambda p: p.stat().st_mtime)


# --------------------------------------------------------------------------- #
# 3.  In-process call to infer_pairs.py                                       #
# --------------------------------------------------------------------------- #
def run_single_inference(
    ckpt: Path,
    model_type: str,
    embedding_h5: Path,
    pairs_parquet: Path,
    truth_col: str,
    query_col: str,
    target_col: str,
    output_npz: Path,
    batch_size: int,
    device_arg: str | None,
):
    """
    Import infer_pairs as a *module* and call its main() with an argv list.
    This avoids spawning a new Python interpreter for every combination.
    """
    argv = [
        "--checkpoint",
        str(ckpt),
        "--model-type",
        model_type,
        "--pairs",
        str(pairs_parquet),
        "--hdf5",
        str(embedding_h5),
        "--truth-col",
        truth_col,
        "--query-col",
        query_col,
        "--target-col",
        target_col,
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(output_npz.parent),
    ]
    if device_arg:
        argv += ["--device", device_arg]
    # infer_pairs.main() parses its own CLI – mimic that
    infer_mod.main(argv)


# --------------------------------------------------------------------------- #
# 4.  Main driver                                                             #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_cli()

    # sanity checks
    if not args.emb_dir.is_dir():
        sys.exit(f"[ERR] embedding dir not found: {args.emb_dir}")
    if not args.pairs.is_file():
        sys.exit(f"[ERR] pair parquet not found: {args.pairs}")
    if not args.models_root.is_dir():
        sys.exit(f"[ERR] models_root not found: {args.models_root}")
    args.output.mkdir(parents=True, exist_ok=True)

    embedding_files = sorted(args.emb_dir.glob("*.h5"))
    if not embedding_files:
        sys.exit(f"[ERR] no *.h5 files in {args.emb_dir}")

    for param in args.truth_params:
        print(f"\n=== PARAM '{param}' ===")
        param_out_dir = args.output / param
        param_out_dir.mkdir(exist_ok=True)

        for emb_h5 in embedding_files:
            emb_name = emb_h5.stem
            print(f"  ↳ embedding: {emb_name}")

            ckpt = find_best_ckpt(args.models_root, args.model_types, param, emb_name)
            if ckpt is None:
                print("    • WARNING: no checkpoint found - skipped")
                continue

            # infer_pairs expects the *exact* model_type that matches the ckpt path
            # Layout relative to models_root: <model_type>/<param>/<embedding>/.../checkpoints/best-*.ckpt
            try:
                model_type = ckpt.relative_to(args.models_root).parts[0]  # <model_type>
            except Exception:
                model_type = args.model_types[0]  # fall back; should not happen

            # Match infer_pairs output naming
            out_npz = param_out_dir / f"{args.pairs.stem}_{emb_name}_pred.npz"
            if out_npz.exists():
                print("    • predictions already exist – skipping")
                continue

            print(f"    • using  {model_type}/{param}/{emb_name}  →  {ckpt.name}")
            run_single_inference(
                ckpt=ckpt,
                model_type=model_type,
                embedding_h5=emb_h5,
                pairs_parquet=args.pairs,
                truth_col=param,
                query_col=args.query_col,
                target_col=args.target_col,
                output_npz=out_npz,
                batch_size=args.batch_size,
                device_arg=args.device,
            )

    print("\n✓ batch inference complete.")


if __name__ == "__main__":
    main()
