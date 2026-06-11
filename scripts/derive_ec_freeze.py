"""One-shot: derive the EC-positive subset freeze from the canonical-set label TSV.

Re-derives the EC-positive cohort (NOT assumed to be 33 — the wildcard-exclude
default may drop class-only entries) and writes freeze/ec_positive_subset_319.json.
Run once; the artifact is committed. Re-run with --overwrite to refresh intentionally.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.ec_freeze import derive_ec_freeze, verify_ec_freeze, write_ec_freeze
from evaluation.label_adapters import parse_ec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="Canonical-set label TSV (UniProt export).")
    ap.add_argument("--ec-col", default=None, help="Structured EC column, if present.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t", dtype=str)
    labels = parse_ec(df, ec_col=args.ec_col, wildcard_policy="exclude")
    manifest = derive_ec_freeze(
        labels, derived_from="canonical_set_319",
        source_tsv=args.tsv, wildcard_policy="exclude", ec_col=args.ec_col,
    )
    verify_ec_freeze(manifest, labels)  # round-trip self-check
    written = write_ec_freeze(manifest, "freeze", overwrite=args.overwrite)
    print(f"wrote {written}: n_proteins={manifest['n_proteins']} "
          f"sha={manifest['content_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
