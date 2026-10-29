#!/usr/bin/env python3
"""How many evaluated New2024 proteins have a close homolog among the probes' training proteins?

    uv run python scripts/paper/new2024_training_homologs.py \
        --pairs data/processed/2024_new_e1/sets/test.parquet \
        --fasta data/interm/2024_new/newly_discovered_dissimilar_proteins_2024.fasta \
        --sprot data/raw/sprot_2024/sprot.fasta \
        --splits data/processed/sprot_pre2024_e1/sets \
        --out <paper_inputs>/d14_2026-10-16

Why: the paper calls New2024 "dissimilar to the probes' training proteins" and quotes how many
of its evaluated proteins have a full-length homolog there (>=50% identity, >=80% coverage of
both). The original 2026-10-09 search was not saved, so this re-runs it reproducibly.

How: the proteins of the evaluated New2024 pairs are searched with MMseqs2 (-s 7.5, E <= 1e-3)
against all of Swiss-Prot 2024_01, a superset of the training proteins. Each hit's target is
labelled with the split(s) it appears in, read from the pair tables in --splits. Those may be the
10% pair sample when the full split tables are not at hand: a protein absent from the sample is
labelled "unlabelled", never assumed to be a training protein, so the training count is a floor.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

COLUMNS = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--sprot", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.pairs)
    evaluated = set(pairs["query"]) | set(pairs["target"])
    records = args.fasta.read_text().split(">")[1:]
    query_fasta = "".join(
        f">{r.split()[0]}\n{''.join(r.splitlines()[1:])}\n" for r in records if r.split()[0] in evaluated
    )

    with tempfile.TemporaryDirectory() as tmp:
        qpath, hits = Path(tmp) / "queries.fasta", Path(tmp) / "hits.m8"
        qpath.write_text(query_fasta)
        subprocess.run(
            ["mmseqs", "easy-search", str(qpath), str(args.sprot), str(hits), str(Path(tmp) / "work"),
             "-s", "7.5", "-e", "1e-3", "--threads", str(args.threads), "-v", "1",
             "--format-output", ",".join(COLUMNS)],
            check=True,
        )
        h = pd.read_csv(hits, sep="\t", names=COLUMNS)
    h["target"] = h["target"].str.split("|").str[1].fillna(h["target"])

    split_of: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        t = pd.read_parquet(args.splits / f"{split}.parquet", columns=["query", "target"])
        for acc in set(t["query"]) | set(t["target"]):
            split_of.setdefault(acc, []).append(split)
    h["split"] = h["target"].map(lambda a: ",".join(split_of.get(a, ["unlabelled"])))

    full = h[(h["qcov"] >= 0.8) & (h["tcov"] >= 0.8)]
    args.out.mkdir(parents=True, exist_ok=True)
    full[full["fident"] >= 0.3].sort_values(["query", "fident"], ascending=[True, False]).to_csv(
        args.out / "new2024_training_homologs.tsv", sep="\t", index=False)

    n = len(evaluated)
    for thr in (0.3, 0.5):
        any_sp = full[full["fident"] >= thr]["query"].nunique()
        train = full[(full["fident"] >= thr) & full["split"].str.contains("train")]["query"].nunique()
        print(f">= {thr:.0%} identity, >= 80% coverage of both: {any_sp}/{n} have a Swiss-Prot hit, "
              f"{train}/{n} a hit labelled as a training protein")


if __name__ == "__main__":
    main()
