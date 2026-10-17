#!/usr/bin/env python3
"""What share of a test set has a close relative among the probes' training proteins?

    uv run python scripts/paper/split_homologs.py \
        --queries <ids.txt> --train-ids <train_ids.txt> --sprot data/raw/sprot_2024/sprot.fasta \
        --label swissprot_test --max-len 1022 --out <paper_inputs>/d14_2026-10-16/split_homologs

Why: the Swiss-Prot split keeps MMseqs2 clusters apart, but cluster membership only requires
similarity to a cluster's representative, so a test protein can still have a close relative in a
training cluster. New2024 is meant to be far from the training proteins. This measures both test
sets the same way, against the same reference: the proteins of the probes' training pairs.

How: queries (and the training proteins) are read from a FASTA (Swiss-Prot 2024_01, or
--queries-fasta for proteins outside it, e.g. New2024), restricted to --max-len residues, and
searched with MMseqs2 (-s 7.5, E <= 1e-3) against the training proteins. A query counts at a
threshold if any hit reaches that identity with >= 80% coverage of both sequences. Writes
<label>_summary.json and <label>_best_hits.tsv (best hit per query).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

COLUMNS = ["query", "target", "fident", "qcov", "tcov"]


def read_fasta(path: Path) -> dict[str, str]:
    seqs, cur, buf = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                seqs[cur] = "".join(buf)
            head = line[1:].strip()
            cur, buf = (head.split("|")[1] if "|" in head else head.split()[0]), []
        else:
            buf.append(line.strip())
    if cur:
        seqs[cur] = "".join(buf)
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queries", type=Path, required=True, help="query accessions, one per line")
    ap.add_argument("--queries-fasta", type=Path, help="FASTA holding the queries if not in --sprot")
    ap.add_argument("--train-ids", type=Path, required=True, help="training-pair proteins, one per line")
    ap.add_argument("--sprot", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    sprot = read_fasta(args.sprot)
    qseqs = read_fasta(args.queries_fasta) if args.queries_fasta else sprot
    wanted = [a for a in args.queries.read_text().split() if a]
    queries = {a: qseqs[a] for a in wanted if a in qseqs and len(qseqs[a]) <= args.max_len}
    train = {a: sprot[a] for a in args.train_ids.read_text().split() if a in sprot}

    with tempfile.TemporaryDirectory() as tmp:
        qf, tf, hits = (Path(tmp) / n for n in ("q.fasta", "t.fasta", "hits.m8"))
        qf.write_text("".join(f">{a}\n{s}\n" for a, s in queries.items()))
        tf.write_text("".join(f">{a}\n{s}\n" for a, s in train.items()))
        subprocess.run(
            ["mmseqs", "easy-search", str(qf), str(tf), str(hits), str(Path(tmp) / "work"),
             "-s", "7.5", "-e", "1e-3", "--threads", str(args.threads), "-v", "1",
             "--format-output", ",".join(COLUMNS)],
            check=True,
        )
        h = pd.read_csv(hits, sep="\t", names=COLUMNS)

    h = h[(h["qcov"] >= 0.8) & (h["tcov"] >= 0.8) & (h["query"] != h["target"])]
    best = h.sort_values("fident", ascending=False).drop_duplicates("query")
    n = len(queries)
    summary = {"label": args.label, "n_queries": n, "n_train": len(train), "max_len": args.max_len,
               "requested": len(wanted)}
    for thr in (0.3, 0.5):
        k = int((best["fident"] >= thr).sum())
        summary[f"ge{int(thr * 100)}"] = {"n": k, "share": round(k / n, 4)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=1))
    best.to_csv(args.out / f"{args.label}_best_hits.tsv", sep="\t", index=False)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
