#!/usr/bin/env python3
"""Mark pair-table rows whose two accessions carry the same sequence.

Why this is not just ``distance == 0``.  Swiss-Prot deposits the same sequence under
more than one accession (P0A8N5/P0A8N6, 505 aa, byte-identical), and those pairs sit at
distance exactly 0 in every embedding space.  They are the spike at zero, and under
min-max normalisation they pin the minimum of every row.  Excluding them by testing the
distance is circular -- the figure's own y-axis decides what the figure shows -- and it
is also inaccurate: ``prott5`` and ``prottucker`` are stored as float16, so 2,130 of the
23,369 identical-sequence pairs in the 10% subset come out at 0.0004-0.0056 instead of
0.0 and survive a ``== 0`` filter, while an arm with a compressed range (ankh_large tops
out at 0.379) can round genuinely distinct pairs down to 0.0 and lose them.

Identity is a property of the sequences.  This computes it from the sequences: one hash
per accession from the cohort FASTA, one boolean per pair row.  Every arm then drops the
same rows, which also settles the "rows normalise over different pair sets" caveat.

    python scripts/ridge_identical_pairs.py \
        --fasta $DSS/.../sprot.fasta --sets-dir $DSS/.../sets --out-dir $DSS/ridge_identical
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import polars as pl

SPLITS = ("train", "val", "test")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sequence_group_ids(fasta: Path) -> dict[str, int]:
    """Map accession -> a small integer that is equal iff the sequences are equal."""
    groups: dict[bytes, int] = {}
    out: dict[str, int] = {}
    acc: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if acc is None:
            return
        digest = hashlib.sha1("".join(chunks).encode()).digest()
        out[acc] = groups.setdefault(digest, len(groups))

    with fasta.open() as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                # ">A0A009IHW8" or ">sp|P12345|NAME" -- take the first whitespace token,
                # then the middle field if it is a pipe-delimited header.
                head = line[1:].split()[0]
                acc = head.split("|")[1] if head.count("|") >= 2 else head
                chunks = []
            else:
                chunks.append(line.strip())
        flush()
    log(f"read {len(out):,} accessions, {len(groups):,} distinct sequences "
        f"({len(out) - len(groups):,} redundant)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", required=True, type=Path)
    ap.add_argument("--sets-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gid = sequence_group_ids(args.fasta)
    stats = {"fasta": str(args.fasta), "n_accessions": len(gid), "splits": {}}

    for split in args.splits:
        t0 = time.time()
        pairs = pl.read_parquet(args.sets_dir / f"{split}.parquet", columns=["query", "target"])
        # replace_strict is elementwise, so the result stays pinned to the pair-table row
        # order the distance parquets were written in.  A join would have to be trusted
        # not to reorder; this does not require trust.
        q = pairs["query"].replace_strict(gid, default=-1, return_dtype=pl.Int64).to_numpy()
        t = pairs["target"].replace_strict(gid, default=-1, return_dtype=pl.Int64).to_numpy()
        unknown = int(((q < 0) | (t < 0)).sum())
        same = (q == t) & (q >= 0)
        n_same = int(same.sum())
        out = args.out_dir / f"{split}_identical.parquet"
        pl.DataFrame({"identical": same}).write_parquet(out, compression="zstd")
        # The same exclusion, keyed by accession rather than by row position, so it can
        # be applied to any other table built from these pairs -- notably the 10% subset,
        # which is a row subset of train and cannot use a positional mask.
        pairs.filter(pl.Series(same)).write_parquet(
            args.out_dir / f"{split}_identical_keyed.parquet", compression="zstd"
        )
        stats["splits"][split] = {
            "n_pairs": len(pairs),
            "n_identical_sequence": n_same,
            "frac": n_same / len(pairs),
            "n_accession_not_in_fasta": unknown,
            "seconds": round(time.time() - t0, 1),
        }
        log(f"{split}: {n_same:,}/{len(pairs):,} identical-sequence pairs "
            f"({100 * n_same / len(pairs):.3f}%), unknown ids {unknown:,} "
            f"[{time.time() - t0:.1f}s]")
        del q, t, same, pairs

    total = sum(v["n_identical_sequence"] for v in stats["splits"].values())
    total_pairs = sum(v["n_pairs"] for v in stats["splits"].values())
    stats["total"] = {"n_pairs": total_pairs, "n_identical_sequence": total,
                      "frac": total / total_pairs}
    (args.out_dir / "identical_pairs.json").write_text(json.dumps(stats, indent=2))
    log(f"TOTAL {total:,}/{total_pairs:,} ({100 * total / total_pairs:.3f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
