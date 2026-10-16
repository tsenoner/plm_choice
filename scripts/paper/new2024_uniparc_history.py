#!/usr/bin/env python3
"""Which New2024 test proteins had a public sequence record before UniProt release 2024_01?

    uv run python scripts/paper/new2024_uniparc_history.py \
        --pairs data/processed/2024_new_e1/sets/test.parquet --out <paper_inputs>/d14_2026-10-16

Why: New2024 keeps UniRef50 2025_01 clusters none of whose members (compared by UniProtKB
accession and UniParc identifier, not by sequence) appear in any 2024_01 cluster. That check
cannot see a sequence UniParc already held from another source (e.g. RefSeq, PDB) before
2024_01. This asks UniParc, for every protein in the evaluated New2024 pairs, when its sequence
was first cross-referenced from any source (``oldestCrossRefCreated``), and counts those older
than the 2024_01 release (2024-01-24). Network access to rest.uniprot.org is required.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

RELEASE_2024_01 = "2024-01-24"
URL = "https://rest.uniprot.org/uniparc/search?query={q}&format=json&size=500"


def fetch(accessions: list[str]) -> dict[str, dict]:
    query = " OR ".join(f"uniprotkb:{a}" for a in accessions)
    with urllib.request.urlopen(URL.format(q=urllib.parse.quote(query)), timeout=60) as r:
        results = json.load(r)["results"]
    out = {}
    for rec in results:
        accs = {x.get("id") for x in rec.get("uniParcCrossReferences", [])} | {
            a.get("accession") for a in rec.get("uniProtKBAccessions", []) if isinstance(a, dict)
        }
        for a in accessions:
            if a in accs or a in json.dumps(rec):
                out[a] = {"upi": rec["uniParcId"], "oldest_xref": rec.get("oldestCrossRefCreated")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pairs = pd.read_parquet(args.pairs)
    accs = sorted(set(pairs["query"]) | set(pairs["target"]))
    rows = {}
    for i in range(0, len(accs), 50):
        batch = accs[i:i + 50]
        rows.update(fetch(batch))
        time.sleep(0.5)
    df = pd.DataFrame([{"accession": a, **rows.get(a, {"upi": None, "oldest_xref": None})} for a in accs])
    df["before_2024_01"] = df["oldest_xref"].notna() & (df["oldest_xref"] < RELEASE_2024_01)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "new2024_uniparc_history.tsv", sep="\t", index=False)
    print(f"{len(accs)} proteins in the pairs; {df['upi'].notna().sum()} with a UniParc record; "
          f"{int(df['before_2024_01'].sum())} first cross-referenced before {RELEASE_2024_01}")
    print(df[df["before_2024_01"]].to_string(index=False))


if __name__ == "__main__":
    main()
