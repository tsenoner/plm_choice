#!/usr/bin/env python3
"""How much of the variant-`all` accuracy is handed over by identical sequences.

Neither cohort is deduplicated at 100% identity: they are stratified by CATH superfamily
and length, which does not look at the sequence. A query whose exact twin is in the cohort
is a free point for every arm under variant ``all`` — the strict variants remove it (an
identical sequence has fident 1.0), so this is a ceiling on the variant-`all` numbers only.
Writes ``duplicate_sequences.json``; SUMMARY.md quotes it rather than a typed number.

    python side_measurements/duplicate_sequences.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


# These paths were absolute to one machine. They are environment variables now, so an
# unset one fails here by name rather than as a FileNotFoundError further down.
import os


def _need(var: str) -> str:
    """The value of `var`, or a message naming what to set."""
    try:
        return os.environ[var]
    except KeyError:
        raise SystemExit(f"set {var} before running this script") from None


ARTEFACTS = _need("PAPER_ARTEFACTS")

RESULTS = Path(f"{ARTEFACTS}/functional_2026-09-17")
BACKUP = Path(ARTEFACTS)
RUNS = {
    "ec": (BACKUP / "c1_strat_2026-09-15/ec_strat_v2_freeze.json", RESULTS / "ec_v2_transfer", "exact"),
    "go": (RESULTS / "go_mf_cohort_freeze.json", RESULTS / "go_mf_transfer", "f1"),
}


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    pid, buf = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if pid:
                seqs[pid] = "".join(buf)
            pid, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if pid:
        seqs[pid] = "".join(buf)
    return seqs


def main() -> int:
    seqs = read_fasta(RESULTS / "union.fasta")
    out: dict[str, dict] = {"union_fasta_records": len(seqs)}
    for kind, (freeze, run_dir, score) in RUNS.items():
        ids = json.loads(freeze.read_text())["ids"]
        by_seq: dict[str, list[str]] = defaultdict(list)
        for pid in ids:
            by_seq[seqs[pid]].append(pid)
        groups = [g for g in by_seq.values() if len(g) > 1]
        dup = {p for g in groups for p in g}
        block: dict = {
            "n": len(ids),
            "distinct_sequences": len(by_seq),
            "n_in_a_duplicate_group": len(dup),
            "share_in_a_duplicate_group": len(dup) / len(ids),
            "n_groups": len(groups),
            "largest_group": max((len(g) for g in groups), default=0),
            "score": score,
            "per_arm": {},
        }
        frame = pd.read_parquet(
            run_dir / "per_query.parquet",
            columns=["arm", "distance", "variant", "query_id", f"score_{score}"],
        )
        frame = frame[(frame["variant"] == "all") & (frame["distance"] == "euclidean")]
        frame["dup"] = frame["query_id"].isin(dup)
        for arm, rows in frame.groupby("arm", observed=True):
            if rows.empty:
                continue
            block["per_arm"][str(arm)] = {
                "overall": float(rows[f"score_{score}"].mean()),
                "duplicated_queries": float(rows[rows["dup"]][f"score_{score}"].mean()),
                "other_queries": float(rows[~rows["dup"]][f"score_{score}"].mean()),
                "n_duplicated": int(rows["dup"].sum()),
            }
        inflation = {
            arm: v["overall"] - v["other_queries"] for arm, v in block["per_arm"].items()
        }
        worst = max(inflation, key=inflation.get)
        block["max_inflation"] = {"arm": worst, "delta": inflation[worst]}
        out[kind] = block
    path = Path(__file__).with_name("duplicate_sequences.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_arm"}
                      for k, v in out.items() if isinstance(v, dict)}, indent=2))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
