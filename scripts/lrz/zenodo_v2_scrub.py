#!/usr/bin/env python3
"""Rewrite the staged Zenodo v2 archives into their public form.

The archives were staged straight from the working trees, so they carried the authors' machine
and cluster paths, the cluster username in every tar header, Slurm job ids, wandb ids and
internal folder labels. This reads each staged tar read-only and writes a cleaned copy under its
published name into <stage>/clean/, renaming folders and files, rewriting the fields that held
paths, and checking the result. The cluster's working trees are never touched.

    ~/plm_choice/.venv/bin/python scripts/lrz/zenodo_v2_scrub.py              # all eight archives
    ~/plm_choice/.venv/bin/python scripts/lrz/zenodo_v2_scrub.py --only pairs_counts.tar

Checks, each fatal: every checkpoint reloads with identical weights and hyperparameters (and,
for the first one per grid, identical outputs through the repo's FNNPredictor); every rewritten
JSON/YAML parses; each manifest's inputs are keys of its sha256 table; no member name or text
member contains a blocked string; no blocked string of six or more bytes occurs anywhere in the
output file; every header has owner 0/0.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath

import yaml

DSS = Path("/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2")
REPO = Path(__file__).resolve().parents[2]

# staged name -> (published name, top-level folder renames)
ARCHIVES = {
    "pair_tables_splits.tar": ("pairs_splits.tar", {"sprot_pre2024_e1": "swissprot_pairs"}),
    "pair_table_counts.tar": ("pairs_counts.tar", {}),
    "identical_sequence_mask.tar": ("pairs_identical_mask.tar", {"ridge_identical": "identical_pairs"}),
    "probes_full100_p10.tar": ("probes_native.tar", {"full100_p10": "probes_native"}),
    "probes_pca_cohort.tar": ("probes_pca128.tar", {"pca_cohort": "probes_pca128"}),
    "probes_randinit_full100.tar": ("probes_untrained.tar", {"randinit_full100": "probes_untrained"}),
    "functional_annotation.tar": ("analysis_functional_annotation.tar", {}),
    "distance_histograms.tar": ("analysis_distance_histograms.tar", {}),
    "new2024.tar": ("testset_new2024.tar", {}),
}
EXCLUSION_LIST = REPO / "docs/zenodo/new2024_exclude_public_before_training.txt"

# renames below the top level (applied to member paths and to paths inside manifests)
SUB_RENAMES = [
    ("pair_table_counts/sprot_pre2024_e1.json", "pair_table_counts/swissprot_pairs.json"),
    ("ec_v2_transfer", "ec_transfer"),
    ("go_mf_transfer_noPB", "go_mf_transfer_no_protein_binding"),
    ("slices_union", "embedding_slices"),
    ("ec_labels_v2.tsv", "ec_labels.tsv"),
    ("ec_strat_v2_freeze.json", "ec_cohort_freeze.json"),
    ("ec_tau_b_v2_euclidean.csv", "ec_tau_b_euclidean.csv"),
]

# where a rewritten hparams.yaml points, relative to the folder the archives are unpacked in
PROBE_EMBEDDINGS = {
    "probes_native": "swissprot_pairs/embeddings",
    "probes_pca128": "swissprot_pairs/embeddings_pca128",
    "probes_untrained": "swissprot_pairs/embeddings_untrained",
}

BLOCK = [
    "/Users", "/dss", "/dev/shm", "/private/tmp", "/opt/homebrew", "ge45ted2", "tsenoner",
    "pr63ci", "claude", "internal_backup", "full100", "pca_cohort", "randinit_full100",
    # "ridge_identical"/"ridge_full", not "ridge_": the Gene Ontology cites "GOC:mtg_cambridge_2009"
    "sprot_pre2024_e1", "ridge_identical", "ridge_full", "_v2", "noPB", "sfcap", "published_tau", "wandb",
    "c1_strat", "functional_2026", "slices_union", "exclusion freeze",
]
# substrings long enough that chance hits in binary data are negligible (scanned over whole files)
BLOCK_RAW = [b.encode() for b in BLOCK if len(b) >= 6]
TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".txt", ".csv", ".tsv", ".md", ".log"}

_ckpt_checked: set[str] = set()


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def blocked(text: str) -> list[str]:
    return [b for b in BLOCK if b in text]


def rename_member(name: str, top: dict[str, str]) -> str:
    first, _, rest = name.partition("/")
    name = top.get(first, first) + ("/" + rest if rest else "")
    for old, new in SUB_RENAMES:
        name = name.replace(old, new)
    return name


def public_path(p: str) -> str:
    """A machine path from a manifest -> the archive-relative name it is deposited under."""
    if p.startswith("/"):
        for anchor in ("functional_2026-09-17/", "c1_strat_2026-09-15/", "c1_strat_offload/functional/"):
            if anchor in p:
                p = p.split(anchor, 1)[1]
                break
        else:
            p = p.rstrip("/").rsplit("/", 1)[-1]
    for old, new in SUB_RENAMES:
        p = p.replace(old, new)
    return p


def rewrite_strings(obj):
    """Apply public_path to every string, dict key included."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = public_path(k) if isinstance(k, str) else k
            if nk in out:
                fail(f"two manifest keys map to {nk!r}")
            out[nk] = rewrite_strings(v)
        return out
    if isinstance(obj, list):
        return [rewrite_strings(v) for v in obj]
    if isinstance(obj, str):
        return public_path(obj)
    return obj


def dump_json(obj) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode()


# --------------------------------------------------------------------------------------------
# per-member rewrites
# --------------------------------------------------------------------------------------------

def rewrite_hparams(raw: bytes, new_name: str) -> bytes:
    h = yaml.safe_load(raw)
    parts = PurePosixPath(new_name).parts            # grid/readout/target/arm/hparams.yaml
    grid, arm = parts[0], parts[3]
    stem = PurePosixPath(str(h["embedding_file"])).stem
    if stem != arm:
        fail(f"{new_name}: embedding_file stem {stem!r} != arm folder {arm!r}")
    h.pop("dataset", None)
    h["embedding_file"] = f"{PROBE_EMBEDDINGS[grid]}/{arm}.h5"
    h["data_dir"] = "swissprot_pairs/sets"
    return yaml.safe_dump(h, sort_keys=False, default_flow_style=False).encode()


def rewrite_training_complete(raw: bytes) -> bytes:
    return re.sub(r"Best model: /\S*/checkpoints/", "Best model: checkpoints/", raw.decode()).encode()


def rewrite_checkpoint(raw: bytes, new_name: str) -> bytes:
    import torch

    rel = lambda p: ("checkpoints/" + p.rsplit("/", 1)[-1]) if p else p  # noqa: E731
    orig = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    n_cb = 0
    for key, st in ck.get("callbacks", {}).items():
        if key.startswith("ModelCheckpoint"):
            n_cb += 1
            st["dirpath"] = "checkpoints"
            st["best_model_path"] = rel(st.get("best_model_path", ""))
            st["kth_best_model_path"] = rel(st.get("kth_best_model_path", ""))
            st["last_model_path"] = ""          # last.ckpt is not deposited
            st["best_k_models"] = {rel(k): v for k, v in st.get("best_k_models", {}).items()}
    if n_cb != 1:
        fail(f"{new_name}: expected one ModelCheckpoint callback, found {n_cb}")
    buf = io.BytesIO()
    torch.save(ck, buf)
    new = buf.getvalue()

    back = torch.load(io.BytesIO(new), map_location="cpu", weights_only=False)
    if set(back) != set(orig):
        fail(f"{new_name}: checkpoint keys changed")
    sd0, sd1 = orig["state_dict"], back["state_dict"]
    if sd0.keys() != sd1.keys() or not all(torch.equal(sd0[k], sd1[k]) for k in sd0):
        fail(f"{new_name}: state_dict changed")
    if back.get("hyper_parameters") != orig.get("hyper_parameters"):
        fail(f"{new_name}: hyper_parameters changed")
    meta = repr({k: v for k, v in back.items() if k != "state_dict"})
    if blocked(meta) or any(b in new for b in BLOCK_RAW):
        fail(f"{new_name}: blocked string left in checkpoint: {blocked(meta)}")

    grid = PurePosixPath(new_name).parts[0]
    if grid not in _ckpt_checked:            # once per grid: the model's outputs must not move
        sys.path.insert(0, str(REPO / "src"))
        from training.models import FNNPredictor

        torch.manual_seed(0)
        x = torch.randn(64, orig["hyper_parameters"]["embedding_size"])
        y0 = FNNPredictor.load_from_checkpoint(io.BytesIO(raw), map_location="cpu").eval()(x, x.flip(0))
        y1 = FNNPredictor.load_from_checkpoint(io.BytesIO(new), map_location="cpu").eval()(x, x.flip(0))
        if not torch.equal(y0, y1):
            fail(f"{new_name}: reloaded model gives different outputs")
        print(f"  reload check ok: {new_name}")
        _ckpt_checked.add(grid)
    return new


def rewrite_union_summary(raw: bytes) -> bytes:
    d = json.loads(raw)
    toks = d["command"].split()
    out = ["mmseqs"]
    for t in toks[1:]:
        if "/" in t:
            base = t.rstrip("/").rsplit("/", 1)[-1]
            out.append("tmp" if ("scratchpad" in t or base.startswith("mmseqs_")) else base)
        else:
            out.append(t)
    d["command"] = " ".join(out)
    d = rewrite_strings(d)
    print(f"  command: {d['command']}")
    return dump_json(d)


def rewrite_freeze(raw: bytes, which: str) -> bytes:
    d = json.loads(raw)
    d["set_name"] = {"ec": "ec_cohort", "go": "go_mf_cohort"}[which]
    if "filter" in d:
        d["filter"] = (d["filter"].replace("not in embedding exclusion freeze", "in the 526,871-protein cohort")
                       .replace(" (== CONTAINS on GoEvidenceType)", ""))
    src = d.get("sources")
    if isinstance(src, dict):
        new_src = {}
        for k, v in src.items():
            k = "n_excluded_from_cohort" if k == "exclusion_freeze_n_excluded" else k
            new_src[k] = public_path(v).rsplit("/", 1)[-1] if isinstance(v, str) and "/" in v else v
        d["sources"] = new_src
    return dump_json(d)


def rewrite_manifest(raw: bytes, freeze_sha: dict[str, str], name: str) -> bytes:
    d = rewrite_strings(json.loads(raw))
    sha = d.get("sha256", {})
    for fname, digest in freeze_sha.items():
        if fname in sha:
            sha[fname] = digest
    inputs = d.get("inputs", {})
    vals = inputs.values() if isinstance(inputs, dict) else inputs
    missing = [v for v in vals if isinstance(v, str) and v not in sha and not v.endswith("embedding_slices")]
    if missing:
        fail(f"{name}: inputs without a sha256 entry: {missing}")
    return dump_json(d)


def drop_columns(raw: bytes, cols: set[str]) -> bytes:
    rows = list(csv.reader(io.StringIO(raw.decode())))
    keep = [i for i, c in enumerate(rows[0]) if c not in cols]
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    for r in rows:
        w.writerow([r[i] for i in keep])
    return out.getvalue().encode()


def rewrite_histogram(raw: bytes, new_name: str) -> bytes:
    d = json.loads(raw)
    d["source"] = ("random_pairs/random_pairs_distances.parquet" if "/random_pairs/" in new_name
                   else "swissprot_pairs/sets")
    d.pop("peak_rss_gb", None)
    return dump_json(d)


def rewrite_small_json(raw: bytes, new_name: str) -> bytes:
    d = json.loads(raw)
    if new_name.endswith("pair_table_counts/swissprot_pairs.json"):
        d["sets_dir"] = "swissprot_pairs/sets"
    if new_name.endswith("identical_pairs/identical_pairs.json"):
        d["fasta"] = "sprot.fasta (Swiss-Prot release 2024_01)"
    return dump_json(d)


# --------------------------------------------------------------------------------------------
# archive pass
# --------------------------------------------------------------------------------------------

def header(ti: tarfile.TarInfo, name: str, size: int | None = None) -> tarfile.TarInfo:
    t = copy.copy(ti)
    t.name = name
    t.pax_headers = {}                       # extended headers can repeat owner names and paths
    t.uid = t.gid = 0
    t.uname = t.gname = ""
    t.mode = 0o755 if ti.isdir() else 0o644
    if size is not None:
        t.size = size
    return t


def transform(old: str, new: str, raw: bytes, ctx: dict) -> bytes | None:
    """Return rewritten bytes, or None to keep the member unchanged."""
    base = PurePosixPath(new).name
    if base == "hparams.yaml":
        return rewrite_hparams(raw, new)
    if base == "training_complete.txt":
        return rewrite_training_complete(raw)
    if base.endswith(".ckpt"):
        return rewrite_checkpoint(raw, new)
    if base == "union_allvsall.summary.json":
        return rewrite_union_summary(raw)
    if base == "ec_cohort_freeze.json":
        return ctx["freezes"]["ec_cohort_freeze.json"]
    if base == "go_mf_cohort_freeze.json":
        return ctx["freezes"]["go_mf_cohort_freeze.json"]
    if base == "manifest.json":
        return rewrite_manifest(raw, ctx["freeze_sha"], new)
    if base == "ec_tau_b_euclidean.csv":
        return drop_columns(raw, {"published_tau_b", "abs_diff"})
    if new.startswith("distance_histograms/") and base.endswith(".json"):
        return rewrite_histogram(raw, new)
    if base in ("swissprot_pairs.json", "identical_pairs.json"):
        return rewrite_small_json(raw, new)
    if base == EXCLUSION_LIST.name:          # the published header, without the work file it cited
        out = EXCLUSION_LIST.read_bytes()
        ids = lambda b: [ln.split()[0] for ln in b.decode().splitlines() if ln and not ln.startswith("#")]  # noqa: E731
        if ids(out) != ids(raw):
            fail(f"{new}: the repo's exclusion list names other accessions than the staged one")
        return out
    return None


def needs_bytes(new: str) -> bool:
    base = PurePosixPath(new).name
    return base.endswith(".ckpt") or PurePosixPath(new).suffix in TEXT_SUFFIXES


def scrub(src: Path, dst: Path, top: dict[str, str]) -> dict:
    ctx: dict = {"freezes": {}, "freeze_sha": {}}
    if src.name == "functional_annotation.tar":     # freezes first: the manifests carry their hashes
        with tarfile.open(src, "r:") as tf:
            for m in tf.getmembers():
                if m.name.endswith("ec_strat_v2_freeze.json"):
                    ctx["freezes"]["ec_cohort_freeze.json"] = rewrite_freeze(tf.extractfile(m).read(), "ec")
                elif m.name.endswith("go_mf_cohort_freeze.json"):
                    ctx["freezes"]["go_mf_cohort_freeze.json"] = rewrite_freeze(tf.extractfile(m).read(), "go")
        ctx["freeze_sha"] = {k: hashlib.sha256(v).hexdigest() for k, v in ctx["freezes"].items()}

    stats = {"in": 0, "out": 0, "dropped": 0, "rewritten": 0}
    part = dst.with_suffix(".tar.part")
    with tarfile.open(src, "r|") as tin, tarfile.open(part, "w", format=tarfile.GNU_FORMAT) as tout:
        for ti in tin:
            stats["in"] += 1
            new = rename_member(ti.name, top)
            parts = PurePosixPath(new).parts
            if parts[-1] == "wandb_run_id.txt" or (
                    ti.isdir() and len(parts) == 5 and parts[1] == "euclidean" and parts[4] == "checkpoints"):
                stats["dropped"] += 1        # offline-log ids; the raw-distance cells' empty checkpoints/
                continue
            if parts[1:2] == ("euclidean",) and "checkpoints" in parts[4:5] and not ti.isdir():
                fail(f"{new}: a file under a raw-distance cell's checkpoints/")
            if blocked(new):
                fail(f"member name still carries {blocked(new)}: {new}")
            if ti.isdir():
                tout.addfile(header(ti, new))
            elif not ti.isfile():
                fail(f"unexpected member type: {ti.name}")
            elif needs_bytes(new):
                raw = tin.extractfile(ti).read()
                out = transform(ti.name, new, raw, ctx)
                if out is not None:
                    stats["rewritten"] += 1
                    raw = out
                if PurePosixPath(new).suffix in TEXT_SUFFIXES:
                    text = raw.decode("utf-8", errors="replace")
                    if blocked(text):
                        fail(f"{new}: blocked string left: {blocked(text)}")
                    if new.endswith((".json",)):
                        json.loads(text)
                    if new.endswith((".yaml", ".yml")):
                        yaml.safe_load(text)
                tout.addfile(header(ti, new, len(raw)), io.BytesIO(raw))
            else:
                tout.addfile(header(ti, new), tin.extractfile(ti))
            stats["out"] += 1
    part.rename(dst)
    return stats


def verify(dst: Path) -> str:
    with tarfile.open(dst, "r:") as tf:
        for m in tf.getmembers():
            if (m.uid, m.gid, m.uname, m.gname) != (0, 0, "", ""):
                fail(f"{dst.name}: {m.name} has owner {m.uname}/{m.gname} {m.uid}/{m.gid}")
    md5 = hashlib.md5()
    tail = b""
    keep = max(len(b) for b in BLOCK_RAW) - 1
    with dst.open("rb") as fh:
        while chunk := fh.read(1 << 26):
            md5.update(chunk)
            window = tail + chunk
            hits = [b.decode() for b in BLOCK_RAW if b in window]
            if hits:
                fail(f"{dst.name}: raw bytes contain {hits}")
            tail = window[-keep:]
    return md5.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", type=Path, default=DSS / "zenodo_stage")
    ap.add_argument("--only", nargs="*", default=None, help="staged archive names to process")
    args = ap.parse_args()
    out_dir = args.stage / "clean"
    out_dir.mkdir(exist_ok=True)
    todo = args.only or list(ARCHIVES)
    for staged in todo:
        published, top = ARCHIVES[staged]
        src, dst = args.stage / staged, out_dir / published
        print(f"{staged} -> clean/{published}", flush=True)
        stats = scrub(src, dst, top)
        md5 = verify(dst)
        print(f"  members {stats['in']} -> {stats['out']} (dropped {stats['dropped']}, "
              f"rewritten {stats['rewritten']}); {dst.stat().st_size:,} bytes; md5 {md5}", flush=True)
        (out_dir / f"{published}.md5").write_text(f"{md5}  {published}\n")
    print("all archives rewritten and checked")


if __name__ == "__main__":
    main()
