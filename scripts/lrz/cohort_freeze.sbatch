#!/bin/bash
# Derive the shared-cohort exclusion freeze from the <=2000-residue embedding arms (M-12).
#
# PROVENANCE, 2026-09-21: this job has never been submitted -- `sacct` has no record of it
# on any window. The freeze it describes DOES exist and is committed
# (freeze/embedding_excluded_proteins.json): it was produced by running the underlying
# Python directly on the login node, which is affordable because the work is an HDF5 key
# walk rather than compute. So this file is a recipe that has never been executed as
# written. Run it with `sbatch --test-only` before trusting it, and do not read a green
# job here as confirmation that the deposited freeze came from it.
#
#     cd ~/plm_choice && mkdir -p logs
#     sbatch --test-only scripts/lrz/cohort_freeze.sbatch
#     sbatch            scripts/lrz/cohort_freeze.sbatch
#
# ---------------------------------------------------------------------------
# WHY THIS RUNS AT ALL. `freeze/embedding_key_coverage_cohort2k.json` says of its own
# counts: "these counts are what that job will produce" -- they are a PROJECTION of the
# pre-cut deposit through the <=2000 rule, never a measurement of the directory
# build_cohort_embeddings.sbatch actually wrote. This job is the first measurement of
# the physical cohort, and it materialises the id list the projection only counted.
#
# WHY sbatch AND NOT THE LOGIN NODE. The scan itself is cheap (~50 s measured: the
# default driver reads esm2_3b.h5, 5.75 GB, in 11.5 s). Memory is the reason. Fifteen
# key sets accumulate at ~42 MB each while one arm is held in RAM by the core driver,
# and the two 3.53 GB arms put the peak at ~4.1 GB against the login node's 4 GiB
# per-user cgroup -- a ~4% margin on a limit SHARED with every other process this
# account has on that node. --mem=32G makes it a non-question, and lrz_setup.md bars
# heavy login-node work on policy grounds regardless.
#
# WHAT IT WRITES, BESIDES THE FREEZE. load_h5_keysets caches each arm's key list as a
# `<stem>.keys.txt` sidecar next to the .h5 (h5_keys.py:71-74) -- ~3.8 MB x 15 ~= 57 MB
# of new files in the DSS embeddings directory, invalidated on (size, mtime). That is
# deliberate: it makes every later coverage query instant.
#
# THE FREEZE DOES NOT STAY HERE. It is written to a scratch path under $HOME, verified,
# and then copied to the workstation to be committed. Writing straight into
# ~/plm_choice/freeze/ would leave a cluster-only artifact that the repo never sees, and
# a FAILED verification would leave a wrong freeze on disk that the next run then
# refuses to overwrite (write_exclusion_freeze:217).
# ---------------------------------------------------------------------------
#SBATCH -J plm-cohort-freeze
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu                    # MANDATORY: lrz-cpu rejects the default QOS
#SBATCH --time=00:30:00              # MANDATORY: default is 1h; the scan is ~50 s
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G                    # >= largest arm (5.75 GB) + 15 key sets, plus slack
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
H5DIR="$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k"
OUT="$HOME/cohort_freeze_out/embedding_excluded_proteins.json"

# The three numbers this job must reproduce, and where they come from.
#   540,881 = Swiss-Prot proteins of <=2000 aa (the cohort2k cut)
#   526,871 = Swiss-Prot proteins of <=1022 aa (ESM-1b's positional cap; CLEAN inherits it)
#    14,010 = the difference, i.e. EXACTLY the proteins of 1023-2000 residues
# The hash was derived independently on the workstation from sprot.fasta.fai alone,
# BEFORE this job ran. If the h5-derived exclusion reproduces it, two independent
# constructions of the cohort agree and the projection is confirmed by measurement.
EXPECT_N=14010
EXPECT_UNIVERSE=540881
EXPECT_INTERSECTION=526871
EXPECT_SHA=b9fba30708b2074129ff4f2c1b08e4d9f65c07de0ac3745e6e6351cbfbecf10e

cd "$HOME/plm_choice"
mkdir -p "$(dirname "$OUT")"
# Idempotent reruns: the scratch copy is ours, so clearing it is safe and keeps
# write_exclusion_freeze's no-clobber rule meaningful for the repo path it protects.
rm -f "$OUT"

echo "h5-dir: $H5DIR"
ls -la "$H5DIR"/*.h5 | wc -l
echo "--- scanning ---"
/usr/bin/time -v ./.venv/bin/python -m plm_choice data cohort-freeze \
    --h5-dir "$H5DIR" --out "$OUT" 2>&1 | grep -Ev '^\s+(Signals|Voluntary|Involuntary|Swaps|File system|Socket|Page size|Average)'

echo "--- verifying ---"
./.venv/bin/python - "$OUT" "$EXPECT_N" "$EXPECT_UNIVERSE" "$EXPECT_INTERSECTION" "$EXPECT_SHA" <<'PY'
import json, sys
from shared.protein_cohort import verify_exclusion, content_hash

path, n, universe, intersection, sha = sys.argv[1], *map(int, sys.argv[2:5]), sys.argv[5]
blob = json.loads(open(path).read())

# verify_exclusion re-derives the hash and the counts from excluded_ids alone, so it
# catches a manifest that disagrees with its own id list. It cannot catch a manifest
# that is internally consistent but describes the WRONG directory -- that is what the
# four expectations below are for.
verify_exclusion(blob)

fail = []
for label, got, want in (
    ("n_excluded", blob["n_excluded"], n),
    ("universe", blob["universe"], universe),
    ("intersection_all", blob["intersection_all"], intersection),
):
    print(f"  {label:18s} {got:>9,}  expected {want:>9,}  {'OK' if got == want else 'MISMATCH'}")
    if got != want:
        fail.append(label)

got_sha = content_hash(blob["excluded_ids"])
ok = got_sha == sha
print(f"  content_sha256     {got_sha}\n  expected           {sha}  {'OK' if ok else 'MISMATCH'}")
if not ok:
    fail.append("content_sha256")

print(f"  arms scanned       {len(blob['arms'])}: {', '.join(blob['arms'])}")
short = {a: c for a, c in blob["counts"].items() if c != blob["universe"]}
print(f"  short arms         {short or 'none'}")

if fail:
    # Delete the bad freeze rather than leaving it to be copied or to block a rerun.
    import os
    os.remove(path)
    sys.exit(f"FAILED on {', '.join(fail)} -- bad freeze removed, nothing to copy")
print("all four expectations met")
PY

echo "--- done ---"
ls -la "$OUT"
echo "Copy it back with:"
echo "  scp ai:$OUT freeze/embedding_excluded_proteins.json"
