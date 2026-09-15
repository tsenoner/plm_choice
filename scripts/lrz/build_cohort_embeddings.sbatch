#!/bin/bash
# Build the uniform <=2000-residue embedding cohort (M-12 / M-CUT2K).
#
#     cd ~/plm_choice && mkdir -p logs
#     sbatch --test-only --array=0-14 scripts/lrz/build_cohort_embeddings.sbatch
#     sbatch            --array=0-14%5 scripts/lrz/build_cohort_embeddings.sbatch
#
# ---------------------------------------------------------------------------
# ONE pass does three jobs at once, because each is a full rewrite anyway:
#   1. CONSOLIDATE  -- 15 arms live in data/zenodo_17469268/ while
#      run_experiments.py:49 globs <cohort>/embeddings/*.h5 and would find 2 files,
#      print "Found 2 embedding files" and report success (blocker B2).
#   2. CUT to <=2000 aa -- the cohort becomes 540,881 of 542,238 proteins. This is
#      already the pipeline's own --max_seq_len default, and esm3_open/esmc_300m/
#      esmc_600m sit at exactly 540,881, i.e. they were built to this rule already.
#   3. REPACK -- `del f[key]` unlinks but does NOT reclaim space, so deleting in place
#      would ship a new Zenodo deposit at the old size with dead blocks inside.
#      Writing a fresh file is the repack.
#
# NOT cutting to ESM-1b's 1022-token cap: that would cost every arm a further 14,010
# proteins to accommodate one model's positional limit. clean/esm1b stay at 526,871
# and the discrepancy is documented (97.41% of the cohort).
#
# Sources are read-only; the pristine Zenodo pull is never modified. esm2_3b is taken
# from the completed working copy (542,187 keys), not the stale deposit (435,298).
# ---------------------------------------------------------------------------
#SBATCH -J plm-cohort-2k
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu                    # MANDATORY: lrz-cpu rejects the default QOS
#SBATCH --time=08:00:00              # MANDATORY: default is 1h
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # >= largest arm (4.6 GB) for the core driver, plus slack
#SBATCH --mail-type=END,FAIL         # never poll squeue in a loop
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
DEPOSIT="$DSS/data/zenodo_17469268"
WORK="$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings"
FASTA="$DSS/plm_choice_data/data/raw/sprot_2024/sprot.fasta"
OUTDIR="$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k"

# The cohort's defining threshold, in ONE place -- it is also the pipeline's own
# --max_seq_len default (embedding_generation.py). Overridable, but changing it
# changes what "the cohort" means, so change it here and nowhere else.
MAX_AA="${MAX_AA:-2000}"

ARMS=(ankh_base ankh_large clean esm1b esm2_150m esm2_35m esm2_3b esm2_650m
      esm2_8m esm3_open esmc_300m esmc_600m prott5 prottucker random_1024)
N=${#ARMS[@]}
IDX=${SLURM_ARRAY_TASK_ID:?set --array}
if [ "$IDX" -ge "$N" ]; then echo "index $IDX out of range (0-$((N-1)))"; exit 2; fi
ARM=${ARMS[$IDX]}

# esm2_3b's completed file is the working copy; every other arm comes from the deposit.
if [ "$ARM" = "esm2_3b" ]; then SRC="$WORK/$ARM.h5"; else SRC="$DEPOSIT/$ARM.h5"; fi

cd "$HOME/plm_choice"
mkdir -p "$OUTDIR"
echo "arm=$ARM  src=$SRC"

./.venv/bin/python - "$ARM" "$SRC" "$FASTA" "$OUTDIR" "$MAX_AA" <<'PY'
import os, sys, h5py
from evaluation.canonical_set import parse_fasta

arm, src, fasta, outdir, max_aa = sys.argv[1:6]
max_aa = int(max_aa)
out = os.path.join(outdir, f"{arm}.h5")
tmp = out + ".partial"

# --- the cohort: every protein of <=MAX_AA residues -----------------------------
# parse_fasta defines the id as the first whitespace token after '>', which is
# EXACTLY the convention the HDF5 keys were written with (embedding_generation.py:605,
# `base_header = header.split()[0]`). A local regex that pulled the bare accession out
# of `sp|P12345|NAME` would disagree with every key in the file, and the job could not
# notice: the expected-count assert below is derived from `keep` too, so 0 == 0 passes
# and the arm ships empty.
keep = {pid for pid, seq in parse_fasta(fasta) if len(seq) <= max_aa}
print(f"cohort (<={max_aa} aa): {len(keep):,}", flush=True)

# --- read with the core driver: one sequential pass, not scattered metadata ------
# Measured 57-67x faster on these superblock-v0 symbol-table groups. --mem=32G is
# well above the largest arm (4.6 GB), so no size guard is needed inside the job.
src_size = os.path.getsize(src)
with h5py.File(src, "r", driver="core", backing_store=False) as fin:
    src_keys = list(fin.keys())
    print(f"source keys: {len(src_keys):,}  ({src_size/1e9:.2f} GB)", flush=True)
    selected = [k for k in src_keys if k in keep]
    print(f"writing {len(selected):,}  (dropping {len(src_keys)-len(selected):,} > {max_aa} aa)",
          flush=True)
    if os.path.exists(tmp):
        os.remove(tmp)
    with h5py.File(tmp, "w") as fout:
        for i, k in enumerate(selected):
            d = fin[k]                      # bind once: two H5Dopen per key cost ~23%
            fout.create_dataset(k, data=d[:], dtype=d.dtype)
            if i and i % 100_000 == 0:
                print(f"  {i:,}/{len(selected):,}", flush=True)

# --- verify BEFORE publishing the name ------------------------------------------
# One open, one traversal: re-opening to re-enumerate is the scattered-metadata read
# this script's core driver exists to avoid (60-100 s per file on GPFS).
with h5py.File(tmp, "r") as f:
    out_keys = set(f)
    first = f[next(iter(out_keys))]
    dtype, shape = first.dtype, first.shape
n_out = len(out_keys)
# `selected` is already `keep & src_keys` (HDF5 names are unique), so this needs no
# second intersection -- and it lets the 542k-name src_keys list be freed earlier.
assert n_out == len(selected), f"{arm}: wrote {n_out}, expected {len(selected)}"
assert not (out_keys - keep), f"{arm}: output contains a >{max_aa} aa protein"
os.replace(tmp, out)
print(f"OK {arm}: {n_out:,} keys, dtype={dtype}, shape={shape} -> {out}")
PY
echo "DONE $ARM"
