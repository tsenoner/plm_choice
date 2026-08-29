#!/bin/bash
# Complete the ESM2-3B embeddings (Track E / E10 · M-9) on LRZ AI Systems.
#
#     cd ~/plm_choice && mkdir -p logs
#     sbatch --test-only scripts/lrz/complete_esm2_3b.sbatch   # validate first
#     sbatch            scripts/lrz/complete_esm2_3b.sbatch
#
# ---------------------------------------------------------------------------
# WHY: esm2_3b.h5 holds 435,298 of the cohort's 542,238 proteins (80.3%).
# Because src/shared/datasets.py:99-105 drops a pair when EITHER protein is
# missing, the loss is quadratic: esm2_3b was trained and scored on 558,947 test
# pairs where its competitors got 872,572 (64.06%), yet is published at rank #10.
#
# It is NOT a length filter and NOT an OOM on long sequences -- measured, the
# present and missing length distributions are indistinguishable:
#     present  n=435,298  median 295  mean 353.3  p95 860  >1024 2.8%  max 13,477
#     missing  n=106,940  median 296  mean 354.2  p95 864  >1024 2.8%  max  5,412
# The longest protein in the cohort was embedded successfully. This was simply an
# interrupted run, so a plain append-mode resume is the correct fix.
# ---------------------------------------------------------------------------
# PARTITION: measured with `sbatch --test-only` on 2026-08-05, all for this job --
#     lrz-dgx-a100-40x8-mig   2026-08-05T15:12   <- chosen (same day)
#     lrz-v100x2              2026-08-08         16 GB: too small for a 3B model in fp32
#     lrz-hgx-h100-94x4       2026-08-18
#     lrz-hgx-a100-80x4       2026-08-19
#     lrz-dgx-a100-80x8       2026-09-14
# Shorter --time did NOT move any estimate, so the queue position -- not backfill --
# is the binding constraint. MIG is 13 days sooner than the next option.
# MIG requires BOTH `--qos=mig` and the typed gres `gpu:3g.20gb:1`; a bare
# `--gres=gpu:1` there fails with "Invalid qos specification".
#SBATCH -J plm-esm2-3b-complete
#SBATCH -p lrz-dgx-a100-40x8-mig
#SBATCH --qos=mig
#SBATCH --gres=gpu:3g.20gb:1         # MANDATORY: without --gres the job sits PD forever (QOSMinGRES)
#SBATCH --time=06:00:00              # MANDATORY: default 1h, QOS caps at 2 days
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL         # never poll squeue in a loop -- that risks a user-ID ban
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
FASTA="$DSS/plm_choice_data/data/raw/sprot_2024/sprot.fasta"
PRISTINE="$DSS/data/zenodo_17469268/esm2_3b.h5"       # the md5-verified Zenodo deposit
WORKDIR="$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings"
OUT="$WORKDIR/esm2_3b.h5"
EXPECTED=542238

cd "$HOME/plm_choice"
mkdir -p "$WORKDIR"

# Never append into the verified Zenodo copy -- work on a duplicate so the
# deposit stays byte-identical to its published md5.
if [ ! -f "$OUT" ]; then
    echo "seeding working copy from the Zenodo deposit (4.6 GB)..."
    cp "$PRISTINE" "$OUT.partial" && mv "$OUT.partial" "$OUT"
fi

echo "=== BEFORE ==="
./.venv/bin/python -c "
import h5py; f=h5py.File('$OUT','r'); print('keys:', len(f.keys())); f.close()"

# --max_seq_len 6000: the DEFAULT IS 2000 and silently skips longer sequences.
#   The existing file contains 1,092 proteins >2000 (max 13,477), so the default
#   would leave the arm incomplete in a new way. The missing set maxes at 5,412,
#   so 6000 covers every one of them.
# --token_budget: length-bucketed batching. Pinned as numerically equivalent to
#   the single-sequence path (max difference 4.8e-07, float32 rounding noise).
# NO --bf16: the existing 435,298 vectors are float32. Mixing precision inside
#   one file would make this arm internally inconsistent -- the exact class of
#   confound the reviewers objected to.
./.venv/bin/python src/data_preparation/embeddings/embedding_generation.py \
    "$FASTA" esm2_3b \
    --output_hdf5_file "$OUT" \
    --weights_dir "$DSS/cache/huggingface" \
    --max_seq_len 6000 \
    --token_budget 8192
# token_budget 8192, not 16384: the MIG slice is 20 GB and ESM2-3B's fp32 weights
# alone are ~11 GB. Hidden states cost ~10 MB per 1k tokens per live layer at
# d=2560, so 8192 keeps a comfortable margin. If this OOMs, halve it -- MIG
# starts same-day, so a retry is cheap; a larger partition is 13 days away.

echo "=== AFTER ==="
# Do NOT trust exit 0: the append writer's skip branch counts a skip as a
# success (embedding_generation.py:965), so a misconfigured resume writes
# nothing, logs full success and exits 0. Assert the key count instead.
./.venv/bin/python -c "
import h5py, sys
f=h5py.File('$OUT','r'); n=len(f.keys()); f.close()
print('keys:', n, 'expected:', $EXPECTED)
sys.exit(0 if n == $EXPECTED else 1)"
echo 'esm2_3b COMPLETE'
